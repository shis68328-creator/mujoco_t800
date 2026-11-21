"""09cuda.py —— 09.py 的 Ubuntu + NVIDIA（RTX 5070 Ti）加速版。

和 09.py 的区别：
  1. 用 SubprocVecEnv 多进程并行采样（默认 16 个环境），把 CPU 多核吃满
  2. 默认 --device cuda，PPO 的网络更新放到 5070 Ti 上
  3. 可选用 TensorBoard 记录训练曲线（--tensorboard）
  4. 每个环境用 MuJoCo 自己算物理，仍是 CPU；GPU 主要加速网络部分

运行（Ubuntu，先 conda activate t800）：
    # 先冒烟测试，确认 16 个进程能起来
    python 09cuda.py --smoke

    # 正式训练（10M 步；envs 建议设成 CPU 物理核心数左右）
    python 09cuda.py --steps 10000000 --envs 16 --device cuda --tensorboard

    # 断点续训
    python 09cuda.py --steps 30000000 --envs 16 --device cuda --resume

    # 回放训练好的策略（需要在有显示器的 Ubuntu 上跑）
    python 09cuda.py --play

注意：
  - 从零训练人形走路很慢，这个脚本负责把训练吞吐拉满，但能否走起来
    还取决于奖励设计、训练步数和调参。
  - --demo（加载 SDK 预训练策略）仍然在 09.py 里，用 09.py --demo 运行。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

# T800 模型路径（和 09.py 相同）
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "engineai_robotics_native_sdk-main/assets/resource/robot/t800/xml/serial_t800.xml"
)

# 12 个腿关节，按这个顺序作为动作/观测的排列
LEG_JOINTS = [
    "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R", "J07_HIP_ROLL_R", "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
]

PHYSICS_DT = 0.002
FRAME_SKIP = 10          # 50 Hz 控制
CTRL_DT = PHYSICS_DT * FRAME_SKIP
EPISODE_STEPS = 750      # 15 秒
TARGET_SPEED = 0.8
ACTION_SCALE = 0.35
KP = 1000.0
KD = 60.0
PHASE_FREQ = 1.5

# 父进程不建模型；每个子进程会各自建一份（fork 后互不影响）
_MODEL: mujoco.MjModel | None = None


def get_model() -> mujoco.MjModel:
    """加载 T800 模型并补地面（每个进程各一份）。"""
    global _MODEL
    if _MODEL is None:
        spec = mujoco.MjSpec.from_file(str(MODEL_PATH))
        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [20.0, 20.0, 0.1]
        floor.rgba = [0.45, 0.48, 0.52, 1.0]
        model = spec.compile()
        model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
        model.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]
        _MODEL = model
    return _MODEL


class T800WalkEnv(gym.Env):
    """T800 行走环境（动作：12 个腿关节目标角偏移）。"""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.model = get_model()
        self.data = mujoco.MjData(self.model)

        self.act_joint = self.model.actuator_trnid[:, 0]
        self.act_qadr = self.model.jnt_qposadr[self.act_joint]
        self.act_vadr = self.model.jnt_dofadr[self.act_joint]
        self.default_q = self.model.qpos0[self.act_qadr].copy()
        self.tau_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.tau_high = self.model.actuator_ctrlrange[:, 1].copy()

        self.leg_act_idx = []
        self.leg_qadr = []
        self.leg_vadr = []
        for name in LEG_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            act = int(np.where(self.act_joint == jid)[0][0])
            self.leg_act_idx.append(act)
            self.leg_qadr.append(int(self.model.jnt_qposadr[jid]))
            self.leg_vadr.append(int(self.model.jnt_dofadr[jid]))
        self.leg_act_idx = np.array(self.leg_act_idx, dtype=int)
        self.leg_qadr = np.array(self.leg_qadr, dtype=int)
        self.leg_vadr = np.array(self.leg_vadr, dtype=int)
        self.leg_default_q = self.model.qpos0[self.leg_qadr].copy()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        obs_dim = 1 + 3 + 3 + 3 + 12 + 12 + 12 + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self.step_count = 0
        self.prev_action = np.zeros(12, dtype=np.float64)
        self.phase = 0.0
        self.mat = np.zeros(9)

    def _get_obs(self) -> np.ndarray:
        data = self.data
        mujoco.mju_quat2Mat(self.mat, data.qpos[3:7])
        rot = self.mat.reshape(3, 3)
        v_body = rot.T @ data.qvel[0:3]
        w_body = data.qvel[3:6]
        gravity_body = -rot[2, :]

        q_leg = data.qpos[self.leg_qadr] - self.leg_default_q
        qd_leg = data.qvel[self.leg_vadr]
        height = data.qpos[2]

        angle = 2.0 * math.pi * self.phase
        clock = np.array([math.sin(angle), math.cos(angle)])

        obs = np.concatenate([
            [height], gravity_body, v_body, w_body,
            q_leg, qd_leg, self.prev_action, clock,
        ])
        return obs.astype(np.float32)

    def _compute_reward(self, action: np.ndarray, tau: np.ndarray) -> float:
        data = self.data
        rot = self.mat.reshape(3, 3)
        v_body = rot.T @ data.qvel[0:3]
        up_z = float(rot[2, 2])
        z = float(data.qpos[2])

        vx, vy = float(v_body[0]), float(v_body[1])
        wz = float(data.qvel[5])

        r_forward = 2.0 * float(np.clip(vx, 0.0, 1.5 * TARGET_SPEED))
        r_lateral = 0.2 * math.exp(-4.0 * vy ** 2)
        r_upright = 0.4 * max(0.0, up_z)
        r_height = 0.2 * math.exp(-8.0 * (z - 1.0) ** 2)
        r_alive = 0.1

        r_torque = 1e-6 * float(np.sum(tau ** 2))
        r_action_rate = 0.02 * float(np.sum((action - self.prev_action) ** 2))
        r_joint_vel = 1e-4 * float(np.sum(data.qvel[self.leg_vadr] ** 2))
        r_yaw = 0.05 * wz ** 2

        return (r_forward + r_lateral + r_upright + r_height + r_alive
                - r_torque - r_action_rate - r_joint_vel - r_yaw)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        target = self.default_q.copy()
        target[self.leg_act_idx] += action * ACTION_SCALE

        tau = np.zeros(self.model.nu)
        for _ in range(FRAME_SKIP):
            q = self.data.qpos[self.act_qadr]
            qd = self.data.qvel[self.act_vadr]
            tau = KP * (target - q) - KD * qd
            tau = np.clip(tau, self.tau_low, self.tau_high)
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        self.phase = (self.phase + CTRL_DT * PHASE_FREQ) % 1.0

        obs = self._get_obs()
        reward = self._compute_reward(action, tau)

        mujoco.mju_quat2Mat(self.mat, self.data.qpos[3:7])
        rot = self.mat.reshape(3, 3)
        up_z = float(rot[2, 2])
        z = float(self.data.qpos[2])
        y = float(self.data.qpos[1])

        terminated = bool(
            z < 0.65 or up_z < 0.5 or abs(y) > 1.2
            or not np.isfinite(obs).all()
        )
        truncated = self.step_count >= EPISODE_STEPS

        self.prev_action = action
        return obs, float(reward), terminated, truncated, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        qpos = self.model.qpos0.copy()
        qpos[0] = 0.0
        qpos[1] = 0.0
        qpos[2] = 1.03
        qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        qpos[self.leg_qadr] += self.np_random.uniform(-0.05, 0.05, size=12)

        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.phase = 0.0
        self.prev_action = np.zeros(12, dtype=np.float64)
        return self._get_obs(), {}


class EnvFactory:
    """可被多进程 pickle 的环境工厂（SubprocVecEnv 需要）。"""

    def __init__(self, rank: int, seed: int):
        self.rank = rank
        self.seed = seed

    def __call__(self):
        env = Monitor(T800WalkEnv())
        env.reset(seed=self.seed)
        env.action_space.seed(self.seed)
        return env


class SaveCallback(BaseCallback):
    """按“总环境步数”保存模型和 VecNormalize 统计量。"""

    def __init__(self, save_dir: Path, save_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.next_save = save_freq

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_save:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(self.save_dir / "ppo_t800_walk")
            vec_env = self.model.get_env()
            if isinstance(vec_env, VecNormalize):
                vec_env.save(self.save_dir / "vecnormalize.pkl")
            print(f"[保存] {self.num_timesteps} 步 -> {self.save_dir}", flush=True)
            self.next_save += self.save_freq
        return True


def build_vec_env(args) -> VecNormalize:
    """按参数创建并行环境，并包上观测/奖励归一化。"""
    factories = [EnvFactory(i, args.seed + i) for i in range(args.envs)]

    if args.vec_env == "subproc":
        # Linux 下用 fork 最快；macOS 用 spawn 避免 fork 问题
        start_method = "fork" if sys.platform.startswith("linux") else "spawn"
        vec_env = SubprocVecEnv(factories, start_method=start_method)
    else:
        vec_env = DummyVecEnv(factories)

    return VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)


def train(args) -> None:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 环境先创建（fork 发生在 CUDA 初始化之前，避免 CUDA + fork 的坑）
    vec_env = build_vec_env(args)

    tb_log = str(save_dir / "tb") if args.tensorboard else None

    if args.resume and (save_dir / "ppo_t800_walk.zip").exists():
        vec_env = VecNormalize.load(save_dir / "vecnormalize.pkl", vec_env)
        model = PPO.load(
            save_dir / "ppo_t800_walk", env=vec_env,
            device=args.device, tensorboard_log=tb_log,
        )
        print(f"[续训] 从 {save_dir / 'ppo_t800_walk.zip'} 继续", flush=True)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.003,
            vf_coef=0.5,
            max_grad_norm=1.0,
            policy_kwargs=dict(
                net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
                activation_fn=torch.nn.Tanh,
                log_std_init=0.0,
            ),
            device=args.device,
            verbose=1,
            seed=args.seed,
            tensorboard_log=tb_log,
        )

    total = args.steps
    print(f"[训练] 环境={args.envs}（{args.vec_env}），总步数={total}，设备={args.device}", flush=True)
    print(f"[训练] n_steps={args.n_steps}，batch_size={args.batch_size}，"
          f"TensorBoard={bool(tb_log)}", flush=True)

    callback = SaveCallback(save_dir, save_freq=args.save_freq)
    t0 = time.time()
    model.learn(total_timesteps=total, callback=callback, progress_bar=False)
    elapsed = time.time() - t0

    model.save(save_dir / "ppo_t800_walk")
    vec_env.save(save_dir / "vecnormalize.pkl")
    vec_env.close()

    print(f"[完成] 用时 {elapsed:.1f}s，平均 {total / max(elapsed, 1e-9):.0f} 步/秒", flush=True)
    print(f"[完成] 模型保存到 {save_dir}", flush=True)
    if tb_log:
        print(f"[完成] TensorBoard：tensorboard --logdir {tb_log}", flush=True)


def play(args) -> None:
    """加载训练好的模型，在 viewer 里走。"""
    import mujoco.viewer

    save_dir = Path(args.save_dir)
    model_file = save_dir / "ppo_t800_walk.zip"
    norm_file = save_dir / "vecnormalize.pkl"
    if not model_file.exists() or not norm_file.exists():
        print(f"找不到模型：{model_file} 或 {norm_file}")
        return

    vec_env = DummyVecEnv([EnvFactory(0, args.seed)])
    vec_env = VecNormalize.load(norm_file, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(model_file, device=args.device)
    obs = vec_env.reset()
    base_env = vec_env.envs[0].unwrapped  # 去掉 Monitor 包装，拿 model/data

    try:
        with mujoco.viewer.launch_passive(base_env.model, base_env.data) as viewer:
            viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -10.0
            viewer.cam.azimuth = 135.0
            print("策略已加载，关闭窗口退出。", flush=True)

            next_time = time.time()
            while viewer.is_running():
                action, _ = model.predict(obs, deterministic=True)
                obs, _, _, _ = vec_env.step(action)
                viewer.sync()
                next_time += CTRL_DT
                lag = next_time - time.time()
                if lag > 0:
                    time.sleep(min(lag, 0.01))
    except Exception as exc:
        print(f"无法打开 3D 窗口：{exc}")
        print("Linux 需要图形界面；远程无显示器时请用 ssh -X 或在 Ubuntu 本机运行。")


def main() -> int:
    parser = argparse.ArgumentParser(description="T800 行走 PPO 训练（CUDA + 多进程）")
    parser.add_argument("--steps", type=int, default=10_000_000, help="总训练步数")
    parser.add_argument("--envs", type=int, default=16, help="并行环境数")
    parser.add_argument("--vec-env", choices=["subproc", "dummy"], default="subproc",
                        help="并行方式：subproc 多进程（默认）或 dummy 单进程")
    parser.add_argument("--n-steps", type=int, default=512,
                        help="每个环境每次 rollout 的步数")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="PPO 小批量大小")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")
    parser.add_argument("--save-dir", type=str, default="t800_walk_model_cuda")
    parser.add_argument("--save-freq", type=int, default=500_000,
                        help="每多少环境步保存一次")
    parser.add_argument("--resume", action="store_true", help="断点续训")
    parser.add_argument("--smoke", action="store_true", help="冒烟测试：2 万步")
    parser.add_argument("--tensorboard", action="store_true", help="开启 TensorBoard 日志")
    parser.add_argument("--play", action="store_true", help="回放训练好的策略")
    args = parser.parse_args()

    # 没有 CUDA 就自动退回 CPU，避免直接报错
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[警告] 没有检测到 CUDA，自动改用 CPU", flush=True)
        args.device = "cpu"
    if args.device == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}", flush=True)

    if args.smoke:
        args.steps = 20_000
        args.save_dir = args.save_dir + "_smoke"

    if args.play:
        play(args)
        return 0

    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
