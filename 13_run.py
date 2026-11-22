"""13_run.py —— 训练 T800 跑步（PPO + 跑步奖励 + 速度课程）。

和 09cuda.py（走路）的区别：
  1. 奖励改成“跑步”：
       - 速度跟踪（目标速度更高）
       - 双脚同时离地（腾空相）给奖励
       - 高速时双脚同时着地（双支撑）扣分
       - 抬脚、屈膝、摆臂幅度更大
  2. 速度课程：目标速度从 --target-start 开始，每 --curriculum-every 步
     加 0.2 m/s，逐步升到 --target-max
  3. 可以从之前的走路 PPO 模型热启动：
       --init-dir t800_walk_model_v2
     同时加载模型和 VecNormalize 统计量

注意：
  跑步比走路难得多，从零训练很可能需要几千万步甚至更久；
  这个脚本负责把奖励、课程和吞吐搭好，不保证几百万步就能跑起来。

用法（Ubuntu）：
    # 冒烟测试
    python 13_run.py --smoke --envs 8 --device cuda

    # 正式训练（先 500 万步看看）
    python 13_run.py --steps 5000000 --envs 16 --device cuda --tensorboard

    # 从走路模型热启动
    python 13_run.py --steps 5000000 --envs 16 --device cuda \
        --init-dir t800_walk_model_v2

    # 回放
    python 13_run.py --play --play-speed 2.0
"""

import argparse
import math
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

ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    ROOT
    / "engineai_robotics_native_sdk-main/assets/resource/robot/t800/xml/serial_t800.xml"
)

LEG_JOINTS = [
    "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R", "J07_HIP_ROLL_R", "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
]

PHYSICS_DT = 0.002
FRAME_SKIP = 10
CTRL_DT = PHYSICS_DT * FRAME_SKIP    # 50 Hz
EPISODE_STEPS = 500                  # 10 秒
ACTION_SCALE = 0.45                  # 跑步步幅比走路大
KP = 1000.0
KD = 60.0
PHASE_FREQ = 2.0                     # 跑步步频比走路快

FOOT_BODIES = ("LINK_FOOT_L", "LINK_FOOT_R")
FOOT_CONTACT_Z = 0.095
ARM_SWING = 0.35
ELBOW_BEND = -0.3

TARGET_START = 1.0
TARGET_MAX = 3.0

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        spec = mujoco.MjSpec.from_file(str(MODEL_PATH))
        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [40.0, 40.0, 0.1]
        floor.rgba = [0.45, 0.48, 0.52, 1.0]
        model = spec.compile()
        model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
        model.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]
        _MODEL = model
    return _MODEL


class T800RunEnv(gym.Env):
    """T800 跑步环境：12 个腿关节动作，奖励跑步速度/腾空/抬脚。"""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.model = get_model()
        self.data = mujoco.MjData(self.model)

        self.act_joint = self.model.actuator_trnid[:, 0]
        self.act_qadr = self.model.jnt_qposadr[self.act_joint]
        self.act_vadr = self.model.jnt_dofadr[self.act_joint]
        self.default_q = self.model.qpos0[self.act_qadr].copy()
        self.tau_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.tau_high = self.model.actuator_ctrlrange[:, 1].copy()

        self.leg_act_idx, self.leg_qadr, self.leg_vadr = [], [], []
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

        self.foot_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in FOOT_BODIES
        ]
        self.prev_foot_pos = None

        self.knee_qadr = np.array([
            int(self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ])
            for name in ("J03_KNEE_PITCH_L", "J09_KNEE_PITCH_R")
        ])

        def act_id(joint_name):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            return int(np.where(self.act_joint == jid)[0][0])

        self.arm_sh_l = act_id("J13_SHOULDER_PITCH_L")
        self.arm_sh_r = act_id("J18_SHOULDER_PITCH_R")
        self.arm_elbow_l = act_id("J16_ELBOW_PITCH_L")
        self.arm_elbow_r = act_id("J21_ELBOW_PITCH_R")

        self.action_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        obs_dim = 1 + 3 + 3 + 3 + 12 + 12 + 12 + 2 + 1  # 49
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self.step_count = 0
        self.prev_action = np.zeros(12, dtype=np.float64)
        self.phase = 0.0
        self.target_speed = TARGET_START
        self.mat = np.zeros(9)

    def set_target_speed(self, speed):
        """课程回调通过这个方法调整目标速度。"""
        self.target_speed = float(speed)

    def _get_obs(self):
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

        return np.concatenate([
            [height], gravity_body, v_body, w_body,
            q_leg, qd_leg, self.prev_action, clock,
            [self.target_speed],
        ]).astype(np.float32)

    def _compute_reward(self, action, tau, foot_pos, foot_vel):
        data = self.data
        rot = self.mat.reshape(3, 3)
        v_body = rot.T @ data.qvel[0:3]
        up_z = float(rot[2, 2])
        z = float(data.qpos[2])

        vx, vy = float(v_body[0]), float(v_body[1])
        wz = float(data.qvel[5])
        target = self.target_speed

        # 速度跟踪：跑步目标速度更高，std 也放大一点
        r_track = 3.0 * math.exp(-((vx - target) ** 2) / 0.5)
        r_lateral = 0.2 * math.exp(-4.0 * vy ** 2)
        r_upright = 0.2 * max(0.0, up_z)
        r_height = 0.1 * math.exp(-8.0 * (z - 1.0) ** 2)
        r_alive = 0.1

        foot_z = foot_pos[:, 2]
        foot_speed_xy = np.linalg.norm(foot_vel[:, :2], axis=1)
        contact = foot_z < FOOT_CONTACT_Z
        airborne = ~contact
        num_contact = int(np.sum(contact))

        # 跑步必须有腾空相：双脚离地给奖励
        r_flight = 0.6 if (num_contact == 0 and vx > 0.5) else 0.0
        # 高速时还双脚着地（双支撑）说明在走，扣分
        r_double_pen = -0.3 if (num_contact == 2 and target > 1.2) else 0.0

        # 单脚支撑时奖励抬脚和屈膝
        single = num_contact == 1
        lift = np.clip((foot_z - 0.09) / 0.08, 0.0, 1.0)
        r_lift = 0.2 * float(np.sum(lift)) if single else 0.0
        knee = data.qpos[self.knee_qadr]
        knee_bend = np.clip((knee - 0.15) / 0.25, 0.0, 1.0)
        r_knee = 0.2 * float(np.sum(knee_bend)) if single else 0.0

        # 支撑脚滑动惩罚
        r_slide = -0.5 * float(np.sum(foot_speed_xy[contact] ** 2))
        # 垂直弹跳不要太大（跑步有腾空，但不要原地蹦）
        r_vertical = -0.2 * float(data.qvel[2] ** 2)

        r_torque = 1e-6 * float(np.sum(tau ** 2))
        r_action_rate = 0.02 * float(np.sum((action - self.prev_action) ** 2))
        r_joint_vel = 1e-4 * float(np.sum(data.qvel[self.leg_vadr] ** 2))
        r_yaw = 0.05 * wz ** 2

        return (r_track + r_lateral + r_upright + r_height + r_alive
                + r_flight + r_double_pen + r_lift + r_knee + r_slide
                + r_vertical
                - r_torque - r_action_rate - r_joint_vel - r_yaw)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        target = self.default_q.copy()
        target[self.leg_act_idx] += action * ACTION_SCALE

        # 摆臂幅度比走路大，频率也更快
        swing = math.sin(2.0 * math.pi * self.phase)
        target[self.arm_sh_l] += ARM_SWING * swing
        target[self.arm_sh_r] -= ARM_SWING * swing
        target[self.arm_elbow_l] = ELBOW_BEND
        target[self.arm_elbow_r] = ELBOW_BEND

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

        foot_pos = self.data.xpos[self.foot_ids].copy()
        if self.prev_foot_pos is None:
            foot_vel = np.zeros((2, 3))
        else:
            foot_vel = (foot_pos - self.prev_foot_pos) / CTRL_DT
        self.prev_foot_pos = foot_pos

        obs = self._get_obs()
        reward = self._compute_reward(action, tau, foot_pos, foot_vel)

        mujoco.mju_quat2Mat(self.mat, self.data.qpos[3:7])
        rot = self.mat.reshape(3, 3)
        up_z = float(rot[2, 2])
        z = float(self.data.qpos[2])
        y = float(self.data.qpos[1])

        terminated = bool(
            z < 0.5 or up_z < 0.45 or abs(y) > 1.5
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
        self.prev_foot_pos = self.data.xpos[self.foot_ids].copy()
        return self._get_obs(), {}


class EnvFactory:
    def __init__(self, rank, seed, target_speed):
        self.rank = rank
        self.seed = seed
        self.target_speed = target_speed

    def __call__(self):
        env = Monitor(T800RunEnv())
        env.reset(seed=self.seed)
        env.action_space.seed(self.seed)
        env.unwrapped.set_target_speed(self.target_speed)
        return env


class SaveCallback(BaseCallback):
    def __init__(self, save_dir, save_freq, verbose=0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.next_save = save_freq

    def _on_step(self):
        if self.num_timesteps >= self.next_save:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(self.save_dir / "ppo_t800_run")
            vec_env = self.model.get_env()
            if isinstance(vec_env, VecNormalize):
                vec_env.save(self.save_dir / "vecnormalize.pkl")
            print(f"[保存] {self.num_timesteps} 步 -> {self.save_dir}", flush=True)
            self.next_save += self.save_freq
        return True


class CurriculumCallback(BaseCallback):
    """每训练一定步数，把所有环境的目标速度提高 0.2 m/s。"""

    def __init__(self, start, max_speed, step, every, verbose=1):
        super().__init__(verbose)
        self.target = start
        self.max_speed = max_speed
        self.step_size = step
        self.every = every
        self.next_update = every

    def _on_step(self):
        if self.num_timesteps >= self.next_update and self.target < self.max_speed:
            self.target = min(self.max_speed, self.target + self.step_size)
            self.training_env.env_method("set_target_speed", self.target)
            print(f"[课程] 目标速度 -> {self.target:.2f} m/s "
                  f"(t={self.num_timesteps} 步)", flush=True)
            self.next_update += self.every
        return True


def build_vec_env(args):
    factories = [EnvFactory(i, args.seed + i, args.target_start)
                 for i in range(args.envs)]
    if args.vec_env == "subproc":
        start_method = "fork" if sys.platform.startswith("linux") else "spawn"
        vec_env = SubprocVecEnv(factories, start_method=start_method)
    else:
        vec_env = DummyVecEnv(factories)
    return VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)


def train(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    vec_env = build_vec_env(args)

    tb_log = str(save_dir / "tb") if args.tensorboard else None

    # 热启动：从走路模型同时加载网络和观测归一化统计
    if args.init_dir:
        init_dir = Path(args.init_dir)
        model_file = init_dir / "ppo_t800_walk.zip"
        norm_file = init_dir / "vecnormalize.pkl"
        if model_file.exists() and norm_file.exists():
            vec_env = VecNormalize.load(norm_file, vec_env)
            model = PPO.load(model_file, env=vec_env, device=args.device,
                             tensorboard_log=tb_log)
            print(f"[热启动] 从走路模型 {init_dir} 初始化", flush=True)
        else:
            print(f"[警告] 找不到 {model_file} 或 {norm_file}，改为从头训练")
            args.init_dir = None

    if not args.init_dir:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=5e-4,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=1.0,
            policy_kwargs=dict(
                net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
                activation_fn=torch.nn.ELU,
                log_std_init=0.0,
            ),
            device=args.device,
            verbose=1,
            seed=args.seed,
            tensorboard_log=tb_log,
        )

    print(f"[训练] 环境={args.envs}，总步数={args.steps}，设备={args.device}", flush=True)
    print(f"[课程] {args.target_start:.1f} -> {args.target_max:.1f} m/s，"
          f"每 {args.curriculum_every} 步 +0.2", flush=True)

    callbacks = [SaveCallback(save_dir, args.save_freq)]
    if args.curriculum:
        callbacks.append(
            CurriculumCallback(args.target_start, args.target_max, 0.2,
                               args.curriculum_every)
        )

    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=False)
    elapsed = time.time() - t0

    model.save(save_dir / "ppo_t800_run")
    vec_env.save(save_dir / "vecnormalize.pkl")
    vec_env.close()
    print(f"[完成] 用时 {elapsed:.1f}s，平均 {args.steps / max(elapsed, 1e-9):.0f} 步/秒", flush=True)


def play(args):
    import mujoco.viewer

    save_dir = Path(args.save_dir)
    model_file = save_dir / "ppo_t800_run.zip"
    norm_file = save_dir / "vecnormalize.pkl"
    if not model_file.exists() or not norm_file.exists():
        print(f"找不到模型：{model_file} 或 {norm_file}")
        return

    env = T800RunEnv()
    env.set_target_speed(args.play_speed)
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize.load(norm_file, venv)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(model_file, device=args.device)
    obs = venv.reset()
    base_env = venv.envs[0].unwrapped

    try:
        with mujoco.viewer.launch_passive(base_env.model, base_env.data) as viewer:
            viewer.cam.distance = 3.5
            viewer.cam.elevation = -10.0
            viewer.cam.azimuth = 135.0
            print(f"跑步策略已加载，目标速度 {args.play_speed:.1f} m/s；关闭窗口退出。")
            next_time = time.time()
            while viewer.is_running():
                action, _ = model.predict(obs, deterministic=True)
                obs, _, _, _ = venv.step(action)
                viewer.cam.lookat[:] = [base_env.data.qpos[0], 0.0, 1.0]
                viewer.sync()
                next_time += CTRL_DT
                lag = next_time - time.time()
                if lag > 0:
                    time.sleep(min(lag, 0.01))
    except Exception as exc:
        print(f"无法打开 3D 窗口：{exc}")
        print("没有显示器时请用 --headless 训练，或 ssh -X。")


def main():
    parser = argparse.ArgumentParser(description="T800 跑步 PPO 训练")
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--vec-env", choices=["subproc", "dummy"], default="subproc")
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default="t800_run_model")
    parser.add_argument("--save-freq", type=int, default=500_000)
    parser.add_argument("--target-start", type=float, default=TARGET_START)
    parser.add_argument("--target-max", type=float, default=TARGET_MAX)
    parser.add_argument("--curriculum", action="store_true", default=True)
    parser.add_argument("--no-curriculum", dest="curriculum", action="store_false")
    parser.add_argument("--curriculum-every", type=int, default=300_000)
    parser.add_argument("--init-dir", type=str, default="",
                        help="走路模型目录（同时加载 ppo_t800_walk.zip 和 vecnormalize.pkl）")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--play-speed", type=float, default=2.0)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[警告] 没有 CUDA，改用 CPU")
        args.device = "cpu"

    if args.smoke:
        args.steps = 20_000
        args.save_dir += "_smoke"

    if args.play:
        play(args)
        return 0

    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
