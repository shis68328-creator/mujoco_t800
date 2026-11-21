"""09.py —— 用强化学习（PPO）训练 T800 在 MuJoCo 里走路。

整体流程：
  1. 写一个 Gymnasium 环境 T800WalkEnv：
       - 动作：12 条腿的关节目标角偏移（PD 控制跟踪）
       - 观测：高度、姿态、速度、12 个腿关节角/角速度、上一步动作、步态相位
       - 奖励：前进速度、保持直立、保持高度、存活，减去力矩/动作变化/关节速度惩罚
       - 失败条件：倾斜过大或高度过低
  2. 用 Stable-Baselines3 的 PPO 训练，VecNormalize 做观测/奖励归一化
  3. 训练完保存模型，可以用 --play 在 MuJoCo viewer 里回放

运行：
    conda activate mujoco
    cd /Users/akihisa/mujoco

    # 1) 先快速冒烟测试，确认代码能跑（约 2 万步）
    python 09.py --smoke

    # 2) 正式训练（时间较长，建议先跑 100 万步看看）
    python 09.py --steps 1000000 --envs 8

    # 3) 断点续训
    python 09.py --steps 3000000 --resume

    # 4) 用训练好的策略在 3D 窗口里走（macOS 必须用 mjpython）
    mjpython 09.py --play

说明：
  从零训练人形机器人走路需要大量时间和调参，这个脚本给出的是完整、
  可运行的训练流程；真正“走稳”通常需要几百万到几千万步，根据机器
  性能可能要跑几十分钟到几小时。
"""

from __future__ import annotations

import argparse
import math
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# T800 模型路径（和其他脚本一致）
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "engineai_robotics_native_sdk-main/assets/resource/robot/t800/xml/serial_t800.xml"
)

# 参与走路的 12 个腿关节（按这个顺序作为动作/观测的排列顺序）
LEG_JOINTS = [
    "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R", "J07_HIP_ROLL_R", "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
]

# 物理参数
PHYSICS_DT = 0.002     # 模型自带的时间步长
FRAME_SKIP = 10        # 每 10 个物理步做一次控制 => 控制频率 50 Hz
CTRL_DT = PHYSICS_DT * FRAME_SKIP
EPISODE_STEPS = 750    # 每个 episode 最多 750 个控制步 = 15 秒
TARGET_SPEED = 0.8     # 期望前进速度 m/s
ACTION_SCALE = 0.35    # 动作 [-1,1] 映射成关节目标角偏移 ±0.35 rad
KP = 1000.0            # 关节 PD 比例增益
KD = 60.0              # 关节 PD 微分增益
PHASE_FREQ = 1.5       # 步态相位频率 Hz（给策略提供一个周期信号）

# 全局共享一个 MuJoCo 模型（模型只读，多个环境可以共用）
_MODEL: mujoco.MjModel | None = None


def get_model() -> mujoco.MjModel:
    """加载 T800 模型并补一块地面（只加载一次）。"""
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
    """T800 行走环境：动作是 12 个腿关节的目标角偏移。"""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.model = get_model()
        self.data = mujoco.MjData(self.model)

        # 执行器 -> 关节 -> qpos/qvel 下标
        self.act_joint = self.model.actuator_trnid[:, 0]          # 每个电机控制的关节
        self.act_qadr = self.model.jnt_qposadr[self.act_joint]   # 该关节角度下标
        self.act_vadr = self.model.jnt_dofadr[self.act_joint]    # 该关节速度下标
        self.default_q = self.model.qpos0[self.act_qadr].copy()  # 目标角度默认取初始姿态
        self.tau_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.tau_high = self.model.actuator_ctrlrange[:, 1].copy()

        # 找到 12 个腿关节对应的执行器下标，以及它们在 qpos/qvel 里的下标
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

        # 动作空间：12 个腿关节，范围 [-1, 1]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)

        # 观测空间：高度(1) + 投影重力(3) + 线速度(3) + 角速度(3)
        #            + 腿关节角(12) + 腿关节速度(12) + 上一步动作(12) + 相位(2)
        obs_dim = 1 + 3 + 3 + 3 + 12 + 12 + 12 + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # 运行时变量
        self.step_count = 0
        self.prev_action = np.zeros(12, dtype=np.float64)
        self.phase = 0.0
        self.mat = np.zeros(9)

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """拼出当前观测向量。"""
        data = self.data
        quat = data.qpos[3:7]
        mujoco.mju_quat2Mat(self.mat, quat)
        rot = self.mat.reshape(3, 3)         # body -> world 旋转矩阵
        v_world = data.qvel[0:3]             # 世界系线速度
        v_body = rot.T @ v_world             # 转到机体坐标系
        w_body = data.qvel[3:6]              # 自由关节角速度本来就在机体系
        gravity_body = -rot[2, :]            # 投影重力（机体坐标系下的重力方向）

        q_leg = data.qpos[self.leg_qadr] - self.leg_default_q
        qd_leg = data.qvel[self.leg_vadr]
        height = data.qpos[2]

        phase_angle = 2.0 * math.pi * self.phase
        clock = np.array([math.sin(phase_angle), math.cos(phase_angle)])

        obs = np.concatenate([
            [height],
            gravity_body,
            v_body,
            w_body,
            q_leg,
            qd_leg,
            self.prev_action,
            clock,
        ])
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    def _compute_reward(self, action: np.ndarray, tau: np.ndarray) -> float:
        """根据速度、姿态、高度和能量消耗计算奖励。"""
        data = self.data
        rot = self.mat.reshape(3, 3)
        v_body = rot.T @ data.qvel[0:3]
        up_z = float(rot[2, 2])             # 机体 z 轴在世界系中的 z 分量，1 表示完全竖直
        z = float(data.qpos[2])

        vx, vy = float(v_body[0]), float(v_body[1])
        wz = float(data.qvel[5])

        # 前进速度奖励：只要往前走就给正奖励，走错方向/站着不动奖励低
        r_forward = 2.0 * float(np.clip(vx, 0.0, 1.2))
        r_lateral = 0.2 * math.exp(-4.0 * vy ** 2)
        r_upright = 0.4 * max(0.0, up_z)
        r_height = 0.2 * math.exp(-8.0 * (z - 1.0) ** 2)
        r_alive = 0.1

        # 惩罚：力矩、动作变化、关节速度、偏航角速度
        r_torque = 1e-6 * float(np.sum(tau ** 2))
        r_action_rate = 0.02 * float(np.sum((action - self.prev_action) ** 2))
        r_joint_vel = 1e-4 * float(np.sum(data.qvel[self.leg_vadr] ** 2))
        r_yaw = 0.05 * wz ** 2

        return r_forward + r_lateral + r_upright + r_height + r_alive \
            - r_torque - r_action_rate - r_joint_vel - r_yaw

    # ------------------------------------------------------------------
    def step(self, action):
        """执行一个控制步（内部做 FRAME_SKIP 次物理步）。"""
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # 目标角度 = 默认角度 + 动作 × 缩放系数
        target = self.default_q.copy()
        target[self.leg_act_idx] += action * ACTION_SCALE

        tau = np.zeros(self.model.nu)
        # 每个物理步都重新算一次 PD，让控制更平滑
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

        # 判断失败：高度太低、倾斜太大、横向跑太远、数值异常
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

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        """重置到站立姿态，加一点随机扰动。"""
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        qpos = self.model.qpos0.copy()
        qpos[0] = 0.0
        qpos[1] = 0.0
        qpos[2] = 1.03
        qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        # 腿关节加小噪声，让每次初始姿态略有不同
        qpos[self.leg_qadr] += self.np_random.uniform(-0.05, 0.05, size=12)

        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.phase = 0.0
        self.prev_action = np.zeros(12, dtype=np.float64)
        return self._get_obs(), {}


# ----------------------------------------------------------------------
# 训练 / 回放
# ----------------------------------------------------------------------
class SaveCallback(BaseCallback):
    """每隔一定步数保存模型和归一化统计量。"""

    def __init__(self, save_dir: Path, save_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(self.save_dir / "ppo_t800_walk")
            vec_env = self.model.get_env()
            if isinstance(vec_env, VecNormalize):
                vec_env.save(self.save_dir / "vecnormalize.pkl")
            print(f"[保存] 已训练 {self.num_timesteps} 步 -> {self.save_dir}", flush=True)
        return True


def make_env(seed: int):
    """返回一个环境构造函数，给 DummyVecEnv 用。"""

    def thunk():
        env = Monitor(T800WalkEnv())
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return thunk


def train(args) -> None:
    """构建环境并训练 PPO。"""
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 8 个并行环境（DummyVecEnv 在单进程里顺序执行，简单可靠）
    env_fns = [make_env(args.seed + i) for i in range(args.envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 断点续训：先加载观测归一化统计，再加载模型
    if args.resume and (save_dir / "ppo_t800_walk.zip").exists():
        vec_env = VecNormalize.load(save_dir / "vecnormalize.pkl", vec_env)
        model = PPO.load(save_dir / "ppo_t800_walk", env=vec_env, device=args.device)
        print(f"[续训] 从 {save_dir / 'ppo_t800_walk.zip'} 继续", flush=True)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=2048,
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
                log_std_init=0.0,   # 初始动作噪声大一点，鼓励探索步态
            ),
            device=args.device,
            verbose=1,
            seed=args.seed,
        )

    total = args.steps
    print(f"[训练] 环境数={args.envs}，总步数={total}，设备={args.device}", flush=True)
    print("[训练] 提示：人形机器人走路训练很慢，可先小步数验证流程。", flush=True)

    callback = SaveCallback(save_dir, save_freq=args.save_freq)
    model.learn(total_timesteps=total, callback=callback, progress_bar=False)

    model.save(save_dir / "ppo_t800_walk")
    vec_env.save(save_dir / "vecnormalize.pkl")
    vec_env.close()
    print(f"[完成] 模型已保存到 {save_dir}", flush=True)


def play(args) -> None:
    """加载训练好的策略，在 MuJoCo viewer 里走。"""
    save_dir = Path(args.save_dir)
    model_file = save_dir / "ppo_t800_walk.zip"
    norm_file = save_dir / "vecnormalize.pkl"
    if not model_file.exists() or not norm_file.exists():
        print(f"找不到模型：{model_file} 或 {norm_file}，请先训练。")
        return

    import mujoco.viewer

    vec_env = DummyVecEnv([lambda: T800WalkEnv()])
    vec_env = VecNormalize.load(norm_file, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(model_file, device=args.device)

    obs = vec_env.reset()
    base_env = vec_env.envs[0]

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

                # 按真实时间播放（每个控制步 0.02 秒）
                next_time += CTRL_DT
                lag = next_time - time.time()
                if lag > 0:
                    time.sleep(min(lag, 0.01))
    except Exception as exc:
        print(f"无法打开 3D 窗口：{exc}")
        print("macOS 需要用 mjpython 运行：")
        print("  /opt/anaconda3/envs/mujoco/bin/mjpython 09.py --play")



# ----------------------------------------------------------------------
# 直接加载 SDK 自带的预训练走路策略做演示（不是本脚本训练出来的）
# ----------------------------------------------------------------------
def pretrained_demo(args) -> None:
    """加载 SDK 里的 t800_...mnn 策略，在 MuJoCo viewer 里走起来。"""
    import mujoco.viewer

    policy_path = (
        Path(__file__).resolve().parent
        / "engineai_robotics_native_sdk-main/assets/config/t800/rl_walking_example"
        / "policy/t800_260318_150533_60000.mnn"
    )
    if not policy_path.exists():
        print(f"找不到预训练策略：{policy_path}")
        return
    try:
        import MNN
    except ImportError:
        print("当前环境没有 MNN，无法运行预训练策略。")
        return

    # 这些参数来自 SDK 的 rl_walking_example/default.yaml
    default = np.array(
        [-0.06, 0, 0, 0.12, -0.06, 0] + [-0.06, 0, 0, 0.12, -0.06, 0] + [0.0]
        + [0.0, 0.15, 0.0, -0.25, 0.0] + [0.0, -0.15, 0.0, -0.25, 0.0] + [0.0, 0.0]
    )
    kp = np.array(
        [180, 100, 100, 180, 40, 40] + [180, 100, 100, 180, 40, 40] + [100]
        + [60, 50, 50, 60, 50] + [60, 50, 50, 60, 50] + [100, 100], dtype=float
    )
    kd = np.array(
        [5, 3, 3, 5, 0.3, 0.3] + [5, 3, 3, 5, 0.3, 0.3] + [5]
        + [0.3] * 5 + [0.3] * 5 + [1, 1], dtype=float
    )
    action_scale = np.array(
        [0.5, 0.2, 0.2, 0.5, 0.5, 0.2] + [0.5, 0.2, 0.2, 0.5, 0.5, 0.2] + [0.2]
        + [0.2, 0.2, 0.05, 0.2, 0.05] + [0.2, 0.2, 0.05, 0.2, 0.05] + [0.2, 0.2]
    )
    soft = np.array([0.8] * 12 + [0.9] + [0.9] * 5 + [0.9] * 5 + [0.9] * 2)

    # 22 个参与策略的关节（跳过 J12 躯干偏航、J23/J24 头部）
    active = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22])

    model = get_model()
    data = mujoco.MjData(model)

    act_joint = model.actuator_trnid[:, 0]
    act_qadr = model.jnt_qposadr[act_joint]
    act_vadr = model.jnt_dofadr[act_joint]
    tau_lo = model.actuator_ctrlrange[:, 0] * soft
    tau_hi = model.actuator_ctrlrange[:, 1] * soft

    # 加载 MNN 策略
    net = MNN.Interpreter(str(policy_path))
    sess = net.createSession()
    inp = net.getSessionInput(sess)
    out = net.getSessionOutput(sess)

    def infer(obs):
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        tensor = MNN.Tensor(inp.getShape(), MNN.Halide_Type_Float, x,
                            MNN.Tensor_DimensionType_Caffe)
        inp.copyFrom(tensor)
        net.runSession(sess)
        host = MNN.Tensor(out.getShape(), MNN.Halide_Type_Float,
                          np.zeros(out.getShape(), np.float32),
                          MNN.Tensor_DimensionType_Caffe)
        out.copyToHostTensor(host)
        return np.asarray(host.getData(), dtype=np.float64).reshape(-1)

    # 初始站立姿态
    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0.0
    data.qpos[2] = 1.03
    data.qpos[3] = 1.0
    data.qpos[7:] = default
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    hist = np.zeros((15, 72))
    last_action = np.zeros(22)
    first = True
    command = np.array([0.6, 0.0, 0.0])  # 前进 0.6 m/s
    mat = np.zeros(9)

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -10.0
            viewer.cam.azimuth = 135.0
            viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

            print("预训练策略已加载，T800 开始行走；关闭窗口退出。", flush=True)
            next_time = time.time()
            next_log = 1.0
            x0 = float(data.qpos[0])

            while viewer.is_running():
                # 组装 72 维单帧观测
                quat = data.qpos[3:7]
                mujoco.mju_quat2Mat(mat, quat)
                rot = mat.reshape(3, 3)
                q = data.qpos[7:32]
                qd = data.qvel[6:31]

                frame = np.zeros(72)
                frame[0:22] = q[active] - default[active]
                frame[22:44] = qd[active] * 0.05
                frame[44:66] = last_action
                frame[66:69] = data.qvel[3:6]
                frame[69:72] = -rot[2, :]

                if first:
                    hist[:] = frame
                    first = False
                else:
                    hist[:-1] = hist[1:]
                    hist[-1] = frame

                obs = np.concatenate([hist.reshape(-1), command])
                action = np.clip(infer(obs), -100.0, 100.0)
                last_action = action.copy()

                target = default.copy()
                target[active] += action * action_scale[active]

                # 100 Hz 控制：每 5 个物理步（0.01 s）更新一次 PD
                for _ in range(5):
                    qa = data.qpos[act_qadr]
                    qda = data.qvel[act_vadr]
                    tau = np.clip(kp * (target - qa) - kd * qda, tau_lo, tau_hi)
                    data.ctrl[:] = tau
                    mujoco.mj_step(model, data)

                # 相机跟着机器人走
                viewer.cam.lookat[:] = [data.qpos[0], data.qpos[1], 1.0]
                viewer.sync()

                if data.time >= next_log:
                    print(f"t={data.time:5.1f}s  x={data.qpos[0] - x0:+.2f} m  "
                          f"z={data.qpos[2]:.3f} m  倾斜={math.degrees(math.acos(max(-1, min(1, float(rot[2, 2]))))):.1f}°",
                          flush=True)
                    next_log += 1.0

                next_time += 0.01
                lag = next_time - time.time()
                if lag > 0:
                    time.sleep(min(lag, 0.01))
    except Exception as exc:
        print(f"无法打开 3D 窗口：{exc}")
        print("macOS 需要用 mjpython 运行：")
        print("  /opt/anaconda3/envs/mujoco/bin/mjpython 09.py --demo")

# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="T800 行走 PPO 训练")
    parser.add_argument("--steps", type=int, default=1_000_000,
                        help="总训练步数（默认 100 万）")
    parser.add_argument("--envs", type=int, default=8,
                        help="并行环境数（默认 8）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--device", type=str, default="cpu",
                        help="PyTorch 设备：cpu 或 mps（默认 cpu）")
    parser.add_argument("--save-dir", type=str, default="t800_walk_model",
                        help="模型保存目录")
    parser.add_argument("--save-freq", type=int, default=100_000,
                        help="每多少步保存一次")
    parser.add_argument("--resume", action="store_true", help="从 save-dir 断点续训")
    parser.add_argument("--smoke", action="store_true", help="冒烟测试：只训练 2 万步")
    parser.add_argument("--play", action="store_true", help="回放训练好的策略")
    parser.add_argument("--demo", action="store_true",
                        help="直接加载 SDK 预训练策略演示走路（不需要训练）")
    args = parser.parse_args()

    if args.smoke:
        args.steps = 20_000
        args.save_dir = args.save_dir + "_smoke"

    if args.demo:
        pretrained_demo(args)
        return 0

    if args.play:
        play(args)
        return 0

    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
