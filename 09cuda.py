"""09cuda.py —— Ubuntu + NVIDIA 显卡上的 T800 行走 PPO 训练脚本（详细注释版）。

======================================================================
一、这个脚本在干什么
======================================================================
用强化学习（PPO）训练 T800 人形机器人在 MuJoCo 里走路。整体数据流：

    命令行参数
        ↓
    创建 N 个 MuJoCo 环境（每个环境一套 model + data）
        ↓
    SubprocVecEnv 把 N 个环境放进 N 个子进程，并行采样
        ↓
    VecNormalize 对观测和奖励做归一化
        ↓
    PPO 用采样数据更新神经网络
        ↓
    保存模型（.zip）和归一化统计量（.pkl）

======================================================================
二、脚本里用到的包，分别是干什么的
======================================================================
1. argparse
   标准库，用来解析命令行参数。
   例如 `python 09cuda.py --steps 2000000 --envs 16`，
   argparse 会把 `--steps` 读成整数 2000000，把 `--envs` 读成 16，
   存进 args 对象，后面用 args.steps、args.envs 访问。
   没有它的话，就只能改代码里的变量再运行。

2. gymnasium（代码里 import 成 gym）
   强化学习环境的标准接口库（OpenAI Gym 的继任者）。
   它规定了一个环境必须长什么样：
       reset()  -> 返回 (初始观测, 附加信息)
       step(a)  -> 返回 (新观测, 奖励, terminated, truncated, 附加信息)
   并且用 spaces.Box 描述“观测是几个数、动作是几个数、范围多少”。
   Stable-Baselines3 只认这个接口，所以我们的 T800WalkEnv 继承 gym.Env。

3. numpy
   数值计算。MuJoCo 的 qpos/qvel、观测向量、奖励计算都是 numpy 数组。

4. mujoco
   物理引擎。MjModel 是“模型”（质量、关节、几何体，只读），
   MjData 是“运行时状态”（位置、速度、时间）。
   mj_step() 推进一步物理仿真，mju_quat2Mat() 把四元数转旋转矩阵。

5. torch（PyTorch）
   深度学习框架。这里主要给 Stable-Baselines3 的 PPO 提供神经网络，
   以及用 torch.cuda.is_available() 判断显卡是否可用。

6. stable_baselines3（SB3）
   封装好的强化学习算法库，这里用它的 PPO，以及：
     - Monitor：记录每个 episode 的回报和长度，方便打印日志
     - DummyVecEnv：单进程里顺序跑多个环境
     - SubprocVecEnv：多进程并行跑多个环境（本脚本默认）
     - VecNormalize：对观测/奖励做滑动平均归一化
     - BaseCallback：训练过程中的回调钩子（用来定时保存模型）

7. 其他标准库
   math：sin/cos/exp/pi
   sys：判断平台、退出程序
   time：计时、控制播放速度
   pathlib.Path：拼接文件路径

======================================================================
三、怎么运行（Ubuntu，先 conda activate mujoco）
======================================================================
    # 冒烟测试：2 万步，确认 16 个进程能起来
    python 09cuda.py --smoke

    # 正式训练：200 万步
    python 09cuda.py --steps 2000000 --envs 16 --device cuda --tensorboard

    # 断点续训：在已有模型上再训练 500 万步
    python 09cuda.py --steps 5000000 --envs 16 --device cuda --resume

    # 回放训练好的策略（需要图形界面）
    python 09cuda.py --play

注意：
  - MuJoCo 的物理计算在 CPU 上，GPU 只负责 PPO 的网络更新。
    所以 nvidia-smi 里 GPU 利用率低是正常的，不代表没配置好。
  - 从零训练人形走路很慢，脚本只负责把训练吞吐拉满；
    能不能走好，主要看奖励函数和训练步数。
  - 加载 SDK 预训练策略的 --demo 在 09.py 里，不在这个脚本里。

======================================================================
四、这一版参考了哪些训练代码
======================================================================
参考本机的 rl_training-main（Deep Robotics，Isaac Lab 版）：
  - track_lin_vel_xy_exp：用 exp 形式做速度跟踪
  - feet_slide：支撑脚在地面滑动要惩罚
  - feet_height：摆动腿抬脚高度要给奖励
  - feet_air_time / 单脚支撑：鼓励左右脚交替迈步
参考它的 rsl_rl PPO 配置：
  - 熵系数 0.01，鼓励探索
  - 5 个 epoch，而不是 10 个
  - ELU 激活函数

因此这一版加了：
  - 每回合随机目标速度 command，并写进观测
  - 脚底滑动惩罚、抬脚高度奖励、单脚支撑奖励
  - 新的 PPO 超参数

注意：观测维度从 48 变成 49，旧模型不能直接 resume，
请用新的保存目录（默认 t800_walk_model_v3）从头训练。

======================================================================
五、针对“跳着走、膝盖不弯、上半身不动”的改动
======================================================================
  - 双脚同时离地要惩罚（抑制蹦跳），双脚着地给一点奖励
  - 基座竖直方向速度平方惩罚，抑制上下弹跳
  - 摆动腿膝盖弯曲给奖励，鼓励抬腿时屈膝
  - 肩关节按步态相位反相摆动，肘关节保持微弯，
    这样上半身会自然摆起来（不需要额外动作维度）
"""

from __future__ import annotations  # 让类型注解可以写 MjModel | None 这种语法

import argparse   # 解析命令行参数
import math       # 数学函数：sin、cos、exp、pi
import sys        # 判断操作系统、退出程序
import time       # 计时、控制 viewer 播放速度
from pathlib import Path  # 跨平台路径拼接

import mujoco             # MuJoCo 物理引擎
import numpy as np        # 数值计算
import torch              # PyTorch，给 PPO 提供神经网络
import gymnasium as gym   # 强化学习环境标准接口
from gymnasium import spaces  # 定义观测空间/动作空间

# Stable-Baselines3：现成的 PPO 实现 + 各种环境包装器
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

# ======================================================================
# 一、路径和超参数
# ======================================================================

# T800 的 MJCF 模型文件（和 09.py 相同）
MODEL_PATH = (
    Path(__file__).resolve().parent  # 本文件所在目录
    / "engineai_robotics_native_sdk-main/assets/resource/robot/t800/xml/serial_t800.xml"
)

# 参与走路的 12 个腿关节；动作和观测都按这个顺序排列
LEG_JOINTS = [
    "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R", "J07_HIP_ROLL_R", "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
]

PHYSICS_DT = 0.002          # MuJoCo 物理步长，2 ms（模型 XML 里也是这个值）
FRAME_SKIP = 10             # 每个控制步内部做 10 次物理步
CTRL_DT = PHYSICS_DT * FRAME_SKIP  # 控制周期 = 20 ms，即 50 Hz
EPISODE_STEPS = 750         # 一个 episode 最多 750 个控制步 = 15 秒
TARGET_SPEED = 0.8          # 默认期望前进速度 m/s
COMMAND_MIN = 0.4           # 每回合随机目标速度的下限
COMMAND_MAX = 1.0           # 每回合随机目标速度的上限
ACTION_SCALE = 0.35         # 动作 [-1,1] 乘这个系数，变成关节目标角偏移
KP = 1000.0                 # 关节 PD 比例增益
KD = 60.0                   # 关节 PD 微分增益
PHASE_FREQ = 1.5            # 步态相位频率 Hz，给策略一个周期时钟信号

# 两个脚对应的 body 名字：用来算脚的高度、速度、是否着地
FOOT_BODIES = ("LINK_FOOT_L", "LINK_FOOT_R")
FOOT_CONTACT_Z = 0.095      # 脚 body 高度低于这个值，认为脚踩在地上

ARM_SWING = 0.25            # 肩关节前后摆动幅度（rad）
ELBOW_BEND = -0.25          # 肘关节常弯角度（rad），避免手臂笔直

# 每个进程各自缓存一个 MuJoCo 模型；父进程不建，子进程 fork 后各自建
_MODEL: mujoco.MjModel | None = None


def get_model() -> mujoco.MjModel:
    """加载 T800 模型并补一块地面；每个进程只加载一次。"""
    global _MODEL
    if _MODEL is None:
        # MjSpec 是“模型草稿”，可以在编译前继续加东西
        spec = mujoco.MjSpec.from_file(str(MODEL_PATH))

        # 原模型没有地面，这里补一块 20m × 20m 的平面
        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [20.0, 20.0, 0.1]
        floor.rgba = [0.45, 0.48, 0.52, 1.0]

        # 编译成真正能仿真的 MjModel
        model = spec.compile()

        # 调亮头灯，回放时不至于黑乎乎的
        model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
        model.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]

        _MODEL = model
    return _MODEL


# ======================================================================
# 二、Gymnasium 环境：把 MuJoCo 包装成强化学习环境
# ======================================================================
class T800WalkEnv(gym.Env):
    """T800 行走环境。

    动作空间：12 维，[-1,1]，对应 12 个腿关节的目标角偏移。
    观测空间：49 维，下面 _get_obs() 里逐项说明。
    """

    # Gymnasium 用 metadata 描述渲染模式；这里不做窗口渲染，所以是空列表
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()

        # MjModel 是只读的模型，MjData 是这个环境独有的运行时状态
        self.model = get_model()
        self.data = mujoco.MjData(self.model)

        # ---- 建立“执行器(电机) -> 关节 -> qpos/qvel 下标”的对应关系 ----
        # actuator_trnid[:, 0] 表示每个电机控制哪个关节
        self.act_joint = self.model.actuator_trnid[:, 0]
        # jnt_qposadr[j] 是这个关节角度在 qpos 数组里的起始下标
        self.act_qadr = self.model.jnt_qposadr[self.act_joint]
        # jnt_dofadr[j] 是这个关节速度在 qvel 数组里的起始下标
        self.act_vadr = self.model.jnt_dofadr[self.act_joint]
        # 初始姿态下每个电机的目标角度
        self.default_q = self.model.qpos0[self.act_qadr].copy()
        # 每个电机的力矩上下限（来自 XML 的 ctrlrange）
        self.tau_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.tau_high = self.model.actuator_ctrlrange[:, 1].copy()

        # ---- 单独找出 12 个腿关节的下标，方便取观测/下动作 ----
        self.leg_act_idx = []  # 腿关节对应的电机编号
        self.leg_qadr = []     # 腿关节在 qpos 里的下标
        self.leg_vadr = []     # 腿关节在 qvel 里的下标
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

        # ---- 动作空间：12 个连续值，范围 [-1, 1] ----
        # Box 表示“每个维度是连续实数”，PPO 会输出这 12 个数的均值
        self.action_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)

        # ---- 观测空间：49 维连续值，范围不设上下限 ----
        # 组成：高度 1 + 投影重力 3 + 线速度 3 + 角速度 3
        #       + 腿关节角 12 + 腿关节速度 12 + 上一步动作 12 + 相位 2
        #       + 目标速度 1
        obs_dim = 1 + 3 + 3 + 3 + 12 + 12 + 12 + 2 + 1  # = 49
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ---- 脚部 body 的编号，用来算抬脚高度和脚底滑动 ----
        self.foot_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in FOOT_BODIES
        ]
        self.prev_foot_pos = None  # 上一步脚的位置，用来算脚的速度

        # 膝盖的 qpos 下标（左、右），用来奖励屈膝
        self.knee_qadr = np.array([
            int(self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ])
            for name in ("J03_KNEE_PITCH_L", "J09_KNEE_PITCH_R")
        ])

        # 肩 pitch 和肘 pitch 对应的执行器下标，用来做脚本化摆臂
        def actuator_id(joint_name):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            return int(np.where(self.act_joint == jid)[0][0])

        self.arm_sh_l = actuator_id("J13_SHOULDER_PITCH_L")
        self.arm_sh_r = actuator_id("J18_SHOULDER_PITCH_R")
        self.arm_elbow_l = actuator_id("J16_ELBOW_PITCH_L")
        self.arm_elbow_r = actuator_id("J21_ELBOW_PITCH_R")

        # ---- 运行时的辅助变量 ----
        self.step_count = 0                          # 当前 episode 走了几个控制步
        self.prev_action = np.zeros(12, dtype=np.float64)  # 上一步动作（给观测用）
        self.phase = 0.0                             # 步态相位，0~1 循环
        self.command = TARGET_SPEED                  # 当前回合的目标速度
        self.mat = np.zeros(9)                       # 复用的 3x3 旋转矩阵缓冲

    def _get_obs(self) -> np.ndarray:
        """拼出 49 维观测向量。"""
        data = self.data

        # 把基座四元数 (w,x,y,z) 转成 3x3 旋转矩阵（行优先展开成 9 个数）
        mujoco.mju_quat2Mat(self.mat, data.qpos[3:7])
        rot = self.mat.reshape(3, 3)  # body -> world 的旋转矩阵

        # 世界系线速度转到机体系：v_body = R^T * v_world
        v_body = rot.T @ data.qvel[0:3]
        # 自由关节的角速度本来就在机体系，直接用
        w_body = data.qvel[3:6]
        # 投影重力：世界系重力 (0,0,-1) 转到机体系 = -R 的第三行
        gravity_body = -rot[2, :]

        # 腿关节相对默认姿态的角度，以及角速度
        q_leg = data.qpos[self.leg_qadr] - self.leg_default_q
        qd_leg = data.qvel[self.leg_vadr]
        # 基座高度（qpos[2] 是自由关节的 z）
        height = data.qpos[2]

        # 步态相位信号：sin/cos 成对出现，避免 0/1 跳变
        angle = 2.0 * math.pi * self.phase
        clock = np.array([math.sin(angle), math.cos(angle)])

        # 把所有分量拼成一个 49 维向量
        obs = np.concatenate([
            [height],       # 1
            gravity_body,   # 3
            v_body,         # 3
            w_body,         # 3
            q_leg,          # 12
            qd_leg,         # 12
            self.prev_action,  # 12
            clock,          # 2
            [self.command],  # 1：本回合要跟踪的目标速度
        ])
        return obs.astype(np.float32)

    def _compute_reward(self, action: np.ndarray, tau: np.ndarray,
                        foot_pos: np.ndarray, foot_vel: np.ndarray) -> float:
        """根据速度、姿态、高度、脚部动作、能量消耗算一个标量奖励。"""
        data = self.data
        rot = self.mat.reshape(3, 3)
        v_body = rot.T @ data.qvel[0:3]
        up_z = float(rot[2, 2])   # 机身 z 轴在世界系的 z 分量，1=完全直立
        z = float(data.qpos[2])   # 基座高度

        vx, vy = float(v_body[0]), float(v_body[1])
        wz = float(data.qvel[5])  # 偏航角速度

        # ---- 1. 速度跟踪 ----
        # 和参考代码的 track_lin_vel_xy_exp 一样，用 exp(-误差²/std²)
        # command 是每回合随机采样的目标速度，策略要在观测里看到它
        r_track = 2.5 * math.exp(-((vx - self.command) ** 2) / 0.16)
        # 横向速度尽量为 0
        r_lateral = 0.3 * math.exp(-4.0 * vy ** 2)

        # ---- 2. 姿态和高度 ----
        r_upright = 0.4 * max(0.0, up_z)
        r_height = 0.2 * math.exp(-8.0 * (z - 1.0) ** 2)
        r_alive = 0.1

        # ---- 3. 脚部奖励（参考 feet_slide / feet_height / 单脚支撑） ----
        foot_z = foot_pos[:, 2]                              # 两只脚的高度
        foot_speed_xy = np.linalg.norm(foot_vel[:, :2], axis=1)  # 脚的水平速度
        contact = foot_z < FOOT_CONTACT_Z                    # True=这只脚踩在地上
        swing = ~contact                                     # True=这只脚在摆动

        # 支撑脚在地面滑动要惩罚：脚贴地但水平速度大 => 扣分
        r_slide = -0.5 * float(np.sum(foot_speed_xy[contact] ** 2))
        # 单脚支撑：一只脚在摆、另一只脚着地
        single_support = int(np.sum(contact)) == 1

        # 摆动脚抬到 9~15 cm 给奖励，拖脚不给（权重别太大，避免高抬腿）
        lift = np.clip((foot_z - 0.09) / 0.06, 0.0, 1.0)
        r_lift = 0.15 * float(np.sum(lift[swing])) if single_support else 0.0
        # 单脚支撑奖励：鼓励交替迈步
        r_single_support = 0.15 if single_support else 0.0

        # ---- 3.1 抑制蹦跳 ----
        # 双脚同时离地：像小跳，扣分
        r_flight = -0.4 if int(np.sum(contact)) == 0 else 0.0
        # 双脚同时着地：走路时常见的双支撑，给一点奖励
        r_double_support = 0.1 if int(np.sum(contact)) == 2 else 0.0
        # 基座上下弹跳：竖直速度平方惩罚
        r_vertical = -0.4 * float(data.qvel[2] ** 2)

        # ---- 3.2 摆动腿屈膝奖励 ----
        # 抬腿时膝盖要弯（>0.1 rad 才开始给分，0.3 rad 满分）
        knee_angle = data.qpos[self.knee_qadr]
        knee_bend = np.clip((knee_angle - 0.1) / 0.2, 0.0, 1.0)
        # 只在单脚支撑时奖励屈膝，避免“双脚离地+双双屈膝”骗奖励
        r_knee = 0.2 * float(np.sum(knee_bend[swing])) if single_support else 0.0

        # ---- 4. 平滑和能耗惩罚 ----
        r_torque = 1e-6 * float(np.sum(tau ** 2))
        r_action_rate = 0.02 * float(np.sum((action - self.prev_action) ** 2))
        r_joint_vel = 1e-4 * float(np.sum(data.qvel[self.leg_vadr] ** 2))
        r_yaw = 0.05 * wz ** 2

        return (r_track + r_lateral + r_upright + r_height + r_alive
                + r_slide + r_lift + r_single_support
                + r_flight + r_double_support + r_vertical + r_knee
                - r_torque - r_action_rate - r_joint_vel - r_yaw)

    def step(self, action):
        """执行一个控制步，返回 Gymnasium 规定的 5 元组。

        返回：(观测, 奖励, terminated, truncated, info)
          terminated=True：真正失败（摔倒、出界、数值异常）
          truncated=True ：到时间上限（15 秒），不是失败
        """
        # 动作裁剪到 [-1, 1]，防止策略输出越界
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # 目标角度 = 默认角度 + 动作 × 缩放系数
        target = self.default_q.copy()
        target[self.leg_act_idx] += action * ACTION_SCALE

        # 脚本化上半身：肩膀随步态相位前后摆，肘关节保持微弯。
        # 这不占用动作维度，策略只专注腿，但看起来不会像两根木棍。
        swing = math.sin(2.0 * math.pi * self.phase)
        target[self.arm_sh_l] += ARM_SWING * swing
        target[self.arm_sh_r] -= ARM_SWING * swing
        target[self.arm_elbow_l] = ELBOW_BEND
        target[self.arm_elbow_r] = ELBOW_BEND

        # 一个控制步内部做 FRAME_SKIP 次物理步（50 Hz 控制，500 Hz 物理）
        tau = np.zeros(self.model.nu)
        for _ in range(FRAME_SKIP):
            # 当前关节角/角速度
            q = self.data.qpos[self.act_qadr]
            qd = self.data.qvel[self.act_vadr]
            # PD 控制：力矩 = kp*(目标-当前) - kd*当前速度
            tau = KP * (target - q) - KD * qd
            # 按电机力矩上限裁剪
            tau = np.clip(tau, self.tau_low, self.tau_high)
            # 写入控制量并推进一步物理仿真
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)

        # 控制步计数 +1，相位前进（0~1 循环）
        self.step_count += 1
        self.phase = (self.phase + CTRL_DT * PHASE_FREQ) % 1.0

        # 算两只脚的速度：用相邻两步的位置差除以控制周期
        foot_pos = self.data.xpos[self.foot_ids].copy()
        if self.prev_foot_pos is None:
            foot_vel = np.zeros((2, 3))
        else:
            foot_vel = (foot_pos - self.prev_foot_pos) / CTRL_DT
        self.prev_foot_pos = foot_pos

        # 算观测和奖励
        obs = self._get_obs()
        reward = self._compute_reward(action, tau, foot_pos, foot_vel)

        # 判断是否失败：太低、太歪、横向跑太远、出现 NaN
        mujoco.mju_quat2Mat(self.mat, self.data.qpos[3:7])
        rot = self.mat.reshape(3, 3)
        up_z = float(rot[2, 2])
        z = float(self.data.qpos[2])
        y = float(self.data.qpos[1])

        terminated = bool(
            z < 0.65 or up_z < 0.5 or abs(y) > 1.2
            or not np.isfinite(obs).all()
        )
        # 到 750 个控制步就截断（时间上限，不是失败）
        truncated = self.step_count >= EPISODE_STEPS

        # 保存动作，下一步观测会用到
        self.prev_action = action
        return obs, float(reward), terminated, truncated, {}

    def reset(self, *, seed=None, options=None):
        """重置环境，返回 (初始观测, 附加信息)。"""
        # 调用父类 reset，初始化随机数生成器 self.np_random
        super().reset(seed=seed)
        # 把 MjData 恢复到模型默认状态
        mujoco.mj_resetData(self.model, self.data)

        # 在默认姿态基础上稍作随机化，避免每次都从完全相同的状态开始
        qpos = self.model.qpos0.copy()
        qpos[0] = 0.0       # 基座 x
        qpos[1] = 0.0       # 基座 y
        qpos[2] = 1.03      # 基座高度（约等于站立高度）
        qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 基座四元数：无旋转

        # 12 个腿关节各加 ±0.05 rad 的随机抖动
        qpos[self.leg_qadr] += self.np_random.uniform(-0.05, 0.05, size=12)

        # 每回合随机一个目标速度，让策略学会跟踪不同速度
        self.command = float(self.np_random.uniform(COMMAND_MIN, COMMAND_MAX))

        # 写回状态，速度清零，控制量清零
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        # 根据 qpos 重新计算所有派生量（坐标、碰撞等）
        mujoco.mj_forward(self.model, self.data)

        # 重置内部计数器
        self.step_count = 0
        self.phase = 0.0
        self.prev_action = np.zeros(12, dtype=np.float64)
        self.prev_foot_pos = self.data.xpos[self.foot_ids].copy()
        return self._get_obs(), {}


# ======================================================================
# 三、多进程环境工厂
# ======================================================================
class EnvFactory:
    """可被多进程 pickle 的环境工厂。

    SubprocVecEnv 会把每个 EnvFactory 对象交给一个子进程，
    子进程执行 env_factory() 才真正创建环境。
    所以这里只保存 rank/seed，不在父进程里创建 MuJoCo 对象。
    """

    def __init__(self, rank: int, seed: int):
        self.rank = rank
        self.seed = seed

    def __call__(self):
        # Monitor 包装后，SB3 能统计 episode 的回报和长度
        env = Monitor(T800WalkEnv())
        # 给环境设随机种子，保证可复现
        env.reset(seed=self.seed)
        env.action_space.seed(self.seed)
        return env


# ======================================================================
# 四、训练回调：定期保存模型
# ======================================================================
class SaveCallback(BaseCallback):
    """Stable-Baselines3 的回调：每训练一定步数就保存模型。

    BaseCallback 会在每个 rollout 步之后调用 _on_step()，
    返回 True 表示继续训练。
    """

    def __init__(self, save_dir: Path, save_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.next_save = save_freq  # 下一次保存的步数阈值

    def _on_step(self) -> bool:
        # num_timesteps 是“所有环境加起来”的总步数
        if self.num_timesteps >= self.next_save:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            # 保存 PPO 的神经网络
            self.model.save(self.save_dir / "ppo_t800_walk")
            # 归一化统计量必须和模型一起保存，否则回放时观测分布对不上
            vec_env = self.model.get_env()
            if isinstance(vec_env, VecNormalize):
                vec_env.save(self.save_dir / "vecnormalize.pkl")
            print(f"[保存] {self.num_timesteps} 步 -> {self.save_dir}", flush=True)
            self.next_save += self.save_freq
        return True


# ======================================================================
# 五、创建并行环境
# ======================================================================
def build_vec_env(args) -> VecNormalize:
    """根据命令行参数创建并行环境，并包上 VecNormalize。"""
    # 每个环境一个工厂对象，rank 只用于区分，seed 保证各环境初始状态不同
    factories = [EnvFactory(i, args.seed + i) for i in range(args.envs)]

    if args.vec_env == "subproc":
        # SubprocVecEnv：每个环境一个子进程，真正并行
        # Linux 用 fork 启动最快；macOS 必须用 spawn，否则 MuJoCo 可能崩溃
        start_method = "fork" if sys.platform.startswith("linux") else "spawn"
        vec_env = SubprocVecEnv(factories, start_method=start_method)
    else:
        # DummyVecEnv：单进程里顺序跑多个环境，调试用，速度慢
        vec_env = DummyVecEnv(factories)

    # VecNormalize：对观测和奖励做滑动平均标准化
    #   norm_obs=True    观测减去均值、除以标准差
    #   norm_reward=True 奖励做同样的缩放（PPO 对奖励尺度很敏感）
    #   clip_obs=10      观测标准化后裁剪到 [-10, 10]，防止异常值
    return VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)


# ======================================================================
# 六、训练
# ======================================================================
def train(args) -> None:
    """创建环境、PPO，并开始训练。"""
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 先创建环境。注意要在 CUDA 初始化之前 fork 子进程，
    # 否则子进程继承 CUDA 上下文可能出问题。
    vec_env = build_vec_env(args)

    # TensorBoard 日志目录；只有 --tensorboard 时才启用
    tb_log = str(save_dir / "tb") if args.tensorboard else None

    if args.resume and (save_dir / "ppo_t800_walk.zip").exists():
        # ---------- 断点续训 ----------
        # 先加载观测归一化统计量，再加载网络
        vec_env = VecNormalize.load(save_dir / "vecnormalize.pkl", vec_env)
        model = PPO.load(
            save_dir / "ppo_t800_walk", env=vec_env,
            device=args.device, tensorboard_log=tb_log,
        )
        print(f"[续训] 从 {save_dir / 'ppo_t800_walk.zip'} 继续", flush=True)
    else:
        # ---------- 从头训练 ----------
        # PPO 的关键参数说明：
        #   learning_rate : 学习率，5e-4（参考配置用 1e-3 + 自适应，这里取中间值）
        #   n_steps       : 每个环境每次 rollout 收集多少步
        #   batch_size    : 每次梯度更新用多少样本（要 <= n_steps*环境数）
        #   n_epochs      : 同一批数据重复训练几轮
        #   gamma         : 折扣因子，越接近 1 越看重长期回报
        #   gae_lambda    : GAE 的平滑系数
        #   clip_range    : PPO 的裁剪范围，限制每次策略变化幅度
        #   ent_coef      : 熵奖励系数，越大越鼓励探索
        #   vf_coef       : 价值函数损失权重
        #   max_grad_norm : 梯度裁剪，防止梯度爆炸
        model = PPO(
            "MlpPolicy",  # 用多层感知机（MLP）作为策略网络
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
            # policy_kwargs 控制网络结构：
            #   pi/vf 分别是策略网络和价值网络的隐藏层
            #   ELU 激活函数，参考 rsl_rl 的配置
            #   log_std_init=0 表示初始动作标准差是 e^0 = 1
            policy_kwargs=dict(
                net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
                activation_fn=torch.nn.ELU,
                log_std_init=0.0,
            ),
            device=args.device,      # "cuda" 或 "cpu"
            verbose=1,               # 打印训练表格
            seed=args.seed,
            tensorboard_log=tb_log,
        )

    total = args.steps
    print(f"[训练] 环境={args.envs}（{args.vec_env}），总步数={total}，设备={args.device}", flush=True)
    print(f"[训练] n_steps={args.n_steps}，batch_size={args.batch_size}，"
          f"TensorBoard={bool(tb_log)}", flush=True)

    # 每 save_freq 步保存一次模型
    callback = SaveCallback(save_dir, save_freq=args.save_freq)
    t0 = time.time()
    # learn() 内部循环：采样 -> 算 GAE -> 更新网络 -> 重复
    model.learn(total_timesteps=total, callback=callback, progress_bar=False)
    elapsed = time.time() - t0

    # 训练结束再保存一次
    model.save(save_dir / "ppo_t800_walk")
    vec_env.save(save_dir / "vecnormalize.pkl")
    vec_env.close()

    print(f"[完成] 用时 {elapsed:.1f}s，平均 {total / max(elapsed, 1e-9):.0f} 步/秒", flush=True)
    print(f"[完成] 模型保存到 {save_dir}", flush=True)
    if tb_log:
        print(f"[完成] TensorBoard：tensorboard --logdir {tb_log}", flush=True)


# ======================================================================
# 七、回放训练好的策略
# ======================================================================
def play(args) -> None:
    """加载训练好的模型和归一化统计，在 MuJoCo viewer 里走。"""
    import mujoco.viewer  # 只有回放才需要窗口，放函数里导入

    save_dir = Path(args.save_dir)
    model_file = save_dir / "ppo_t800_walk.zip"
    norm_file = save_dir / "vecnormalize.pkl"
    if not model_file.exists() or not norm_file.exists():
        print(f"找不到模型：{model_file} 或 {norm_file}")
        return

    # 回放只需要一个环境
    vec_env = DummyVecEnv([EnvFactory(0, args.seed)])
    # 加载训练时的观测归一化统计
    vec_env = VecNormalize.load(norm_file, vec_env)
    vec_env.training = False     # 不再更新统计量
    vec_env.norm_reward = False  # 回放不需要奖励

    model = PPO.load(model_file, device=args.device)
    obs = vec_env.reset()
    # Monitor 包装了两层，unwrapped 之后才是 T800WalkEnv
    base_env = vec_env.envs[0].unwrapped

    try:
        with mujoco.viewer.launch_passive(base_env.model, base_env.data) as viewer:
            # 初始相机
            viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -10.0
            viewer.cam.azimuth = 135.0
            print("策略已加载，关闭窗口退出。", flush=True)

            next_time = time.time()
            while viewer.is_running():
                # deterministic=True：用策略均值，不加探索噪声
                action, _ = model.predict(obs, deterministic=True)
                obs, _, _, _ = vec_env.step(action)
                viewer.sync()

                # 按真实时间播放（一个控制步 20ms）
                next_time += CTRL_DT
                lag = next_time - time.time()
                if lag > 0:
                    time.sleep(min(lag, 0.01))
    except Exception as exc:
        print(f"无法打开 3D 窗口：{exc}")
        print("Linux 需要图形界面；远程无显示器时请用 ssh -X 或在 Ubuntu 本机运行。")


# ======================================================================
# 八、命令行入口
# ======================================================================
def main() -> int:
    """解析命令行参数，决定训练还是回放。

    argparse 的用法：
      parser = argparse.ArgumentParser(...)   创建解析器
      parser.add_argument("--steps", ...)     声明一个参数
      args = parser.parse_args()              解析 sys.argv
      args.steps                              取参数值

    type=int 会把命令行字符串转成整数；
    action="store_true" 表示这是一个开关，出现就是 True，不出现就是 False。
    """
    parser = argparse.ArgumentParser(description="T800 行走 PPO 训练（CUDA + 多进程）")
    parser.add_argument("--steps", type=int, default=10_000_000, help="总训练步数")
    parser.add_argument("--envs", type=int, default=16, help="并行环境数")
    parser.add_argument("--vec-env", choices=["subproc", "dummy"], default="subproc",
                        help="并行方式：subproc 多进程（默认）或 dummy 单进程")
    parser.add_argument("--n-steps", type=int, default=512,
                        help="每个环境每次 rollout 的步数")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="PPO 小批量大小")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")
    parser.add_argument("--save-dir", type=str, default="t800_walk_model_v3",
                        help="模型保存目录")
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

    # 冒烟测试：只跑 2 万步，模型存到单独的目录，避免覆盖正式训练
    if args.smoke:
        args.steps = 20_000
        args.save_dir = args.save_dir + "_smoke"

    if args.play:
        play(args)
        return 0

    train(args)
    return 0


if __name__ == "__main__":
    # 直接运行本文件时才执行；被 import 时不执行
    # SystemExit 会把 main() 的返回值（0）作为进程退出码
    raise SystemExit(main())
