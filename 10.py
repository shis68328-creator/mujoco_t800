"""10.py —— 加载 SDK 训练好的 MNN 走路策略，让 T800 在 MuJoCo 里走起来。

======================================================================
一、这个脚本是干什么的
======================================================================
不训练，只做“推理”：
    读取 MuJoCo 模型和 MNN 策略
        -> 每个控制周期拼出 1083 维观测
        -> 交给 MNN 网络算出 22 维动作
        -> 动作换算成 25 个关节的目标角
        -> 用 PD 控制让关节跟踪目标角
        -> 推进 5 个物理步（0.01 秒），循环

配置完全对齐 SDK 里的：
    engineai_robotics_native_sdk-main/assets/config/t800/rl_walking_example/default.yaml
以及它对应的策略文件：
    .../rl_walking_example/policy/t800_260318_150533_60000.mnn

======================================================================
二、名词表（每个专有名词单独解释）
======================================================================
【MJCF】
    MuJoCo 的 XML 模型格式，描述机器人有哪些连杆、关节、几何体、电机。

【MjSpec】
    MuJoCo 解析 XML 后得到的“模型草稿”。可以在编译前继续加地面、改参数。
    `MjSpec.from_file(path)` 返回一个 MjSpec 对象。

【MjModel】
    MjSpec 编译后的“只读模型”，包含质量、关节上下限、电机增益等固定参数。
    `spec.compile()` 返回 MjModel。仿真过程中不会变。

【MjData】
    “运行时状态”，保存 qpos、qvel、ctrl、time、每个 body 的世界坐标等。
    `MjData(model)` 返回 MjData。每次 mj_step 都会更新它。

【qpos】
    位置向量（position）。长度 nq = 32：
      qpos[0:3]   基座在世界系的位置 x,y,z
      qpos[3:7]   基座姿态四元数 w,x,y,z
      qpos[7:32]  25 个关节的角度（弧度）
    自由关节占 7 个数，每个旋转关节占 1 个数。

【qvel】
    速度向量（velocity）。长度 nv = 31：
      qvel[0:3]   基座线速度（世界系）
      qvel[3:6]   基座角速度（机体系）
      qvel[6:31]  25 个关节的角速度（弧度/秒）
    自由关节占 6 个数，每个旋转关节占 1 个数。

【DOF / 自由度】
    独立运动方向的数量，等于 nv。T800 的 nv=31。
    自由关节 6 个自由度 + 25 个关节 = 31。

【free joint（自由关节）】
    让基座可以在空间中平移 + 旋转的关节，有 6 个自由度。
    它没有名字，qpos 固定占前 7 个位置。

【四元数 quaternion】
    表示三维旋转的 4 个数 (w, x, y, z)，比欧拉角更稳定、不会万向锁。
    MuJoCo 的 qpos[3:7] 就是基座姿态的四元数。

【旋转矩阵】
    3x3 矩阵 R，把“机体系”的向量转到“世界系”。
    `mju_quat2Mat(mat, quat)` 把四元数转成 9 个数（行优先的 3x3 矩阵）。

【投影重力 projected gravity】
    把世界系的重力方向 (0,0,-1) 转到机体系，得到 3 个数。
    机器人完全直立时约等于 (0,0,-1)；倾斜时 x/y 分量会变化。
    策略用它感知“我歪了多少”。

【actuator / 执行器 / 电机】
    驱动机器人的元件。T800 有 25 个力矩电机，每个控制一个关节。
    `model.actuator_trnid[:,0]` 记录每个电机控制哪个关节。

【ctrl】
    控制输入向量，长度 25。这里每个数就是一个电机的目标力矩（N·m）。
    代码里写 `data.ctrl[:] = tau`，mj_step 会照着这个力矩去驱动。

【ctrlrange】
    每个电机允许的力矩范围。代码里再乘 80%/90% 的“软限制”。
    `model.actuator_ctrlrange[:,0]` 是下限，`[:,1]` 是上限。

【PD 控制】
    最常用的关节控制方法：
        力矩 = kp × (目标角度 - 当前角度) - kd × 当前角速度
    第一项把关节拉向目标，第二项抑制抖动。
    kp 越大越“硬”，kd 越大越“稳”。

【action / 动作】
    策略网络输出的 22 个数，是“归一化的关节偏移”，不是角度本身。
    代码里 `action * ACTION_SCALE` 之后才变成弧度。

【action_scale】
    把策略输出映射到实际关节角偏移的系数。
    例如髋 pitch 的 scale=0.5，动作 1.0 就对应 0.5 rad 的目标偏移。

【observation / 观测】
    喂给策略的输入向量。这个策略要 1083 个数：
        15 帧 × 72 维单帧 + 3 维速度指令 = 1080 + 3 = 1083。

【history / 历史帧】
    只给一帧观测，策略不知道“速度”和“趋势”；
    把过去 15 帧拼起来，策略就能从变化中推断运动状态。

【command / 速度指令】
    3 个数 [vx, vy, wz]：期望前进速度、横移速度、转向角速度。
    这里固定 [0.6, 0, 0]，即只让它往前走。

【MNN】
    阿里巴巴开源的推理引擎。这里只用来“加载已经训练好的神经网络并前向计算”，
    不涉及训练。策略文件是 .mnn 格式。

【Interpreter / Session / Tensor】
    Interpreter：加载 .mnn 文件。
    Session：一次推理会话，保存运行时的中间结果。
    Tensor：多维数组，网络的输入/输出都是 Tensor。
    `copyFrom` 把 Python 数据拷进输入 Tensor；
    `runSession` 执行前向计算；
    `copyToHostTensor` 把输出拷回 Python。

【inference / 推理】
    只做“输入 -> 网络 -> 输出”，不更新权重，和训练相反。

【tilt / 倾斜角】
    机身 z 轴和世界 z 轴的夹角。直立时 0°，躺下时接近 90°。

【Caffe 维度顺序】
    MNN 创建 Tensor 时的维度约定，shape=(1, 1083) 表示 1 条样本、1083 个特征。

======================================================================
三、用法
======================================================================
    # 无窗口跑 10 秒，测走了多远（SSH 也能用）
    python 10.py --headless

    # 指定目标速度和时长
    python 10.py --headless --command 0.6 --seconds 15

    # 打开 3D 窗口看它走（macOS 用 mjpython）
    python 10.py
    mjpython 10.py

依赖：mujoco、numpy、MNN（pip install MNN==3.6.2）
"""

import argparse          # 解析命令行参数，返回 Namespace
import math              # acos / degrees
import sys               # sys.exit：设置退出码
import time              # 计时、控制窗口刷新节奏
from pathlib import Path  # 拼接文件路径

import mujoco             # 物理引擎
import numpy as np        # 数组运算

# 本文件所在目录，用来拼模型和策略的路径
ROOT = Path(__file__).resolve().parent

# T800 的 MJCF 模型文件
MODEL_PATH = (
    ROOT
    / "engineai_robotics_native_sdk-main/assets/resource/robot/t800/xml/serial_t800.xml"
)

# SDK 训练好的 MNN 走路策略
POLICY_PATH = (
    ROOT
    / "engineai_robotics_native_sdk-main/assets/config/t800/rl_walking_example"
    / "policy/t800_260318_150533_60000.mnn"
)

# ======================================================================
# 以下数值全部来自 rl_walking_example/default.yaml
# ======================================================================

# 参与策略的 22 个关节编号。
# 25 个关节里跳过 J12（躯干偏航）、J23/J24（头部），它们不参与走路。
ACTIVE = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22])

# 25 个关节的默认角度（弧度）。
# 顺序：左腿 6 个、右腿 6 个、躯干 1 个、左臂 5 个、右臂 5 个、头 2 个。
DEFAULT_Q = np.array(
    [-0.06, 0, 0, 0.12, -0.06, 0]
    + [-0.06, 0, 0, 0.12, -0.06, 0]
    + [0.0]
    + [0.0, 0.15, 0.0, -0.25, 0.0]
    + [0.0, -0.15, 0.0, -0.25, 0.0]
    + [0.0, 0.0]
)

# PD 比例增益 kp：每个关节一个值，共 25 个。
# kp 越大，关节越“硬”，越用力保持目标角度。
KP = np.array(
    [180, 100, 100, 180, 40, 40]
    + [180, 100, 100, 180, 40, 40]
    + [100]
    + [60, 50, 50, 60, 50]
    + [60, 50, 50, 60, 50]
    + [100, 100],
    dtype=float,
)

# PD 微分增益 kd：抑制关节速度，防止来回摆动。
KD = np.array(
    [5, 3, 3, 5, 0.3, 0.3]
    + [5, 3, 3, 5, 0.3, 0.3]
    + [5]
    + [0.3] * 5
    + [0.3] * 5
    + [1, 1],
    dtype=float,
)

# 动作缩放：策略输出的 22 个数 × 这个系数，再加上默认关节角。
ACTION_SCALE = np.array(
    [0.5, 0.2, 0.2, 0.5, 0.5, 0.2]
    + [0.5, 0.2, 0.2, 0.5, 0.5, 0.2]
    + [0.2]
    + [0.2, 0.2, 0.05, 0.2, 0.05]
    + [0.2, 0.2, 0.05, 0.2, 0.05]
    + [0.2, 0.2]
)

# 软力矩限制：腿只用到电机上限的 80%，其他关节 90%。
# 这样留一点余量，控制更稳，也能保护电机。
SOFT_LIMIT = np.array([0.8] * 12 + [0.9] + [0.9] * 5 + [0.9] * 5 + [0.9] * 2)

OBS_STEPS = 15        # 策略需要 15 帧历史
OBS_PER_STEP = 72     # 每帧 72 个数
CONTROL_DT = 0.01     # 控制周期 10ms，即 100 Hz
PHYSICS_DT = 0.002    # MuJoCo 物理步长 2ms
FRAME_SKIP = int(round(CONTROL_DT / PHYSICS_DT))  # 每个控制周期走 5 个物理步


class Walker:
    """把 MuJoCo 模型、MNN 策略、观测历史封装在一起。"""

    def __init__(self, policy_path: Path):
        """加载模型、地面、MNN 策略，并准备好各种下标和缓冲区。

        参数：
            policy_path：.mnn 策略文件路径
        返回：
            无（构造函数）
        """
        # ---- 1. 模型 ----
        # MjSpec 是“模型草稿”，可以在编译前加地面
        spec = mujoco.MjSpec.from_file(str(MODEL_PATH))

        # 原模型没有地面，加一块 30m × 30m 的平面，防止走远了掉下去
        floor = spec.worldbody.add_geom()   # 返回一个几何体对象
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE  # 平面类型
        floor.size = [30.0, 30.0, 0.1]
        floor.rgba = [0.45, 0.48, 0.52, 1.0]      # 灰色

        # 编译成只读 MjModel
        self.model = spec.compile()

        # 调亮头灯，窗口里看得清
        self.model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
        self.model.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]

        # MjData：运行时状态（qpos、qvel、ctrl、time 等）
        self.data = mujoco.MjData(self.model)

        # ---- 2. 建立“电机 -> 关节 -> qpos/qvel 下标”的对应 ----
        # actuator_trnid[:,0]：形状 (25,)，第 i 个电机控制哪个关节
        self.act_joint = self.model.actuator_trnid[:, 0]

        # jnt_qposadr[j]：关节 j 的角度在 qpos 里的下标
        self.act_qadr = self.model.jnt_qposadr[self.act_joint]

        # jnt_dofadr[j]：关节 j 的速度在 qvel 里的下标
        self.act_vadr = self.model.jnt_dofadr[self.act_joint]

        # 每个电机的力矩上下限，再乘软限制
        self.tau_low = self.model.actuator_ctrlrange[:, 0] * SOFT_LIMIT
        self.tau_high = self.model.actuator_ctrlrange[:, 1] * SOFT_LIMIT

        # ---- 3. 加载 MNN 策略 ----
        try:
            import MNN
        except ImportError as exc:
            raise RuntimeError(
                "当前环境没有 MNN，请先执行：pip install MNN==3.6.2"
            ) from exc

        self.MNN = MNN
        # Interpreter：加载 .mnn 文件
        self.net = MNN.Interpreter(str(policy_path))
        # Session：一次推理会话
        self.session = self.net.createSession()
        # 输入/输出 Tensor
        self.input_tensor = self.net.getSessionInput(self.session)
        self.output_tensor = self.net.getSessionOutput(self.session)

        # ---- 4. 观测历史、动作、指令 ----
        # hist：15 帧 × 72 维；reshape(-1) 后就是 1080 维
        self.hist = np.zeros((OBS_STEPS, OBS_PER_STEP))
        # 上一帧策略输出，22 维
        self.last_action = np.zeros(22)
        # 第一帧时要把 15 行历史都填成当前帧
        self.first_frame = True
        # 速度指令 [vx, vy, wz]：前进、横移、转向
        self.command = np.array([0.6, 0.0, 0.0])

        # 复用 3x3 旋转矩阵缓冲，避免每帧新建
        self.mat = np.zeros(9)

    # ------------------------------------------------------------------
    def reset(self):
        """把机器人恢复成默认站立姿态。

        返回：
            无。会原地修改 self.data、self.hist、self.last_action。
        """
        # mj_resetData：把 MjData 恢复成模型默认状态，返回 None
        mujoco.mj_resetData(self.model, self.data)

        # 手动设置初始 qpos
        qpos = np.zeros(self.model.nq)
        qpos[2] = 1.03          # 基座高度 1.03m
        qpos[3] = 1.0           # 四元数 w=1，表示没有旋转
        qpos[7:] = DEFAULT_Q    # 25 个关节角

        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        # mj_forward：根据 qpos 重新计算所有坐标/碰撞，返回 None
        mujoco.mj_forward(self.model, self.data)

        # 清空历史和动作
        self.hist[:] = 0.0
        self.last_action[:] = 0.0
        self.first_frame = True

    # ------------------------------------------------------------------
    def _build_frame(self):
        """拼出一帧 72 维观测，并更新 15 帧历史。

        返回：
            np.ndarray，形状 (1083,)：
            15×72 的历史 + 3 维速度指令。
        """
        data = self.data

        # 四元数 -> 3x3 旋转矩阵（写入 self.mat，函数返回 None）
        mujoco.mju_quat2Mat(self.mat, data.qpos[3:7])
        rot = self.mat.reshape(3, 3)   # body -> world 的旋转矩阵

        q = data.qpos[7:32]    # 25 个关节角
        qd = data.qvel[6:31]   # 25 个关节角速度

        # 单帧 72 维
        frame = np.zeros(OBS_PER_STEP)
        frame[0:22] = q[ACTIVE] - DEFAULT_Q[ACTIVE]  # 关节角相对默认角
        frame[22:44] = qd[ACTIVE] * 0.05             # 关节速度，缩放 0.05
        frame[44:66] = self.last_action              # 上一步动作，22 维
        frame[66:69] = data.qvel[3:6]                # 基座角速度，3 维
        frame[69:72] = -rot[2, :]                    # 投影重力，3 维

        # 维护历史：第一帧时 15 行都填当前帧，之后左移、末尾放新帧
        if self.first_frame:
            self.hist[:] = frame
            self.first_frame = False
        else:
            self.hist[:-1] = self.hist[1:]
            self.hist[-1] = frame

        # 拼接：15×72=1080，再加 3 维指令，共 1083
        return np.concatenate([self.hist.reshape(-1), self.command])

    # ------------------------------------------------------------------
    def _infer(self, obs):
        """把观测送进 MNN 网络，得到 22 维动作。

        参数：
            obs：形状 (1083,) 的 numpy 数组
        返回：
            np.ndarray，形状 (22,)，已经裁剪到 [-100, 100]
        """
        MNN = self.MNN

        # 输入 Tensor 的形状一般是 (1, 1083)
        # reshape(1,-1) 增加一个 batch 维度
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)

        # 创建输入 Tensor，把数据拷进去
        inp = MNN.Tensor(
            self.input_tensor.getShape(),     # 网络期望的形状
            MNN.Halide_Type_Float,            # 数据类型 float32
            x,                                # 数据
            MNN.Tensor_DimensionType_Caffe,   # 维度约定
        )
        self.input_tensor.copyFrom(inp)       # 返回 None
        self.net.runSession(self.session)     # 执行前向推理，返回 None

        # 创建输出 Tensor，把结果拷回来
        out = MNN.Tensor(
            self.output_tensor.getShape(),
            MNN.Halide_Type_Float,
            np.zeros(self.output_tensor.getShape(), np.float32),
            MNN.Tensor_DimensionType_Caffe,
        )
        self.output_tensor.copyToHostTensor(out)   # 返回 None
        action = np.asarray(out.getData(), dtype=np.float64).reshape(-1)

        # 和 C++ runner 一样把动作裁剪到 [-100, 100]
        return np.clip(action, -100.0, 100.0)

    # ------------------------------------------------------------------
    def step(self):
        """执行一个控制步（0.01 秒）。

        返回：
            无。会原地更新 self.data（机器人状态）和 self.last_action。
        """
        # 1. 拼观测：1083 维
        obs = self._build_frame()

        # 2. MNN 推理：22 维动作
        action = self._infer(obs)
        self.last_action = action.copy()

        # 3. 动作 -> 25 个关节目标角
        #    target 初始等于默认角，只把 22 个 active 关节加上偏移
        target = DEFAULT_Q.copy()
        target[ACTIVE] += action * ACTION_SCALE[ACTIVE]

        # 4. PD 控制，一个控制周期内做 5 个物理步
        for _ in range(FRAME_SKIP):
            # 当前关节角、角速度
            q = self.data.qpos[self.act_qadr]
            qd = self.data.qvel[self.act_vadr]

            # PD 公式：力矩 = kp*(目标-当前) - kd*当前速度
            tau = KP * (target - q) - KD * qd

            # 按软力矩限制裁剪
            tau = np.clip(tau, self.tau_low, self.tau_high)

            # 写入 ctrl，然后推进一步物理
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------------------------
    def body_state(self):
        """读基座的位置、高度和倾斜角。

        返回：
            (x, z, tilt)
            x    : 基座世界 x 坐标，float
            z    : 基座高度，float
            tilt : 倾斜角（度），float
        """
        mujoco.mju_quat2Mat(self.mat, self.data.qpos[3:7])
        # mat[8] 是机身 z 轴在世界系里的 z 分量；直立时=1，躺下时=0
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(self.mat[8])))))
        return float(self.data.qpos[0]), float(self.data.qpos[2]), tilt


def run_headless(walker, seconds, command):
    """不开窗口，跑指定秒数并打印走了多远。

    参数：
        walker  : Walker 实例
        seconds : 仿真多少秒
        command : 目标前进速度 m/s
    返回：
        无，结果打印到终端。
    """
    walker.command = np.array([command, 0.0, 0.0])
    walker.reset()

    x0 = float(walker.data.qpos[0])   # 起点
    max_x = x0
    steps = int(round(seconds / CONTROL_DT))  # 总控制步数
    fell = False

    print(f"开始：command={command:.2f} m/s，时长={seconds:.0f}s")
    for i in range(steps):
        walker.step()
        x, z, tilt = walker.body_state()
        max_x = max(max_x, x)

        # 每 100 步（1 秒）打印一次
        if (i + 1) % 100 == 0:
            print(f" t={(i + 1) * CONTROL_DT:5.1f}s  x={x - x0:+.2f}m  "
                  f"z={z:.2f}m  tilt={tilt:.1f}°")

        # 倾斜超过 60° 或高度低于 0.5m，认为摔倒
        if tilt > 60.0 or z < 0.5:
            fell = True
            print(f" 在 t={(i + 1) * CONTROL_DT:.2f}s 摔倒")
            break

    x, z, tilt = walker.body_state()
    dist = x - x0
    print(f"\n结束：实际前进 {dist:.2f} m，最大 {max_x - x0:.2f} m，"
          f"平均 {dist / max(seconds, 1e-9):.2f} m/s，"
          f"高度 {z:.2f} m，倾斜 {tilt:.1f}°，{'摔倒' if fell else '未摔倒'}")


def run_viewer(walker, command):
    """打开 MuJoCo 3D 窗口，实时看机器人走。

    参数：
        walker  : Walker 实例
        command : 目标前进速度 m/s
    返回：
        无。关闭窗口后函数结束。
    """
    import mujoco.viewer  # 只有开窗口才需要，放到函数里导入

    walker.command = np.array([command, 0.0, 0.0])
    walker.reset()

    try:
        # launch_passive 返回一个上下文管理器，with 结束后自动关窗口
        with mujoco.viewer.launch_passive(walker.model, walker.data) as viewer:
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -10.0
            viewer.cam.azimuth = 135.0
            viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

            print(f"策略已加载：command={command:.2f} m/s；关闭窗口退出。")
            x0 = float(walker.data.qpos[0])
            next_time = time.time()   # 墙上时间，用来按真实速度播放
            next_log = 1.0

            while viewer.is_running():   # 返回 bool，关窗口后变 False
                walker.step()

                x, z, tilt = walker.body_state()
                # 相机跟着机器人走
                viewer.cam.lookat[:] = [x, 0.0, 1.0]
                viewer.sync()            # 刷新画面，返回 None

                t = walker.data.time
                if t >= next_log:
                    print(f" t={t:5.1f}s  x={x - x0:+.2f}m  z={z:.2f}m  tilt={tilt:.1f}°")
                    next_log += 1.0

                # 按真实时间播放：每个控制步 0.01 秒
                next_time += CONTROL_DT
                lag = next_time - time.time()
                if lag > 0:
                    time.sleep(min(lag, 0.01))
    except Exception as exc:
        print(f"无法打开 3D 窗口：{exc}")
        print("没有显示器时请用：python 10.py --headless")


def main():
    """解析命令行参数，决定无窗口评估还是开窗口。

    返回：
        int：0 表示成功，1 表示失败（路径不对/加载失败）。
    """
    # argparse：把命令行里的 --headless、--seconds 等变成 args 的属性
    parser = argparse.ArgumentParser(description="T800 预训练 MNN 走路策略")
    parser.add_argument("--headless", action="store_true",
                        help="不开窗口，跑完打印距离（SSH 可用）")
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="无窗口模式跑多少秒")
    parser.add_argument("--command", type=float, default=0.6,
                        help="目标前进速度 m/s（默认 0.6）")
    parser.add_argument("--policy", type=str, default=str(POLICY_PATH),
                        help="MNN 策略文件路径")
    args = parser.parse_args()   # 返回 Namespace

    policy = Path(args.policy)
    if not policy.exists():      # exists() 返回 bool
        print(f"找不到策略文件：{policy}")
        return 1

    try:
        walker = Walker(policy)
    except Exception as exc:
        print(f"加载失败：{exc}")
        return 1

    if args.headless:
        run_headless(walker, args.seconds, args.command)
    else:
        run_viewer(walker, args.command)
    return 0


if __name__ == "__main__":
    # 直接运行本文件时执行；sys.exit 把返回值当退出码
    sys.exit(main())
