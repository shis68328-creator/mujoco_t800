"""11_dance.py —— 加载 SDK 的 T800 跳舞策略（dance.mnn），在 MuJoCo 里跳舞。

配置和策略来自：
    engineai_robotics_native_sdk-main/assets/config/t800/rl_dance_example/
        default.yaml
        policies/dance.mnn
        trajectories/dance.npz

和 10.py（走路）的区别：
  - 观测不是“历史帧”，而是参考舞蹈轨迹的命令 + 当前状态
  - 策略输出 25 个关节的动作，叠加到默认关节角上
  - 参考轨迹 dance.npz 提供每个时刻的目标关节角/速度，供策略跟踪

用法：
    # 无窗口播放一遍，打印过程（SSH 可用）
    python 11_dance.py --headless

    # 打开 3D 窗口看跳舞（macOS 用 mjpython）
    python 11_dance.py
    mjpython 11_dance.py

    # 循环跳
    python 11_dance.py --loop

依赖：mujoco、numpy、MNN（pip install MNN==3.6.2）
"""

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    ROOT
    / "engineai_robotics_native_sdk-main/assets/resource/robot/t800/xml/serial_t800.xml"
)
DANCE_DIR = (
    ROOT / "engineai_robotics_native_sdk-main/assets/config/t800/rl_dance_example"
)
POLICY_PATH = DANCE_DIR / "policies/dance.mnn"
TRAJ_PATH = DANCE_DIR / "trajectories/dance.npz"

# 以下参数来自 rl_dance_example/default.yaml
# 注意：这里的关节顺序不是 J00~J24，而是策略训练时用的顺序
JOINT_NAMES = [
    "J00_HIP_PITCH_L", "J06_HIP_PITCH_R", "J12_TORSO_YAW",
    "J01_HIP_ROLL_L", "J07_HIP_ROLL_R",
    "J13_SHOULDER_PITCH_L", "J18_SHOULDER_PITCH_R",
    "J23_HEAD_PITCH", "J02_HIP_YAW_L", "J08_HIP_YAW_R",
    "J14_SHOULDER_ROLL_L", "J19_SHOULDER_ROLL_R", "J24_HEAD_YAW",
    "J03_KNEE_PITCH_L", "J09_KNEE_PITCH_R",
    "J15_SHOULDER_YAW_L", "J20_SHOULDER_YAW_R",
    "J04_ANKLE_PITCH_L", "J10_ANKLE_PITCH_R",
    "J16_ELBOW_PITCH_L", "J21_ELBOW_PITCH_R",
    "J05_ANKLE_ROLL_L", "J11_ANKLE_ROLL_R",
    "J17_ELBOW_YAW_L", "J22_ELBOW_YAW_R",
]

DEFAULT_Q = np.array([
    -0.06, -0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.15, -0.15, 0.0, 0.12, 0.12, 0.0, 0.0, -0.06, -0.06,
    -0.25, -0.25, 0.0, 0.0, 0.0, 0.0,
])

KP = np.array([
    180.0, 180.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 100.0, 100.0,
    50.0, 50.0, 50.0, 180.0, 180.0, 50.0, 50.0, 40.0, 40.0, 50.0,
    50.0, 40.0, 40.0, 50.0, 50.0,
])

KD = np.array([
    5.0, 5.0, 3.0, 3.0, 3.0, 0.3, 0.3, 0.3, 3.0, 3.0,
    0.3, 0.3, 0.3, 5.0, 5.0, 0.3, 0.3, 2.0, 2.0, 0.3,
    0.3, 2.0, 2.0, 0.3, 0.3,
])

ACTION_SCALE = np.array([
    0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
    0.2, 0.2, 0.2, 0.5, 0.5, 0.05, 0.05, 0.5, 0.5,
    0.2, 0.2, 0.2, 0.2, 0.0, 0.0,
])

OBS_DIM = 134      # 50 + 6 + 3 + 25 + 25 + 25
ACTION_DIM = 25
PHYSICS_DT = 0.002
# 轨迹 fps=50，所以控制周期取 0.02 秒（50 Hz）
CONTROL_DT = 0.02
FRAME_SKIP = int(round(CONTROL_DT / PHYSICS_DT))  # 10 个物理步


def quat_to_mat(quat, mat):
    """四元数 (w,x,y,z) -> 3x3 旋转矩阵，写入 mat 并返回 reshape 后的视图。"""
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


def yaw_rot(yaw):
    """绕 z 轴旋转 yaw 弧度的 3x3 矩阵。"""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Dance:
    """封装 MuJoCo 模型、MNN 策略和参考舞蹈轨迹。"""

    def __init__(self, policy_path: Path, traj_path: Path):
        # ---- 模型 + 地面 ----
        spec = mujoco.MjSpec.from_file(str(MODEL_PATH))
        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [30.0, 30.0, 0.1]
        floor.rgba = [0.45, 0.48, 0.52, 1.0]

        self.model = spec.compile()
        self.model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
        self.model.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]
        self.data = mujoco.MjData(self.model)

        # ---- 关节映射：策略顺序 -> MuJoCo 下标 ----
        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ]
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in self.joint_ids])
        self.vadr = np.array([self.model.jnt_dofadr[j] for j in self.joint_ids])

        # 25 个关节对应的执行器（电机）下标
        self.act_joint = self.model.actuator_trnid[:, 0]
        self.act_idx = np.array([
            int(np.where(self.act_joint == j)[0][0]) for j in self.joint_ids
        ])
        self.act_qadr = self.model.jnt_qposadr[self.act_joint]
        self.act_vadr = self.model.jnt_dofadr[self.act_joint]

        # 每个电机的 PD 增益：先把策略顺序的 kp/kd 填进去
        self.kp = np.full(self.model.nu, 100.0)
        self.kd = np.full(self.model.nu, 2.0)
        self.kp[self.act_idx] = KP
        self.kd[self.act_idx] = KD

        # 力矩上下限
        self.tau_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.tau_high = self.model.actuator_ctrlrange[:, 1].copy()

        # ---- 参考舞蹈轨迹 ----
        traj = np.load(traj_path)
        self.ref_pos = traj["joint_pos"].astype(np.float64)  # (1024, 25)
        self.ref_vel = traj["joint_vel"].astype(np.float64)  # (1024, 25)
        self.ref_quat = traj["body_quat_w"].astype(np.float64)  # (1024, 30, 4)
        self.max_step = len(self.ref_pos) - 1

        # ---- MNN 策略 ----
        try:
            import MNN
        except ImportError as exc:
            raise RuntimeError("当前环境没有 MNN，请先执行：pip install MNN==3.6.2") from exc
        self.MNN = MNN
        self.net = MNN.Interpreter(str(policy_path))
        self.session = self.net.createSession()
        self.input_tensor = self.net.getSessionInput(self.session)
        self.output_tensor = self.net.getSessionOutput(self.session)

        # ---- 运行时状态 ----
        self.mat = np.zeros(9)
        self.policy_step = 0
        self.last_action = np.zeros(ACTION_DIM)
        self.is_first = True
        # 第一帧 yaw 对齐
        self.ref_yaw_rot = np.eye(3)
        self.body_yaw_rot = np.eye(3)

    # ------------------------------------------------------------------
    def reset(self):
        """回到默认姿态，policy_step 归零。"""
        mujoco.mj_resetData(self.model, self.data)

        qpos = np.zeros(self.model.nq)
        qpos[2] = 1.03
        qpos[3] = 1.0
        # 按策略顺序把默认角写进 qpos
        qpos[self.qadr] = DEFAULT_Q
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.policy_step = 0
        self.last_action[:] = 0.0
        self.is_first = True

    # ------------------------------------------------------------------
    def _yaw_alignment(self):
        """第一帧：计算参考轨迹和机器人当前的 yaw，后面用来对齐。"""
        R_ref = quat_to_mat(self.ref_quat[0, 0], self.mat)
        ref_yaw = math.atan2(R_ref[1, 0], R_ref[0, 0])
        R_body = quat_to_mat(self.data.qpos[3:7], self.mat)
        body_yaw = math.atan2(R_body[1, 0], R_body[0, 0])
        self.ref_yaw_rot = yaw_rot(ref_yaw)
        self.body_yaw_rot = yaw_rot(body_yaw)

    # ------------------------------------------------------------------
    def _observation(self):
        """拼 134 维观测：command 50 + 姿态 6 + 角速度 3 + 关节 25+25 + 动作 25。"""
        step = min(self.policy_step, self.max_step)

        # 1) command：参考关节角 25 + 参考关节速度 25
        command = np.concatenate([self.ref_pos[step], self.ref_vel[step]])

        # 2) motion_anchor_ori_b：参考姿态和当前姿态在机体坐标系的差
        R_ref = quat_to_mat(self.ref_quat[step, 0], self.mat).copy()
        R_body = quat_to_mat(self.data.qpos[3:7], self.mat)
        ref_aligned = self.ref_yaw_rot.T @ R_ref
        body_aligned = self.body_yaw_rot.T @ R_body
        M = body_aligned.T @ ref_aligned
        # C++ 里是 leftCols(2).reshaped<RowMajor>()，即前两列按行展开
        anchor_ori = np.array([M[0, 0], M[0, 1],
                               M[1, 0], M[1, 1],
                               M[2, 0], M[2, 1]])

        # 3) 基座角速度
        base_ang_vel = self.data.qvel[3:6]

        # 4) 关节角（相对默认）和关节速度，按策略顺序
        joint_pos = self.data.qpos[self.qadr] - DEFAULT_Q
        joint_vel = self.data.qvel[self.vadr]

        # 5) 上一步动作
        return np.concatenate([
            command,          # 50
            anchor_ori,       # 6
            base_ang_vel,     # 3
            joint_pos,        # 25
            joint_vel,        # 25
            self.last_action, # 25
        ])

    # ------------------------------------------------------------------
    def _infer(self, obs):
        """MNN 前向推理，返回 25 维动作。"""
        MNN = self.MNN
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        inp = MNN.Tensor(
            self.input_tensor.getShape(), MNN.Halide_Type_Float, x,
            MNN.Tensor_DimensionType_Caffe,
        )
        self.input_tensor.copyFrom(inp)
        self.net.runSession(self.session)

        out = MNN.Tensor(
            self.output_tensor.getShape(), MNN.Halide_Type_Float,
            np.zeros(self.output_tensor.getShape(), np.float32),
            MNN.Tensor_DimensionType_Caffe,
        )
        self.output_tensor.copyToHostTensor(out)
        return np.asarray(out.getData(), dtype=np.float64).reshape(-1)

    # ------------------------------------------------------------------
    def step(self):
        """走一个控制步（0.02 秒）。"""
        if self.is_first:
            self._yaw_alignment()
            self.is_first = False

        obs = self._observation()
        action = self._infer(obs)
        self.last_action = action.copy()

        # resident_control=false：q_des = default + action*action_scale
        target = np.zeros(self.model.nu)
        target[self.act_idx] = DEFAULT_Q + action * ACTION_SCALE

        for _ in range(FRAME_SKIP):
            q = self.data.qpos[self.act_qadr]
            qd = self.data.qvel[self.act_vadr]
            tau = self.kp * (target - q) - self.kd * qd
            tau = np.clip(tau, self.tau_low, self.tau_high)
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)

        self.policy_step = min(self.policy_step + 1, self.max_step)

    # ------------------------------------------------------------------
    def finished(self):
        return self.policy_step >= self.max_step

    def body_state(self):
        """返回 (x, y, z, tilt_deg)。"""
        R = quat_to_mat(self.data.qpos[3:7], self.mat)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(R[2, 2])))))
        return (float(self.data.qpos[0]), float(self.data.qpos[1]),
                float(self.data.qpos[2]), tilt)


def run_headless(dance, loop):
    """无窗口播放，打印进度。"""
    dance.reset()
    print(f"开始跳舞：共 {dance.max_step + 1} 帧，控制周期 {CONTROL_DT:.3f}s")
    total = 0
    while True:
        dance.step()
        total += 1
        if total % 50 == 0:
            x, y, z, tilt = dance.body_state()
            print(f" frame={dance.policy_step:4d}  t={total * CONTROL_DT:5.1f}s  "
                  f"pos=({x:+.2f},{y:+.2f},{z:.2f})  tilt={tilt:.1f}°")
        if dance.finished() and not loop:
            break

    x, y, z, tilt = dance.body_state()
    print(f"\n结束：共 {total} 个控制步，最终位置 ({x:+.2f},{y:+.2f},{z:.2f})，倾斜 {tilt:.1f}°")


def run_viewer(dance, loop):
    """开窗口跳舞。"""
    import mujoco.viewer

    dance.reset()
    try:
        with mujoco.viewer.launch_passive(dance.model, dance.data) as viewer:
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -10.0
            viewer.cam.azimuth = 135.0
            viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
            print("跳舞策略已加载；关闭窗口退出。")

            next_time = time.time()
            while viewer.is_running():
                dance.step()
                x, y, z, tilt = dance.body_state()
                viewer.cam.lookat[:] = [x, y, 1.0]
                viewer.sync()

                if dance.finished():
                    if loop:
                        dance.reset()
                    else:
                        print("舞蹈播放完毕，关闭窗口退出。")
                        while viewer.is_running():
                            viewer.sync()
                            time.sleep(0.02)
                        break

                next_time += CONTROL_DT
                lag = next_time - time.time()
                if lag > 0:
                    time.sleep(min(lag, 0.01))
    except Exception as exc:
        print(f"无法打开 3D 窗口：{exc}")
        print("没有显示器时请用：python 11_dance.py --headless")


def main():
    parser = argparse.ArgumentParser(description="T800 MNN 跳舞策略")
    parser.add_argument("--headless", action="store_true", help="不开窗口，打印过程")
    parser.add_argument("--loop", action="store_true", help="循环播放舞蹈")
    parser.add_argument("--policy", type=str, default=str(POLICY_PATH))
    parser.add_argument("--traj", type=str, default=str(TRAJ_PATH))
    args = parser.parse_args()

    policy = Path(args.policy)
    traj = Path(args.traj)
    if not policy.exists() or not traj.exists():
        print(f"找不到文件：{policy} 或 {traj}")
        return 1

    try:
        dance = Dance(policy, traj)
    except Exception as exc:
        print(f"加载失败：{exc}")
        return 1

    if args.headless:
        run_headless(dance, args.loop)
    else:
        run_viewer(dance, args.loop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
