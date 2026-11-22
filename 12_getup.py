"""12_getup.py —— 调用 SDK 的“躺下 / 起身”两个 MNN 策略。

SDK 里除了走路、跳舞，还有两个全身动作策略：

    rl_stance_to_supine   站立 -> 躺下
        policy/T800_stand_to_supine.mnn
        trajectory/T800_stand_to_supine.csv

    rl_supine_to_stance   躺下 -> 站立（起身）
        policy/T800_supine_to_stance.mnn
        trajectory/T800_supine_to_stance.npy

两个策略的观测都是 430 维：
    5 帧历史 × 81 + 25 维目标
    单帧 81 = 关节角 25 + 关节速度 25 + 上一动作 25 + 角速度 3 + 投影重力 3
    目标 25 = 当前关节角 - 参考轨迹关节角

动作：目标角 = 参考轨迹关节角 + 策略输出 × 0.25

用法：
    # 无窗口：先躺下，再起身
    python 12_getup.py --headless

    # 只看躺下 / 只看起身
    python 12_getup.py --mode lie --headless
    python 12_getup.py --mode getup --headless

    # 打开 3D 窗口（macOS 用 mjpython）
    mjpython 12_getup.py
    mjpython 12_getup.py --mode getup
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
CFG = ROOT / "engineai_robotics_native_sdk-main/assets/config/t800"

# 25 个关节，顺序就是策略训练时的顺序（J00~J24）
JOINT_NAMES = [
    "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R", "J07_HIP_ROLL_R", "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
    "J12_TORSO_YAW",
    "J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L", "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L",
    "J18_SHOULDER_PITCH_R", "J19_SHOULDER_ROLL_R", "J20_SHOULDER_YAW_R",
    "J21_ELBOW_PITCH_R", "J22_ELBOW_YAW_R",
    "J23_HEAD_PITCH", "J24_HEAD_YAW",
]

# 两个策略共用的 qd 掩码和力矩上限（来自各自的 default.yaml）
QD_MASK = np.array([1, 1, 1, 1, 0, 0] * 2 + [1] + [1] * 5 + [1] * 5 + [1, 1], dtype=float)
TAU_MAX = np.array(
    [415, 370, 222, 415, 160, 160] * 2
    + [222]
    + [160, 160, 160, 160, 52]
    + [160, 160, 160, 160, 52]
    + [52, 52],
    dtype=float,
)

OBS_STEPS = 5
DT = 0.01            # 100 Hz（task_motion 里两个动作的 period 都是 0.01）
PHYSICS_DT = 0.002
FRAME_SKIP = int(round(DT / PHYSICS_DT))  # 5 个物理步


# ----------------------------------------------------------------------
# 两个动作的配置
# ----------------------------------------------------------------------
LIE_CFG = {
    "name": "站立 -> 躺下",
    "policy": CFG / "rl_stance_to_supine/policy/T800_stand_to_supine.mnn",
    "traj": CFG / "rl_stance_to_supine/trajectory/T800_stand_to_supine.csv",
    "traj_type": "csv",
    "frame_start": 0,
    "frame_end": 70,
    "init": "stand",
    "default_q": np.array([
        -0.7732778293, 0.2109002427, -0.2140328236, 0.9002003037,
        -0.2894360952, -0.1873647404,
        -0.6147637016, -0.1066712524, 0.0556522214, 0.8299468884,
        -0.3985402688, 0.1808947799,
        0.0235501909,
        -0.6900132027, 0.9303021872, -0.1659253576, -1.9041238125,
        -0.2584387427,
        -1.1696804212, -0.1520707048, 0.6273249496, -1.8817574626,
        0.2462114231,
        0.1889251056, 0.6451696690,
    ]),
    "kp": np.array([180, 95, 34, 180, 12, 12] * 2 + [34]
                   + [24] * 5 + [24] * 5 + [2.5, 2.5], dtype=float),
    "kd": np.array([21, 11, 4, 21, 1.5, 1.5] * 2 + [4]
                   + [1.5] * 5 + [1.5] * 5 + [0.5, 0.5], dtype=float),
    "scale": np.full(25, 0.25),
}

GETUP_CFG = {
    "name": "躺下 -> 站立（起身）",
    "policy": CFG / "rl_supine_to_stance/policy/T800_supine_to_stance.mnn",
    "traj": CFG / "rl_supine_to_stance/trajectory/T800_supine_to_stance.npy",
    "traj_type": "npy",
    "frame_start": 90,
    "frame_end": 180,
    "init": "supine",
    "default_q": np.zeros(25),
    "kp": np.array([280.4, 95.3, 34.3, 280.4, 12.6, 12.6] * 2 + [34.3]
                   + [22.4] * 5 + [22.4] * 5 + [2.4, 2.4], dtype=float),
    "kd": np.array([41.1, 21.1, 4.0, 41.1, 1.5, 1.5] * 2 + [4.0]
                   + [2.0] * 5 + [2.0] * 5 + [0.3, 0.3], dtype=float),
    "scale": np.full(25, 0.25),
}


def load_reference(cfg):
    """读取参考轨迹并插值到 100 Hz。

    CSV：按表头里的关节名取 25 列。
    NPY：每行前 3 个是 base 位置、接着 4 个四元数、再 25 个关节角；
         这里只取关节角。
    返回：(参考关节角 [N,25], 原始数据)
    """
    if cfg["traj_type"] == "csv":
        with open(cfg["traj"]) as f:
            header = f.readline().strip().split(",")
        data = np.loadtxt(cfg["traj"], delimiter=",", skiprows=1)
        cols = [header.index(name) for name in JOINT_NAMES]
        raw = data[cfg["frame_start"]:cfg["frame_end"] + 1][:, cols]
        raw = raw.astype(np.float64)
        src_dt = 1.0 / 30.0
        base = None
    else:
        data = np.load(cfg["traj"]).astype(np.float64)
        raw = data[cfg["frame_start"]:cfg["frame_end"] + 1, 7:32]
        src_dt = 1.0 / 30.0
        base = data[cfg["frame_start"], 0:7].copy()

    total_time = (len(raw) - 1) * src_dt
    n = int(round(total_time / DT)) + 1
    t_src = np.arange(len(raw)) * src_dt
    t_dst = np.arange(n) * DT
    ref = np.zeros((n, 25))
    for j in range(25):
        ref[:, j] = np.interp(t_dst, t_src, raw[:, j])
    return ref, base


class MimicSkill:
    """加载一个 MNN 动作策略，并维护 5 帧观测历史。"""

    def __init__(self, model, data, cfg):
        self.model = model
        self.data = data
        self.cfg = cfg
        self.ref, self.base = load_reference(cfg)

        # 关节/执行器映射
        self.joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.joint_ids])
        self.vadr = np.array([model.jnt_dofadr[j] for j in self.joint_ids])
        self.act_joint = model.actuator_trnid[:, 0]
        self.act_idx = np.array([
            int(np.where(self.act_joint == j)[0][0]) for j in self.joint_ids
        ])
        self.act_qadr = model.jnt_qposadr[self.act_joint]
        self.act_vadr = model.jnt_dofadr[self.act_joint]

        # 每个电机的 PD 增益
        self.kp = np.full(model.nu, 100.0)
        self.kd = np.full(model.nu, 2.0)
        self.kp[self.act_idx] = cfg["kp"]
        self.kd[self.act_idx] = cfg["kd"]

        # MNN 策略
        try:
            import MNN
        except ImportError as exc:
            raise RuntimeError("当前环境没有 MNN，请先执行：pip install MNN==3.6.2") from exc
        self.MNN = MNN
        self.net = MNN.Interpreter(str(cfg["policy"]))
        self.session = self.net.createSession()
        self.input_tensor = self.net.getSessionInput(self.session)
        self.output_tensor = self.net.getSessionOutput(self.session)

        # 观测历史：q / qd / action / w / gravity 各 5 帧
        self.q_hist = np.zeros((OBS_STEPS, 25))
        self.qd_hist = np.zeros((OBS_STEPS, 25))
        self.a_hist = np.zeros((OBS_STEPS, 25))
        self.w_hist = np.zeros((OBS_STEPS, 3))
        self.g_hist = np.zeros((OBS_STEPS, 3))
        self.first = True
        self.last_action = np.zeros(25)
        self.step_idx = 0
        self.mat = np.zeros(9)

    # ------------------------------------------------------------------
    def reset(self):
        """按动作类型设置初始姿态。"""
        mujoco.mj_resetData(self.model, self.data)
        qpos = np.zeros(self.model.nq)

        if self.cfg["init"] == "stand":
            qpos[2] = 1.03
            qpos[3] = 1.0
            qpos[self.qadr] = 0.0
        else:
            # 从 NPY 第一帧取躺下的 base 位姿和关节角
            qpos[0:3] = self.base[0:3]
            qpos[3:7] = self.base[3:7]
            qpos[self.qadr] = self.ref[0]

        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.q_hist[:] = 0.0
        self.qd_hist[:] = 0.0
        self.a_hist[:] = 0.0
        self.w_hist[:] = 0.0
        self.g_hist[:] = 0.0
        self.first = True
        self.last_action[:] = 0.0
        self.step_idx = 0

    # ------------------------------------------------------------------
    def _obs(self):
        """拼 430 维观测。"""
        step = min(self.step_idx, len(self.ref) - 1)
        q = self.data.qpos[self.qadr]
        qd = self.data.qvel[self.vadr] * QD_MASK

        mujoco.mju_quat2Mat(self.mat, self.data.qpos[3:7])
        R = self.mat.reshape(3, 3)
        w = self.data.qvel[3:6]
        gravity = -R[2, :]

        q_diff = q - self.cfg["default_q"]
        qd_scaled = qd * 0.05

        if self.first:
            self.q_hist[:] = q_diff
            self.qd_hist[:] = qd_scaled
            self.a_hist[:] = self.last_action
            self.w_hist[:] = w
            self.g_hist[:] = gravity
            self.first = False
        else:
            self.q_hist[:-1] = self.q_hist[1:]
            self.q_hist[-1] = q_diff
            self.qd_hist[:-1] = self.qd_hist[1:]
            self.qd_hist[-1] = qd_scaled
            self.a_hist[:-1] = self.a_hist[1:]
            self.a_hist[-1] = self.last_action
            self.w_hist[:-1] = self.w_hist[1:]
            self.w_hist[-1] = w
            self.g_hist[:-1] = self.g_hist[1:]
            self.g_hist[-1] = gravity

        goal = q - self.ref[step]
        return np.concatenate([
            self.q_hist.reshape(-1),    # 125
            self.qd_hist.reshape(-1),   # 125
            self.a_hist.reshape(-1),    # 125
            self.w_hist.reshape(-1),    # 15
            self.g_hist.reshape(-1),    # 15
            goal,                       # 25
        ])

    # ------------------------------------------------------------------
    def _infer(self, obs):
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
        obs = self._obs()
        action = np.clip(self._infer(obs), -100.0, 100.0)
        self.last_action = action.copy()

        step = min(self.step_idx, len(self.ref) - 1)
        target = np.zeros(self.model.nu)
        target[self.act_idx] = self.ref[step] + action * self.cfg["scale"]

        for _ in range(FRAME_SKIP):
            q = self.data.qpos[self.act_qadr]
            qd = self.data.qvel[self.act_vadr]
            tau = self.kp * (target - q) - self.kd * qd
            tau = np.clip(tau, -TAU_MAX, TAU_MAX)
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)

        self.step_idx += 1

    def finished(self):
        return self.step_idx >= len(self.ref)

    def state(self):
        mujoco.mju_quat2Mat(self.mat, self.data.qpos[3:7])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(self.mat[8])))))
        return (float(self.data.qpos[0]), float(self.data.qpos[1]),
                float(self.data.qpos[2]), tilt)


def build_skills():
    """创建共用一个 model/data 的两个技能。"""
    spec = mujoco.MjSpec.from_file(str(MODEL_PATH))
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [30.0, 30.0, 0.1]
    floor.rgba = [0.45, 0.48, 0.52, 1.0]
    model = spec.compile()
    model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    model.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]
    data = mujoco.MjData(model)

    lie = MimicSkill(model, data, LIE_CFG)
    getup = MimicSkill(model, data, GETUP_CFG)
    return model, data, {"lie": lie, "getup": getup}


def run_headless(skills, order):
    for key in order:
        skill = skills[key]
        skill.reset()
        print(f"\n开始：{skill.cfg['name']}，共 {len(skill.ref)} 帧")
        while not skill.finished():
            skill.step()
            if skill.step_idx % 50 == 0:
                x, y, z, tilt = skill.state()
                print(f"  t={skill.step_idx * DT:5.1f}s  pos=({x:+.2f},{y:+.2f},{z:.2f})  "
                      f"tilt={tilt:5.1f}°")
        x, y, z, tilt = skill.state()
        print(f"结束：{skill.cfg['name']} -> pos=({x:+.2f},{y:+.2f},{z:.2f})，倾斜 {tilt:.1f}°")


def run_viewer(skills, order):
    import mujoco.viewer

    model = skills[order[0]].model
    data = skills[order[0]].data

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 3.2
            viewer.cam.elevation = -10.0
            viewer.cam.azimuth = 135.0
            viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
            print("策略已加载；关闭窗口退出。")

            for key in order:
                skill = skills[key]
                skill.reset()
                print(f"播放：{skill.cfg['name']}")

                next_time = time.time()
                while viewer.is_running() and not skill.finished():
                    skill.step()
                    x, y, z, tilt = skill.state()
                    viewer.cam.lookat[:] = [x, y, 1.0]
                    viewer.sync()

                    next_time += DT
                    lag = next_time - time.time()
                    if lag > 0:
                        time.sleep(min(lag, 0.01))

            # 保持最后姿态，直到用户关窗口
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)
    except Exception as exc:
        print(f"无法打开 3D 窗口：{exc}")
        print("没有显示器时请用：python 12_getup.py --headless")


def main():
    parser = argparse.ArgumentParser(description="T800 躺下 / 起身策略")
    parser.add_argument("--mode", choices=["lie", "getup", "both"], default="both",
                        help="lie=躺下，getup=起身，both=先躺后起")
    parser.add_argument("--headless", action="store_true", help="不开窗口，打印过程")
    args = parser.parse_args()

    order = {"lie": ["lie"], "getup": ["getup"], "both": ["lie", "getup"]}[args.mode]

    for key in order:
        cfg = LIE_CFG if key == "lie" else GETUP_CFG
        if not cfg["policy"].exists() or not cfg["traj"].exists():
            print(f"找不到文件：{cfg['policy']} 或 {cfg['traj']}")
            return 1

    try:
        model, data, skills = build_skills()
    except Exception as exc:
        print(f"加载失败：{exc}")
        return 1

    if args.headless:
        run_headless(skills, order)
    else:
        run_viewer(skills, order)
    return 0


if __name__ == "__main__":
    sys.exit(main())
