"""07.py —— 给 T800 加上 PD 控制，让它站着，并每 100 ms 打印一次运动信息。

控制思路（关节空间 PD）：
    电机力矩 = kp × (目标角度 - 当前角度) - kd × 当前角速度
目标角度取模型初始姿态（qpos0，所有关节为 0）。
kp 选得足够大，才能托住 85 kg 的机器人；太小会直接软倒。

运行（macOS 必须用 mjpython）：
    conda activate mujoco
    cd /Users/akihisa/mujoco
    mjpython 07.py

窗口关闭后程序结束。
"""

import math                       # 弧度转角度
import sys                        # sys.exit：设置程序退出码
import time                       # 控制刷新节奏
from pathlib import Path          # 拼接模型路径

import numpy as np                # 求速度大小、保存旋转矩阵
import mujoco                     # MuJoCo 物理引擎
import mujoco.viewer              # 3D 可视化窗口

# T800 模型文件的位置
MODEL_PATH = (
    Path(__file__).resolve().parent  # 本文件（07.py）所在目录
    / "engineai_robotics_native_sdk-main/assets/resource/robot/t800/xml/serial_t800.xml"  # SDK 里的模型
)

# PD 参数：比例增益 kp、微分增益 kd
KP = 1000.0   # 角度误差 0.1 rad 时，输出 100 N·m 力矩
KD = 60.0     # 阻尼项，抑制摆动

# 打印间隔（秒）
LOG_INTERVAL = 0.1


def quat_to_euler(quat):
    """把四元数 (w, x, y, z) 换成欧拉角 (roll, pitch, yaw)，单位是弧度。"""
    mat = np.zeros(9)                 # 3x3 旋转矩阵，按行展开
    mujoco.mju_quat2Mat(mat, quat)    # 四元数 -> 旋转矩阵
    pitch = math.asin(max(-1.0, min(1.0, -mat[6])))  # R[2,0] = -sin(pitch)
    roll = math.atan2(mat[7], mat[8])                # R[2,1] / R[2,2]
    yaw = math.atan2(mat[3], mat[0])                 # R[1,0] / R[0,0]
    return roll, pitch, yaw


def fmt3(vec):
    """把 3 个数格式化成 (x, y, z) 字符串。"""
    return f"({vec[0]:+.3f}, {vec[1]:+.3f}, {vec[2]:+.3f})"


def print_state(data, tag=""):
    """打印一行运动信息：时间、位置、姿态、倾斜角、速度大小。"""
    pos = data.qpos[0:3]                  # 基座位置 (x, y, z)
    quat = data.qpos[3:7]                 # 基座四元数 (w, x, y, z)
    roll, pitch, yaw = quat_to_euler(quat)

    # 倾斜角：机器人“头顶方向”偏离竖直方向多少度
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    up_z = max(-1.0, min(1.0, float(mat[8])))  # 旋转矩阵第 3 行第 3 列 = 头顶方向的 z 分量
    tilt = math.acos(up_z)

    v = float(np.linalg.norm(data.qvel[0:3]))  # 基座线速度大小
    print(
        f" t={data.time:6.3f}s  pos={fmt3(pos)}  "
        f"rpy=({math.degrees(roll):+5.1f}, {math.degrees(pitch):+5.1f}, "
        f"{math.degrees(yaw):+5.1f})°  tilt={math.degrees(tilt):4.1f}°  |v|={v:.3f} m/s"
        + tag,
        flush=True,
    )


def main():
    """加载模型、加地面、用 PD 控制让机器人站稳并持续打印状态。"""
    # 1. 读取模型
    spec = mujoco.MjSpec.from_file(str(MODEL_PATH))

    # 2. 加一块地面（模型本身没有地面）
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [5.0, 5.0, 0.1]
    floor.rgba = [0.45, 0.48, 0.52, 1.0]

    # 3. 编译模型
    model = spec.compile()

    # 4. 调亮头灯，方便观察
    model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    model.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]

    # 5. 创建数据，先算一次正向运动学
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # 6. 建立“执行器 -> 关节”的对应关系，后面用 qpos/qvel 算 PD
    joint_ids = model.actuator_trnid[:, 0]       # 每个电机控制哪个关节
    qpos_adr = model.jnt_qposadr[joint_ids]      # 该关节角度在 qpos 里的下标
    qvel_adr = model.jnt_dofadr[joint_ids]       # 该关节角速度在 qvel 里的下标
    q_target = model.qpos0[qpos_adr].copy()      # 目标角度 = 模型的初始姿态

    print(f"T800 已加载，PD 参数 kp={KP:g}, kd={KD:g}", flush=True)
    print("时间(s)      位置 pos (m)                姿态 rpy (°)                倾斜角     速度", flush=True)
    print("-" * 100, flush=True)
    print_state(data)  # 先打印 t=0 的初始状态

    # 7. 打开 3D 窗口（passive 模式，需要自己调用 mj_step）
    try:
        viewer_ctx = mujoco.viewer.launch_passive(model, data)
    except Exception as exc:
        print(f"\n无法打开 3D 窗口：{exc}")
        print("macOS 需要用 mjpython 运行：")
        print("  /opt/anaconda3/envs/mujoco/bin/mjpython 07.py")
        return 1

    with viewer_ctx as viewer:
        # 8. 相机对准站立位置
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        viewer.cam.distance = 2.6
        viewer.cam.elevation = -8.0
        viewer.cam.azimuth = 135.0

        print("PD 控制已启动：机器人应保持站立；关闭窗口退出。", flush=True)

        # 9. 控制 + 仿真循环
        next_log = LOG_INTERVAL       # 下一次打印的时间点
        wall_start = time.time()      # 墙上时间，用来让动画按真实速度播放

        while viewer.is_running():
            # 按真实时间推进：应该仿真到第几秒
            target_time = time.time() - wall_start
            while data.time < target_time and viewer.is_running():
                # 读取当前关节角和角速度
                q = data.qpos[qpos_adr]
                qd = data.qvel[qvel_adr]

                # PD 公式：力矩 = kp*(目标-当前) - kd*当前速度
                data.ctrl[:] = KP * (q_target - q) - KD * qd

                # 推进一步物理仿真
                mujoco.mj_step(model, data)

            viewer.sync()  # 刷新画面

            # 10. 每 100 ms 打印一次运动信息
            if data.time >= next_log:
                print_state(data)
                next_log += LOG_INTERVAL

            # 11. 如果倾斜超过 60°，说明已经摔倒，停止控制
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, data.qpos[3:7])
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(mat[8])))))
            if tilt > 60.0:
                print(f"\n机器人倾倒了（tilt = {tilt:.1f}°），停止仿真。", flush=True)
                break

            time.sleep(0.003)  # 稍微歇一下，避免 CPU 空转

    print("程序结束。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # 直接运行 07.py 时执行 main()
