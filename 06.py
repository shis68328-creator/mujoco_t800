"""06.py —— 打印 T800 机器人的初始运动状态。

“运动状态”主要看两组数据：
  qpos：所有位置量（基座位姿 + 25 个关节角）
  qvel：所有速度量（基座线速度/角速度 + 25 个关节角速度）

刚创建 MjData 时，机器人处于模型定义的初始姿态，所有速度都是 0。

运行：
    conda activate mujoco
    cd /Users/akihisa/mujoco
    python 06.py
"""

import math  # 用来把弧度换算成角度
from pathlib import Path  # 拼接模型文件的路径

import mujoco  # 读取模型、创建数据

# T800 模型文件的位置（和其他脚本相同）
MODEL_PATH = (
    Path(__file__).resolve().parent  # 本文件（06.py）所在目录
    / "engineai_robotics_native_sdk-main/assets/resource/robot/t800/xml/serial_t800.xml"  # SDK 里的模型
)


def fmt3(vec):
    """把 3 个数（x, y, z）格式化成带正负号、4 位小数的字符串。"""
    return f"({vec[0]:+9.4f}, {vec[1]:+9.4f}, {vec[2]:+9.4f})"


def fmt4(vec):
    """把 4 个数（四元数 w, x, y, z）格式化成同样风格的字符串。"""
    return f"({vec[0]:+7.4f}, {vec[1]:+7.4f}, {vec[2]:+7.4f}, {vec[3]:+7.4f})"


def main():
    """读取模型，打印初始时刻的位置、姿态和速度。"""
    # 1. 读取模型，并创建一份运行时数据
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)  # 新创建的 MjData 会自动填入模型的初始状态

    # 2. 执行一次正向运动学，把坐标、传感器等派生数据都算出来
    mujoco.mj_forward(model, data)

    # 3. 打印基本信息和仿真时间
    print("=" * 60)
    print(" T800 初始运动状态")
    print("=" * 60)
    print(f" 仿真时间 t = {data.time:.3f} s")
    print(f" 位置量个数 nq = {model.nq}，速度量个数 nv = {model.nv}，电机个数 nu = {model.nu}")
    print(f" 重力 = {fmt3(model.opt.gravity)} m/s²")

    # 4. 基座使用自由关节（free joint），它的 qpos 占 7 个数：
    #    前 3 个是位置 x/y/z，后 4 个是姿态四元数 w/x/y/z
    free_joint_id = 0  # 本模型里 0 号关节就是基座的自由关节
    qadr = int(model.jnt_qposadr[free_joint_id])  # 基座 qpos 的起始下标
    vadr = int(model.jnt_dofadr[free_joint_id])   # 基座 qvel 的起始下标

    print("\n 基座（LINK_BASE，自由关节）")
    print(f"   位置       (x, y, z)       = {fmt3(data.qpos[qadr:qadr + 3])} m")
    print(f"   姿态四元数 (w, x, y, z)   = {fmt4(data.qpos[qadr + 3:qadr + 7])}")
    print(f"   线速度     (vx, vy, vz)    = {fmt3(data.qvel[vadr:vadr + 3])} m/s")
    print(f"   角速度     (wx, wy, wz)    = {fmt3(data.qvel[vadr + 3:vadr + 6])} rad/s")

    # 5. 打印 25 个旋转关节的初始角度和角速度
    print("\n 关节初始状态（25 个旋转关节，不含基座自由关节）")
    print("序号  " + "关节名称" + " " * 13 + " " + "qpos(rad) " + " " + "角度(°)  " + " " + "qvel(rad/s) ")
    print("-" * 66)

    for j in range(model.njnt):
        # 跳过基座的自由关节，它已经在上面单独打印过了
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue

        # 取关节名和它在 qpos/qvel 里的下标
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        qpos_adr = int(model.jnt_qposadr[j])
        qvel_adr = int(model.jnt_dofadr[j])

        # 取当前位置（弧度）和当前速度（弧度/秒）
        q = float(data.qpos[qpos_adr])
        v = float(data.qvel[qvel_adr])

        # 打印：弧度、角度、角速度
        print(f"{j:>4}  {name:<21} {q:>10.4f} {math.degrees(q):>9.2f} {v:>12.4f}")

    # 6. 最后给出一个直观结论
    print()
    print("说明：初始时刻所有关节角速度和基座速度都是 0，机器人处于静止状态；")
    print("      角度 = 弧度 × 180 / π，例如 1 rad ≈ 57.3°。")


if __name__ == "__main__":
    main()  # 直接运行 06.py 时执行 main()
