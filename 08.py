"""08.py —— 遍历 T800 的每个关节，打印它对应的执行器 ctrlrange。

说明：
  ctrlrange 是执行器控制量的允许范围（XML 里的写法是 ctrlrange）。
  T800 的 25 个执行器都是 gear=1 的力矩电机，所以 ctrlrange 的单位是 N·m。
  基座的自由关节没有电机，打印时会显示 "-"。

运行（不需要 3D 窗口，普通 python 即可）：
    conda activate mujoco
    cd /Users/akihisa/mujoco
    python 08.py
"""

from pathlib import Path  # 拼接模型文件路径

import mujoco  # 读取模型、查询关节和执行器属性

# T800 模型文件的位置（和其他脚本相同）
MODEL_PATH = (
    Path(__file__).resolve().parent  # 本文件（08.py）所在目录
    / "engineai_robotics_native_sdk-main/assets/resource/robot/t800/xml/serial_t800.xml"  # SDK 里的模型
)


def main():
    """加载模型，逐个关节打印执行器名称和 ctrlrange。"""
    # 1. 读取模型（只查静态参数，不需要 MjData）
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    # 2. 建立“关节 -> 执行器”的对应表
    #    actuator_trnid[a, 0] 表示第 a 个执行器控制哪个关节
    actuator_of_joint = {}
    for a in range(model.nu):
        joint_id = int(model.actuator_trnid[a, 0])
        actuator_of_joint[joint_id] = a

    # 3. 打印统计信息和表头
    print(f"T800 共 {model.njnt} 个关节，其中 {model.nu} 个带电机")
    print()
    print("序号  " + "关节名称" + " " * 13 + " " + "执行器名称" + " " * 16 + " " + "ctrlrange (N·m)")
    print("-" * 88)

    # 4. 逐个关节遍历
    for j in range(model.njnt):
        # 4.1 取关节名字；基座自由关节没有名字
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name is None:
            name = "(未命名)"

        # 4.2 找这个关节对应的执行器
        actuator_id = actuator_of_joint.get(j)

        if actuator_id is None:
            # 没有执行器：基座自由关节
            actuator_name = "-"
            ctrl_range = "-"
        else:
            # 有执行器：取名字和 ctrlrange
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            if model.actuator_ctrllimited[actuator_id]:
                low, high = model.actuator_ctrlrange[actuator_id]
                ctrl_range = f"[{low:+.1f}, {high:+.1f}]"
            else:
                ctrl_range = "不限幅"

        # 4.3 按列打印一行
        print(f"{j:>4}  {name:<21} {actuator_name:<26} {ctrl_range:<22}")

    # 5. 补充说明和整体统计
    limits = [
        model.actuator_ctrlrange[a]
        for a in range(model.nu)
        if model.actuator_ctrllimited[a]
    ]
    max_abs = max(abs(float(lim)) for pair in limits for lim in pair) if limits else 0.0

    print()
    print("说明：")
    print("  ctrlrange 就是代码里 data.ctrl 的允许范围。")
    print("  本模型的电机 gear=1，所以 ctrl 的数值等于关节力矩，单位是 N·m。")
    print(f"  所有电机的力矩限幅绝对值最大为 {max_abs:.1f} N·m。")
    print("  第 0 号关节是基座的自由关节，不用电机驱动，所以没有 ctrlrange。")


if __name__ == "__main__":
    main()  # 直接运行 08.py 时执行 main()
