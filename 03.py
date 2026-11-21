"""03.py —— 打印 T800 机器人所有关节和连杆的信息。

运行（这个脚本不需要 3D 窗口，普通 python 就能跑）：
    conda activate mujoco
    cd /Users/akihisa/mujoco
    python 03.py
"""

from pathlib import Path  # Path：用来拼接模型文件的路径

import mujoco  # MuJoCo：读取模型、查询关节属性

# T800 模型文件的位置（和 02.py 用的是同一个模型）
MODEL_PATH = (
    Path(__file__).resolve().parent  # 本文件（03.py）所在的目录
    / "engineai_robotics_native_sdk-main/assets/resource/robot/t800/xml/serial_t800.xml"  # SDK 里的模型文件
)

# MuJoCo 用数字表示关节类型，这里把它们翻译成简短的英文，方便表格对齐
JOINT_TYPE_SHORT = {
    mujoco.mjtJoint.mjJNT_FREE: "free",    # 自由关节：基座可在空间中平移 + 旋转
    mujoco.mjtJoint.mjJNT_BALL: "ball",    # 球关节：绕一个点向任意方向转动
    mujoco.mjtJoint.mjJNT_SLIDE: "slide",  # 滑动关节：沿一根轴直线移动
    mujoco.mjtJoint.mjJNT_HINGE: "hinge",  # 旋转关节：绕一根轴转动（机器人最常用）
}


def main():
    """加载模型，并逐个打印所有关节。"""
    # 1. 从 XML 文件读取模型（只查看静态信息，所以不需要创建 MjData）
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    # 2. 打印关节总数；njnt 是 joint 的缩写
    print(f"T800 共 {model.njnt} 个关节")
    print(f"（nq = {model.nq} 个位置量，nv = {model.nv} 个速度量）")

    # 3. 打印表头。{...:<20} 的意思是“左对齐，占 20 个字符宽”
    # 表头里的中文字符占两个显示宽度，这里手动补空格，让表头和数据对齐
    print()
    print("序号  " + "关节名称" + " " * 13 + "类型" + " " * 3 + "所属刚体" + " " * 15 + "限位 [rad]")
    print("-" * 78)

    # 4. 逐个遍历关节：i 从 0 到 25（共 26 个）
    for i in range(model.njnt):
        # 4.1 取关节名称；没有名字的关节会返回 None
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name is None:  # 本模型的第 0 个关节是没名字的自由关节
            name = "(未命名)"

        # 4.2 取关节类型：jnt_type[i] 是数字，翻译成 "free"/"hinge" 等
        joint_type = JOINT_TYPE_SHORT[mujoco.mjtJoint(model.jnt_type[i])]

        # 4.3 取这个关节连接在哪个刚体（body）上，再查出刚体的名字
        body_id = model.jnt_bodyid[i]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

        # 4.4 取关节的限位（能转动的角度范围）；没有限位就显示 "-"
        if model.jnt_limited[i]:
            low, high = model.jnt_range[i]  # low/high 单位是弧度
            limit = f"[{low:+.3f}, {high:+.3f}]"
        else:
            limit = "-"

        # 4.5 按列打印这一行
        print(f"{i:>4}  {name:<20} {joint_type:<6} {body_name:<22} {limit}")

    # 5. 打印连杆（body）信息
    #    连杆就是机器人的一段“骨架”：每个 body 通过一个关节连接到它的父 body 上
    print("\n" + "=" * 78)
    print(f"连杆（body）信息：共 {model.nbody - 1} 个（不含 world），"
          f"总质量 {mujoco.mj_getTotalmass(model):.3f} kg")
    print()
    # 表头同样手动补空格，让中英文混排也能对齐
    print("序号  " + "连杆名称" + " " * 14 + " " + "父连杆" + " " * 16 + " "
          + "质量(kg)" + "  " + "几何" + "  " + "关节")
    print("-" * 78)

    # 5.1 遍历所有 body；0 号永远是 world（整个世界），所以从 1 开始
    for i in range(1, model.nbody):
        # 5.2 取连杆名称
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        # 5.3 取父连杆：body_parentid[i] 记录父 body 的编号，再查名字
        parent_id = model.body_parentid[i]
        parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
        # 5.4 取质量（kg）、几何体数量、关节数量
        mass = model.body_mass[i]        # 这段骨架的质量
        ngeom = model.body_geomnum[i]    # 几何体数量（含显示用和碰撞用）
        njnt = model.body_jntnum[i]      # 挂在这个连杆上的关节数量
        # 5.5 按列打印这一行
        print(f"{i:>4}  {name:<22} {parent_name:<22} {mass:>8.3f}  {ngeom:>4}  {njnt:>4}")

    # 6. 最后解释表格里的术语，方便第一次看的人
    print()
    print("说明：")
    print("  free  = 自由关节，基座可以整体平移和旋转（整个机器人只有这一个）")
    print("  hinge = 旋转关节，绕一根轴转动（其余 25 个都是这种）")
    print("  限位单位是弧度（rad），1 rad ≈ 57.3°；没写限位（-）的关节不受角度限制")
    print("  连杆的“几何”包含显示用和碰撞用的几何体；脚、手腕末端没有关节，所以关节数为 0")


if __name__ == "__main__":
    main()  # 直接运行 03.py 时执行 main()
