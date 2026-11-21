"""04.py —— 打印 T800 机器人所有 body（连杆）的位置。

运行（不需要 3D 窗口，普通 python 即可）：
    conda activate mujoco
    cd /Users/akihisa/mujoco
    python 04.py

表格里有两组位置：
  body_pos：相对父连杆的位置，模型里写死的固定值
  xpos    ：世界坐标，由 mj_forward 根据当前关节角计算出来
"""

from pathlib import Path  # Path：用来拼接模型文件的路径

import mujoco  # MuJoCo：读取模型、计算位置

# T800 模型文件的位置（和 02.py、03.py 用的是同一个模型）
MODEL_PATH = (
    Path(__file__).resolve().parent  # 本文件（04.py）所在的目录
    / "engineai_robotics_native_sdk-main/assets/resource/robot/t800/xml/serial_t800.xml"  # SDK 里的模型文件
)


def fmt_vec(vec):
    """把 [x, y, z] 三个数格式化成对齐的字符串。"""
    # vec[0]/vec[1]/vec[2] 就是 x/y/z；:+8.4f 表示带正负号、共 8 格宽、保留 4 位小数
    return f"({vec[0]:+8.4f}, {vec[1]:+8.4f}, {vec[2]:+8.4f})"


def main():
    """加载模型，并逐个打印所有 body 的位置。"""
    # 1. 读取模型
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    # 2. 创建运行时数据，并执行一次正向运动学
    #    mj_forward 会根据当前关节角（初始全是 0）算出每个 body 的世界坐标 xpos
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # 3. 打印总数和列的含义
    print(f"T800 共 {model.nbody} 个 body（包含 world）")
    print("body_pos = 相对父连杆的位置（模型里的固定值）")
    print("xpos     = 世界坐标（由 mj_forward 根据当前关节角算出）")
    print()

    # 4. 打印表头；中文字符占两格宽，这里手动补空格让表头和数据对齐
    print("编号  " + "名称" + " " * 18 + " " + "body_pos（相对父连杆）" + " " * 14 + " " + "xpos（世界坐标）")
    print("-" * 100)

    # 5. 遍历所有 body：i 从 0 到 30
    for i in range(model.nbody):
        # 5.1 取 body 名称；正常都有名字，拿不到时显示 "(未命名)"
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name is None:
            name = "(未命名)"

        # 5.2 取两种位置：
        #     body_pos 是相对父连杆的偏移，不随关节转动变化
        #     xpos 是世界坐标，关节一转它就会变
        pos = model.body_pos[i]
        xpos = data.xpos[i]

        # 5.3 按列打印这一行；:<36 表示左对齐、占 36 个字符宽
        print(f"{i:>4}  {name:<22} {fmt_vec(pos):<36} {fmt_vec(xpos)}")

    # 6. 补充一句说明，避免把 world 当成机器人零件
    print()
    print("说明：world 是 0 号 body，代表整个世界；它没有父连杆，所以 body_pos 是 (0, 0, 0)。")


if __name__ == "__main__":
    main()  # 直接运行 04.py 时执行 main()
