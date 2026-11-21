"""05.py —— 用 matplotlib 画 T800 机器人的三维骨架。

做法：
  1. 用 MuJoCo 读取模型，通过 mj_forward 算出每个 body（连杆）的世界坐标
  2. 把“父连杆 -> 子连杆”用直线连起来，就得到一副三维骨架
  3. 用 matplotlib 的 3D 坐标轴画出来

运行：
    conda activate mujoco
    cd /Users/akihisa/mujoco
    python 05.py

运行后会弹出 matplotlib 窗口（可以用鼠标拖动旋转视角），
同时把图片保存为 t800_skeleton.png。
"""

from pathlib import Path  # 拼接模型路径、保存图片路径

import matplotlib  # 画图库

# 让图中的中文正常显示：优先使用 macOS 自带的中文字体
matplotlib.rcParams["font.sans-serif"] = [
    "PingFang SC",       # macOS 苹方
    "Hiragino Sans GB",  # macOS 冬青黑体
    "Heiti SC",          # macOS 黑体
    "Arial Unicode MS",  # 备选
    "DejaVu Sans",       # Linux 上常见的备选
]
matplotlib.rcParams["axes.unicode_minus"] = False  # 让坐标轴上的负号正常显示

import matplotlib.pyplot as plt  # noqa: E402  画图函数（要放在 rcParams 设置之后）
from matplotlib.lines import Line2D  # noqa: E402  用来手工创建图例
import numpy as np  # noqa: E402  用来构造地面网格

import mujoco  # 读取模型、计算 body 的世界坐标

# T800 模型文件的位置（和 02/03/04.py 相同）
MODEL_PATH = (
    Path(__file__).resolve().parent  # 本文件（05.py）所在目录
    / "engineai_robotics_native_sdk-main/assets/resource/robot/t800/xml/serial_t800.xml"  # SDK 里的模型
)


def short_joint_name(name):
    """把 J13_SHOULDER_PITCH_L 缩写成 SH_P_L 这种短名字，避免图上文字太挤。"""
    name = name.split("_", 1)[1]        # 去掉 "J13_" 前缀
    name = name.replace("SHOULDER", "SH")  # 肩膀
    name = name.replace("ELBOW", "ELB")    # 肘
    name = name.replace("ANKLE", "ANK")    # 踝
    name = name.replace("KNEE_PITCH", "KNEE")  # 膝盖
    name = name.replace("_PITCH", "_P")    # Pitch（俯仰）缩写为 P
    name = name.replace("_ROLL", "_R")     # Roll（横滚）缩写为 R
    name = name.replace("_YAW", "_Y")      # Yaw（偏航）缩写为 Y
    return name


def main():
    """读取模型坐标，画三维骨架并保存图片。"""
    # 1. 读取 T800 模型
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    # 2. 算一次正向运动学，得到每个 body 在世界坐标系里的位置 xpos
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    points = data.xpos  # 形状是 (31, 3)，每一行是 [x, y, z]

    # 3. 按连杆名字决定颜色：_L 结尾蓝色，_R 结尾红色，其余灰色
    colors = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if name.endswith("_L"):
            colors.append("#1f77b4")  # 左侧：蓝
        elif name.endswith("_R"):
            colors.append("#d62728")  # 右侧：红
        else:
            colors.append("#444444")  # 躯干、头部、基座：深灰

    # 4. 计算显示范围（四周留 10 cm 边距）
    body_points = points[1:]  # 不包含 world
    margin = 0.1
    xmin = body_points[:, 0].min() - margin
    xmax = body_points[:, 0].max() + margin
    ymin = body_points[:, 1].min() - margin
    ymax = body_points[:, 1].max() + margin
    zmax = body_points[:, 2].max() + margin

    # 5. 创建一张图和三维坐标轴
    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(111, projection="3d")  # projection="3d" 是关键
    ax.set_title("T800 机器人三维骨架与关节名称（初始站立姿态）", fontsize=16, pad=20)

    # 6. 在 z=0 处画一块半透明地面，帮助判断机器人的高度
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, 2), np.linspace(ymin, ymax, 2))
    ax.plot_surface(
        gx, gy, np.zeros_like(gx),
        color="#d9d9d9", alpha=0.25, linewidth=0, antialiased=False, zorder=0,
    )

    # 7. 逐条画“父连杆 -> 子连杆”的连线，这就是三维骨架
    for i in range(1, model.nbody):  # 0 号是 world，从 1 号基座开始
        parent_id = model.body_parentid[i]  # 当前连杆的父连杆编号
        if parent_id == 0:  # 不画 world 到基座的那根线，否则会多一条竖线
            continue
        ax.plot(
            [points[parent_id, 0], points[i, 0]],  # 两个端点的 x
            [points[parent_id, 1], points[i, 1]],  # 两个端点的 y
            [points[parent_id, 2], points[i, 2]],  # 两个端点的 z
            color=colors[i],                       # 连杆颜色：左蓝、右红、中间灰
            linewidth=2.5,                         # 线宽
            solid_capstyle="round",
            zorder=3,
        )

    # 8. 每个 body 的坐标点画一个小球，近似表示关节位置
    ax.scatter(
        points[1:, 0], points[1:, 1], points[1:, 2],
        s=25, c=colors[1:], depthshade=True, zorder=4,
    )

    # 9. 在每个关节旁边写上关节名字（去掉 Jxx_ 前缀，文字更短）
    for j in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if joint_name is None:  # 0 号自由关节没有名字，跳过
            continue
        short_name = short_joint_name(joint_name)  # "J13_SHOULDER_PITCH_L" -> "SH_P_L"
        body_id = int(model.jnt_bodyid[j])         # 这个关节挂在哪个 body 上
        x, y, z = data.xanchor[j]                  # 关节在世界坐标系中的位置

        # 左侧标签往 +y 挪一点、右侧往 -y 挪一点，避免和躯干上的文字挤在一起
        if short_name.endswith("_L"):
            y += 0.03
        elif short_name.endswith("_R"):
            y -= 0.03

        # 肩、踝、髋的 pitch/roll 关节位置挨得很近，把文字上下错开一点
        core = joint_name.split("_", 1)[1]  # 去掉 "J23_" 前缀，例如 "HEAD_PITCH"
        if core.endswith(("_L", "_R")):     # 再去掉表示左/右的后缀
            core = core[:-2]
        z_offset = 0.02
        if core.startswith("HEAD"):
            z_offset = 0.05 if core.endswith("_YAW") else -0.02
        elif core.endswith("_PITCH"):
            z_offset = 0.055
        elif core.endswith("_ROLL"):
            z_offset = 0.005

        ax.text(
            x, y, z + z_offset, short_name,        # 文字放在关节附近
            fontsize=6.5,
            color=colors[body_id],
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.08", facecolor="white",
                      edgecolor="none", alpha=0.5),
            zorder=5,
        )

    # 10. 设置坐标轴范围、刻度和标签
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(0.0, zmax)

    ax.set_xticks([-0.1, 0.0, 0.1, 0.2, 0.3])
    ax.set_yticks([-0.3, -0.15, 0.0, 0.15, 0.3])
    ax.set_zticks([0.0, 0.4, 0.8, 1.2, 1.6])

    ax.set_xlabel("X (m)  前后")
    ax.set_ylabel("Y (m)  左右")
    ax.set_zlabel("Z (m)  上下")

    # 11. 设置三维坐标盒子的长宽高比例；适当加宽，避免整个人又细又长
    ax.set_box_aspect((1.0, 1.0, 2.2))

    # 12. 设置初始视角：仰角 18°，方位角 -55°，是一个斜前方的 3/4 视角
    ax.view_init(elev=18, azim=-55)

    # 13. 去掉坐标轴的灰色背景板，只保留网格线，看起来更干净
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor("#cccccc")
    ax.grid(True, linestyle="--", alpha=0.4)

    # 14. 在画布底部加一个图例，说明颜色含义
    legend_handles = [
        Line2D([0], [0], color="#1f77b4", linewidth=3, label="左侧（_L）"),
        Line2D([0], [0], color="#d62728", linewidth=3, label="右侧（_R）"),
        Line2D([0], [0], color="#444444", linewidth=3, label="躯干 / 头部 / 基座"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False)
    fig.text(
        0.5, 0.06,
        "缩写：P=Pitch 俯仰    R=Roll 横滚    Y=Yaw 偏航    "
        "SH=Shoulder 肩    ELB=Elbow 肘    ANK=Ankle 踝",
        ha="center", va="center", fontsize=9, color="#555555",
    )
    fig.subplots_adjust(bottom=0.13, top=0.92)  # 给标题、图例和缩写说明留空间

    # 15. 保存图片到 05.py 同目录
    output = Path(__file__).with_name("t800_skeleton.png")
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"三维骨架已保存：{output}")

    # 16. 弹出窗口显示图片（可以用鼠标拖动旋转；关闭窗口后程序结束）
    plt.show()


if __name__ == "__main__":
    main()  # 直接运行 05.py 时执行 main()
