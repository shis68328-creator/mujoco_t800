"""02.py —— 用 MuJoCo viewer 展示 T800，并让它在重力作用下摔倒。

流程：
  1. 加载 T800 模型，并补一块地面
  2. 打开 passive（被动）3D 窗口
  3. 每帧调用 mj_step 推进物理，机器人会在重力作用下自然倒下
  4. 大约 4 秒后打印基座的位置 pos、四元数 quat、欧拉角（含 yaw）
  5. 窗口保持打开，直到你手动关闭

运行（macOS 必须用 mjpython，不能直接用 python）：
    conda activate mujoco
    cd /Users/akihisa/mujoco
    mjpython 02.py

为什么 macOS 要用 mjpython：窗口底层使用苹果的 Cocoa 框架，
它要求窗口必须在“主线程”创建；普通 python 启动时主线程被解释器占用，
mjpython 是 MuJoCo 自带的启动器，专门解决这个问题。
Windows / Linux 直接用 python 02.py 即可。
"""

import math                       # 用来把弧度换算成角度
import sys                        # sys.exit：让程序以指定状态码退出（0 成功，1 失败）
import time                       # time.sleep：控制画面刷新间隔，避免占满 CPU
from pathlib import Path          # Path：跨平台地拼接、寻找文件路径

import numpy as np                # 用来保存 3x3 旋转矩阵
import mujoco                     # MuJoCo 物理引擎（模型、数据、仿真都在这里）
import mujoco.viewer              # MuJoCo 自带的 3D 可视化窗口

# 拼出 T800 模型文件（.xml）的完整路径
MODEL_PATH = (
    Path(__file__).resolve().parent  # __file__ 是本文件；resolve() 转绝对路径，parent 取所在目录
    / "engineai_robotics_native_sdk-main/assets/resource/robot/t800/xml/serial_t800.xml"  # 继续进入 SDK 里的模型目录
)  # 括号只是为了把一行写不下的代码分成多行


def quat_to_euler(quat):
    """把四元数 (w, x, y, z) 换算成欧拉角 (roll, pitch, yaw)，单位是弧度。"""
    mat = np.zeros(9)                 # 3x3 旋转矩阵，按行展开成 9 个数
    mujoco.mju_quat2Mat(mat, quat)    # MuJoCo 自带函数：四元数 -> 旋转矩阵

    # 按 ZYX 顺序（先 yaw、再 pitch、最后 roll）从矩阵里反解
    pitch = math.asin(max(-1.0, min(1.0, -mat[6])))  # R[2,0] = -sin(pitch)
    roll = math.atan2(mat[7], mat[8])                # R[2,1] / R[2,2]
    yaw = math.atan2(mat[3], mat[0])                 # R[1,0] / R[0,0]
    return roll, pitch, yaw


def main():
    """程序入口：加载模型、打开窗口、让机器人摔倒并打印状态。"""
    # 1. 读取 XML 模型文件，得到 MjSpec ——可以理解为“还没编译的模型草稿”
    spec = mujoco.MjSpec.from_file(str(MODEL_PATH))  # from_file 要求传字符串形式的路径

    # 2. 这个模型文件里没有地面，机器人会悬在空中，所以手动加一块地板
    floor = spec.worldbody.add_geom()  # worldbody 是场景的根节点；add_geom 往场景里加一个几何体
    floor.name = "floor"               # 给地面起个名字，方便调试时辨认
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE  # 类型设为“平面”：向四周无限延伸的地面
    floor.size = [5.0, 5.0, 0.1]       # x、y 方向各延伸 5 米；第三个值只影响显示，不影响碰撞
    floor.rgba = [0.45, 0.48, 0.52, 1.0]  # 颜色：红、绿、蓝、透明度，每个数都在 0~1 之间

    # 3. 把“草稿”编译成真正能用于仿真的模型
    model = spec.compile()  # 编译后模型参数就不能随便改了，计算效率更高

    # 4. 调亮 MuJoCo 自带的“头灯”，否则深色机器人看起来黑乎乎的
    # 注意 [:] 的写法：表示“原地修改数组里的三个数”，不能直接写 = [...] 整体替换
    model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]  # 环境光：均匀照亮物体所有表面
    model.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]     # 漫反射光：让物体表面产生明暗层次

    # 5. 创建运行时数据对象：机器人的位置、速度、时间等都会存在 data 里
    data = mujoco.MjData(model)      # 同一个 model 可以创建多个 data，用来跑不同的仿真
    mujoco.mj_forward(model, data)   # 先算一遍坐标、碰撞等派生数据，保证窗口第一帧姿势正确

    # 打印模型信息：nbody 是 body 数量（包含 world），nu 是电机（执行器）数量
    # 字符串前面的 f 表示“格式化字符串”，花括号里的内容会被替换成实际数值
    print(f"T800 已加载：{model.nbody} 个 body，{model.nu} 个电机", flush=True)

    # 6. 打开 3D 窗口。passive（被动）模式：窗口只负责显示，不会自己推进仿真
    try:
        viewer_ctx = mujoco.viewer.launch_passive(model, data)  # 成功就返回一个窗口句柄
    except Exception as exc:  # 没有图形环境，或 macOS 上没用 mjpython，就会进到这里
        print(f"\n无法打开 3D 窗口：{exc}")  # 把具体错误原因打印出来
        print("macOS 需要用 mjpython 运行：")  # 告诉用户怎么解决
        print("  /opt/anaconda3/envs/mujoco/bin/mjpython 02.py")
        return 1  # 返回 1 表示程序出错

    # 7. with 语句：进入时拿到 viewer 对象；退出时自动关闭窗口、释放资源
    with viewer_ctx as viewer:
        # 8. 设置打开窗口时的初始相机位置（之后可以用鼠标随意调整）
        viewer.cam.lookat[:] = [0.2, 0.2, 0.6]  # 相机注视的点：机器人摔倒区域附近
        viewer.cam.distance = 3.4               # 相机到注视点的距离，单位米；越大画面越远
        viewer.cam.elevation = -12.0            # 俯仰角（度）：负值表示从稍高处往下看
        viewer.cam.azimuth = 135.0              # 水平旋转角（度）：135° 是斜前方视角

        # 提示用户鼠标怎么操作
        print("机器人将在重力作用下摔倒；鼠标左键旋转，右键平移，滚轮缩放。", flush=True)

        # 9. 开始推进物理：调用 mj_step 后，重力、接触、阻尼才会真正生效
        sim_end = 4.0                 # 一共仿真 4 秒，足够摔倒并基本停稳
        wall_start = time.time()      # 记录墙上时间，用来让动画按真实速度播放

        while viewer.is_running() and data.time < sim_end:
            # 按真实时间计算“现在应该仿真到第几秒”，最多到 sim_end
            target_time = min(sim_end, time.time() - wall_start)
            # 一直单步推进，直到追上真实时间
            while data.time < target_time:
                mujoco.mj_step(model, data)  # 核心：推进一步物理仿真

            viewer.sync()             # 把最新画面画到窗口里
            time.sleep(0.005)         # 稍微歇 5 毫秒，避免 CPU 空转

        # 10. 摔倒结束后，读取基座的位置和姿态并打印
        #     基座是自由关节，qpos 前 3 个数是位置，后 4 个数是四元数
        pos = data.qpos[0:3]          # (x, y, z)
        quat = data.qpos[3:7]         # (w, x, y, z)
        roll, pitch, yaw = quat_to_euler(quat)  # 换算成欧拉角

        print("\n" + "=" * 56)
        print(f" 摔倒后的基座状态（t = {data.time:.2f} s）")
        print("=" * 56)
        print(f" pos  (x, y, z)          = ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) m")
        print(f" quat (w, x, y, z)       = ({quat[0]:+.4f}, {quat[1]:+.4f}, "
              f"{quat[2]:+.4f}, {quat[3]:+.4f})")
        print(f" euler(roll, pitch, yaw) = ({math.degrees(roll):+.1f}°, "
              f"{math.degrees(pitch):+.1f}°, {math.degrees(yaw):+.1f}°)")
        print(f" yaw  (偏航角)           = {math.degrees(yaw):+.1f}°")
        print("=" * 56, flush=True)

        # 11. 窗口继续开着，方便观察最终姿态；关闭窗口后程序结束
        print("摔倒结束，关闭窗口退出。", flush=True)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)

    return 0  # 返回 0 表示正常结束


# 12. 只有“直接运行 02.py”时才会执行下面两行；被别的文件 import 时不会执行
if __name__ == "__main__":
    sys.exit(main())  # 用 main() 的返回值作为退出码：0 成功，非 0 失败
