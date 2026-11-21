import mujoco # type: ignore

XML = """
<mujoco model="free_fall">
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/> pyth
    <body name="box" pos="0 0 1">
      <freejoint name="box_free"/>
      <geom type="box" size="0.1 0.1 0.1" density="500"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

print("自由落体：初始高度 1.000 m，每 200 ms 打印一次")
print(" 时间(s)   高度(m)   速度(m/s)")

next_log = 0.0
while data.time <= 2.0 + 1e-9:
    if data.time >= next_log - 1e-9:
        print(f" {data.time:7.3f} {data.qpos[2]:9.4f} {data.qvel[2]:12.3f}")
        next_log += 0.2
    mujoco.mj_step(model, data)
