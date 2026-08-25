import subprocess

import mujoco
import mujoco.viewer
from OpenGL.GL import glGetString, GL_RENDERER
import numpy as np
import time
 
m = mujoco.GLContext(640, 640)
m.make_current()

renderer = glGetString(GL_RENDERER).decode()
# Reject-list, not an allow-list: naming the exact GPU fails on any other
# machine that is perfectly fine. These two strings are the Windows
# software fallbacks (CLAUDE.md, environment facts).
SOFTWARE_GL = ("GDI Generic", "Microsoft Basic Render Driver")
assert not any(s in renderer for s in SOFTWARE_GL), f"software GL, not hardware: {renderer!r}"

model = mujoco.MjModel.from_xml_string(
    '<mujoco><visual><global offwidth="640" offheight="640"/><quality offsamples="0" shadowsize="0"/></visual>'
    '<worldbody>'
    ' <light/>'
    '  <body pos="0 0 0">'
    '    <geom type="sphere" size="0.1"/>'
    '  </body>'
    '</worldbody></mujoco>'
)
data = mujoco.MjData(model)

renderContext = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, renderContext)
assert renderContext.currentBuffer == mujoco.mjtFramebuffer.mjFB_OFFSCREEN, \
    f"offscreen not selected: got {renderContext.currentBuffer}"


scene = mujoco.MjvScene(model, maxgeom = 1000) 

cam = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, cam)

opt = mujoco.MjvOption()
mujoco.mjv_defaultOption(opt)

mujoco.mj_forward(model, data)

mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)

viewport = mujoco.MjrRect(0, 0, 64, 64)

mujoco.mjr_render(viewport, scene, renderContext)
mujoco.mjr_finish()

rgb_img = np.empty((64, 64, 3), dtype=np.uint8)
depth_img = np.empty((64, 64), dtype=np.float32)

mujoco.mjr_readPixels(rgb_img, depth_img, viewport, renderContext)
assert rgb_img.max() > 0, "framebuffer empty - nothing rendered"

print(f"RGB image shape: {rgb_img.shape}, dtype: {rgb_img.dtype}")
print(f"Depth image shape: {depth_img.shape}, dtype: {depth_img.dtype}")
print(f"Max rgb value: {np.max(rgb_img)}, Min rgb value: {np.min(rgb_img)}")
print(f"Mean rgb value: {rgb_img.mean(axis=(0, 1))}")  # Mean color value
print(f"Max depth value: {np.max(depth_img)}, Min depth value: {np.min(depth_img)}")

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     while viewer.is_running():
#         step_start = time.time()

#         # Advance the physics forward
#         mujoco.mj_step(model, data)

#         # Sync the updated physics state with the viewer UI
#         viewer.sync()

#         # Throttle loop to match real-time physics speed
#         time_until_next_step = model.opt.timestep - (time.time() - step_start)
#         if time_until_next_step > 0:
#             time.sleep(time_until_next_step)



def gpu_state():
    return subprocess.run(
        ["nvidia-smi",
         "--query-gpu=pstate,clocks.current.sm,power.draw",
         "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()

def time_readback(width, height, want_depth, render_each=False , N=1000, warmup=500):
    vp = mujoco.MjrRect(0, 0, width, height)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    depth = np.empty((height, width), dtype=np.float32) if want_depth else None

    mujoco.mjr_render(vp, scene, renderContext)
    mujoco.mjr_finish()

    for _ in range(warmup):
        if render_each:
            mujoco.mjr_render(vp, scene, renderContext)
        mujoco.mjr_readPixels(rgb, depth, vp, renderContext)

    s = np.empty(N, dtype=np.int64)
    for i in range(N):
        t0 = time.perf_counter_ns()
        if render_each:
            mujoco.mjr_render(vp, scene, renderContext)
        mujoco.mjr_readPixels(rgb, depth, vp, renderContext)
        s[i] = time.perf_counter_ns() - t0

    return np.median(s) / 1000.0

REPS = 5
arms = {"64 depth": (64, 64, True, False), "64 nodepth": (64, 64, False, False), "64 depth+render": (64, 64, True, True)}
out = {k: [] for k in arms}
before = gpu_state()
for _ in range(REPS):
    for k, (w, h, d, r) in arms.items():
        out[k].append(time_readback(w, h, d, r))
after = gpu_state()
for k, v in out.items():
    v = np.array(v)
    print(f"{k:12s} med {np.median(v):7.2f} us   "
          f"spread {(v.max() - v.min()) / np.median(v) * 100:5.1f}%")
    
print(model.vis.quality.shadowsize, model.vis.quality.offsamples)