"""Time taken by each step of producing one dataset frame, on the real scene.

Answers the last open timing question: `mjv_updateScene`. The readback probe
timed render and readback on a single sphere; this times everything on
`scene/arm_blocks.xml` with the camera the generator actually uses.

Times each call, then the whole frame, rather than one end-to-end frames/s
loop, which would hide which step dominates.
"""
import pathlib
import time

import mujoco
import numpy as np
from OpenGL.GL import GL_RENDERER, glGetString

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENE = ROOT / "scene" / "arm_blocks.xml"
assert SCENE.exists(), f"scene file not found: {SCENE}"

W = H = 64

ctx = mujoco.GLContext(W, H)
ctx.make_current()
renderer = glGetString(GL_RENDERER).decode()
# Reject known bad renderers rather than allow only this GPU, which would fail
# on any other good machine. These two are the Windows software renderers.
SOFTWARE_GL = ("GDI Generic", "Microsoft Basic Render Driver")
assert not any(s in renderer for s in SOFTWARE_GL), f"software GL, not hardware: {renderer!r}"

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)
assert (model.vis.global_.offwidth, model.vis.global_.offheight) == (W, H), \
    "offscreen buffer must match the capture size exactly"

rc = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, rc)
assert rc.currentBuffer == mujoco.mjtFramebuffer.mjFB_OFFSCREEN, \
    f"offscreen not selected: got {rc.currentBuffer}"

# The XML's named camera, not a free camera: a free camera shows a different
# view, so the numbers would describe a scene we never capture.
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "main")
assert cam.fixedcamid >= 0, "camera 'main' not found - mj_name2id returns -1 silently"

# The defaults already turn off every overlay. Zeroing the flag array would
# also clear mjVIS_STATIC, and the table and other world geoms would vanish.
opt = mujoco.MjvOption()
mujoco.mjv_defaultOption(opt)

scene = mujoco.MjvScene(model, maxgeom=1000)
vp = mujoco.MjrRect(0, 0, W, H)
rgb = np.empty((H, W, 3), dtype=np.uint8)
seg = np.empty((H, W, 3), dtype=np.uint8)

data.ctrl[:] = model.actuator_ctrlrange[:, 1]

# Required before the first mjv_updateScene. Positions and orientations in
# mjData are zero until mj_forward runs, so the scene gets a correct-looking
# geom count and renders a completely black frame. No error, and checking the
# geom count does not catch it.
mujoco.mj_forward(model, data)


def update():
    mujoco.mjv_updateScene(model, data, opt, None, cam,
                           mujoco.mjtCatBit.mjCAT_ALL, scene)


def draw(buf):
    mujoco.mjr_render(vp, scene, rc)
    mujoco.mjr_readPixels(buf, None, vp, rc)   # readPixels waits for the render; no mjr_finish


def segment(on):
    scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = on
    scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = on


def f_update():
    update()


def f_onepass():
    update()
    draw(rgb)


def f_twopass():
    """The real frame: RGB pass, then the segmentation pass for per-block visible-pixel counts."""
    update()
    draw(rgb)
    segment(1)
    draw(seg)
    segment(0)


def f_full():
    """Frame plus the physics step that precedes it."""
    mujoco.mj_step(model, data)
    f_twopass()


ARMS = {
    "mjv_updateScene": f_update,
    "1-pass frame": f_onepass,
    "2-pass frame": f_twopass,
    "step + 2-pass": f_full,
}

REPS, N, WARMUP, RESET_EVERY = 5, 1000, 500, 200


def time_arm(fn, N, warmup, reset_every):
    def restart():
        mujoco.mj_resetData(model, data)
        data.ctrl[:] = model.actuator_ctrlrange[:, 1]

    restart()
    for i in range(warmup):
        mujoco.mj_step(model, data)
        fn()
        if (i + 1) % reset_every == 0:
            restart()

    t = np.empty(N, dtype=np.int64)
    for i in range(N):
        t0 = time.perf_counter_ns()
        fn()
        t[i] = time.perf_counter_ns() - t0
        # Step the sim outside the timer for variants that do not step
        # themselves, so every variant sees a moving scene, not a frozen one.
        if fn is not f_full:
            mujoco.mj_step(model, data)
        if (i + 1) % reset_every == 0:
            restart()
    return t


update()
draw(rgb)
assert rgb.max() > 0, "framebuffer empty - nothing rendered"
segment(1)
draw(seg)
segment(0)
assert seg.max() > 0, "segmentation buffer empty"
print(f"renderer {renderer}")
print(f"scene    {model.ngeom} geoms, {scene.ngeom} in mjvScene, {W}x{H}\n")

out = {k: [] for k in ARMS}
for _ in range(REPS):
    for k, fn in ARMS.items():
        out[k].append(time_arm(fn, N, WARMUP, RESET_EVERY))

med = {}
for k, reps in out.items():
    t = np.concatenate(reps)
    v = np.array([np.median(r) for r in reps])
    med[k] = np.median(t) / 1000.0
    print(f"{k:16s} med {med[k]:7.2f} p99 {np.percentile(t, 99)/1000:7.2f}"
          f" max {t.max()/1000:8.2f} us  spread {(v.max()-v.min())/np.median(v)*100:4.1f}%")

frame = med["step + 2-pass"]
print(f"\nframe {frame:.1f} us  |  budget 2000 us (P-6, 500 fps)"
      f"  |  headroom {2000/frame:.1f}x  |  {1e6/frame:,.0f} fps")
print(f"300k frames at this rate: {300_000 * frame / 1e6 / 60:.1f} min single-threaded")
assert frame < 2000, f"frame {frame:.1f} us exceeds the 2000 us P-6 budget"
