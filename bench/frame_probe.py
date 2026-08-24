"""Per-call cost of every term in one dataset frame, against the real scene.

Closes the last day-1 unknown: `mjv_updateScene`. The readback probe timed
render and readback against an inline sphere; this times all of it against
`scene/arm_blocks.xml` with the camera the generator will actually use.

Per-call, then the assembled frame - not a Python end-to-end fps loop, which
hides which term dominates.
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
assert "RTX 5060" in renderer, f"not hardware GL: got {renderer!r}"

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)
assert (model.vis.global_.offwidth, model.vis.global_.offheight) == (W, H), \
    "offscreen buffer must match the capture size exactly"

rc = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, rc)
assert rc.currentBuffer == mujoco.mjtFramebuffer.mjFB_OFFSCREEN, \
    f"offscreen not selected: got {rc.currentBuffer}"

# The XML's named camera, not a free camera: a free camera frames a different
# view and would make these numbers describe a scene we never capture.
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "main")
assert cam.fixedcamid >= 0, "camera 'main' not found - mj_name2id returns -1 silently"

# Defaults already leave every decoration off; zeroing the array also clears
# mjVIS_STATIC and worldbody geoms stop drawing (logged 2026-08-23).
opt = mujoco.MjvOption()
mujoco.mjv_defaultOption(opt)

scene = mujoco.MjvScene(model, maxgeom=1000)
vp = mujoco.MjrRect(0, 0, W, H)
rgb = np.empty((H, W, 3), dtype=np.uint8)
seg = np.empty((H, W, 3), dtype=np.uint8)

data.ctrl[:] = model.actuator_ctrlrange[:, 1]

# Required before the first mjv_updateScene. mjData's derived xpos/xmat are zero
# until forward dynamics runs, so the scene builds with a correct-looking
# scene.ngeom and renders an entirely black frame. Silent - checking the geom
# count does not catch it.
mujoco.mj_forward(model, data)


def update():
    mujoco.mjv_updateScene(model, data, opt, None, cam,
                           mujoco.mjtCatBit.mjCAT_ALL, scene)


def draw(buf):
    mujoco.mjr_render(vp, scene, rc)
    mujoco.mjr_readPixels(buf, None, vp, rc)   # readPixels syncs; no mjr_finish needed


def segment(on):
    scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = on
    scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = on


def f_update():
    update()


def f_onepass():
    update()
    draw(rgb)


def f_twopass():
    """The real frame: RGB pass, then segmentation pass for the F-7 counts."""
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
        # Advance outside the timer for the arms that do not step themselves,
        # so every arm sees a moving scene rather than a frozen one.
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
