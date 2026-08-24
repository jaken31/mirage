import sys

import mujoco
import numpy as np
from PIL import Image

XML = sys.argv[1] if len(sys.argv) > 1 else "scene/arm_blocks.xml"
OUT = "bench/preview.png"
SIZE = 64          # what the pipeline captures; keep the palette count honest
SETTLE_STEPS = 100  # let the blocks come to rest before looking
ZOOM = 8           # ponytail: 64px is unviewable, so the PNG is upscaled nearest-neighbour
                   # while the colour count uses the original pixels

model = mujoco.MjModel.from_xml_path(XML)

buf_w, buf_h = model.vis.global_.offwidth, model.vis.global_.offheight
assert buf_w >= SIZE and buf_h >= SIZE, \
    f"offscreen buffer {buf_w}x{buf_h} is smaller than {SIZE}: set <visual><global offwidth/offheight>"

gl = mujoco.GLContext(buf_w, buf_h)
gl.make_current()

ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, ctx)
assert ctx.currentBuffer == mujoco.mjtFramebuffer.mjFB_OFFSCREEN, "offscreen buffer not selected"

data = mujoco.MjData(model)
for _ in range(SETTLE_STEPS):
    mujoco.mj_step(model, data)

cam = mujoco.MjvCamera()
if model.ncam:
    # Preview what the dataset will see: the first named camera, not the free camera.
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = 0
    which = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, 0)
else:
    mujoco.mjv_defaultFreeCamera(model, cam)
    which = "free camera (no <camera> in scene)"

opt = mujoco.MjvOption()
mujoco.mjv_defaultOption(opt)
# Defaults already have every decoration off (contact dots, joint axes, COM, actuators).
# Do not zero the whole array: that clears mjVIS_STATIC and worldbody geoms stop drawing.
# mjVIS_TEXTURE stays on deliberately, so a textured material shows up in the colour count.

scene = mujoco.MjvScene(model, maxgeom=1000)
mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)

viewport = mujoco.MjrRect(0, 0, SIZE, SIZE)
mujoco.mjr_render(viewport, scene, ctx)
mujoco.mjr_finish()

rgb = np.empty((SIZE, SIZE, 3), dtype=np.uint8)
mujoco.mjr_readPixels(rgb, None, viewport, ctx)
assert rgb.max() > 0, "framebuffer empty - nothing rendered"

colours = np.unique(rgb.reshape(-1, 3), axis=0)
verdict = "ok" if len(colours) <= 24 else "OVER 24"
print(f"{XML}  via {which}")
print(f"{SIZE}x{SIZE}  offsamples={model.vis.quality.offsamples} "
      f"shadowsize={model.vis.quality.shadowsize}  colours={len(colours)} {verdict}")
if len(colours) <= 24:
    for c in colours:
        print("   ", tuple(int(v) for v in c))

Image.fromarray(np.flipud(rgb)).resize((SIZE * ZOOM, SIZE * ZOOM), Image.NEAREST).save(OUT)
print("wrote", OUT)
