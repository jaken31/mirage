"""Live sandbox: move the arm by hand and watch the tokenizer rebuild the camera feed.

    python bench/sandbox.py                    # default R1 gate checkpoint
    python bench/sandbox.py --run RUN_ID

Three panels update live: what the dataset camera sees at 64x64, what the
tokenizer rebuilds from its token ids, and the absolute difference, with the
live PSNR in the title. One slider per actuator, so you can put the arm
anywhere and see where the reconstruction breaks down. Space pauses.

Nothing here is a measurement. The real check is
`python -m mirage.fsq --eval RUN_ID`, and PSNR on a hand-posed frame is not
comparable to it.

ponytail: no 3D viewer window. `mujoco.viewer` needs its own GLFW context next
to the offscreen one; for a 3D view, run `python -m mujoco.viewer
--mjcf=scene/arm_blocks.xml` in a second terminal.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from matplotlib.widgets import Slider
from OpenGL.GL import GL_RENDERER, glGetString

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mirage import config                      # noqa: E402
from mirage.fsq import PEAK                    # noqa: E402
from mirage.fsq_eval import load_run           # noqa: E402

SIZE = 64
SETTLE_STEPS = 100
STEPS_PER_FRAME = 10   # sim steps between redraws; roughly real time at 60 redraws/s
R1_DEFAULT = "20260829-005439-r1"


def render(model, data, ctx, scene, opt, cam, viewport, rgb):
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, ctx)
    mujoco.mjr_finish()
    mujoco.mjr_readPixels(rgb, None, viewport, ctx)
    # readPixels returns rows bottom-up; training frames are flipped right-side
    # up (see mirage.data.preload), so flip here or the model sees it upside down.
    return np.flipud(rgb).copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=R1_DEFAULT, help="run id under runs/")
    ap.add_argument("--selfcheck", action="store_true",
                    help="run one frame headless and exit, instead of opening the window")
    ap.add_argument("--config", default=str(ROOT / "mirage" / "configs" / "base.json"))
    args = ap.parse_args()

    cfg = config.load(args.config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, knobs = load_run(args.run, cfg, device=dev)
    print(f"{args.run}: levels={knobs['levels']} attention={knobs['attention']} "
          f"quantize={knobs['quantize']} on {dev}")

    mj = mujoco.MjModel.from_xml_path(cfg.sim["scene_xml"])
    assert mj.vis.global_.offwidth >= SIZE and mj.vis.global_.offheight >= SIZE
    gl = mujoco.GLContext(mj.vis.global_.offwidth, mj.vis.global_.offheight)
    gl.make_current()
    ctx = mujoco.MjrContext(mj, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, ctx)

    # Same check as bench/frame_probe.py: these two are the Windows software
    # renderers, and with one of them every number and pixel below is meaningless.
    gl_renderer = glGetString(GL_RENDERER).decode()
    assert not any(s in gl_renderer for s in ("GDI Generic", "Microsoft Basic Render Driver")),         f"software GL, not hardware: {gl_renderer!r}"
    print("GL_RENDERER:", gl_renderer)

    d = mujoco.MjData(mj)
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(mj, d)

    cam = mujoco.MjvCamera()
    if mj.ncam:
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = 0
    else:
        mujoco.mjv_defaultFreeCamera(mj, cam)
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    scene = mujoco.MjvScene(mj, maxgeom=1000)
    viewport = mujoco.MjrRect(0, 0, SIZE, SIZE)
    rgb = np.empty((SIZE, SIZE, 3), dtype=np.uint8)

    frame = render(mj, d, ctx, scene, opt, cam, viewport, rgb)
    assert frame.max() > 0, "framebuffer empty - nothing rendered"

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.6))
    fig.subplots_adjust(bottom=0.10 + 0.06 * mj.nu, top=0.86)
    ims = [ax.imshow(frame, interpolation="nearest") for ax in axes]
    for ax, name in zip(axes, ("camera 64x64", "reconstruction", "|difference|")):
        ax.set_title(name, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    sliders = []
    for i in range(mj.nu):
        lo, hi = mj.actuator_ctrlrange[i]
        sax = fig.add_axes([0.15, 0.03 + 0.06 * i, 0.7, 0.03])
        name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}"
        sliders.append(Slider(sax, name, float(lo), float(hi), valinit=0.0))

    state = {"paused": False}

    def on_key(event):
        if event.key == " ":
            state["paused"] = not state["paused"]

    fig.canvas.mpl_connect("key_press_event", on_key)

    def tick(_=None):
        if state["paused"]:
            return
        for i, s in enumerate(sliders):
            d.ctrl[i] = s.val
        for _ in range(STEPS_PER_FRAME):
            mujoco.mj_step(mj, d)
        cur = render(mj, d, ctx, scene, opt, cam, viewport, rgb)

        # The model works in [0, 1] and output is converted to uint8, the same as
        # `reconstruction_psnr`, so this PSNR is the same kind of number the gate
        # reports.
        x = torch.from_numpy(cur).to(dev).float().permute(2, 0, 1)[None] / PEAK
        with torch.no_grad():
            out = tok(x)
        rec = ((out[0] * PEAK).round().clamp(0, PEAK)
               .permute(1, 2, 0).to(torch.uint8).cpu().numpy())

        sse = float(((cur.astype(np.float64) - rec.astype(np.float64)) ** 2).sum())
        db = 10.0 * np.log10(PEAK * PEAK / (sse / cur.size)) if sse > 0 else float("inf")
        ims[0].set_data(cur)
        ims[1].set_data(rec)
        ims[2].set_data(np.abs(cur.astype(np.int16) - rec.astype(np.int16)).astype(np.uint8))
        fig.suptitle(f"{args.run}   live PSNR {db:.2f} dB   "
                     f"(space = pause; not a gate number)", fontsize=10)
        fig.canvas.draw_idle()

    if args.selfcheck:
        tick()
        assert ims[1].get_array().shape == (SIZE, SIZE, 3)
        print("selfcheck ok:", fig._suptitle.get_text())
        return

    timer = fig.canvas.new_timer(interval=33)
    timer.add_callback(tick)
    timer.start()
    plt.show()


if __name__ == "__main__":
    main()
