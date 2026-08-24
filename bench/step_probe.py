import mujoco
import pathlib 
import numpy as np
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENE = ROOT / "scene" / "arm_blocks.xml"
assert SCENE.exists(), f"scene file not found: {SCENE}"
model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)

assert model.nq == 23, f"expected 23 dofs, got {model.nq}"
assert model.nu == 2, f"expected 2 actuators, got {model.nu}"
assert model.opt.timestep == 0.002, f"expected timestep 0.002, got {model.opt.timestep}"

assert model.actuator_ctrllimited.all(), "ctrlrange is meaningless without crtllimited"
DRIVE = model.actuator_ctrlrange[:, 1].copy()

def time_steps(drive, N, warmup, reset_every):
    def restart():
        mujoco.mj_resetData(model, data)
        if drive is not None:
            data.ctrl[:] = drive
    restart()

    for i in range(warmup):
        mujoco.mj_step(model, data)
        if (i + 1) % reset_every == 0:
            restart()
    t = np.empty(N, dtype=np.int64)
    nc = np.empty(N, dtype=np.int32)

    for i in range(N):
        t0 = time.perf_counter_ns()
        mujoco.mj_step(model, data)
        t[i] = time.perf_counter_ns() - t0
        nc[i] = data.ncon
        if (i + 1) % reset_every == 0:
            restart()

    return t, nc

REPS, N, warmup, reset_every = 5, 1000, 500, 200
arms = {"idle": None, "driven": DRIVE}

out = {k: [] for k in arms}
for _ in range(REPS):
    for k, drive in arms.items():
        out[k].append(time_steps(drive, N, warmup, reset_every))

for k, reps in out.items():
    t = np.concatenate([r[0] for r in reps])
    nc = np.concatenate([r[1] for r in reps])
    v = np.array([np.median(r[0]) for r in reps])          # one median per rep
    frac = (nc > 0).mean()
    print(f"{k:8s} med {np.median(t)/1000:7.2f} p99 {np.percentile(t, 99)/1000:7.2f}"
          f" max {t.max()/1000:8.2f} us  contacts {frac:5.1%}  ncon max {nc.max():3d}"
          f"  spread {(v.max() - v.min()) / np.median(v) * 100:4.1f}%")

    if k == "driven":
        assert frac > 0.3, f"contact fraction {frac:.1%} too low - measurement invalid"
        med = np.median(t) / 1000.0
        # 1850 us: the 2000 us frame P-6 allows at 500 fps, less render+readback and margin
        print(f"driven median {med:.1f} us  |  budget 1850 us  "
              f"|  headroom {1850 / med:.0f}x  |  {1e6 / med:,.0f} steps/s")