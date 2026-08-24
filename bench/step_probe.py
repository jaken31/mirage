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

# Ids by prefix, not by hardcoded name: F-6/F-7 iterations are expected to edit
# the scene, and a hardcoded list would silently drop a fourth block.
NAMES = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)]
ARM = np.array([i for i, n in enumerate(NAMES) if n and n.startswith("link")])
BLOCK = np.array([i for i, n in enumerate(NAMES) if n and n.startswith("block")])
assert ARM.size and BLOCK.size, f"no arm/block geoms found among {NAMES}"

def arm_block_contacts(d):
    """Contact rows pairing an arm link with a block.

    Excludes the table, which rests under every block and is why a plain
    `ncon > 0` guard was vacuous. `d.contact` is already sliced to `ncon`
    (verified, mujoco 3.12.0), so no truncation. Column order is not
    guaranteed, hence both directions.
    """
    g = np.asarray(d.contact.geom)                      # (ncon, 2)
    a, b = np.isin(g, ARM), np.isin(g, BLOCK)
    return int(((a[:, 0] & b[:, 1]) | (b[:, 0] & a[:, 1])).sum())

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
    ab = np.empty(N, dtype=np.int32)

    for i in range(N):
        t0 = time.perf_counter_ns()
        mujoco.mj_step(model, data)
        t[i] = time.perf_counter_ns() - t0
        nc[i] = data.ncon
        ab[i] = arm_block_contacts(data)      # costs more than mj_step; outside the timer
        if (i + 1) % reset_every == 0:
            restart()

    return t, nc, ab

REPS, N, warmup, reset_every = 5, 1000, 500, 200
arms = {"idle": None, "driven": DRIVE}

out = {k: [] for k in arms}
for _ in range(REPS):
    for k, drive in arms.items():
        out[k].append(time_steps(drive, N, warmup, reset_every))

for k, reps in out.items():
    t = np.concatenate([r[0] for r in reps])
    nc = np.concatenate([r[1] for r in reps])
    ab = np.concatenate([r[2] for r in reps])
    v = np.array([np.median(r[0]) for r in reps])          # one median per rep
    frac = (ab > 0).mean()
    print(f"{k:8s} med {np.median(t)/1000:7.2f} p99 {np.percentile(t, 99)/1000:7.2f}"
          f" max {t.max()/1000:8.2f} us  arm-block {frac:5.1%}  ncon max {nc.max():3d}"
          f"  spread {(v.max() - v.min()) / np.median(v) * 100:4.1f}%")

    if k == "driven":
        # Catches "the arm never touched a block". Below this is a finding about
        # arm reach or reset cadence - do not lower it to make the run pass.
        assert frac > 0.10, f"arm-block contact in only {frac:.1%} of steps - measurement invalid"
        med = np.median(t) / 1000.0
        # 1850 us: the 2000 us frame P-6 allows at 500 fps, less render+readback and margin
        print(f"driven median {med:.1f} us  |  budget 1850 us  "
              f"|  headroom {1850 / med:.0f}x  |  {1e6 / med:,.0f} steps/s")