"""Settles `sim.action_hold_steps`, which has been a guess since it was written.

Three questions, in order:

  A. How long does a joint actually take to reverse?  Drive it from rest and read
     the step count at 63% of terminal velocity - one time constant.  The
     architecture doc estimated ~15 steps from `inertia / damping` using
     `dof_armature = 0.01`.  Armature is the term *added* to the mass-matrix
     diagonal, not the diagonal, so that estimate omits the link inertia.

  B. What does the hold cost Q-4?  Q-4 scores action-following as
     `sign(theta_t+1 - theta_t)` against the commanded sign.  For roughly one
     settling time after every sign flip the joint is still moving the old way,
     so those frames disagree with the command through no fault of any model.
     Too short a hold and **Q-4 is unreachable by any model** - a dataset defect
     that would present as a modelling failure.

     Reported per position-within-hold, not only as an average, because the
     average hides the shape, and the shape is what picking a hold trades against.

  C. Free rider: the fraction of ctx-length windows containing no action change.
     A window with a constant action carries no evidence of what the action
     *does*.  Measured at 58.7% on the shipped dataset; this reports how it moves
     with the hold.  D2 in `docs/phase0_debt_checklist.md`.

Drives the **random** half of the policy - uniform draws over the action space,
held for `hold` steps.  That is exactly what `Policy` does on a random episode.
The scripted half is C++ and is not reimplemented here: two implementations of one
policy drift, and the physics answering A and B does not depend on how the action
was chosen.  What does depend on it is C, which reads worse under the scripted
half - it re-derives `sign(gain)` on each draw and usually gets the same corner
back.  The shipped-dataset number covers that; see the note at the end.

    python bench/hold_probe.py
"""

import json
import pathlib

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENE = ROOT / "scene" / "arm_blocks.xml"
CONFIG = ROOT / "mirage" / "configs" / "base.json"
assert SCENE.exists(), f"scene file not found: {SCENE}"
assert CONFIG.exists(), f"config not found: {CONFIG}"

cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
SHIPPED_HOLD = cfg["sim"]["action_hold_steps"]
STEPS_PER_EPISODE = cfg["sim"]["steps_per_episode"]
CTX = cfg["data"]["ctx"]

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)

assert model.nu == 2, f"expected 2 actuators, got {model.nu}"
assert model.actuator_ctrllimited.all(), "ctrlrange is meaningless without ctrllimited"

DT = model.opt.timestep
LEVELS = 3                                   # matches kActionLevels in sim/policy.h
N_ACTIONS = LEVELS ** model.nu
DRIVE_HI = model.actuator_ctrlrange[:, 1].copy()
DRIVE_LO = model.actuator_ctrlrange[:, 0].copy()

# Joint i is base-3 digit i, least significant first - sim/policy.h.  Reproduced
# rather than imported because there is no binding; `policy_self_check` is what
# guarantees the C++ side round-trips, and this only has to agree with the
# encoding, which is eight lines and fully specified in that header.
DIGITS = np.array([[(a // LEVELS ** j) % LEVELS - 1 for j in range(model.nu)]
                   for a in range(N_ACTIONS)])          # (9, 2) in {-1, 0, +1}

# Read, never computed - jnt_qposadr and jnt_dofadr diverge for free joints.
QADR = model.jnt_qposadr[:model.nu].copy()
DADR = model.jnt_dofadr[:model.nu].copy()


def signs_to_ctrl(signs):
    """Decode per-joint signs onto ctrl.  Mirrors `action_to_control`: the two
    driven directions map onto ctrlrange's ends rather than literal -1/+1, and
    neutral is zero clipped into that range."""
    return np.where(signs > 0, DRIVE_HI, np.where(signs < 0, DRIVE_LO, 0.0))


# --------------------------------------------------------------------------
# A. Settling time
# --------------------------------------------------------------------------

# Horizon is short on purpose.  Terminal velocity is gear*ctrl/damping ~ 4 rad/s,
# so a 600-step run travels ~4.8 rad - past joint1's 5 rad range.  A joint parked
# against its stop has terminal velocity ~0, which makes the 63% crossing fire on
# step 1 and reports a settling time of one step for a joint that takes twelve.
# 150 steps is 5.5 time constants for the slowest joint and ~1.2 rad of travel.
SETTLE_HORIZON = 150


def settling_steps(joint, q1, horizon=SETTLE_HORIZON):
    """Steps until |qvel[joint]| first reaches 63% of terminal, driving that joint
    alone at full torque from rest with link1 folded to `q1`.

    Terminal is the mean of the last 10% of the horizon rather than the final
    sample, so one noisy step cannot set the target.  A limited joint starts at
    its low stop so it has the whole range to accelerate through, and the run
    asserts it never reached the far stop - against a stop the measurement is
    meaningless, not merely noisy."""
    mujoco.mj_resetData(model, data)
    data.qpos[QADR[1]] = q1
    if model.jnt_limited[joint]:
        lo, hi = model.jnt_range[joint]
        data.qpos[QADR[joint]] = lo
    mujoco.mj_forward(model, data)           # derived xpos/xmat are zero until this runs
    ctrl = np.zeros(model.nu)
    ctrl[joint] = DRIVE_HI[joint]
    data.ctrl[:] = ctrl

    v = np.empty(horizon)
    for i in range(horizon):
        mujoco.mj_step(model, data)
        v[i] = abs(data.qvel[DADR[joint]])

    if model.jnt_limited[joint]:
        lo, hi = model.jnt_range[joint]
        reached = data.qpos[QADR[joint]]
        assert reached < hi - 0.05, (
            f"joint {joint} hit its far stop at {reached:.3f} of {hi} within "
            f"{horizon} steps - shorten SETTLE_HORIZON, the terminal velocity "
            f"read from a parked joint is not a terminal velocity")

    terminal = v[int(horizon * 0.9):].mean()
    if terminal <= 0:
        return None, 0.0
    crossed = np.flatnonzero(v >= 0.63 * terminal)
    return (int(crossed[0]) + 1 if crossed.size else None), terminal


print("=" * 78)
print("A. Settling time - steps to 63% of terminal velocity")
print("=" * 78)
print(f"timestep {DT * 1000:.0f} ms   dof_damping {model.dof_damping[:2]}   "
      f"dof_armature {model.dof_armature[:2]}")
print()

full = np.zeros((model.nv, model.nv))
print(f"{'link1 angle':>12} | {'M00':>8} {'est':>6} {'measured':>9} | "
      f"{'M11':>8} {'est':>6} {'measured':>9}")
print("-" * 78)
tau_meas = {0: [], 1: []}
for q1 in (0.0, 1.25, 2.5, -2.5):
    data.qpos[:] = 0
    data.qpos[QADR[1]] = q1
    mujoco.mj_forward(model, data)
    mujoco.mj_fullM(model, data, full)
    row = f"{q1:>12.2f} |"
    for j in (0, 1):
        est = full[j, j] / model.dof_damping[j] / DT
        meas, _ = settling_steps(j, q1)
        tau_meas[j].append(meas)
        row += f" {full[j, j]:8.5f} {est:6.1f} {meas:9d} |"
    print(row)

# M11 is identical at every link1 angle above - joint1's own inertia does not
# depend on the angle of the link it carries, only joint0's does. So joint1 has
# one settling time, not a spread, and the four rows agreeing is the check.
assert len(set(tau_meas[1])) == 1, f"joint1 varies by configuration: {tau_meas[1]}"

TAU0 = max(t for t in tau_meas[0] if t)
TAU1 = max(t for t in tau_meas[1] if t)
print()
print(f"Worst case over the workspace: joint0 {TAU0} steps, joint1 {TAU1} steps.")
print("The architecture doc's estimate was ~15 steps, taken from armature alone.")
print(f"Shipped `action_hold_steps` is {SHIPPED_HOLD}.")
print("`est` is the first-order M/b prediction; `measured` is the 63% crossing.")


# --------------------------------------------------------------------------
# B + C. Hold sweep
# --------------------------------------------------------------------------

def sweep(hold, episodes, rng):
    """Random held actions.  Returns per-offset agreement on driven digits, the
    overall agreement, and the fraction of ctx windows carrying an action change."""
    hit = np.zeros(hold, dtype=np.int64)     # agreements by position within hold
    tot = np.zeros(hold, dtype=np.int64)
    win_total = win_changed = 0

    for _ in range(episodes):
        mujoco.mj_resetData(model, data)
        data.qpos[QADR[0]] = rng.uniform(-np.pi, np.pi)
        data.qpos[QADR[1]] = rng.uniform(-2.5, 2.5)
        mujoco.mj_forward(model, data)

        actions = np.empty(STEPS_PER_EPISODE, dtype=np.int64)
        held = -1
        for t in range(STEPS_PER_EPISODE):
            off = t % hold
            if off == 0:
                held = int(rng.integers(N_ACTIONS))
                data.ctrl[:] = signs_to_ctrl(DIGITS[held])
            actions[t] = held

            before = data.qpos[QADR].copy()
            mujoco.mj_step(model, data)
            delta = data.qpos[QADR] - before

            cmd = DIGITS[held]
            driven = cmd != 0
            if driven.any():
                hit[off] += int((np.sign(delta[driven]) == cmd[driven]).sum())
                tot[off] += int(driven.sum())

        d = (actions[1:] != actions[:-1]).astype(np.int64)
        c = np.r_[0, np.cumsum(d)]
        n = STEPS_PER_EPISODE - CTX + 1
        win_total += n
        win_changed += int((c[CTX - 1:CTX - 1 + n] - c[:n] > 0).sum())

    per_off = np.divide(hit, tot, out=np.zeros(hold), where=tot > 0)
    return per_off, hit.sum() / max(tot.sum(), 1), win_changed / win_total


HOLDS = [8, 12, 15, 20, 25, 30, 35, 45, 60]
EPISODES = 60

print()
print("=" * 78)
print("B. Action-following agreement vs hold   (Q-4 wants >= 90%)")
print("=" * 78)
print(f"{EPISODES} episodes x {STEPS_PER_EPISODE} steps per hold, random half of the policy.")
print(f"`transient` is the mean over the first {TAU0} frames of each hold - one joint0")
print("settling time, the frames Q-4 cannot win. `settled` is the remainder.")
print()
print(f"{'hold':>5} | {'overall':>8} {'transient':>10} {'settled':>9} | "
      f"{'window has change':>18} | {'Q-4':>5}")
print("-" * 78)

rows = []
for h in HOLDS:
    rng = np.random.default_rng(0)           # same stream per hold - controlled comparison
    per_off, overall, changed = sweep(h, EPISODES, rng)
    k = min(TAU0, h)
    early = per_off[:k].mean()
    late = per_off[k:].mean() if h > k else float("nan")
    rows.append((h, overall, early, late, changed))
    mark = "  <- shipped" if h == SHIPPED_HOLD else ""
    print(f"{h:>5} | {overall:>7.1%} {early:>10.1%} "
          f"{'      n/a' if np.isnan(late) else format(late, '>9.1%')} | "
          f"{changed:>17.1%} | {'PASS' if overall >= 0.90 else 'fail':>5}{mark}")

ok = [r for r in rows if r[1] >= 0.90]
print()
print(f"Agreement crosses 90% at hold >= {ok[0][0]}." if ok
      else "Agreement never reaches 90% in the swept range.")

print()
print("=" * 78)
print("C. Constant-action windows (D2)")
print("=" * 78)
print(f"ctx = {CTX}. A window with no action change carries no evidence of what the")
print("action does. The column above is the RANDOM half only, and is therefore the")
print("optimistic bound: the shipped 300k set reads 58.7% constant, because the")
print("scripted half re-derives sign(gain) on each draw and usually repeats,")
print("stretching its effective change interval to ~34 steps.")
print()
print(f"`action_hold_steps < ctx` ({CTX}) is the only setting where every window")
print("contains a boundary by construction. Note that it pulls against B directly:")
print("a shorter hold spends a larger share of frames in the transient.")


# --------------------------------------------------------------------------
# D. Is the trade forced?
# --------------------------------------------------------------------------
# B and C pull opposite ways: agreement wants a long hold, window coverage wants
# a short one, and at the shipped physics nothing satisfies both.  But the
# transient length is not a constant of the universe - it is tau = M / damping,
# while the arm's speed is v_term = gear * ctrl / damping.  Scaling gear and
# damping together holds the speed and shrinks the transient, which moves the
# frontier instead of sliding along it.

print()
print("=" * 78)
print("D. Is the trade forced?   tau = M/damping,  v_term = gear*ctrl/damping")
print("=" * 78)
print("Scaling gear and damping together keeps the arm's speed and shortens its")
print("response. This is a scene change - it moves `data_hash` and needs F-5, F-6")
print("and F-7 re-verified - so it is measured here, not applied.")
print()
print(f"{'gear':>5} {'damp':>5} | {'v_term':>7} {'tau0':>5} {'hold':>5} | "
      f"{'agreement':>10} {'win w/ change':>14} | both")
print("-" * 78)

GEAR0 = model.actuator_gear[:model.nu, 0].copy()
DAMP0 = model.dof_damping[:model.nu].copy()
try:
    for scale in (1, 2, 3):
        model.actuator_gear[:model.nu, 0] = GEAR0 * scale
        model.dof_damping[:model.nu] = DAMP0 * scale
        tau0, v_term = None, None
        for q1 in (0.0, 1.25, 2.5, -2.5):       # worst case over the workspace
            steps, term = settling_steps(0, q1)
            if steps is not None and (tau0 is None or steps > tau0):
                tau0, v_term = steps, term
        for hold in (12, 15, 20):
            rng = np.random.default_rng(0)
            _, overall, changed = sweep(hold, EPISODES // 2, rng)
            both = "YES" if overall >= 0.90 and changed >= 0.80 else ""
            mark = "  <- shipped" if scale == 1 and hold == SHIPPED_HOLD else ""
            print(f"{GEAR0[0] * scale:>5.1f} {DAMP0[0] * scale:>5.2f} | {v_term:>7.2f} "
                  f"{tau0:>5} {hold:>5} | {overall:>9.1%} {changed:>13.1%} | "
                  f"{both:>4}{mark}")
finally:
    model.actuator_gear[:model.nu, 0] = GEAR0   # never leave the model mutated
    model.dof_damping[:model.nu] = DAMP0

print()
print("Terminal velocity is flat down the table, so the arm is not being made")
print("faster - only more responsive. Caveat before acting on this: the")
print("window-with-change column is the RANDOM half. The shipped 50/50 mix reads")
print("41.3% at hold 20 against this table's 62.1%, so the scripted half roughly")
print("halves it, and that cause is untouched by any physics change.")


def _self_check():
    """The encoding reproduced here must match sim/policy.h, and both measurements
    must respond in the direction the physics requires."""
    assert N_ACTIONS == 9, N_ACTIONS
    assert DIGITS.shape == (9, 2), DIGITS.shape
    # digit i is joint i, least significant first; 4 is the neutral pair
    assert (DIGITS[4] == 0).all(), DIGITS[4]
    assert (DIGITS[0] == -1).all(), DIGITS[0]
    assert (DIGITS[8] == +1).all(), DIGITS[8]
    assert len({tuple(d) for d in DIGITS}) == 9, "actions must decode to distinct pairs"
    # a longer hold amortises the transient over more frames, so agreement rises
    assert rows[-1][1] > rows[0][1], (rows[0], rows[-1])
    # a longer hold means fewer boundaries, so fewer windows carry a change
    assert rows[-1][4] < rows[0][4], (rows[0], rows[-1])
    # a 63% crossing at one step would mean the measurement resolved nothing
    assert TAU0 > 1 and TAU1 > 1, (TAU0, TAU1)
    print("\nself-check ok")


_self_check()
