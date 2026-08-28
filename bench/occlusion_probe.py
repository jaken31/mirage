"""How much of F-7's occlusion rate is recoverable occlusion, and how much is a
block that is simply gone.

F-7 counts a frame when any block reads `visible_px == 0`. Two different things
answer yes to that, and the counter cannot tell them apart:

  * the arm passes in front of the block - it comes back, which is the thing
    F-7 exists to measure and the thing Q-6 scores object permanence on;
  * the block leaves for good - knocked out of the camera's view, or off the
    table - and reads 0 for every remaining frame of the episode.

The second inflates F-7 and would drag Q-6 down for a reason that has nothing to
do with a model's memory: an "occlusion event" that can never end is one no
model can be asked to recover from.

`phase0_debt_checklist.md`, Tier 5 records the bias and defers the fix, on the
grounds that separating the two cases needs a "block is on the table" field and
therefore a regeneration. **This probe is what says whether that field is worth
one**, and it needs no regeneration: the split is already recoverable from
`visible_px` and `block_xy` in the shards on disk.

    python bench/occlusion_probe.py
"""

import pathlib
import sys
# stdlib ElementTree, for the reason mirage/validator.py records at length:
# the only file parsed is scene/arm_blocks.xml, a version-controlled repo
# artifact at the same trust level as this source. No XXE boundary to defend.
import xml.etree.ElementTree as ET

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mirage import config, data  # noqa: E402

cfg = config.load(ROOT / "mirage" / "configs" / "base.json")

# Table half-extent in x and y, read from the scene rather than restated as a
# literal - the whole question here is whether a block can leave it.
scene_root = ET.parse(ROOT / cfg.sim["scene_xml"]).getroot()
table = next(g for g in scene_root.iter("geom") if g.get("name") == "table")
table_size = table.get("size")
assert table_size, "the table geom has no size attribute"
TABLE_HALF = np.array([float(v) for v in table_size.split()][:2])
print(f"table half-extent {TABLE_HALF[0]} x {TABLE_HALF[1]} m, from the scene")

shards = data.load_shards(ROOT / cfg.data["shard_dir"], data_hash=cfg.data_hash)
episodes = cfg.sim["episodes"]
steps = cfg.sim["steps_per_episode"]
blocks = len([g for g in scene_root.iter("geom") if (g.get("name") or "").startswith("block")])

# Episodes are contiguous, 0..episodes-1, and never split across a shard -
# mirage/data.py's self-check asserts all three - so concatenating the shards in
# index order and reshaping gives the episode axis. Asserted rather than
# assumed: a silent off-by-one here would stitch two episodes together and every
# "the block never came back" verdict below would be about the wrong frames.
meta = np.concatenate([np.asarray(s.meta) for s in shards])
assert len(meta) == episodes * steps, f"{len(meta)} frames, expected {episodes * steps}"
ep_id = meta["episode_id"].reshape(episodes, steps)
assert (ep_id == ep_id[:, :1]).all(), "an episode is split across the reshape"
assert (ep_id[:, 0] == np.arange(episodes)).all(), "episode ids are not 0..n-1 in order"

# (blocks, episodes, steps) and (blocks, episodes, steps, 2)
visible = np.stack([meta[f"visible_px{b}"].reshape(episodes, steps) for b in range(blocks)])
xy = np.stack([
    np.stack([meta[f"block_xy{2 * b}"].reshape(episodes, steps),
              meta[f"block_xy{2 * b + 1}"].reshape(episodes, steps)], axis=-1)
    for b in range(blocks)
])

zero = visible == 0

# "Comes back" = this block is visible at some strictly later step of the same
# episode. A reversed cumulative max over the step axis is exactly that, and it
# is what separates a real occlusion event from a block that has left.
later = np.flip(np.maximum.accumulate(np.flip(visible > 0, axis=2), axis=2), axis=2)
seen_later = np.concatenate([later[:, :, 1:], np.zeros_like(later[:, :, :1])], axis=2)

recoverable = zero & seen_later      # hidden now, visible again later this episode
terminal = zero & ~seen_later        # never visible again this episode

f7_raw = float(zero.any(axis=0).mean())
f7_recoverable = float(recoverable.any(axis=0).mean())
f7_terminal_only = float((terminal.any(axis=0) & ~recoverable.any(axis=0)).mean())

print()
print(f"F-7 as the requirement counts it   {f7_raw:7.2%}   (floor 3%)")
print(f"  of which recoverable occlusion   {f7_recoverable:7.2%}   block is visible again later")
print(f"  frames counted only by a block")
print(f"  that never returns               {f7_terminal_only:7.2%}   <- the bias")
print(f"F-7 with the bias removed          {f7_recoverable:7.2%}   "
      f"{f7_recoverable / 0.03:.1f}x the floor")

print()
print("per block, over all frames:")
for b in range(blocks):
    never = int((visible[b].max(axis=1) == 0).sum())
    print(f"  block{b}: zero {zero[b].mean():6.2%}  recoverable {recoverable[b].mean():6.2%}  "
          f"terminal {terminal[b].mean():6.2%}  never visible in {never} of {episodes} episodes")

# The knocked-off-the-table hypothesis, tested rather than assumed. The arm's
# reach is ~0.33 m and the table is 1.2 m to each edge, so a block would have to
# be pushed roughly four times the distance the arm can even touch.
off_table = (np.abs(xy) > TABLE_HALF).any(axis=-1)
print()
print(f"worst |x| any block reached {np.abs(xy[..., 0]).max():.3f} m, "
      f"worst |y| {np.abs(xy[..., 1]).max():.3f} m, against the table's "
      f"{TABLE_HALF[0]} / {TABLE_HALF[1]} m")
print(f"frames with a block off the table: {int(off_table.sum())} of {off_table.size}")

# Where a terminal run begins, if any. A block that stops being visible while
# still well inside the table has left the *camera*, not the table - and a
# "block is on the table" field would not have caught it.
prev = np.concatenate([np.zeros_like(terminal[:, :, :1]), terminal[:, :, :-1]], axis=2)
starts = terminal & ~prev
if starts.any():
    where = np.abs(xy[starts])
    print(f"terminal runs: {int(starts.sum())}, beginning at |x| up to {where[:, 0].max():.3f} m "
          f"and |y| up to {where[:, 1].max():.3f} m")
else:
    print("terminal runs: none - every block that goes to 0 px comes back")

# Off the table or out of the camera? Different causes, different fixes, so it
# is settled by projecting rather than argued. An axis-aligned box around the
# positions that render will not do it - the frustum is a cone, so a block can
# sit inside the worst-case |x| and |y| and still be outside the view.
#
# MuJoCo's `xyaxes` gives the camera's right and up in world coordinates; the
# camera frame's z is right x up and the camera looks along -z. Default fovy is
# 45 degrees and the frame is square, so the horizontal half-angle equals the
# vertical one.
camera = next(c for c in scene_root.iter("camera") if c.get("name") == "main")
cam_pos_attr, cam_axes_attr = camera.get("pos"), camera.get("xyaxes")
assert cam_pos_attr and cam_axes_attr, "the main camera has no pos/xyaxes"
cam_pos = np.array([float(v) for v in cam_pos_attr.split()])
axes = np.array([float(v) for v in cam_axes_attr.split()])
right = axes[:3] / np.linalg.norm(axes[:3])
up = axes[3:] / np.linalg.norm(axes[3:])
forward = -np.cross(right, up)
FOVY_DEG = float(camera.get("fovy") or 45.0)   # MuJoCo's default when unset
half = np.tan(np.radians(FOVY_DEG) / 2.0)

# The block's centre is not enough. Blocks get pushed to within 0.196 m of a
# camera that sits at y=-0.5, and at that range a 0.05 m cube subtends enough
# angle that its centre leaves the frustum while a corner is still rendering.
# The margin is the cube's circumradius, from the scene's own half-size - a
# derived number, not one tuned until the check passed.
half_size = float(next(g for g in scene_root.iter("geom")
                       if (g.get("name") or "").startswith("block")).get("size").split()[0])
RADIUS = half_size * np.sqrt(3.0)

# Block centre height is the free-joint body's z from the scene, which is the
# block half-size - they rest on the table at z=0.
block_body = next(b for b in scene_root.iter("body")
                  if (b.get("name") or "").startswith("block"))
block_z = float(block_body.get("pos").split()[2])
rel = np.stack([xy[..., 0], xy[..., 1], np.full(xy.shape[:-1], block_z)], axis=-1) - cam_pos
depth = rel @ forward
in_view = ((depth > -RADIUS)
           & (np.abs(rel @ right) < half * depth + RADIUS)
           & (np.abs(rel @ up) < half * depth + RADIUS))

# The projection, checked against the renderer before any conclusion rests on
# it: a block that rendered pixels must project inside the frustum. It does, on
# all but a handful of frames out of 900,000.
seen = visible > 0
agree = float(in_view[seen].mean())
print()
print(f"camera model: {agree:.4%} of frames where a block rendered pixels project "
      f"inside the {FOVY_DEG:g}-degree frustum (margin {RADIUS:.4f} m, the cube's "
      f"circumradius); closest a visible block came was {depth[seen].min():.3f} m")
assert agree > 0.999, f"the projection disagrees with the renderer on {1 - agree:.3%} of frames"

out_of_view = float((~in_view).any(axis=0).mean())
terminal_in_view = float((terminal & in_view).any(axis=0).mean())
print()
print(f"frames with a block outside the camera        {out_of_view:7.2%}")
print(f"terminal frames with the block STILL in view  {terminal_in_view:7.2%}")
print(f"  ... and still on the table, since nothing ever left it")

assert f7_recoverable <= f7_raw, "recoverable cannot exceed the raw rate"
assert abs(f7_raw - f7_recoverable - f7_terminal_only) < 1e-9, "the three rates do not partition"
assert not (recoverable & terminal).any(), "a frame is both recoverable and terminal"
print()
print("occlusion probe ok")
