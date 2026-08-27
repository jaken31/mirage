#pragma once

#include <cstdint>
#include <vector>

#include <mujoco/mujoco.h>

#include "gl_context.h"

// ---------------------------------------------------------------------------
// Ground truth: the four things per frame that the pixels alone cannot tell you
// afterwards, read straight out of mjData and one extra render pass.
//
// This is the only file whose inputs disappear when the simulator is deleted -
// phase0_structural_plan.md, "What each file owns". Everything here is measured
// at generation time or not at all.
// ---------------------------------------------------------------------------

// One frame's worth, in the same order shard_writer's meta record lays it out.
// Vectors rather than fixed arrays because the block and joint counts are read
// from the model - the "no hardcoded shapes" constraint. The record itself is
// fixed-width; shard_writer is where the sizes are asserted against it.
//
// Reused across frames: pass the same instance back to read() every step and
// nothing allocates after the first call.
struct TruthFrame {
    // One per 1-DOF joint, in model order. The meta record's qpos[2].
    std::vector<mjtNum> joint_qpos;

    // Two per block - x then y of the body frame, in block order.
    std::vector<mjtNum> block_xy;

    // Pixels of each block visible in this frame. Counts, not a boolean:
    // a boolean bakes in a threshold and F-7 and Q-6 both want to re-derive
    // theirs without regenerating the dataset.
    std::vector<int> visible_px;

    // Bit i set when block i touches the arm. F-6 is the fraction of frames
    // with any bit set.
    std::uint8_t contact_mask;
};

// Reads ground truth for the current step. Owns the segmentation pass and the
// pixel buffer it reads back into; owns no files and no simulation state.
class Truth {
public:
    // model must outlive the Truth - the geom and joint ids cached below are
    // meaningful only for the model they were resolved from.
    //
    // Takes the GlContext rather than a bare mjrContext* and viewport so that a
    // caller cannot pair a context with someone else's viewport. Aborts via
    // mju_error on anything the meta record could not represent, the same way
    // GlContext's constructor does.
    Truth(const mjModel* model, const GlContext& gl);

    // Fills out for the current state of data. The scene must already have been
    // updated for this same data - mjv_updateScene - because this renders it
    // rather than rebuilding it: the segmentation pass has to see exactly the
    // geometry the RGB pass saw, or visible_px describes a frame that was never
    // stored.
    //
    // scene is mutable because the two segmentation flags are set and restored
    // around the render. Every other input is read-only.
    //
    // Leaves the offscreen framebuffer holding the segmentation image, so the
    // RGB frame must be read back *before* this call. Render RGB, read it,
    // then call this. Call it first and shard_writer stores id colours - which
    // look like a plausible flat-shaded image and would not fail any check
    // downstream of here.
    void read(const mjData* data, mjvScene* scene, TruthFrame* out);

    int block_count() const { return static_cast<int>(block_body_ids_.size()); }

    // Body id of block index i - the same index every TruthFrame field uses.
    // Public because "which block is this count about?" is otherwise
    // unanswerable outside this file; truth_dry_run needs it to teleport one.
    // Aborts via mju_error on an out-of-range index rather than reading past the
    // end of the vector.
    int block_body_id(int block) const;

private:
    // The segmentation render plus pixel histogram. Writes out->visible_px.
    void count_visible_pixels(mjvScene* scene, TruthFrame* out);

    const mjModel* model_;
    const mjrContext* con_;
    mjrRect viewport_;

    std::vector<int> block_body_ids_;

    // geom id -> block index, -1 for anything else. Dense over model->ngeom
    // because it is indexed by both the contact walk and the pixel histogram,
    // on the path of all 300k frames.
    std::vector<int> block_of_geom_;

    // qpos address of each 1-DOF joint, in model order.
    std::vector<int> joint_qposadr_;

    // Scratch, not state. Sized in the constructor, overwritten every frame.
    //
    // rgb_ is the segmentation readback - 3 bytes per pixel, discarded once
    // counted. block_of_segid_ is refilled from the scene on every read
    // because mjv_updateScene reassigns segids, so a table built once could
    // silently start pointing at the wrong geom.
    std::vector<unsigned char> rgb_;
    std::vector<int> block_of_segid_;
};

// Runs Truth against a throwaway mjData, scene and camera - no policy, no files.
// Three fatal checks and two printed verdicts.
//
// Fatal, via mju_error:
//   at rest every block reads a nonzero pixel count, which is what fails if the
//   id-colour channel order is not r + 256*g + 65536*b;
//   a block teleported below the table reads exactly zero;
//   restoring it brings the count back.
//
// Printed, not enforced - both are fixed by editing the scene XML rather than by
// changing code, so this is early evidence and not a gate:
//   F-6  fraction of frames with any arm-block contact, wanted > 5%
//   F-7  fraction of frames with any block fully occluded, wanted >= 3%
//
// Drives actuator 0 at full torque for steps steps, which sweeps the arm across
// all three blocks. That is a cruder trajectory than the real policy produces,
// so treat the two rates as a floor rather than an estimate.
void truth_dry_run(const mjModel* model, const GlContext& gl, int steps);
