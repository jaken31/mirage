#pragma once

#include <cstdint>
#include <vector>

#include <mujoco/mujoco.h>

#include "gl_context.h"

// ---------------------------------------------------------------------------
// Ground truth: the per-frame facts that cannot be recovered from the pixels
// later. Read directly from mjData, plus one extra render pass.
//
// Measure it while generating or never: once the simulator is gone, these
// inputs are gone too.
// ---------------------------------------------------------------------------

// One frame of ground truth, in the same field order as shard_writer's meta
// record. Vectors rather than fixed arrays because the block and joint counts
// come from the model, not from constants. The record on disk is still
// fixed-width; shard_writer checks these sizes against it.
//
// Reuse one instance: pass it back to read() every step and nothing allocates
// after the first call.
struct TruthFrame {
    // One angle per 1-DOF joint, in model order. The meta record's qpos field.
    std::vector<mjtNum> joint_qpos;

    // Two per block: x then y of the block's body frame, in block order.
    std::vector<mjtNum> block_xy;

    // How many pixels of each block are visible in this frame. Stored as counts
    // rather than a yes/no so later checks can pick their own "visible"
    // threshold without regenerating the dataset.
    std::vector<int> visible_px;

    // Bit i is set when block i touches the arm. At most seven blocks (the
    // constructor enforces it) because shard_writer.h reserves bit 7 of the
    // stored byte.
    //
    // This field holds ONLY block bits. The stored byte of the same name also
    // carries a "scripted episode" flag in its high bit, but ShardWriter::append
    // adds that, never this struct. Truth does not know which policy is running,
    // and a field with two meanings is how the contact rate would silently read
    // as "50% plus real contacts".
    std::uint8_t contact_mask;
};

// Reads ground truth for the current step. Owns the segmentation render pass
// and its pixel buffer; owns no files and no simulation state.
class Truth {
public:
    // model must outlive the Truth: the geom and joint ids cached below only
    // mean something for the model they came from.
    //
    // Takes the whole GlContext, not a bare mjrContext* plus viewport, so a
    // caller cannot pair one context with another's viewport. Aborts via
    // mju_error on anything the meta record cannot store, like GlContext does.
    Truth(const mjModel* model, const GlContext& gl);

    // Fills out for the current state of data. The caller must already have
    // updated the scene for this same data (mjv_updateScene). This renders that
    // scene rather than rebuilding it, so the segmentation pass sees exactly the
    // geometry the RGB pass saw; otherwise visible_px would describe a frame
    // that was never stored.
    //
    // scene is non-const only because two render flags are set and then
    // restored. Every other input is read-only.
    //
    // Order matters: this leaves the segmentation image in the offscreen
    // framebuffer, so read back the RGB frame *before* calling it. Get it wrong
    // and shard_writer stores the id-colour image instead, which looks like a
    // plausible flat-shaded frame and fails no later check.
    void read(const mjData* data, mjvScene* scene, TruthFrame* out);

    int block_count() const { return static_cast<int>(block_body_ids_.size()); }

    // How many joint angles a TruthFrame carries. Available before the first
    // read(), which is when shard_writer needs it to size the meta record (the
    // vectors in TruthFrame stay empty until then).
    int joint_count() const { return static_cast<int>(joint_qposadr_.size()); }

    // Body id of block index i, the index every TruthFrame field uses. Public
    // so code outside this file can tell which block a count belongs to;
    // truth_dry_run needs it to move one. Aborts via mju_error on an
    // out-of-range index rather than reading past the vector.
    int block_body_id(int block) const;

private:
    // The segmentation render plus the per-block pixel count. Writes
    // out->visible_px.
    void count_visible_pixels(mjvScene* scene, TruthFrame* out);

    const mjModel* model_;
    const mjrContext* con_;
    mjrRect viewport_;

    std::vector<int> block_body_ids_;

    // geom id -> block index, or -1 for any other geom. A flat array over every
    // geom because both the contact loop and the pixel count look it up, on
    // every one of the 300k frames.
    std::vector<int> block_of_geom_;

    // qpos address of each 1-DOF joint, in model order.
    std::vector<int> joint_qposadr_;

    // Scratch buffers, not state. Sized in the constructor, overwritten every
    // frame.
    //
    // rgb_ receives the segmentation image, 3 bytes per pixel, and is thrown
    // away once counted. block_of_segid_ is rebuilt on every read because
    // mjv_updateScene can reassign segmentation ids, so a table built once
    // could quietly start pointing at the wrong geom.
    std::vector<unsigned char> rgb_;
    std::vector<int> block_of_segid_;
};

// Runs Truth on a throwaway mjData, scene and camera: no policy, no files.
// Three fatal checks and two printed rates.
//
// Fatal, via mju_error:
//   at rest, the most visible block has a nonzero pixel count (this is what
//   fails if the id colour is not decoded as r + 256*g + 65536*b);
//   a block moved below the table reads exactly zero;
//   moving it back restores the original count.
//
// Printed, not enforced. Both are fixed by editing the scene XML, not code, so
// this is an early signal rather than a gate:
//   frames with any arm-block contact, target above 5%
//   frames with any block fully hidden, target at least 3%
//
// Drives actuator 0 at full torque for `steps` steps, which sweeps the arm
// across all three blocks. That is much cruder than the real policy, so read
// both rates as rough signs of life, not estimates.
void truth_dry_run(const mjModel* model, const GlContext& gl, int steps);
