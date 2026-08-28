#include "truth.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace {
    // Block bits available in the meta record's contact_mask. Seven, not eight:
    // shard_writer.h reserves the high bit for the scripted-episode flag.
    constexpr int kContactMaskBits = 7;

    // Largest value the meta record's u16 visible_px field holds, and therefore
    // the largest frame this file can count pixels over.
    constexpr int kVisiblePxMax = 65535;

    // mjv_makeScene's buffer size. Twelve geoms in arm_blocks.xml against a
    // one-off allocation - no reason to compute it.
    constexpr int kMaxSceneGeom = 1000;

    // Where the dry run parks a block to check that an invisible block reads
    // zero: below the table and outside the camera frustum, not merely behind
    // something.
    constexpr mjtNum kOutOfFrameZ = -5.0;

    // Full-torque control for one actuator, used only by the dry run's sweep.
    // Reads ctrlrange rather than writing 1.0 so that editing the XML's drive
    // strength changes the sweep instead of saturating against it - the same
    // rule policy.cpp's ActuatorRange follows.
    mjtNum FullDrive(const mjModel* model, mjtSize i) {
        if (!model->actuator_ctrllimited[i]) {
            return 1.0;
        }
        return model->actuator_ctrlrange[2*i + 1];
    }

    // The dry run's own scene refresh. pert is null because nothing is being
    // dragged; mjCAT_ALL because a geom left out of the scene carries segid -1
    // and would read as permanently occluded.
    void UpdateScene(const mjModel* model, mjData* data, const mjvOption* opt,
                     mjvCamera* camera, mjvScene* scene) {
        mjv_updateScene(model, data, opt, nullptr, camera, mjCAT_ALL, scene);
    }

    // qpos address of a block's free joint, checked rather than assumed - the
    // dry run writes into the z slot and a hinge has no such slot.
    int FreeJointQposAdr(const mjModel* model, int body) {
        const int joint = model->body_jntadr[body];
        if (joint < 0 || model->jnt_type[joint] != mjJNT_FREE) {
            mju_error("block body %d has no free joint, so the dry run cannot "
                      "teleport it out of frame", body);
        }
        return model->jnt_qposadr[joint];
    }
}

Truth::Truth(const mjModel* model, const GlContext& gl)
    : model_(model), con_(gl.context()), viewport_(gl.viewport()) {
    if (viewport_.width <= 0 || viewport_.height <= 0) {
        mju_error("viewport is %d x %d; nothing to count",
                  viewport_.width, viewport_.height);
    }
    if (viewport_.width * viewport_.height > kVisiblePxMax) {
        mju_error("a %d x %d frame has more pixels than the meta record's u16 "
                  "visible_px field can hold (%d)",
                  viewport_.width, viewport_.height, kVisiblePxMax);
    }

    // Discovered by name, exactly as policy.cpp does it and for the same reason:
    // F-6 and F-7 are both fixed by editing the scene XML, so the set of blocks
    // is the thing most likely to change out from under this code. The two loops
    // must agree on the prefix, or block index i names a different block in the
    // meta record than it does in the action stream.
    for (int i = 0; i < model_->nbody; ++i) {
        const char* name = mj_id2name(model_, mjOBJ_BODY, i);
        if (name && std::strncmp(name, "block", 5) == 0) {
            block_body_ids_.push_back(i);
        }
    }
    if (block_body_ids_.empty()) {
        mju_error("no bodies named 'block*' - nothing to measure visibility or "
                  "contact for");
    }
    if (block_count() > kContactMaskBits) {
        mju_error("%d blocks, but contact_mask has %d block bits (the high bit "
                  "is the scripted-episode flag)", block_count(), kContactMaskBits);
    }

    // geom -> block, and by omission geom -> arm. A geom on the world body is
    // scenery (the table); a geom on a moving body that is not a block is arm.
    //
    // ponytail: "not world, not block" is the whole arm test. It is right for
    // any scene where the arm is the only other moving thing. Add a second
    // movable object - a distractor, a second arm - and it reads as arm, which
    // inflates F-6. Derive the arm from the actuated chain if that day comes.
    block_of_geom_.assign(static_cast<std::size_t>(model_->ngeom), -1);
    for (int b = 0; b < block_count(); ++b) {
        const int body = block_body_ids_[static_cast<std::size_t>(b)];
        const int first = model_->body_geomadr[body];
        const int count = model_->body_geomnum[body];
        if (count <= 0) {
            mju_error("block body %d carries no geoms, so it can never be seen "
                      "or touched", body);
        }
        for (int g = first; g < first + count; ++g) {
            block_of_geom_[static_cast<std::size_t>(g)] = b;
        }
    }

    // Every 1-DOF joint, in model order - the two arm hinges here. Derived
    // rather than named so the meta record's qpos field follows the XML; the
    // blocks' free joints have 7 qpos each and are skipped by the type test.
    for (int j = 0; j < model_->njnt; ++j) {
        const int type = model_->jnt_type[j];
        if (type == mjJNT_HINGE || type == mjJNT_SLIDE) {
            joint_qposadr_.push_back(model_->jnt_qposadr[j]);
        }
    }
    if (joint_qposadr_.empty()) {
        mju_error("no hinge or slide joints - nothing to record as qpos");
    }

    rgb_.resize(static_cast<std::size_t>(3 * viewport_.width * viewport_.height));

    std::printf("Truth: %d blocks (body ids", block_count());
    for (const int id : block_body_ids_) {
        std::printf(" %d", id);
    }
    std::printf("), %zu single-DOF joints (qposadr", joint_qposadr_.size());
    for (const int adr : joint_qposadr_) {
        std::printf(" %d", adr);
    }
    std::printf("), segmentation buffer %d x %d\n",
                viewport_.width, viewport_.height);
}

int Truth::block_body_id(int block) const {
    if (block < 0 || block >= block_count()) {
        mju_error("block %d is out of range [0, %d)", block, block_count());
    }
    return block_body_ids_[static_cast<std::size_t>(block)];
}

void Truth::read(const mjData* data, mjvScene* scene, TruthFrame* out) {
    // Resizes are no-ops from the second frame on, which is why one TruthFrame
    // is meant to be reused for the whole run.
    out->joint_qpos.resize(joint_qposadr_.size());
    out->block_xy.resize(static_cast<std::size_t>(2 * block_count()));
    out->visible_px.resize(static_cast<std::size_t>(block_count()));

    for (std::size_t i = 0; i < joint_qposadr_.size(); ++i) {
        out->joint_qpos[i] = data->qpos[joint_qposadr_[i]];
    }

    for (int b = 0; b < block_count(); ++b) {
        const std::size_t block = static_cast<std::size_t>(b);
        const mjtNum* xpos = data->xpos + 3*block_body_ids_[block];
        out->block_xy[2*block] = xpos[0];
        out->block_xy[2*block + 1] = xpos[1];
    }

    // Contact. mjData.contact holds ncon entries whatever the solver did with
    // them, so exclude is checked: a contact in the gap or with no DOFs was
    // detected but generates no force, and F-6 is about the arm actually
    // touching a block.
    out->contact_mask = 0;
    for (int i = 0; i < data->ncon; ++i) {
        const mjContact& contact = data->contact[i];
        if (contact.exclude != 0) {
            continue;
        }
        // Either side of the pair can be the block, so both orderings are
        // tested rather than assuming MuJoCo orders the pair by geom id.
        for (int side = 0; side < 2; ++side) {
            const int block_geom = contact.geom[side];
            const int other_geom = contact.geom[1 - side];
            if (block_geom < 0 || other_geom < 0) {
                continue;  // flex contact - no geom on this side
            }
            const int block = block_of_geom_[static_cast<std::size_t>(block_geom)];
            const bool other_is_arm =
                block_of_geom_[static_cast<std::size_t>(other_geom)] < 0 &&
                model_->geom_bodyid[other_geom] != 0;
            if (block >= 0 && other_is_arm) {
                out->contact_mask |= static_cast<std::uint8_t>(1u << block);
            }
        }
    }

    count_visible_pixels(scene, out);
}

void Truth::count_visible_pixels(mjvScene* scene, TruthFrame* out) {
    // segid -> block, rebuilt every frame. mjv_updateScene reassigns segids, so
    // a table built once in the constructor would silently start naming a
    // different geom the first time the scene's geom order changed. The scene
    // holds ~12 geoms, so this is an assign() over a handful of ints.
    int max_segid = -1;
    for (int i = 0; i < scene->ngeom; ++i) {
        max_segid = std::max(max_segid, scene->geoms[i].segid);
    }
    block_of_segid_.assign(static_cast<std::size_t>(max_segid + 1), -1);
    for (int i = 0; i < scene->ngeom; ++i) {
        const mjvGeom& geom = scene->geoms[i];
        if (geom.objtype != mjOBJ_GEOM || geom.objid < 0 || geom.segid < 0) {
            continue;  // decor, or not shown
        }
        const int block = block_of_geom_[static_cast<std::size_t>(geom.objid)];
        if (block >= 0) {
            block_of_segid_[static_cast<std::size_t>(geom.segid)] = block;
        }
    }

    // The segmentation pass. Flags are saved and restored rather than assumed
    // clear, so a caller who renders segmentation for its own reasons does not
    // get the flags this function wanted left behind.
    const mjtByte saved_segment = scene->flags[mjRND_SEGMENT];
    const mjtByte saved_idcolor = scene->flags[mjRND_IDCOLOR];
    scene->flags[mjRND_SEGMENT] = 1;
    scene->flags[mjRND_IDCOLOR] = 1;
    mjr_render(viewport_, scene, con_);
    // No depth buffer: the id colour is the whole measurement. Skipping it saves
    // 2.4% of the frame budget - world_model_architecture.md, "Render path and
    // occlusion measurement (F-7)".
    mjr_readPixels(rgb_.data(), nullptr, viewport_, con_);
    scene->flags[mjRND_SEGMENT] = saved_segment;
    scene->flags[mjRND_IDCOLOR] = saved_idcolor;

    std::fill(out->visible_px.begin(), out->visible_px.end(), 0);
    const std::size_t pixels = rgb_.size() / 3;
    const std::size_t table_size = block_of_segid_.size();
    for (std::size_t p = 0; p < pixels; ++p) {
        // mjRND_IDCOLOR writes segid+1 across r, g, b low byte first, so an
        // unshown geom and the background both read 0 and the decode is exact
        // rather than a nearest-colour match. This is the one assumption in the
        // file that no header states; truth_dry_run's first check is what holds
        // it - a reversed channel order puts every decoded id outside the table,
        // and every count reads zero.
        const int encoded = rgb_[3*p] |
                            (rgb_[3*p + 1] << 8) |
                            (rgb_[3*p + 2] << 16);
        if (encoded == 0) {
            continue;
        }
        const std::size_t segid = static_cast<std::size_t>(encoded - 1);
        if (segid >= table_size) {
            continue;  // table, arm link, or decor
        }
        const int block = block_of_segid_[segid];
        if (block >= 0) {
            ++out->visible_px[static_cast<std::size_t>(block)];
        }
    }
}

void truth_dry_run(const mjModel* model, const GlContext& gl, int steps) {
    if (steps <= 0) {
        mju_error("truth_dry_run needs at least one step, got %d", steps);
    }

    mjData* data = mj_makeData(model);
    if (!data) {
        mju_error("mj_makeData failed in truth_dry_run");
    }

    mjvScene scene;
    mjv_defaultScene(&scene);
    mjv_makeScene(model, &scene, kMaxSceneGeom);

    mjvOption opt;
    mjv_defaultOption(&opt);

    // The fixed camera from the XML, by index rather than by name: the scene has
    // one camera and it is the capture viewpoint. A free camera here would
    // measure occlusion from a viewpoint no frame is ever rendered from.
    if (model->ncam < 1) {
        mju_error("model has no camera; occlusion is only defined from the "
                  "viewpoint the frames are captured from");
    }
    mjvCamera camera;
    mjv_defaultCamera(&camera);
    camera.type = mjCAMERA_FIXED;
    camera.fixedcamid = 0;

    Truth truth(model, gl);
    TruthFrame frame;
    const int blocks = truth.block_count();
    const int frame_pixels = gl.viewport().width * gl.viewport().height;

    // ---- Check 1: at rest, some block is visible. --------------------------
    // The id-colour decode check. Get the channel order wrong and every decoded
    // id lands outside the segid table, so every count reads zero.
    mj_resetData(model, data);
    mj_forward(model, data);
    UpdateScene(model, data, &opt, &camera, &scene);
    truth.read(data, &scene, &frame);

    std::printf("Truth dry run: at rest visible_px =");
    for (const int count : frame.visible_px) {
        std::printf(" %d", count);
    }
    std::printf(" of %d frame pixels\n", frame_pixels);

    int subject = 0;
    for (int b = 1; b < blocks; ++b) {
        if (frame.visible_px[static_cast<std::size_t>(b)] >
            frame.visible_px[static_cast<std::size_t>(subject)]) {
            subject = b;
        }
    }
    const std::size_t subject_slot = static_cast<std::size_t>(subject);
    const int open_count = frame.visible_px[subject_slot];
    if (open_count == 0) {
        mju_error("no block is visible at rest. Either the id-colour decode is "
                  "wrong - it assumes segid+1 across r,g,b low byte first - or "
                  "the camera cannot see the blocks");
    }

    // ---- Check 2: a block out of frame reads exactly zero. -----------------
    // Exactly zero, not merely smaller: a count that only shrinks would mean
    // pixels are being attributed by something other than the block's own segid.
    const int subject_body = truth.block_body_id(subject);
    const int subject_qposadr = FreeJointQposAdr(model, subject_body);
    const mjtNum saved_z = data->qpos[subject_qposadr + 2];
    data->qpos[subject_qposadr + 2] = kOutOfFrameZ;
    mj_forward(model, data);
    UpdateScene(model, data, &opt, &camera, &scene);
    truth.read(data, &scene, &frame);
    const int hidden_count = frame.visible_px[subject_slot];
    if (hidden_count != 0) {
        mju_error("block %d parked at z=%g still reads %d pixels; the pixel "
                  "histogram is not keyed on that block's segid",
                  subject, kOutOfFrameZ, hidden_count);
    }

    // ---- Check 3: putting it back brings the count back. -------------------
    // Without this, check 2 also passes when the counter is simply stuck at
    // zero after the first frame.
    data->qpos[subject_qposadr + 2] = saved_z;
    mj_forward(model, data);
    UpdateScene(model, data, &opt, &camera, &scene);
    truth.read(data, &scene, &frame);
    const int restored_count = frame.visible_px[subject_slot];
    if (restored_count != open_count) {
        mju_error("block %d read %d pixels in the open, %d after being moved "
                  "out of frame and back; the count is not a pure function of "
                  "the state", subject, open_count, restored_count);
    }
    std::printf("Truth dry run: block %d (body %d) %d px open -> %d hidden -> "
                "%d restored\n",
                subject, subject_body, open_count, hidden_count, restored_count);

    // ---- Verdicts: F-6 contact rate and F-7 occlusion rate. ---------------
    // Actuator 0 at full torque sweeps the arm across all three blocks. That is
    // a cruder trajectory than the real policy produces, and it biases both
    // rates in ways that do not cancel: the contact rate runs high because the
    // arm never stops, and the occlusion rate runs high because a block punted
    // off the table reads zero for every remaining frame. Neither number is an
    // estimate of the real one - they say "the measurement responds", not "the
    // scene passes". Neither is enforced here either: both are fixed by editing
    // the scene XML, and the enforcing check lives in mirage/validator.py over a
    // real shard.
    mj_resetData(model, data);
    for (mjtSize i = 0; i < model->nu; ++i) {
        data->ctrl[i] = (i == 0) ? FullDrive(model, i) : 0.0;
    }

    int contact_frames = 0;
    int occluded_frames = 0;
    std::vector<int> min_visible(static_cast<std::size_t>(blocks), frame_pixels);
    std::vector<int> max_visible(static_cast<std::size_t>(blocks), 0);
    std::vector<int> contact_frames_per_block(static_cast<std::size_t>(blocks), 0);

    for (int t = 0; t < steps; ++t) {
        mj_step(model, data);
        UpdateScene(model, data, &opt, &camera, &scene);
        truth.read(data, &scene, &frame);

        if (frame.contact_mask != 0) {
            ++contact_frames;
        }
        bool any_hidden = false;
        for (int b = 0; b < blocks; ++b) {
            const std::size_t slot = static_cast<std::size_t>(b);
            const int count = frame.visible_px[slot];
            min_visible[slot] = std::min(min_visible[slot], count);
            max_visible[slot] = std::max(max_visible[slot], count);
            if (count == 0) {
                any_hidden = true;
            }
            if ((frame.contact_mask & static_cast<std::uint8_t>(1u << b)) != 0) {
                ++contact_frames_per_block[slot];
            }
        }
        if (any_hidden) {
            ++occluded_frames;
        }
    }

    const double contact_rate = 100.0 * contact_frames / steps;
    const double occlusion_rate = 100.0 * occluded_frames / steps;
    std::printf("Truth dry run: %d steps, joint 0 driven at full torque\n", steps);
    for (int b = 0; b < blocks; ++b) {
        const std::size_t slot = static_cast<std::size_t>(b);
        std::printf("  block %d: visible_px %d..%d, contact in %d frames\n",
                    b, min_visible[slot], max_visible[slot],
                    contact_frames_per_block[slot]);
    }
    std::printf("  F-6 arm-block contact: %.2f%% of frames (want > 5%%) - %s\n",
                contact_rate, contact_rate > 5.0 ? "PASS" : "below threshold");
    std::printf("  F-7 any block fully occluded: %.2f%% of frames "
                "(want >= 3%%) - %s\n",
                occlusion_rate,
                occlusion_rate >= 3.0 ? "PASS" : "below threshold");
    std::printf("  Indicative only, and biased high both ways: one sweep "
                "direction rather than the real policy, and a block knocked off "
                "the table reads zero forever. The enforcing check is the "
                "validator over a shard.\n");

    // One GL error check for the whole run rather than per frame - mjr_getError
    // is a driver round trip and this loop runs it thousands of times otherwise.
    const int gl_error = mjr_getError();
    if (gl_error) {
        mju_error("OpenGL error 0x%x during the truth dry run", gl_error);
    }

    mjv_freeScene(&scene);
    mj_deleteData(data);
}
