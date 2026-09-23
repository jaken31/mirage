#include "policy.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
    // Third seed input for each random stream, and the only one that differs
    // between them. The values are arbitrary but frozen: changing either
    // changes every stream, and no shard on disk would show it. A third stream
    // gets tag 2, never a reused one.
    constexpr int kEpisodeStreamTag = 0;
    constexpr int kStepStreamTag = 1;

    static_assert(kEpisodeStreamTag != kStepStreamTag,
                  "equal stream tags give both engines identical state, so every "
                  "random action would be correlated with the episode coin flip");

    // The control interval an actuator accepts. An unlimited actuator has no
    // meaningful ctrlrange (MuJoCo leaves it unset), so use [-1, 1] instead of
    // reading whatever is there.
    struct ControlRange {
        mjtNum low;
        mjtNum high;
    };

    ControlRange ActuatorRange(const mjModel* model, mjtSize i) {
        if (!model->actuator_ctrllimited[i]) {
            return {-1.0, 1.0};
        }
        return {model->actuator_ctrlrange[2*i], model->actuator_ctrlrange[2*i + 1]};
    }

    // Zero torque, clipped into the actuator's range. Deliberately not the
    // midpoint: on an asymmetric range the midpoint is a constant push, so the
    // "neutral" action would not be neutral.
    mjtNum NeutralControl(ControlRange range) {
        return mju_clip(0.0, range.low, range.high);
    }

    // Three-way sign with a dead zone around zero. The dead zone is the point:
    // without it the reach never commands the neutral direction, because a
    // double is essentially never exactly zero.
    int SignWithDeadband(mjtNum value, mjtNum deadband) {
        if (value > deadband) {
            return +1;
        }
        if (value < -deadband) {
            return -1;
        }
        return 0;
    }

    // Distance in the table plane only. Both hinges turn about z, so the
    // fingertip's height never changes. Including z would add a gap the arm can
    // never close (0.015 m here: block centre at 0.025, tip at 0.04), and the
    // distance might never drop below reach_done_dist. Symptom: the reach never
    // switches target, every scripted episode parks on its first block, and the
    // action mix gets lopsided.
    mjtNum PlanarDistance(const mjtNum a[3], const mjtNum b[3]) {
        const mjtNum dx = a[0] - b[0];
        const mjtNum dy = a[1] - b[1];
        return std::sqrt(dx*dx + dy*dy);
    }

    // One random engine, seeded by mixing three small integers into the 624
    // words mt19937 needs. Adding base_seed + shard_index would not do: shard
    // 0's second stream would equal shard 1's first, so different shards would
    // no longer be independent.
    //
    // The seed_seq is a local, and that is why this function exists.
    // seed_seq::generate always gives the same output, so feeding one seq to
    // both engines would give them identical state, a bug with no symptom. A
    // fresh local per call makes that impossible. Do not move it out.
    //
    // Returns by value: copying ~2.5 KB twice per shard is nothing, and it lets
    // both members be fully built in the initializer list.
    std::mt19937 SeededStream(int base_seed, int shard_index, int tag) {
        std::seed_seq seq{base_seed, shard_index, tag};
        return std::mt19937(seq);
    }
}


int action_count(const mjModel* model) {
    if (model->nu == 0) {
        mju_error("model has no actuators - nothing for the policy to command");
    }

    int count = 1;

    for (mjtSize i = 0; i < model->nu; ++i) {
        count *= kActionLevels;
        if (count > kActionByteMax) {
            mju_error("nu=%lld needs at least %d actions, over the %d that the meta record can store",
                      static_cast<long long>(model->nu), count, kActionByteMax);
        }
    }
    return count;
}

void action_to_control(const mjModel* model, int action, mjtNum* ctrl) {
    const int count = action_count(model);
    if (action < 0 || action >= count) {
        mju_error("action %d is out of range [0, %d)", action, count);
    }
    int remaining = action;
    for (mjtSize i = 0; i < model->nu; ++i) {
        const int digit = remaining % kActionLevels;
        remaining /= kActionLevels;
        const int direction = digit - 1;

        const ControlRange range = ActuatorRange(model, i);
        if (direction < 0) {
            ctrl[i] = range.low;
        } else if (direction > 0) {
            ctrl[i] = range.high;
        } else {
            ctrl[i] = NeutralControl(range);
        }
    }
}

int signs_to_action(const mjModel* model, const int* signs) {
    int action = 0;
    int weight = 1;

    for (mjtSize i = 0; i < model->nu; ++i) {
        if (signs[i] < -1 || signs[i] > 1) {
            mju_error("signs[%lld] = %d is not in [-1, 0, +1]",
                      static_cast<long long>(i), signs[i]);
        }
        action += (signs[i] + 1) * weight;
        weight *= kActionLevels;
    }
    return action;
}

void policy_self_check(const mjModel* model) {
    const int count = action_count(model);
    const int nu = static_cast<int>(model->nu);

    // The encoding needs three *distinct* control values per actuator. A
    // one-directional actuator (ctrlrange [0, 1], say) makes low equal neutral.
    // Check it here so the error names that actuator, rather than a confusing
    // round-trip failure below.
    for (int i = 0; i < nu; ++i) {
        const ControlRange range = ActuatorRange(model, i);
        const mjtNum neutral = NeutralControl(range);
        if (!(range.low < neutral && neutral < range.high)) {
            mju_error("actuator %d has ctrlrange [%g, %g], which cannot represent "
                      "three distinct directions around zero torque",
                      i, range.low, range.high);
        }
    }

    std::vector<mjtNum> ctrl(static_cast<std::size_t>(nu));
    std::vector<int> signs(static_cast<std::size_t>(nu));

    for (int action = 0; action < count; ++action) {
        action_to_control(model, action, ctrl.data());
        for (int i = 0; i < nu; ++i) {
            const ControlRange range = ActuatorRange(model, i);
            const std::size_t k = static_cast<std::size_t>(i);
            // Exact comparison, no tolerance: both sides come from the same
            // computation on the same doubles, so any difference is a bug, not
            // rounding. It also catches a stray fourth value that a tolerance
            // would silently round to one of the three.
            if (ctrl[k] == range.low) {
                signs[k] = -1;
            } else if (ctrl[k] == range.high) {
                signs[k] = +1;
            } else if (ctrl[k] == NeutralControl(range)) {
                signs[k] = 0;
            } else {
                mju_error("action %d wrote ctrl[%d] = %g, which is none of the three "
                          "commandable values [%g, %g, %g]",
                          action, i, ctrl[k],
                          range.low, NeutralControl(range), range.high);
            }
        }
        const int recovered = signs_to_action(model, signs.data());
        if (recovered != action) {
            mju_error("action %d round-trips to %d", action, recovered);
        }
    }
    std::printf("policy_self_check passed: %d actions round-trip\n", count);
}

Policy::Policy(const mjModel* model, int base_seed, int shard_index, PolicyParams params)
    : model_(model),
      episode_rng_(SeededStream(base_seed, shard_index, kEpisodeStreamTag)),
      step_rng_(SeededStream(base_seed, shard_index, kStepStreamTag)),
      params_(params),
      action_count_(action_count(model)),
      tip_body_id_(0),
      tip_local_{0.0, 0.0, 0.0},
      is_scripted_(false),
      target_index_(0),
      hold_remaining_(0),
      held_action_(0) {
    // Checked here, not where they are used, because each one fails silently
    // there: a zero hold redraws every step, a probability outside [0, 1] makes
    // bernoulli_distribution undefined, and a non-positive arrival distance is
    // never reached, so the reach parks on its first target with no error.
    if (params_.action_hold_steps < 1) {
        mju_error("action_hold_steps = %d; an action has to be held for at least "
                  "one step", params_.action_hold_steps);
    }
    if (!(params_.reach_digit_noise_prob >= 0.0 && params_.reach_digit_noise_prob <= 1.0)) {
        mju_error("reach_digit_noise_prob = %g is not a probability",
                  params_.reach_digit_noise_prob);
    }
    // A negative value would flip the comparison in SignWithDeadband, so every
    // gain (even exact zero) reads as a direction: the "only corner actions"
    // bug again. Zero is rejected too, as in config.py's POSITIVE_FLOAT_KEYS:
    // it behaves the same as 1e-9, so allowing it adds a rule and no ability.
    if (!(params_.jacobian_deadband > 0.0)) {
        mju_error("jacobian_deadband = %g m/rad is not positive; to disable the "
                  "dead zone use a small positive value, not 0",
                  params_.jacobian_deadband);
    }
    if (!(params_.reach_done_dist > 0.0)) {
        mju_error("reach_done_dist = %g m; a non-positive arrival distance is "
                  "never met, so the reach never re-targets",
                  params_.reach_done_dist);
    }

    // Find blocks by name prefix, not as three fixed names. Contact and
    // occlusion rates are tuned by editing the scene XML, so the set of blocks
    // is what is most likely to change under this code.
    for (int i = 0; i < model_->nbody; ++i) {
        const char* name = mj_id2name(model_, mjOBJ_BODY, i);
        if (name && std::strncmp(name, "block", 5) == 0) {
            block_body_ids_.push_back(i);
        }
    }
    if (block_body_ids_.empty()) {
        mju_error("no bodies named 'block*' - the scripted reach has nothing to aim at");
    }

    // Map each actuator to its joint. The action encoding is indexed by
    // actuator and the Jacobian by DOF, and actuator_trnid is the only link
    // between them. Read it rather than assume "motor i drives joint i", which
    // happens to be true in this XML but is not a rule.
    actuated_joint_ids_.reserve(static_cast<std::size_t>(model_->nu));
    for (mjtSize i = 0; i < model_->nu; ++i) {
        if (model_->actuator_trntype[i] != mjTRN_JOINT) {
            mju_error("actuator %lld drives something other than a joint; one "
                      "joint per actuator is what the base-3 digit assumes",
                      static_cast<long long>(i));
        }
        const int joint = model_->actuator_trnid[2*i];
        const int type = model_->jnt_type[joint];
        if (type != mjJNT_HINGE && type != mjJNT_SLIDE) {
            mju_error("actuator %lld drives a joint with more than one DOF; both "
                      "the base-3 digit and the Jacobian column assume exactly one",
                      static_cast<long long>(i));
        }
        if (!model_->jnt_limited[joint] && type != mjJNT_HINGE) {
            mju_error("actuator %lld drives an unlimited slide joint, which has "
                      "no bounded range to draw a start pose from",
                      static_cast<long long>(i));
        }
        actuated_joint_ids_.push_back(joint);
    }

    // Assumes a serial chain: the last actuator drives the joint nearest the
    // fingertip. True for this arm and any similar one. A branching mechanism
    // would silently pick the wrong body and would need the tip named in the
    // XML instead.
    tip_body_id_ = model_->jnt_bodyid[actuated_joint_ids_.back()];
    if (model_->body_geomnum[tip_body_id_] != 1) {
        mju_error("body %d carries %d geoms; the fingertip offset is derived from "
                  "exactly one link geom",
                  tip_body_id_, model_->body_geomnum[tip_body_id_]);
    }
    // Far end of the link along its own +x: geom centre plus half-length, which
    // is how both links in arm_blocks.xml are laid out. Computed so resizing a
    // link in the XML moves the tip; a hardcoded 0.15 would keep aiming at
    // where the link used to end.
    const int tip_geom = model_->body_geomadr[tip_body_id_];
    tip_local_[0] = model_->geom_pos[3*tip_geom + 0] + model_->geom_size[3*tip_geom + 0];
    tip_local_[1] = model_->geom_pos[3*tip_geom + 1];
    tip_local_[2] = model_->geom_pos[3*tip_geom + 2];

    jacp_.resize(static_cast<std::size_t>(3 * model_->nv));
    signs_.resize(static_cast<std::size_t>(model_->nu));

    // The two printed draws come from throwaway engines seeded like the
    // members, not from the members. Drawing from episode_rng_ here would
    // advance it and change every shard's sequence: a diagnostic that changes
    // what it measures.
    //
    // Raw engine output, not a distribution: the raw word is what shows the two
    // streams differ, and distributions are the one part of <random> that can
    // differ between standard libraries.
    std::printf("Policy: base_seed %d, shard %d, first draws episode=%lu step=%lu\n",
                base_seed, shard_index,
                static_cast<unsigned long>(SeededStream(base_seed, shard_index, kEpisodeStreamTag)()),
                static_cast<unsigned long>(SeededStream(base_seed, shard_index, kStepStreamTag)()));
    std::printf("Policy: %zu blocks (body ids", block_body_ids_.size());
    for (const int id : block_body_ids_) {
        std::printf(" %d", id);
    }
    std::printf("), %zu actuated joints (qposadr/dofadr", actuated_joint_ids_.size());
    for (const int joint : actuated_joint_ids_) {
        std::printf(" %d/%d", model_->jnt_qposadr[joint], model_->jnt_dofadr[joint]);
    }
    std::printf(")\n");
    std::printf("Policy: fingertip on body %d at body-local (%g %g %g), nv=%lld, "
                "%d actions\n",
                tip_body_id_, tip_local_[0], tip_local_[1], tip_local_[2],
                static_cast<long long>(model_->nv), action_count_);
}

const mjtNum* Policy::target_xpos(const mjData* data) const {
    return data->xpos + 3*block_body_ids_[static_cast<std::size_t>(target_index_)];
}

void Policy::fingertip(const mjData* data, mjtNum out[3]) const {
    // Rotate the body-local offset into world frame, then add the body origin.
    // xmat is the body's 3x3 row-major rotation. It is only valid after
    // mj_forward or mj_step; writing qpos does not update it.
    mjtNum offset[3];
    mju_mulMatVec3(offset, data->xmat + 9*tip_body_id_, tip_local_);
    mju_add3(out, data->xpos + 3*tip_body_id_, offset);
}

mjtNum Policy::target_distance(const mjData* data) const {
    mjtNum tip[3];
    fingertip(data, tip);
    return PlanarDistance(tip, target_xpos(data));
}

void Policy::begin_episode(mjData* data) {
    mj_resetData(model_, data);

    // A random start angle per actuated joint, so 1500 episodes start 1500
    // different ways instead of replaying one start. Blocks stay where the XML
    // puts them: moving a free joint means writing a quaternion, and a
    // non-unit one corrupts the physics without any call failing.
    for (const int joint : actuated_joint_ids_) {
        mjtNum low = -mjPI;
        mjtNum high = mjPI;
        if (model_->jnt_limited[joint]) {
            low = model_->jnt_range[2*joint];
            high = model_->jnt_range[2*joint + 1];
        }
        std::uniform_real_distribution<mjtNum> angle(low, high);
        data->qpos[model_->jnt_qposadr[joint]] = angle(episode_rng_);
    }

    // Required. qpos is the input; xpos, xmat and the Jacobian are computed
    // from it and still hold last episode's values until recomputed. step()
    // reads all three on its first call, so without this every episode's first
    // action would aim using the previous episode's geometry, and nothing would
    // notice because every value is still valid-looking.
    mj_forward(model_, data);

    // Per episode, not per frame: a scripted reach needs many consecutive steps
    // to finish, and a per-frame coin flip would break every reach.
    std::bernoulli_distribution coin(0.5);
    is_scripted_ = coin(episode_rng_);

    std::uniform_int_distribution<std::size_t> pick(0, block_body_ids_.size() - 1);
    target_index_ = static_cast<int>(pick(episode_rng_));

    // Zero, not a full hold, so step() draws a fresh action on its first call.
    // The alternative, a special first-step branch in step(), is one more place
    // to get it wrong.
    hold_remaining_ = 0;
    held_action_ = 0;
}

int Policy::step(const mjData* data) {
    if (hold_remaining_ > 0) {
        --hold_remaining_;
        return held_action_;
    }

    // Noise is not applied here to the whole action. It is applied per joint
    // inside scripted_action, which measured better at every setting: half a
    // steering decision beats none.
    held_action_ = is_scripted_ ? scripted_action(data) : random_action();

    hold_remaining_ = params_.action_hold_steps - 1;
    return held_action_;
}

int Policy::random_action() {
    std::uniform_int_distribution<int> pick(0, action_count_ - 1);
    return pick(step_rng_);
}

void Policy::retarget() {
    // Nothing to switch to, so the reach stays on the block it reached. That
    // is the right behaviour for a one-block scene.
    if (block_body_ids_.size() < 2) {
        return;
    }

    // Pick among the *other* blocks, skipping past the current one. Picking the
    // current target again would count as arrived instantly and park the arm
    // for the rest of the episode, skewing the action mix.
    //
    // step_rng_, not episode_rng_: this runs a trajectory-dependent number of
    // times, and using the episode stream would make the *next* episode's
    // start pose depend on action_hold_steps. See the stream comment in
    // policy.h: episode_rng_ is used in begin_episode and nowhere else.
    std::uniform_int_distribution<std::size_t> pick(0, block_body_ids_.size() - 2);
    std::size_t choice = pick(step_rng_);
    if (choice >= static_cast<std::size_t>(target_index_)) {
        ++choice;
    }
    target_index_ = static_cast<int>(choice);
}

int Policy::scripted_action(const mjData* data) {
    mjtNum tip[3];
    fingertip(data, tip);

    // Switch targets before deciding, so the step that reaches a block already
    // steers toward the next one instead of wasting a hold.
    if (PlanarDistance(tip, target_xpos(data)) < params_.reach_done_dist) {
        retarget();
    }

    mjtNum direction[3];
    mju_sub3(direction, target_xpos(data), tip);
    direction[2] = 0.0;

    // Normalise to unit length so the deadband below is a fixed
    // metres-per-radian threshold, not one that scales with distance to the
    // block. mju_normalize3 returns the original length. Zero means the tip is
    // exactly over the target, which can happen when retarget had nothing to
    // switch to.
    if (mju_normalize3(direction) < mjMINVAL) {
        std::fill(signs_.begin(), signs_.end(), 0);
        return signs_to_action(model_, signs_.data());
    }

    // The fingertip's position Jacobian: 3 rows (x, y, z) by nv columns (one
    // per DOF). Column j says how the tip moves when DOF j turns at unit speed.
    // Valid only because the kinematics are current (mj_forward in
    // begin_episode, mj_step after that).
    mj_jac(model_, data, jacp_.data(), nullptr, tip, tip_body_id_);

    const std::size_t nv = static_cast<std::size_t>(model_->nv);
    for (std::size_t i = 0; i < signs_.size(); ++i) {
        const std::size_t column =
            static_cast<std::size_t>(model_->jnt_dofadr[actuated_joint_ids_[i]]);

        // Row-major 3 x nv: row r, DOF j is jacp[r*nv + j]. Indexing it as
        // [j*3 + r] compiles, runs, and reads the wrong numbers with no error;
        // the arm just wanders. nv is 20 in this scene (the blocks' free joints
        // count too), not 2, so the column must come from jnt_dofadr, not the
        // actuator index.
        //
        // This dot product says how much turning this joint moves the tip
        // toward the target (the transpose-Jacobian rule). Torque with that
        // sign drives the tip toward the target.
        const mjtNum gain = jacp_[column] * direction[0]
                          + jacp_[nv + column] * direction[1]
                          + jacp_[2*nv + column] * direction[2];

        signs_[i] = SignWithDeadband(gain, params_.jacobian_deadband);

        // Noise per joint, applied after the steering decision rather than
        // instead of it. A corrupted joint leaves the other still steering, so
        // the arm keeps roughly its heading while the action mix spreads. With
        // the deadband on, this measured better than replacing the whole
        // action at every setting.
        //
        // Drawn once per decision, not per step: within a held action a
        // per-step coin would do nothing, and the tuning was measured as the
        // fraction of *decisions* corrupted.
        if (params_.reach_digit_noise_prob > 0.0) {
            std::bernoulli_distribution corrupt(params_.reach_digit_noise_prob);
            if (corrupt(step_rng_)) {
                std::uniform_int_distribution<int> pick(-1, +1);
                signs_[i] = pick(step_rng_);
            }
        }
    }

    return signs_to_action(model_, signs_.data());
}

void policy_dry_run(const mjModel* model, PolicyParams params, int base_seed,
                    int shard_index, int episodes, int steps_per_episode) {
    const int count = action_count(model);

    std::vector<int> first_pass;
    first_pass.reserve(static_cast<std::size_t>(episodes) *
                       static_cast<std::size_t>(steps_per_episode));
    std::vector<long long> histogram(static_cast<std::size_t>(count), 0);
    // Closest the tip got to its target in each episode. The arrival count
    // alone cannot tell "the reach is broken" from "reach_done_dist is tighter
    // than the arm can manage"; the median of this tells them apart.
    std::vector<mjtNum> closest_per_episode;
    closest_per_episode.reserve(static_cast<std::size_t>(episodes));
    int reached = 0;

    mjData* data = mj_makeData(model);
    if (!data) {
        mju_error("mj_makeData failed in policy_dry_run");
    }

    // Two passes, each with a freshly built Policy and the same seed and shard.
    // The action sequences must match exactly. A fresh Policy is the only fair
    // test; reusing one would test that the stream continues, not that it
    // restarts identically.
    for (int pass = 0; pass < 2; ++pass) {
        Policy policy(model, base_seed, shard_index, params);
        std::size_t cursor = 0;

        for (int episode = 0; episode < episodes; ++episode) {
            policy.begin_episode(data);
            mjtNum closest = policy.target_distance(data);

            for (int t = 0; t < steps_per_episode; ++t) {
                const int action = policy.step(data);
                action_to_control(model, action, data->ctrl);
                mj_step(model, data);
                closest = std::min(closest, policy.target_distance(data));

                if (pass == 0) {
                    first_pass.push_back(action);
                    ++histogram[static_cast<std::size_t>(action)];
                } else {
                    if (first_pass[cursor] != action) {
                        mju_error("F-4: episode %d step %d gave action %d on the "
                                  "first pass and %d on the second, same seed",
                                  episode, t, first_pass[cursor], action);
                    }
                    ++cursor;
                }
            }

            if (pass == 0) {
                closest_per_episode.push_back(closest);
                if (closest < params.reach_done_dist) {
                    ++reached;
                }
            }
        }
    }

    mj_deleteData(data);

    long long total = 0;
    long long lowest = histogram[0];
    long long highest = histogram[0];
    for (const long long bin : histogram) {
        total += bin;
        lowest = std::min(lowest, bin);
        highest = std::max(highest, bin);
    }

    std::printf("policy_dry_run: %d episodes x %d steps, %lld actions, twice\n",
                episodes, steps_per_episode, total);
    std::printf("  F-4 determinism: both passes identical\n");
    std::printf("  F-5 histogram:");
    for (const long long bin : histogram) {
        std::printf(" %lld", bin);
    }
    std::printf("\n      flat is %lld per bin; observed %lld..%lld\n",
                total / count, lowest, highest);

    // The two balance numbers. min_share checks coverage (is every action
    // common enough to build a balanced evaluation subset) and ratio checks
    // evenness. They fail independently, so both are always printed.
    const double min_share = static_cast<double>(lowest) / static_cast<double>(total);
    const double ratio = lowest > 0
        ? static_cast<double>(highest) / static_cast<double>(lowest)
        : 0.0;
    const bool meets = (min_share >= 0.05) && (lowest > 0) && (ratio <= 2.5);
    std::printf("      min share %.2f%% (>= 5%%), ratio ", 100.0 * min_share);
    if (lowest > 0) {
        std::printf("%.2f (<= 2.5)", ratio);
    } else {
        std::printf("infinite - a bin is empty");
    }
    // Reported, not enforced. The requirement is judged over 2,000 episodes and
    // the startup smoke run is far shorter, so aborting here would apply the
    // threshold to the wrong sample size.
    std::printf(" -> %s%s\n",
                meets ? "meets F-5" : "MISSES F-5",
                episodes >= 2000 ? "" : " (indicative - F-5 wants >= 2,000 episodes)");
    // About half is expected: the scripted episodes arriving. Far below half
    // means the reach is not closing in. Far above means random episodes drift
    // into blocks, so reach_done_dist is too generous to mean anything. Read
    // both numbers together. A median well below a random episode's typical
    // miss means the reach works and only the threshold is tight. A median
    // near it means the reach is not steering at all: check the row-major
    // Jacobian indexing and the sign of the dot product before touching
    // reach_done_dist.
    const std::size_t middle = closest_per_episode.size() / 2;
    std::nth_element(closest_per_episode.begin(),
                     closest_per_episode.begin() + static_cast<std::ptrdiff_t>(middle),
                     closest_per_episode.end());
    std::printf("  reach: %d of %d episodes closed to within %g m (%.1f%%), "
                "~50%% expected; median closest %.3f m\n",
                reached, episodes, params.reach_done_dist,
                100.0 * static_cast<double>(reached) / static_cast<double>(episodes),
                closest_per_episode[middle]);
}
