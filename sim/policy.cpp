#include "policy.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
    // Third input to each stream's seed_seq, and the only one that differs
    // between them. Arbitrary values, but frozen: changing either re-derives
    // every stream, and no shard on disk would report that it had happened.
    // A third stream gets tag 2 - never a recycled one.
    constexpr int kEpisodeStreamTag = 0;
    constexpr int kStepStreamTag = 1;

    static_assert(kEpisodeStreamTag != kStepStreamTag,
                  "equal stream tags give both engines identical state, so every "
                  "random action would be correlated with the episode coin flip");

    // The interval an actuator can actually be commanded over. An unlimited
    // actuator has no meaningful ctrlrange - MuJoCo never fills it in - so fall
    // back to the symmetric unit interval rather than reading whatever is there.
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

    // Zero torque, kept inside the actuator's range. Deliberately not the
    // midpoint: the two coincide only while the range is symmetric, and on an
    // asymmetric range the midpoint is a constant drive, which would leave the
    // neutral action anything but neutral.
    mjtNum NeutralControl(ControlRange range) {
        return mju_clip(0.0, range.low, range.high);
    }

    // Three-way sign with a dead zone around zero. The dead zone is the point -
    // without one the neutral direction is never commanded by the reach, because
    // a double is essentially never exactly zero.
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
    // fingertip cannot change height: including z would add an error the arm can
    // never work off - 0.015 m here, a block centre at 0.025 under a tip at 0.04
    // - and reach_done_dist would then be measured against a floor it never gets
    // below. Symptom if you leave z in: the reach never re-targets, every
    // scripted episode parks on its first block, and F-5's histogram sags.
    mjtNum PlanarDistance(const mjtNum a[3], const mjtNum b[3]) {
        const mjtNum dx = a[0] - b[0];
        const mjtNum dy = a[1] - b[1];
        return std::sqrt(dx*dx + dy*dy);
    }

    // One engine, seeded by diffusing three small integers into the 624 words
    // mt19937 actually needs. base_seed + shard_index alone would not do: shard
    // 0's second stream would be shard 1's first, and "a different shard index
    // diverges" would quietly stop being true.
    //
    // The seed_seq is a local, and that is the point of the function existing at
    // all rather than two blocks in the constructor. seed_seq::generate is pure,
    // so feeding one seq to both engines hands them identical state - a bug with
    // no symptom. A fresh local per call makes that unrepresentable instead of
    // something to remember. Do not hoist it out.
    //
    // Returns by value: ~2.5 KB copied twice per shard, against 50,000 frames,
    // buys both members being fully constructed in the init list with no window
    // where one exists in a default state waiting to be fixed.
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

    // Three *distinct* control values per actuator is what the encoding assumes.
    // A unidirectional actuator - ctrlrange [0, 1], say - collapses low onto
    // neutral, and the round-trip below would then fail with a message about
    // action indices instead of naming the actuator that cannot be commanded
    // three ways.
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
            // Exact comparison, not a tolerance: both sides come from the same
            // computation on the same doubles, so any difference at all is a
            // defect rather than rounding. It also catches a fourth value,
            // which a threshold classifier would silently bucket.
            if (ctrl[i] == range.low) {
                signs[i] = -1;
            } else if (ctrl[i] == range.high) {
                signs[i] = +1;
            } else if (ctrl[i] == NeutralControl(range)) {
                signs[i] = 0;
            } else {
                mju_error("action %d wrote ctrl[%d] = %g, which is none of the three "
                          "commandable values [%g, %g, %g]",
                          action, i, ctrl[i],
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
    // Validated here rather than where they are used, because every one of these
    // fails silently at the point of use: a zero hold redraws every step, a
    // probability outside [0, 1] makes bernoulli_distribution undefined, and a
    // non-positive arrival distance is a condition never met, so the reach parks
    // on its first target for the whole episode and nothing reports an error.
    if (params_.action_hold_steps < 1) {
        mju_error("action_hold_steps = %d; an action has to be held for at least "
                  "one step", params_.action_hold_steps);
    }
    if (!(params_.reach_digit_noise_prob >= 0.0 && params_.reach_digit_noise_prob <= 1.0)) {
        mju_error("reach_digit_noise_prob = %g is not a probability",
                  params_.reach_digit_noise_prob);
    }
    // Negative would invert the comparison in SignWithDeadband and make every
    // gain including exact zero read as a direction - the corner-only bug in
    // disguise. Zero is rejected too, matching config.py's POSITIVE_FLOAT_KEYS:
    // it behaves identically to 1e-9, so allowing it would buy one more rule and
    // no capability.
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

    // Discovered by name, not looked up as three fixed strings: AGENDA's F-6 and
    // F-7 iterations are both fixed by editing the scene XML, so the set of
    // blocks is the thing most likely to change out from under this code.
    for (int i = 0; i < model_->nbody; ++i) {
        const char* name = mj_id2name(model_, mjOBJ_BODY, i);
        if (name && std::strncmp(name, "block", 5) == 0) {
            block_body_ids_.push_back(i);
        }
    }
    if (block_body_ids_.empty()) {
        mju_error("no bodies named 'block*' - the scripted reach has nothing to aim at");
    }

    // The actuator-to-joint map, which is the piece that lets the action encoding
    // and the Jacobian talk to each other: the encoding is indexed by actuator,
    // the Jacobian by DOF, and actuator_trnid is the only thing connecting them.
    // Read rather than assumed - "motor i drives the joint named jointi" holds in
    // this XML and is not a rule.
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

    // Serial chain: the last actuator drives the joint nearest the fingertip.
    // True of scene/arm_blocks.xml and of any arm-shaped successor to it. A
    // branching mechanism would pick the wrong body here without complaining,
    // and would need a tip named in the XML instead.
    tip_body_id_ = model_->jnt_bodyid[actuated_joint_ids_.back()];
    if (model_->body_geomnum[tip_body_id_] != 1) {
        mju_error("body %d carries %d geoms; the fingertip offset is derived from "
                  "exactly one link geom",
                  tip_body_id_, model_->body_geomnum[tip_body_id_]);
    }
    // Far end of the link along its own +x - geom centre plus half-length, which
    // is how both links in arm_blocks.xml are laid out. Derived rather than
    // written down so that resizing a link in the XML moves the tip with it; a
    // hardcoded 0.15 would keep aiming at where the link used to end.
    const int tip_geom = model_->body_geomadr[tip_body_id_];
    tip_local_[0] = model_->geom_pos[3*tip_geom + 0] + model_->geom_size[3*tip_geom + 0];
    tip_local_[1] = model_->geom_pos[3*tip_geom + 1];
    tip_local_[2] = model_->geom_pos[3*tip_geom + 2];

    jacp_.resize(static_cast<std::size_t>(3 * model_->nv));
    signs_.resize(static_cast<std::size_t>(model_->nu));

    // The two stream draws come from throwaway engines built with the same
    // arguments as the members, not from the members themselves. Drawing from
    // episode_rng_ here would advance it, and that draw would be part of every
    // shard's sequence forever - a diagnostic that changes what it measures.
    //
    // Raw engine output, not a distribution: the raw word is the state being
    // claimed decorrelated, and distributions are the one part of <random> that
    // is not portable across standard libraries.
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
    // Rotate the body-local offset into world, then add the body origin. xmat is
    // the body's 3x3 rotation, row-major, and is derived state: valid only after
    // mj_forward or mj_step has run. Writing qpos does not update it.
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

    // A start pose per actuated joint, so 1500 episodes are 1500 openings rather
    // than one opening replayed 1500 times. Blocks stay where the XML puts them:
    // jittering a free joint means writing a quaternion, and an unnormalised one
    // corrupts the physics without ever failing a call.
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

    // Not optional. qpos is the input; xpos, xmat and the Jacobian are derived
    // and still hold the previous episode's values until something recomputes
    // them. step() reads all three on its very first call, so without this the
    // first action of every episode is aimed using the last episode's geometry -
    // which nothing reports, because every value involved is perfectly valid.
    mj_forward(model_, data);

    // Per-episode, not per-frame: a scripted reach needs consecutive steps to
    // complete, and coin-flipping per frame destroys the property the mix exists
    // for. See world_model_architecture.md, "Policy mixing is per-episode".
    std::bernoulli_distribution coin(0.5);
    is_scripted_ = coin(episode_rng_);

    std::uniform_int_distribution<std::size_t> pick(0, block_body_ids_.size() - 1);
    target_index_ = static_cast<int>(pick(episode_rng_));

    // Zero rather than a full hold, so step() draws on its first call. The
    // alternative is a separate "first step of the episode" branch in step(),
    // which is a second place to get the same thing wrong.
    hold_remaining_ = 0;
    held_action_ = 0;
}

int Policy::step(const mjData* data) {
    if (hold_remaining_ > 0) {
        --hold_remaining_;
        return held_action_;
    }

    // The noise used to live here, replacing the whole action. It now lives one
    // level down in scripted_action, one joint at a time - measured better at
    // every setting, because half a steering decision beats none.
    held_action_ = is_scripted_ ? scripted_action(data) : random_action();

    hold_remaining_ = params_.action_hold_steps - 1;
    return held_action_;
}

int Policy::random_action() {
    std::uniform_int_distribution<int> pick(0, action_count_ - 1);
    return pick(step_rng_);
}

void Policy::retarget() {
    // Nothing to switch to. Returning leaves the reach parked on the block it
    // just arrived at, which is the honest behaviour for a one-block scene.
    if (block_body_ids_.size() < 2) {
        return;
    }

    // Draw over the *other* blocks and shift past the current one. Re-picking the
    // current target would leave it instantly arrived again and park the arm for
    // the rest of the episode - the F-5 failure the reach_done_dist note in the
    // architecture doc warns about.
    //
    // step_rng_, not episode_rng_: this fires a trajectory-dependent number of
    // times, and drawing from the episode stream here would make the *next*
    // episode's start pose depend on action_hold_steps. See the stream comment in
    // policy.h - episode_rng_ is drawn from in begin_episode and nowhere else.
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

    // Switch targets before deciding, so the draw that lands on a block already
    // steers toward the next one instead of spending a hold going nowhere.
    if (PlanarDistance(tip, target_xpos(data)) < params_.reach_done_dist) {
        retarget();
    }

    mjtNum direction[3];
    mju_sub3(direction, target_xpos(data), tip);
    direction[2] = 0.0;

    // Unit length, so the deadband below stays a fixed metres-per-radian
    // threshold instead of one that scales with how far away the block happens to
    // be. mju_normalize3 returns the original length; a zero one means the tip is
    // exactly on the target in plan view, reachable when retarget had nothing to
    // switch to.
    if (mju_normalize3(direction) < mjMINVAL) {
        std::fill(signs_.begin(), signs_.end(), 0);
        return signs_to_action(model_, signs_.data());
    }

    // The position Jacobian of the fingertip: 3 rows of Cartesian axes by nv
    // columns of DOFs, answering "spin DOF j at unit speed and the tip moves this
    // way". Valid only because data's kinematics are current - mj_forward in
    // begin_episode, mj_step afterwards.
    mj_jac(model_, data, jacp_.data(), nullptr, tip, tip_body_id_);

    const std::size_t nv = static_cast<std::size_t>(model_->nv);
    for (std::size_t i = 0; i < signs_.size(); ++i) {
        const std::size_t column =
            static_cast<std::size_t>(model_->jnt_dofadr[actuated_joint_ids_[i]]);

        // Row-major 3 x nv: row r, DOF j is jacp[r*nv + j]. Indexing it as
        // [j*3 + r] compiles, runs, and reads a different matrix entirely - there
        // is no error, the arm just wanders. nv is 20 in this scene, not 2, which
        // is why the column has to come from jnt_dofadr rather than from the
        // actuator index.
        //
        // The dot product is the transpose-Jacobian rule: the Jacobian maps joint
        // speed to tip velocity, so its transpose applied to the desired direction
        // says how much each joint pushes the tip that way. Torque in the sign of
        // that quantity drives the tip toward the target.
        const mjtNum gain = jacp_[column] * direction[0]
                          + jacp_[nv + column] * direction[1]
                          + jacp_[2*nv + column] * direction[2];

        signs_[i] = SignWithDeadband(gain, params_.jacobian_deadband);

        // Per joint, and after the steering decision rather than instead of it.
        // A corrupted joint leaves the other still steering, so the arm keeps
        // roughly its heading while the histogram spreads - measured to dominate
        // whole-action replacement at every setting once the deadband is on.
        //
        // Drawn per decision, not per step: inside a held action a per-step coin
        // would change nothing, and the fraction of *decisions* corrupted is the
        // unit the frontier was measured in.
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
    // The closest the tip got to its target in each episode. The arrival count
    // alone cannot tell "the reach is broken" from "reach_done_dist is tighter
    // than the arm's own geometry allows" - the median separates them.
    std::vector<mjtNum> closest_per_episode;
    closest_per_episode.reserve(static_cast<std::size_t>(episodes));
    int reached = 0;

    mjData* data = mj_makeData(model);
    if (!data) {
        mju_error("mj_makeData failed in policy_dry_run");
    }

    // Two passes over a freshly constructed Policy each time. Same seed, same
    // shard: F-4 says the action sequences have to match exactly, and a fresh
    // Policy is the only honest way to ask - reusing one would test that the
    // stream continues, not that it restarts.
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

    // The two numbers F-5 names. min_share guards coverage - can a balanced Q-4
    // eval subset be drawn at all - and ratio guards balance. They fail
    // independently, so both are printed even when one passes.
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
    // Reported, not enforced. F-5 names 2,000 episodes and the startup smoke run
    // is far short of that; aborting on a verdict this run cannot legitimately
    // reach would be enforcing a threshold against the wrong sample size.
    std::printf(" -> %s%s\n",
                meets ? "meets F-5" : "MISSES F-5",
                episodes >= 2000 ? "" : " (indicative - F-5 wants >= 2,000 episodes)");
    // Roughly half is the scripted half arriving. Far below half means the reach
    // is not closing; far above means random episodes are drifting into blocks,
    // which makes reach_done_dist too generous to mean anything. Read the two
    // numbers together: a median well below the random half's typical miss means
    // the reach is closing and only the threshold is tight; a median near it
    // means the reach is not steering at all, and the row-major Jacobian indexing
    // and the sign of the projection are the two places to look before touching
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
