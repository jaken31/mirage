#pragma once

#include <mujoco/mujoco.h>
#include <random>
#include <vector>
// ---------------------------------------------------------------------------
// The action encoding.
//
// An action is a torque direction for every actuated joint, packed into one
// integer. Each joint contributes one base-3 digit:
//
//     action = sum over joints i of (sign_i + 1) * 3^i,  sign_i in {-1, 0, +1}
//
// Joint 0 is the *least* significant digit, and decoding digit i gives
// sign_i = digit - 1. With the two hinges in scene/arm_blocks.xml that is
// 3^2 = 9 actions, numbered 0..8.
//
// Three other places depend on this exact layout without including this
// header: shard_writer stores the index as one byte, mirage/data.py reads it
// back, and the validator's action-following check compares each joint's
// actual direction of motion, sign(theta_t+1 - theta_t), with the commanded
// sign. Reorder the digits and every shard on disk is silently wrong: nothing
// fails to read, the numbers just stop meaning what they say. Change all
// three together with this, or not at all.
// ---------------------------------------------------------------------------

// Directions per joint: -1, 0, +1. Not tunable: the one-byte action field and
// the validator's direction check both assume exactly three levels.
constexpr int kActionLevels = 3;

// Largest value the one-byte action field can hold. The action count is
// kActionLevels^nu, so this limits the model to at most 5 actuators.
constexpr int kActionByteMax = 255;

// kActionLevels^nu, computed from the model rather than hardcoded (the design
// rule is no hardcoded shapes, and the action count is named in it). Also
// enforces nu <= 5: aborts via mju_error rather than return a count the meta
// record cannot store.
int action_count(const mjModel* model);

// Decodes action into ctrl, which must hold model->nu values (pass
// mjData.ctrl). Writes every entry on every call, because MuJoCo keeps ctrl
// across mj_step: a zero must be written, not skipped, or the previous step's
// torque stays on.
//
// The two driven directions map to the low and high ends of the actuator's
// control range, not literal -1/+1, so editing the range in the XML changes
// the drive strength instead of silently clipping against it. Neutral is zero
// torque clipped into the range. Deliberately not the range's midpoint, which
// only equals zero while the range is symmetric.
void action_to_control(const mjModel* model, int action, mjtNum* ctrl);

// Packs one direction per joint into an action index. signs must hold model->nu
// values, each of them -1, 0 or +1.
int signs_to_action(const mjModel* model, const int* signs);

// Proves the action encoding round-trips: every index decodes to controls that
// encode back to the same index. Aborts via mju_error on failure, like
// GlContext's constructor, so there is no return value to check. Called from
// main at startup.
//
// The other determinism check, that the same shard index replays the same
// action sequence, is in policy_dry_run.
void policy_self_check(const mjModel* model);

// The sim.* settings the policy reads, named exactly as in
// mirage/configs/base.json. main parses the config and passes them in, so
// config handling can change without touching this class.
//
// Two settings shape how evenly the actions are spread: jacobian_deadband and
// reach_digit_noise_prob. They are not interchangeable. The deadband reacts to
// the arm's state and costs almost nothing; the noise is random and makes the
// reach worse. Tune the deadband first. The measured trade-off curve for both
// is in docs/world_model_architecture.md.
struct PolicyParams {
  int action_hold_steps;

  // Per *joint*, not per action: the chance that one commanded sign is
  // replaced by a random pick from {-1, 0, +1}. Replacing the whole action was
  // measured to be worse at every setting. Corrupting one joint leaves the
  // other still steering, so the arm stays roughly on course while the action
  // mix spreads out.
  double reach_digit_noise_prob;

  // Metres of fingertip travel per radian of joint rotation below which that
  // joint gets zero torque instead of a direction. Without it the neutral
  // direction is never chosen: the reach commands sign(gain), a double is
  // never exactly zero, so the scripted policy would only ever emit the four
  // corner actions.
  mjtNum jacobian_deadband;

  mjtNum reach_done_dist;
};

// Picks the action for each step. Its state lives at three timescales (shard,
// episode, step) and the members below are grouped that way, so "did I forget
// to reset this in begin_episode?" is answered by reading one block.
class Policy {
public:
  // base_seed and shard_index are separate because combining them is the
  // determinism rule: the same shard index replays exactly, different ones
  // diverge. Doing it here means no caller can seed two shards the same.
  //
  // model must outlive the Policy. The cached ids below only mean something
  // for that model, so the pointer is stored rather than passed per call; a
  // caller cannot hand in a different model and have these ids silently index
  // into it.
  Policy(const mjModel* model, int base_seed, int shard_index, PolicyParams params);

  // Starts an episode: resets data, writes a random start pose, then flips the
  // 50/50 coin that picks random or scripted for the whole episode. Chosen per
  // episode, not per frame, because a scripted reach needs many consecutive
  // steps to finish.
  //
  // Takes a writable mjData because the start pose goes into qpos. The only
  // method here that changes the simulation instead of reading it.
  //
  // Sets every member in the episode and step groups below. The grouping
  // exists to make that easy to check.
  void begin_episode(mjData* data);

  // The action index for this step, in [0, action_count(model)). Reads the
  // fingertip and block positions for the scripted reach but never writes
  // mjData, so the policy cannot change the sim behind the caller's back. Turn
  // the result into controls with action_to_control.
  //
  // Each call advances the hold counter and may draw a random number, so call
  // it exactly once per step, before mj_step.
  int step(const mjData* data);

  // Fingertip-to-target distance in the plane, in metres. Public because the
  // action counts alone cannot show whether the reach is actually closing in:
  // a sign error in the Jacobian math drives the arm away from the block while
  // producing a perfectly healthy-looking action mix. policy_dry_run watches
  // this; the generation loop should log it per episode.
  //
  // Only meaningful in scripted episodes. In a random episode it still
  // reports the distance to the target picked at the start, which nothing is
  // steering toward.
  mjtNum target_distance(const mjData* data) const;

  // Whether this episode is the scripted or random half of the mix. Set by
  // begin_episode, constant until the next one. Public because the meta record
  // stores it; without it, questions like "does most contact come from
  // scripted episodes?" could only be guessed at.
  bool is_scripted() const { return is_scripted_; }

private:
  int random_action();
  int scripted_action(const mjData* data);
  void retarget();
  void fingertip(const mjData* data, mjtNum out[3]) const;
  const mjtNum* target_xpos(const mjData* data) const;

  const mjModel* model_;

  // Shard lifetime: fixed from construction to the shard's last frame.
  //
  // Two random streams, not one. The episode stream draws the coin flip, the
  // start pose and the first target. The step stream draws random actions,
  // the noise substitutions and every later target. The split makes sweeps of
  // action_hold_steps fair comparisons: change the hold and only the step
  // stream changes, so every episode still starts in the same place.
  //
  // This only holds if episode_rng_ is used in exactly one place,
  // begin_episode, with a fixed number of draws. Draw from it mid-episode (on
  // a re-target, say, which happens a trajectory-dependent number of times)
  // and the *next* episode's start pose starts depending on
  // action_hold_steps. Grep for episode_rng_; one function should match.
  //
  // How they are seeded is part of the determinism contract, like the digit
  // layout above: swap the two stream tags in policy.cpp and every shard on
  // disk replays differently, with no error anywhere.
  std::mt19937 episode_rng_;
  std::mt19937 step_rng_;

  PolicyParams params_;

  // kActionLevels^nu, computed once. action_count() loops over the actuators
  // on every call, and this is used on all 300k frames.
  int action_count_;

  std::vector<int> block_body_ids_;

  // The joint each actuator drives, in actuator order. So index i is both
  // digit i of the action and the joint whose Jacobian column decides that
  // digit. Read from actuator_trnid, not looked up by name: the action is
  // indexed by actuator and the Jacobian by DOF, and this is the only link
  // between the two.
  std::vector<int> actuated_joint_ids_;

  // The fingertip: the far end of the last actuated link, as an offset in that
  // body's own frame. Computed from the link's geom rather than hardcoded, so
  // resizing the arm in the XML moves the tip too.
  int tip_body_id_;
  mjtNum tip_local_[3];

  // Scratch buffers, not state: sized in the constructor, overwritten on each
  // scripted step, never read across calls. Members only to keep a 3 x nv
  // allocation out of the per-frame loop.
  std::vector<mjtNum> jacp_;
  std::vector<int> signs_;

  // Episode lifetime: set by begin_episode.
  //
  // is_scripted_ is fixed for the whole episode; that is the point of mixing
  // per episode. target_index_ is not: the reach picks a new one on arrival.
  // It indexes block_body_ids_, not the model's body array.
  bool is_scripted_;
  int target_index_;

  // Step lifetime: the held action, and how many more steps it has to run.
  //
  // Each action is held for action_hold_steps because a joint needs some time
  // to actually start moving the commanded way. The hold is set so the motion
  // matches the command often enough for the action-following check.
  // hold_remaining_ == 0 means "draw a new action on the next call", which is
  // why begin_episode zeroes it instead of filling it.
  int hold_remaining_;
  int held_action_;
};

// Runs the policy on a throwaway mjData (no rendering, no files) and reports
// three things that show begin_episode and step work:
//
//   determinism  two passes with the same seed give identical actions
//   balance      every action gets >= 5% of frames, and the most common is
//                at most 2.5x the least common
//   reach rate   about half of episodes reach their target, meaning the
//                scripted half arrives and the coin flip is fair
//
// Aborts via mju_error if the two passes differ. The balance result is
// printed, not enforced, and marked indicative below 2,000 episodes, the
// sample size the requirement names. Allocates and frees its own mjData.
void policy_dry_run(const mjModel* model, PolicyParams params, int base_seed,
                    int shard_index, int episodes, int steps_per_episode);
