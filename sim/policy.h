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
// Joint i is digit i, so joint 0 is the *least* significant digit, and decoding
// is digit i minus 1. With the two hinges in scene/arm_blocks.xml that is
// 3^2 = 9 actions, 0..8.
//
// Three other places assume this exact layout and none of them include this
// header: shard_writer's meta record stores the index as a u8, mirage/data.py
// reads it back, and the validator's Q-4 check compares
// sign(theta_t+1 - theta_t) against the commanded sign. Reorder the digits and
// every shard already on disk is silently wrong - no read fails, the numbers
// just stop meaning what they say. Change those three with this or not at all.
// ---------------------------------------------------------------------------

// Directions per joint: -1, 0, +1. Not a tunable - the u8 meta field and the
// Q-4 sign check are both written against exactly three levels.
constexpr int kActionLevels = 3;

// Largest value the meta record's u8 action field holds. Since the action count
// is kActionLevels^nu, this is what bounds the model at nu <= 5.
constexpr int kActionByteMax = 255;

// kActionLevels^nu, derived from the model rather than written down - the "no
// hardcoded shapes" constraint in world_model_architecture.md names the action
// count explicitly. This is also where nu <= 5 is enforced: aborts via
// mju_error rather than returning a count the meta record cannot store.
int action_count(const mjModel* model);

// Decodes action into ctrl, which must have room for model->nu values - pass
// mjData.ctrl. Writes every entry on every call, because MuJoCo carries ctrl
// across mj_step: the zero action has to be written, not skipped, or the
// previous step's torque stays applied.
//
// The two driven directions map onto actuator_ctrlrange's low and high ends
// rather than literal -1/+1, so editing ctrlrange in the XML changes the drive
// strength instead of silently saturating against it. The neutral direction is
// zero torque clipped into that range - deliberately not the range's midpoint,
// which is the same number only while the range stays symmetric.
void action_to_control(const mjModel* model, int action, mjtNum* ctrl);

// Packs one direction per joint into an action index. signs must hold model->nu
// values, each of them -1, 0 or +1.
int signs_to_action(const mjModel* model, const int* signs);

// Proves the action encoding round-trips: every index decodes to controls that
// re-encode to the same index. Aborts via mju_error on any failure, the same way
// GlContext's constructor does, so there is no return value to check. Called
// from main at startup.
//
// The other half of F-4 - the same shard index replaying the same action
// sequence - lands here alongside Policy, since there is no generator to replay
// yet.
void policy_self_check(const mjModel* model);

// The sim.* knobs the policy reads, named to match mirage/configs/base.json
// exactly. Passed in rather than read here because config parsing lands with
// shard_writer - when it does, main changes and this class does not.
//
// The two that shape F-5's histogram are jacobian_deadband and
// reach_digit_noise_prob, and they are not interchangeable. The deadband is
// state-dependent and nearly free; the noise is blind and paid for in reach
// quality. Reach for the deadband first. The measured frontier for both is in
// docs/world_model_architecture.md, "F-5's threshold is the knee of a measured
// curve".
struct PolicyParams {
  int action_hold_steps;

  // Per *joint*, not per action: the probability that one commanded sign is
  // replaced by a uniform draw from {-1, 0, +1}. Deliberately not whole-action
  // replacement, which the frontier showed is dominated at every setting - a
  // corrupted joint leaves the other one still steering, so the arm stays
  // roughly on course while the histogram spreads.
  double reach_digit_noise_prob;

  // Metres of fingertip travel per radian of joint rotation, below which that
  // joint is commanded to zero torque rather than a direction. This is what
  // makes the neutral digit reachable at all: the reach commands sign(gain), and
  // a double is never exactly zero, so with no dead zone the scripted half can
  // only ever emit the four corner actions.
  mjtNum jacobian_deadband;

  mjtNum reach_done_dist;
};

// Picks the action for each step. Its state lives at three timescales - shard,
// episode, step - and the member blocks below are grouped that way, so "did I
// forget to reset this in begin_episode?" is answerable by reading one block
// rather than reasoning about the whole class.
class Policy {
public:
  // base_seed and shard_index stay separate because the derivation between them
  // is the determinism rule: same shard index replays, different ones diverge.
  // Keeping it inside the constructor means no caller can seed two shards alike.
  //
  // model must outlive the Policy. The cached ids below are meaningful only for
  // the model they were resolved from, so the pointer is held rather than passed
  // back in per call - a caller then cannot hand a method a different model and
  // have it silently index that one with these ids.
  Policy(const mjModel* model, int base_seed, int shard_index, PolicyParams params);

  // Starts an episode: resets data, draws a start pose into it, then flips the
  // 50/50 coin that fixes random-vs-scripted for the whole episode. Per-episode
  // and not per-frame because a scripted reach needs consecutive steps to
  // complete - world_model_architecture.md, "Policy mixing is per-episode".
  //
  // Takes a writable mjData because the start pose is written into qpos. The
  // only method here that touches the simulation rather than reading it.
  //
  // Every member in the episode and step blocks below is set here. That is the
  // invariant the block grouping exists to make checkable.
  void begin_episode(mjData* data);

  // The action index for the current step, in [0, action_count(model)). Reads
  // mjData for the scripted reach - fingertip and block positions - and never
  // writes it, so a caller cannot have the policy perturb the sim behind its
  // back. Map the result onto mjData.ctrl with action_to_control.
  //
  // Not idempotent: every call advances the hold counter and may draw from the
  // step stream. Call it exactly once per step, and before mj_step.
  int step(const mjData* data);

  // Planar fingertip-to-target distance, in metres. Public because "is the
  // scripted reach actually closing on anything?" is not answerable from the
  // action indices alone - a sign error in the Jacobian projection drives the
  // arm away from the block and produces a perfectly healthy action histogram
  // while doing it. policy_dry_run watches this; the generation loop should log
  // it per episode.
  //
  // Meaningful in scripted episodes. In a random episode it still reports the
  // distance to the target drawn at episode start, which nothing is steering
  // toward.
  mjtNum target_distance(const mjData* data) const;

  // Which half of the 50/50 mix this episode is. Fixed by begin_episode and
  // constant until the next one. Public because the meta record stores it:
  // without it, every per-half question about the dataset - does F-6's contact
  // come from the scripted half? does Q-4 fail on random episodes? - is
  // guesswork from a histogram.
  bool is_scripted() const { return is_scripted_; }

private:
  int random_action();
  int scripted_action(const mjData* data);
  void retarget();
  void fingertip(const mjData* data, mjtNum out[3]) const;
  const mjtNum* target_xpos(const mjData* data) const;

  const mjModel* model_;

  // Shard lifetime: fixed from construction until the last frame of the shard.
  //
  // Two streams, not one. Episode-scale draws the coin flip, the start pose and
  // the first target; step-scale draws random actions, the noise substitution
  // and every re-target after the first. The split is what makes the
  // action_hold_steps sweep a controlled experiment - change the hold and only
  // the step stream reshuffles, so every episode still starts in the same place.
  //
  // The rule that keeps that true: episode_rng_ is drawn from in exactly one
  // place, begin_episode, and the number of draws there does not depend on the
  // trajectory. Draw from it mid-episode - on re-target, say, which fires a
  // trajectory-dependent number of times - and the *next* episode's start pose
  // starts depending on action_hold_steps, which is the coupling the split
  // exists to prevent. Grep for episode_rng_; one function should match.
  //
  // Their seeding is part of the determinism contract, like the base-3 digit
  // layout above: swap the two stream tags in policy.cpp and every shard already
  // on disk replays differently, with nothing reporting an error.
  std::mt19937 episode_rng_;
  std::mt19937 step_rng_;

  PolicyParams params_;

  // kActionLevels^nu, resolved once. action_count() walks nu actuators on every
  // call, and this is on the path of all 300k frames.
  int action_count_;

  std::vector<int> block_body_ids_;

  // The joint each actuator drives, in actuator order - so index i is both the
  // base-3 digit i of the action and the joint whose Jacobian column decides
  // that digit. Read from actuator_trnid rather than looked up by name: the
  // action encoding is indexed by actuator and the Jacobian by DOF, and this is
  // the only thing connecting the two.
  std::vector<int> actuated_joint_ids_;

  // The fingertip: the far end of the last actuated link, as an offset in that
  // body's own frame. Derived from the link's geom rather than written down, so
  // resizing the arm in the XML moves the tip with it.
  int tip_body_id_;
  mjtNum tip_local_[3];

  // Scratch, not state - sized in the constructor, overwritten on every scripted
  // draw, never read across calls. Members only to keep a 3 x nv allocation off
  // the per-frame path.
  std::vector<mjtNum> jacp_;
  std::vector<int> signs_;

  // Episode lifetime: set by begin_episode.
  //
  // is_scripted_ is fixed for the whole episode - that is the point of the
  // per-episode mix. target_index_ is not: the reach re-draws it on arrival, and
  // it indexes block_body_ids_, not mjModel's body array.
  bool is_scripted_;
  int target_index_;

  // Step lifetime: the held action, and how many more steps it has to run.
  //
  // One action is held for action_hold_steps because the joint needs roughly a
  // settling time to actually move the commanded way - Q-4's 90% action-following
  // is what this number is for. hold_remaining_ == 0 means "draw on the next
  // call", which is why begin_episode zeroes it rather than filling it.
  int hold_remaining_;
  int held_action_;
};

// Runs the policy against a throwaway mjData - no rendering, no files - and
// reports the three things that say begin_episode and step work:
//
//   F-4  two passes at the same seed produce an identical action sequence
//   F-5  every action holds >= 5% of frames, and max bin / min bin <= 2.5
//        the fraction of episodes that reach their target is roughly 0.5, which
//        is the scripted half arriving and the coin flip being fair
//
// Aborts via mju_error if the two passes diverge. F-5 is printed with a verdict
// rather than enforced, and the verdict is marked indicative below 2,000
// episodes, which is the sample size F-5 names. Allocates and frees its own
// mjData, so it can run before the generation loop exists.
void policy_dry_run(const mjModel* model, PolicyParams params, int base_seed,
                    int shard_index, int episodes, int steps_per_episode);
