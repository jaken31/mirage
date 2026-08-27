#pragma once

#include <mujoco/mujoco.h>
#include <random>

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
class Policy;
void policy_self_check(const mjModel* model);
