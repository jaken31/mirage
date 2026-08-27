#include "policy.h"

#include <vector>

namespace {
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
}
