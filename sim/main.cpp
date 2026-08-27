#include <cstdio>
#include <cstdlib>

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include "gl_context.h"
#include "policy.h"

int main(int argc, const char** argv) {
    printf("C++ version: %ld\n", __cplusplus);
    printf("MuJoCo:      %s\n", mj_versionString());
    printf("GLFW:        %s\n", glfwGetVersionString());

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <model.xml>\n", argv[0]);
        return EXIT_FAILURE;
    }

    constexpr int error_buffer_size = 1000;
    // Initialised, not just declared: if mj_loadXML fails without writing
    // here, printing an uninitialised array walks off the end of it.
    char error_buffer[error_buffer_size] = "Could not load model";
    mjModel* model = mj_loadXML(argv[1], nullptr, error_buffer, error_buffer_size);
    if (!model) {
        mju_error("Failed to load model from '%s': %s", argv[1], error_buffer);
    }
    policy_self_check(model);

    // Hardcoded until config parsing lands with shard_writer. The values are the
    // sim.* section of mirage/configs/base.json; when the reader exists this
    // block is what it replaces, and Policy does not change.
    const PolicyParams policy_params{
        /*action_hold_steps=*/20,
        /*reach_digit_noise_prob=*/0.15,
        /*jacobian_deadband=*/0.04,
        /*reach_done_dist=*/0.04,
    };

    // steps_per_episode matches sim.steps_per_episode, because the reach needs
    // it: at 200 steps the arm cannot cross to a block inside 0.4 s of sim time
    // and the arrival rate reads 13% instead of ~44%. 200 episodes is enough to
    // see a lopsided histogram, not to tune against one - at hold=20 that is
    // 6,000 draws over 9 bins - F-5 itself names 2,000 episodes, so the verdict
    // this prints is marked indicative. Raise it before moving either knob.
    policy_dry_run(model, policy_params, /*base_seed=*/0, /*shard_index=*/0,
                   /*episodes=*/200, /*steps_per_episode=*/600);

    {
        GlContext context(model);
        // Seed and shard index are hardcoded for the same reason as the params
        // above: base_seed comes from sim.seed, shard_index from the loop that
        // does not exist yet.
        Policy policy(model, /*base_seed=*/0, /*shard_index=*/0, policy_params);

        // The episode and step loops go here.
    }

    mj_deleteModel(model);
    return 0;
}


