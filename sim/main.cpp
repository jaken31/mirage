#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <nlohmann/json.hpp>

#include "gl_context.h"
#include "policy.h"
#include "truth.h"
#include "shard_writer.h"

namespace {
    using json = nlohmann::json;

    // The knobs this binary reads out of mirage/configs/base.json. Only the keys
    // it uses: mirage/config.py owns validating the whole file and computing the
    // hash tree, and duplicating either here would give the project two answers
    // to the same question.
    struct SimConfig {
        std::string scene_xml;
        std::string shard_dir;
        int seed;
        int episodes;
        int steps_per_episode;
        int frames_per_shard;
        int height;
        int width;
        PolicyParams policy;
    };

    // is_number_integer() rather than get<int>(): nlohmann happily narrows a
    // float to an int, so `"episodes": 500.7` would silently become 500. It also
    // reports false for a bool, which get<int>() would turn into 1.
    int PositiveInt(const json& section, const char* section_name, const char* key) {
        const auto found = section.find(key);
        if (found == section.end()) {
            mju_error("config is missing %s.%s", section_name, key);
        }
        if (!found->is_number_integer()) {
            mju_error("%s.%s must be an integer", section_name, key);
        }
        const long long value = found->get<long long>();
        if (value <= 0 || value > 1000000000LL) {
            mju_error("%s.%s is %lld; expected a positive int under 1e9",
                      section_name, key, value);
        }
        return static_cast<int>(value);
    }

    double Number(const json& section, const char* section_name, const char* key) {
        const auto found = section.find(key);
        if (found == section.end()) {
            mju_error("config is missing %s.%s", section_name, key);
        }
        if (!found->is_number() || found->is_boolean()) {
            mju_error("%s.%s must be a number", section_name, key);
        }
        return found->get<double>();
    }

    std::string Text(const json& section, const char* section_name, const char* key) {
        const auto found = section.find(key);
        if (found == section.end() || !found->is_string()) {
            mju_error("config is missing string %s.%s", section_name, key);
        }
        return found->get<std::string>();
    }

    // Paths inside the config are repo-relative, exactly as mirage/config.py
    // reads them, so this binary must be run from the repo root. A wrong working
    // directory fails at mj_loadXML with the path in the message - loud, and not
    // the silent kind where a different scene gets loaded than the one the
    // data_hash covers.
    SimConfig LoadSimConfig(const std::string& path) {
        std::ifstream in(path);
        if (!in) {
            mju_error("could not open config '%s' (run from the repo root)",
                      path.c_str());
        }

        json raw;
        try {
            in >> raw;
        } catch (const json::exception& e) {
            mju_error("config '%s' is not valid JSON: %s", path.c_str(), e.what());
        }
        if (!raw.contains("sim") || !raw.contains("data")) {
            mju_error("config '%s' has no sim or data section", path.c_str());
        }
        const json& sim = raw["sim"];
        const json& data = raw["data"];

        SimConfig cfg;
        cfg.scene_xml = Text(sim, "sim", "scene_xml");
        cfg.shard_dir = Text(data, "data", "shard_dir");

        const auto seed = sim.find("seed");
        if (seed == sim.end() || !seed->is_number_integer() || seed->get<long long>() < 0) {
            mju_error("sim.seed must be a non-negative integer");
        }
        cfg.seed = static_cast<int>(seed->get<long long>());

        cfg.episodes = PositiveInt(sim, "sim", "episodes");
        cfg.steps_per_episode = PositiveInt(sim, "sim", "steps_per_episode");
        cfg.frames_per_shard = PositiveInt(sim, "sim", "frames_per_shard");
        cfg.height = PositiveInt(sim, "sim", "height");
        cfg.width = PositiveInt(sim, "sim", "width");

        cfg.policy.action_hold_steps = PositiveInt(sim, "sim", "action_hold_steps");
        cfg.policy.reach_digit_noise_prob = Number(sim, "sim", "reach_digit_noise_prob");
        cfg.policy.jacobian_deadband = Number(sim, "sim", "jacobian_deadband");
        cfg.policy.reach_done_dist = Number(sim, "sim", "reach_done_dist");

        // step_idx is a u16 in the meta record, so an episode longer than 65535
        // steps would wrap and the dataloader would read a window as if it had
        // restarted mid-episode.
        if (cfg.steps_per_episode > 65535) {
            mju_error("sim.steps_per_episode is %d; the meta record's step_idx is "
                      "a u16", cfg.steps_per_episode);
        }
        if (cfg.frames_per_shard < cfg.steps_per_episode) {
            mju_error("sim.frames_per_shard (%d) is under one episode (%d steps); "
                      "shards rotate on episode boundaries, so a shard could not "
                      "hold even one", cfg.frames_per_shard, cfg.steps_per_episode);
        }
        return cfg;
    }

    // Reads `git rev-parse HEAD` at build time is not available here, and this
    // binary is not the right place to shell out, so provenance arrives as two
    // required flags. Required rather than defaulted: a shard with no data_hash
    // is a shard nothing can trace back to its config.
    struct Args {
        std::string config_path;
        std::string data_hash;
        std::string git_sha;
    };

    bool ParseArgs(int argc, const char** argv, Args* out) {
        if (argc < 2) {
            return false;
        }
        out->config_path = argv[1];
        for (int i = 2; i + 1 < argc; i += 2) {
            const std::string flag = argv[i];
            if (flag == "--data-hash") {
                out->data_hash = argv[i + 1];
            } else if (flag == "--git-sha") {
                out->git_sha = argv[i + 1];
            } else {
                return false;
            }
        }
        return !out->data_hash.empty() && !out->git_sha.empty();
    }
}

int main(int argc, const char** argv) {
    printf("C++ version: %ld\n", __cplusplus);
    printf("MuJoCo:      %s\n", mj_versionString());
    printf("GLFW:        %s\n", glfwGetVersionString());

    Args args;
    if (!ParseArgs(argc, argv, &args)) {
        fprintf(stderr,
                "Usage: %s <config.json> --data-hash <hex> --git-sha <hex>\n"
                "\n"
                "Run from the repo root - config paths are repo-relative. Both\n"
                "flags are required: a shard that cannot name the config and the\n"
                "commit that produced it is a shard nothing can reproduce.\n"
                "\n"
                "  python -c \"from mirage.config import load; "
                "print(load('mirage/configs/base.json').data_hash)\"\n"
                "  git rev-parse HEAD\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const SimConfig cfg = LoadSimConfig(args.config_path);

    constexpr int error_buffer_size = 1000;
    // Initialised, not just declared: if mj_loadXML fails without writing
    // here, printing an uninitialised array walks off the end of it.
    char error_buffer[error_buffer_size] = "Could not load model";
    mjModel* model = mj_loadXML(cfg.scene_xml.c_str(), nullptr, error_buffer,
                                error_buffer_size);
    if (!model) {
        mju_error("Failed to load model from '%s': %s", cfg.scene_xml.c_str(),
                  error_buffer);
    }

    // The render size comes from config, not from the XML, and it has to be
    // written before GlContext exists because mjr_makeContext reads these two
    // fields to size the offscreen framebuffer. They are plain mutable ints on
    // mjModel (mjmodel.h, struct mjVisual_, the global sub-struct).
    //
    // Editing <global offwidth/offheight> in the XML instead would work and
    // would also be wrong: data_hash is sha256(canon(sim) + canon(data) +
    // xml_bytes), so changing the file to enable a 96x96 dataset would change
    // the 64x64 dataset's hash and orphan 300,000 frames that are still
    // correct. This way the XML stays byte-identical and resolution becomes a
    // config-only change. The literal offwidth="64" left in the XML is now
    // decorative - it is overwritten here on every run.
    model->vis.global.offwidth = cfg.width;
    model->vis.global.offheight = cfg.height;

    policy_self_check(model);
    // Needs no model and no context - it writes a throwaway shard to the temp
    // directory - so it runs beside policy_self_check rather than inside the
    // GlContext block.
    shard_writer_self_check();

    // 200 episodes is enough to see a lopsided histogram, not to tune against
    // one; F-5 itself names 2,000, so the verdict this prints is marked
    // indicative. It runs on every generation run because F-4's determinism half
    // has no other home yet, and 240k stepped actions cost a few seconds against
    // a run measured in minutes.
    policy_dry_run(model, cfg.policy, cfg.seed, /*shard_index=*/0,
                   /*episodes=*/200, cfg.steps_per_episode);

    {
        GlContext context(model);
        const mjrRect viewport = context.viewport();
        // This used to be a third corner: config said how big a frame is, the
        // XML said how big the offscreen buffer is, and GlContext checked the
        // buffer against the XML. Now that offwidth/offheight are written from
        // config above, GlContext's check and this one compare the same two
        // numbers, so this is a duplicate rather than an independent corner.
        // Kept because it costs nothing and it is the assertion that fails if
        // anything is ever inserted between that assignment and this block -
        // but do not read it as corroboration. It is one fact, checked twice.
        if (viewport.width != cfg.width || viewport.height != cfg.height) {
            mju_error("config asks for %d x %d frames, the scene renders %d x %d",
                      cfg.width, cfg.height, viewport.width, viewport.height);
        }

        truth_dry_run(model, context, /*steps=*/1500);

        mjData* data = mj_makeData(model);
        if (!data) {
            mju_error("mj_makeData failed");
        }

        mjvScene scene;
        mjv_defaultScene(&scene);
        mjv_makeScene(model, &scene, 1000);

        mjvOption opt;
        mjv_defaultOption(&opt);

        // The XML's fixed camera, by index. Free-camera framing would differ run
        // to run and F-4 asks for bit-identical frames from the same seed.
        if (model->ncam < 1) {
            mju_error("model has no camera to capture from");
        }
        mjvCamera camera;
        mjv_defaultCamera(&camera);
        camera.type = mjCAMERA_FIXED;
        camera.fixedcamid = 0;

        Truth truth(model, context);
        TruthFrame frame;
        std::vector<unsigned char> rgb(
            static_cast<std::size_t>(3 * viewport.width * viewport.height));

        // Shards rotate on episode boundaries, never mid-episode: Policy is
        // seeded per shard, so a shard change mid-episode would reseed the
        // generator halfway through one. Episodes are then spread evenly rather
        // than packed, which costs a little file size and avoids a last shard
        // holding two episodes.
        const int episodes_per_shard = cfg.frames_per_shard / cfg.steps_per_episode;
        const int shards = (cfg.episodes + episodes_per_shard - 1) / episodes_per_shard;
        const int base_episodes = cfg.episodes / shards;
        const int extra_episodes = cfg.episodes % shards;

        const ShardProvenance provenance{args.data_hash, args.git_sha, cfg.seed};

        printf("Generating %d episodes x %d steps = %d frames over %d shards "
               "into '%s'\n",
               cfg.episodes, cfg.steps_per_episode,
               cfg.episodes * cfg.steps_per_episode, shards, cfg.shard_dir.c_str());

        const auto started = std::chrono::steady_clock::now();
        std::int64_t total_frames = 0;
        int episode_id = 0;

        for (int shard = 0; shard < shards; ++shard) {
            const int shard_episodes = base_episodes + (shard < extra_episodes ? 1 : 0);
            ShardWriter writer(cfg.shard_dir, shard, viewport.height, viewport.width,
                               truth.joint_count(),
                               truth.block_count(), provenance);
            Policy policy(model, cfg.seed, shard, cfg.policy);

            for (int e = 0; e < shard_episodes; ++e) {
                policy.begin_episode(data);
                for (int t = 0; t < cfg.steps_per_episode; ++t) {
                    const int action = policy.step(data);
                    action_to_control(model, action, data->ctrl);
                    mj_step(model, data);

                    mjv_updateScene(model, data, &opt, nullptr, &camera, mjCAT_ALL,
                                    &scene);
                    mjr_render(viewport, &scene, context.context());
                    mjr_readPixels(rgb.data(), nullptr, viewport, context.context());

                    // After the RGB readback, never before: the segmentation pass
                    // inside read() leaves the framebuffer holding id colours, so
                    // reversing these two stores the segmentation image as the
                    // dataset and nothing downstream would notice.
                    truth.read(data, &scene, &frame);

                    writer.append(rgb.data(), action, frame, policy.is_scripted(),
                                  static_cast<std::uint32_t>(episode_id),
                                  static_cast<std::uint16_t>(t));
                }
                ++episode_id;
            }

            writer.commit();
            total_frames += writer.frames();
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            printf("  shard %03d: %d episodes, %lld frames, %.1f s elapsed\n",
                   shard, shard_episodes, static_cast<long long>(writer.frames()),
                   elapsed);
        }

        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        printf("Wrote %lld frames in %.1f s = %.0f fps\n",
               static_cast<long long>(total_frames), elapsed,
               elapsed > 0.0 ? static_cast<double>(total_frames) / elapsed : 0.0);

        // Reverse order of creation. The GL context goes last, in GlContext's
        // destructor at the end of this block, because mjr_freeContext needs it
        // still current.
        mjv_freeScene(&scene);
        mj_deleteData(data);
    }

    mj_deleteModel(model);
    return 0;
}
