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

    // The settings this binary reads from mirage/configs/base.json, and only
    // those. mirage/config.py validates the whole file and computes its hash;
    // doing either here too would give two answers to the same question.
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

    // is_number_integer() rather than get<int>(): get<int>() quietly truncates
    // a float, so `"episodes": 500.7` would become 500, and it turns true into 1.
    // is_number_integer() rejects both.
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

    // Paths in the config are relative to the repo root, as mirage/config.py
    // reads them, so run this binary from the repo root. From anywhere else,
    // mj_loadXML fails loudly with the path in the message, rather than silently
    // loading a different scene than the one data_hash describes.
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

        // step_idx is 16 bits in the meta record, so an episode longer than 65535
        // steps would wrap around, and the data loader would think the episode
        // restarted partway through.
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

    // Characters allowed in a config path before it is passed to the shell
    // below. Deliberately narrow, and the same list on both platforms: the pipe
    // runs through cmd.exe on Windows, where `&`, `|`, `^` and `%` are special,
    // and /bin/sh elsewhere, where `$`, `` ` ``, `"`, `;` and `|` are. The path is
    // the one argument that comes from outside. The only listed character either
    // shell treats specially inside double quotes is `\` on /bin/sh, where it can
    // escape another `\` or the closing quote: at worst that garbles the path
    // Python is given, and it never runs a command. Spaces are rejected too:
    // config paths are repo-relative, so a space is more likely a quoting mistake
    // than a real folder name, and failing loudly beats guessing.
    bool IsSafePathArg(const std::string& text) {
        for (const char c : text) {
            const bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
                            (c >= 'A' && c <= 'Z') || c == '_' || c == '-' ||
                            c == '.' || c == '/' || c == '\\' || c == ':';
            if (!ok) {
                return false;
            }
        }
        return !text.empty();
    }

    // The one platform difference in this file. On Windows the underscore names
    // are on purpose: MSVC deprecates plain `popen`, and /W4 /WX would make that
    // warning fatal. Both close functions return 0 only when the command exited
    // with status 0.
    FILE* OpenReadPipe(const char* command) {
#ifdef _WIN32
        return _popen(command, "r");
#else
        return popen(command, "r");
#endif
    }

    int ClosePipe(FILE* pipe) {
#ifdef _WIN32
        return _pclose(pipe);
#else
        return pclose(pipe);
#endif
    }

    // stdout of `command`, or an empty string if it could not be run. stderr
    // stays connected to ours, so a Python traceback is shown instead of being
    // reduced to a return code.
    std::string RunCapture(const std::string& command, bool* ran) {
        *ran = false;
        FILE* pipe = OpenReadPipe(command.c_str());
        if (!pipe) {
            return {};
        }
        std::string out;
        char chunk[256];
        while (std::fgets(chunk, sizeof(chunk), pipe)) {
            out += chunk;
        }
        *ran = (ClosePipe(pipe) == 0);
        return out;
    }

    std::string Trimmed(std::string text) {
        const char* space = " \t\r\n";
        const std::size_t first = text.find_first_not_of(space);
        if (first == std::string::npos) {
            return {};
        }
        return text.substr(first, text.find_last_not_of(space) - first + 1);
    }

    // Asks mirage/config.py what this config hashes to, and refuses to run if
    // the caller passed a different hash.
    //
    // Calls Python instead of hashing in C++, on purpose. data_hash is sha256
    // over Python's canonical JSON, so a C++ copy would have to match Python's
    // float formatting byte for byte, and the day it drifted, identical shards
    // would get different names. Asking the one real implementation cannot
    // drift.
    //
    // Aborts if Python cannot run at all instead of skipping the check: a check
    // that silently turns itself off is exactly the gap it was added to close.
    void VerifyDataHash(const std::string& config_path, const std::string& claimed) {
        if (!IsSafePathArg(config_path)) {
            mju_error("config path '%s' has characters outside [0-9A-Za-z_.:/\\-]; "
                      "it is passed to a command interpreter to verify --data-hash",
                      config_path.c_str());
        }

        // The path is passed as argv[1], not pasted into the -c source, so it can
        // never run as Python. The script has no quotes of its own, and nothing
        // that cmd.exe or /bin/sh would expand inside double quotes, so one
        // quoting works for both.
        const std::string command =
            "python -c \"import sys;from mirage.config import load;"
            "print(load(sys.argv[1]).data_hash)\" \"" + config_path + "\"";

        bool ran = false;
        const std::string actual = Trimmed(RunCapture(command, &ran));
        if (!ran || actual.empty()) {
            mju_error("could not verify --data-hash: `%s` failed. Run from the repo "
                      "root with python on PATH - this check is not optional, an "
                      "unverified data_hash is a shard nothing can trace back",
                      command.c_str());
        }
        if (actual != claimed) {
            mju_error("--data-hash is %s, but %s hashes to %s. The flag is stale: "
                      "every shard this run wrote would claim a config that did not "
                      "produce it", claimed.c_str(), config_path.c_str(),
                      actual.c_str());
        }
        std::printf("data_hash:   %s (verified against %s)\n",
                    actual.c_str(), config_path.c_str());
    }

    // Provenance comes in as two required flags, with no defaults: a shard with
    // no data_hash cannot be traced back to its config.
    //
    // `--data-hash` is *checked* against the config this run loaded (see
    // VerifyDataHash). Without that check, any hex-like string would be stamped
    // into every sidecar and trusted by `mirage/data.load_shards`. The realistic
    // way to hit that is a stale hash from an earlier config, and it is silent:
    // the shards load and claim to come from a config that never produced them.
    //
    // `--git-sha` is not checked. It names the code version, not the config, and
    // nothing in this process could confirm it.
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
                "--data-hash is checked against the config before anything is\n"
                "written, by asking mirage/config.py, so a stale one aborts the\n"
                "run rather than being stamped into every sidecar. That check\n"
                "needs python on PATH. --git-sha is taken on trust.\n"
                "\n"
                "  python -c \"from mirage.config import load; "
                "print(load('mirage/configs/base.json').data_hash)\"\n"
                "  git rev-parse HEAD\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const SimConfig cfg = LoadSimConfig(args.config_path);
    VerifyDataHash(args.config_path, args.data_hash);

    constexpr int error_buffer_size = 1000;
    // Initialised, not just declared: if mj_loadXML fails without writing
    // here, printing an uninitialised array would read past its end.
    char error_buffer[error_buffer_size] = "Could not load model";
    mjModel* model = mj_loadXML(cfg.scene_xml.c_str(), nullptr, error_buffer,
                                error_buffer_size);
    if (!model) {
        mju_error("Failed to load model from '%s': %s", cfg.scene_xml.c_str(),
                  error_buffer);
    }

    // The render size comes from config, not the XML, and must be set before
    // GlContext is created because mjr_makeContext reads these two fields to
    // size the offscreen buffer. They are plain writable ints on mjModel
    // (mjmodel.h, struct mjVisual_, the global sub-struct).
    //
    // Editing offwidth/offheight in the XML would also work, but would be
    // wrong: data_hash covers the XML's bytes, so changing the file to make a
    // 96x96 dataset would change the 64x64 dataset's hash and orphan 300,000
    // frames that are still correct. This way the XML never changes and
    // resolution is a config-only setting. The offwidth="64" still in the XML
    // does nothing; it is overwritten here on every run.
    model->vis.global.offwidth = cfg.width;
    model->vis.global.offheight = cfg.height;

    policy_self_check(model);
    // Needs no model and no GL context (it writes a throwaway shard to the temp
    // directory), so it runs next to policy_self_check rather than inside the
    // GlContext block.
    shard_writer_self_check();

    // 200 episodes is enough to spot a lopsided action mix, not to tune it (the
    // requirement is judged over 2,000), so its verdict is marked indicative.
    // It runs on every generation run because it is the only place the
    // same-seed determinism check runs, and 240k steps take a few seconds in a
    // run that takes minutes.
    policy_dry_run(model, cfg.policy, cfg.seed, /*shard_index=*/0,
                   /*episodes=*/200, cfg.steps_per_episode);

    {
        GlContext context(model);
        const mjrRect viewport = context.viewport();
        // Now that offwidth/offheight are set from config above, this compares
        // the same two numbers GlContext already checked. It is a duplicate, not
        // an independent check. Kept because it costs nothing and would catch
        // anything ever inserted between that assignment and this block. Do not
        // read it as extra confirmation: it is one fact, checked twice.
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

        // The XML's fixed camera, by index. A free camera's framing could differ
        // between runs, and the same seed must give bit-identical frames.
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

        // Start a new shard only between episodes, never mid-episode: Policy is
        // seeded per shard, so switching shard mid-episode would reseed it
        // halfway through. Episodes are spread evenly across shards rather than
        // packed, which avoids a tiny last shard holding two episodes.
        const int episodes_per_shard = cfg.frames_per_shard / cfg.steps_per_episode;
        const int shards = (cfg.episodes + episodes_per_shard - 1) / episodes_per_shard;
        const int base_episodes = cfg.episodes / shards;
        const int extra_episodes = cfg.episodes % shards;

        const ShardProvenance provenance{args.data_hash, args.git_sha, cfg.seed,
                                         context.renderer(), context.version()};

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
                    // inside read() leaves id colours in the framebuffer, so
                    // swapping these two would store the segmentation image as the
                    // dataset, and nothing downstream would notice.
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

        // Free in reverse order of creation. The GL context goes last (GlContext's
        // destructor, at the end of this block) because mjr_freeContext needs it
        // still current.
        mjv_freeScene(&scene);
        mj_deleteData(data);
    }

    mj_deleteModel(model);
    return 0;
}
