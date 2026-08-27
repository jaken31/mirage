#include "shard_writer.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <vector>

#include <mujoco/mujoco.h>

namespace {
    // Upper bound on the meta record, so one stack buffer serves every frame
    // instead of a per-frame allocation on the path of all 300k. 46 bytes for
    // the current scene; this covers ~24 blocks before the constructor
    // complains.
    constexpr int kMaxMetaRecordBytes = 256;

    // Characters allowed in the two provenance strings. They are pasted into the
    // sidecar with no escaping, so this is what keeps a stray quote or backslash
    // from producing a file json.load rejects - or worse, one it accepts with
    // the wrong contents.
    bool IsSafeJsonAtom(const std::string& text) {
        for (const char c : text) {
            const bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
                            (c >= 'A' && c <= 'Z') || c == '_' || c == '-' || c == '.';
            if (!ok) {
                return false;
            }
        }
        return !text.empty();
    }

    std::string ShardPath(const std::string& dir, int shard_index, const char* extension) {
        char name[64];
        std::snprintf(name, sizeof(name), "shard_%03d.%s", shard_index, extension);
        return (std::filesystem::path(dir) / name).string();
    }

    // Little-endian writers. Explicit shifts rather than a memcpy of a struct:
    // padding differs between compilers, and mirage/data.py reads these at fixed
    // offsets with a '<' dtype. Spelling out the byte order makes the file the
    // same on any host, not just this one.
    void PutU8(unsigned char* out, std::uint8_t value) {
        out[0] = value;
    }

    void PutU16(unsigned char* out, std::uint16_t value) {
        out[0] = static_cast<unsigned char>(value & 0xFF);
        out[1] = static_cast<unsigned char>((value >> 8) & 0xFF);
    }

    void PutU32(unsigned char* out, std::uint32_t value) {
        out[0] = static_cast<unsigned char>(value & 0xFF);
        out[1] = static_cast<unsigned char>((value >> 8) & 0xFF);
        out[2] = static_cast<unsigned char>((value >> 16) & 0xFF);
        out[3] = static_cast<unsigned char>((value >> 24) & 0xFF);
    }

    // f32, not the mjtNum double it came from. The 46-byte record is what R-4's
    // budget is computed against, and single precision is ~1e-7 relative - far
    // below anything measurable off a 64x64 frame.
    void PutF32(unsigned char* out, double value) {
        const float narrowed = static_cast<float>(value);
        std::uint32_t bits = 0;
        std::memcpy(&bits, &narrowed, sizeof(bits));
        PutU32(out, bits);
    }

    std::uint16_t GetU16(const unsigned char* in) {
        return static_cast<std::uint16_t>(in[0] | (in[1] << 8));
    }

    std::uint32_t GetU32(const unsigned char* in) {
        return static_cast<std::uint32_t>(in[0]) |
               (static_cast<std::uint32_t>(in[1]) << 8) |
               (static_cast<std::uint32_t>(in[2]) << 16) |
               (static_cast<std::uint32_t>(in[3]) << 24);
    }

    float GetF32(const unsigned char* in) {
        const std::uint32_t bits = GetU32(in);
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
}

bool shard_offset_fits(std::int64_t written, std::int64_t add) {
    if (written < 0 || add < 0) {
        return false;
    }
    return written <= std::numeric_limits<std::int64_t>::max() - add;
}

ShardWriter::ShardWriter(const std::string& dir, int shard_index, int height, int width,
                         int joints, int blocks, ShardProvenance provenance)
    : dir_(dir), shard_index_(shard_index), height_(height), width_(width),
      joints_(joints), blocks_(blocks), provenance_(std::move(provenance)) {
    if (height <= 0 || width <= 0) {
        mju_error("shard %d: frame is %d x %d", shard_index, width, height);
    }
    if (joints <= 0 || blocks <= 0) {
        mju_error("shard %d: %d joints and %d blocks; a record with neither "
                  "describes nothing", shard_index, joints, blocks);
    }
    if (shard_index < 0) {
        mju_error("shard index %d is negative; the filename pads it to three "
                  "digits and a minus sign is not one of them", shard_index);
    }
    if (!IsSafeJsonAtom(provenance_.data_hash) || !IsSafeJsonAtom(provenance_.git_sha)) {
        mju_error("data_hash '%s' or git_sha '%s' has characters outside "
                  "[0-9A-Za-z_.-], which the sidecar writes unescaped",
                  provenance_.data_hash.c_str(), provenance_.git_sha.c_str());
    }

    pixel_bytes_per_frame_ = 3LL * height * width;

    // action u8 + contact_mask u8 + episode_id u32 + step_idx u16 = 8 fixed,
    // then qpos f32 per joint, and block_xy f32 pair + visible_px u16 per block.
    // 46 for 2 joints and 3 blocks, the figure in the architecture doc.
    meta_record_bytes_ = 8 + 4*joints + 10*blocks;
    if (meta_record_bytes_ > kMaxMetaRecordBytes) {
        mju_error("meta record is %d bytes, over the %d the writer's frame "
                  "buffer holds", meta_record_bytes_, kMaxMetaRecordBytes);
    }

    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);
    if (ec && !std::filesystem::is_directory(dir_)) {
        mju_error("could not create shard directory '%s': %s",
                  dir_.c_str(), ec.message().c_str());
    }

    // The stale sidecar goes first, and that ordering matters: a re-run of the
    // same shard index truncates last run's blobs, and if it then died, last
    // run's commit marker would be sitting on top of this run's half-written
    // ones.
    const std::string pixel_path = ShardPath(dir_, shard_index_, "pixels");
    const std::string meta_path = ShardPath(dir_, shard_index_, "meta");
    std::filesystem::remove(ShardPath(dir_, shard_index_, "json"), ec);

    pixels_.open(pixel_path, std::ios::binary | std::ios::trunc);
    if (!pixels_) {
        mju_error("could not open '%s' for writing", pixel_path.c_str());
    }
    meta_.open(meta_path, std::ios::binary | std::ios::trunc);
    if (!meta_) {
        mju_error("could not open '%s' for writing", meta_path.c_str());
    }
}

ShardWriter::~ShardWriter() {
    // No sidecar here, deliberately. Everything else about this destructor is
    // ordinary cleanup; the omission is the design.
    pixels_.close();
    meta_.close();
}

void ShardWriter::append(const unsigned char* rgb, int action, const TruthFrame& truth,
                         std::uint32_t episode_id, std::uint16_t step_idx) {
    if (committed_) {
        mju_error("shard %d: append after commit; the sidecar already claims "
                  "%lld frames", shard_index_, static_cast<long long>(frames_));
    }
    if (!rgb) {
        mju_error("shard %d: null pixel buffer", shard_index_);
    }
    if (action < 0 || action > 255) {
        mju_error("shard %d: action %d does not fit the record's u8",
                  shard_index_, action);
    }
    if (truth.joint_qpos.size() != static_cast<std::size_t>(joints_) ||
        truth.block_xy.size() != static_cast<std::size_t>(2 * blocks_) ||
        truth.visible_px.size() != static_cast<std::size_t>(blocks_)) {
        mju_error("shard %d: TruthFrame carries %zu joints and %zu blocks, the "
                  "record was sized for %d and %d",
                  shard_index_, truth.joint_qpos.size(), truth.visible_px.size(),
                  joints_, blocks_);
    }

    // E-3's bounds assert, at the write site, on 64-bit counters. The predicate
    // lives apart so shard_writer_self_check can watch it reject an overflowing
    // offset without ending the process.
    if (!shard_offset_fits(pixel_bytes_, pixel_bytes_per_frame_) ||
        !shard_offset_fits(meta_bytes_, meta_record_bytes_)) {
        mju_error("shard %d: offset would overflow at frame %lld (%lld pixel "
                  "bytes, %lld meta bytes)", shard_index_,
                  static_cast<long long>(frames_),
                  static_cast<long long>(pixel_bytes_),
                  static_cast<long long>(meta_bytes_));
    }

    unsigned char record[kMaxMetaRecordBytes] = {};
    int at = 0;
    PutU8(record + at, static_cast<std::uint8_t>(action));
    at += 1;
    for (int j = 0; j < joints_; ++j) {
        PutF32(record + at, truth.joint_qpos[static_cast<std::size_t>(j)]);
        at += 4;
    }
    for (int b = 0; b < 2 * blocks_; ++b) {
        PutF32(record + at, truth.block_xy[static_cast<std::size_t>(b)]);
        at += 4;
    }
    for (int b = 0; b < blocks_; ++b) {
        const int count = truth.visible_px[static_cast<std::size_t>(b)];
        if (count < 0 || count > 65535) {
            mju_error("shard %d: visible_px[%d] is %d, outside the record's u16",
                      shard_index_, b, count);
        }
        PutU16(record + at, static_cast<std::uint16_t>(count));
        at += 2;
    }
    PutU8(record + at, truth.contact_mask);
    at += 1;
    PutU32(record + at, episode_id);
    at += 4;
    PutU16(record + at, step_idx);
    at += 2;
    if (at != meta_record_bytes_) {
        mju_error("shard %d: wrote %d record bytes, sized for %d",
                  shard_index_, at, meta_record_bytes_);
    }

    pixels_.write(reinterpret_cast<const char*>(rgb),
                  static_cast<std::streamsize>(pixel_bytes_per_frame_));
    meta_.write(reinterpret_cast<const char*>(record),
                static_cast<std::streamsize>(meta_record_bytes_));
    // Checked every frame rather than at close: a full disk sets failbit here,
    // and finding out at close means not knowing which frame was the last good
    // one. The check is a flag test, not a syscall.
    if (!pixels_ || !meta_) {
        mju_error("shard %d: write failed at frame %lld", shard_index_,
                  static_cast<long long>(frames_));
    }

    pixel_bytes_ += pixel_bytes_per_frame_;
    meta_bytes_ += meta_record_bytes_;
    frames_ += 1;
}

void ShardWriter::commit() {
    if (committed_) {
        mju_error("shard %d: commit called twice", shard_index_);
    }
    if (frames_ == 0) {
        mju_error("shard %d: commit with no frames; an empty shard the loader "
                  "would happily accept is worse than none", shard_index_);
    }

    // close(), not flush(): the sidecar is a commit marker only if the blobs are
    // shut and their buffers are on disk before it appears.
    pixels_.close();
    meta_.close();
    if (!pixels_.good() || !meta_.good()) {
        mju_error("shard %d: closing the blobs failed, so the last writes may "
                  "not have reached disk; no sidecar written", shard_index_);
    }

    write_sidecar();
    committed_ = true;
}

void ShardWriter::write_sidecar() {
    const std::string path = ShardPath(dir_, shard_index_, "json");
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        mju_error("could not open '%s' for writing", path.c_str());
    }

    // Flat by design: every value is an integer or a character-checked atom, so
    // this needs no JSON library and no escaping. Nest anything in here and that
    // stops being true - reach for nlohmann/json at that point rather than
    // growing this.
    //
    // meta_joints and meta_blocks are here so mirage/data.py builds its dtype
    // from the file instead of hardcoding 46 bytes. Same "no hardcoded shapes"
    // rule as everywhere else, applied across the language boundary.
    out << "{\n"
        << "  \"frames\": " << frames_ << ",\n"
        << "  \"height\": " << height_ << ",\n"
        << "  \"width\": " << width_ << ",\n"
        << "  \"channels\": 3,\n"
        << "  \"pixel_dtype\": \"uint8\",\n"
        << "  \"meta_record_bytes\": " << meta_record_bytes_ << ",\n"
        << "  \"meta_joints\": " << joints_ << ",\n"
        << "  \"meta_blocks\": " << blocks_ << ",\n"
        << "  \"seed\": " << provenance_.seed << ",\n"
        << "  \"shard_index\": " << shard_index_ << ",\n"
        << "  \"data_hash\": \"" << provenance_.data_hash << "\",\n"
        << "  \"git_sha\": \"" << provenance_.git_sha << "\"\n"
        << "}\n";

    out.close();
    if (!out.good()) {
        mju_error("shard %d: the sidecar did not close cleanly, so the shard is "
                  "marked complete but may not be", shard_index_);
    }
}

void shard_writer_self_check() {
    // The bounds predicate first, since nothing else matters if it is wrong.
    const std::int64_t max64 = std::numeric_limits<std::int64_t>::max();
    if (!shard_offset_fits(0, 1) || !shard_offset_fits(max64 - 1, 1)) {
        mju_error("shard_offset_fits rejects an offset that fits");
    }
    if (shard_offset_fits(max64, 1) || shard_offset_fits(max64 - 1, 2) ||
        shard_offset_fits(-1, 0) || shard_offset_fits(0, -1)) {
        mju_error("shard_offset_fits accepts an offset that overflows - E-3's "
                  "write-site assert would never fire");
    }

    const int height = 2;
    const int width = 2;
    const int joints = 2;
    const int blocks = 3;
    const int frames = 3;
    const std::int64_t pixel_bytes_per_frame = 3LL * height * width;
    const int record_bytes = 8 + 4*joints + 10*blocks;

    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "mirage_shard_self_check";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);

    const std::string dir_string = dir.string();
    const std::string pixel_path = ShardPath(dir_string, 7, "pixels");
    const std::string meta_path = ShardPath(dir_string, 7, "meta");
    const std::string sidecar_path = ShardPath(dir_string, 7, "json");

    // The values below are dyadic - halves and quarters - so the double to f32
    // narrowing is exact and the comparisons can be ==. Anything else would need
    // a tolerance, and a tolerance would hide a real corruption.
    std::vector<unsigned char> rgb(static_cast<std::size_t>(pixel_bytes_per_frame));
    TruthFrame truth;
    truth.joint_qpos.resize(static_cast<std::size_t>(joints));
    truth.block_xy.resize(static_cast<std::size_t>(2 * blocks));
    truth.visible_px.resize(static_cast<std::size_t>(blocks));

    {
        ShardWriter writer(dir_string, 7, height, width, joints, blocks,
                           ShardProvenance{"deadbeef", "0123abc", 11});
        if (writer.pixel_bytes_per_frame() != pixel_bytes_per_frame ||
            writer.meta_record_bytes() != record_bytes) {
            mju_error("writer sized a frame at %lld pixel bytes and %d record "
                      "bytes, expected %lld and %d",
                      static_cast<long long>(writer.pixel_bytes_per_frame()),
                      writer.meta_record_bytes(),
                      static_cast<long long>(pixel_bytes_per_frame), record_bytes);
        }

        for (int f = 0; f < frames; ++f) {
            for (std::size_t i = 0; i < rgb.size(); ++i) {
                rgb[i] = static_cast<unsigned char>((17*f + static_cast<int>(i)) & 0xFF);
            }
            for (int j = 0; j < joints; ++j) {
                truth.joint_qpos[static_cast<std::size_t>(j)] = 0.5 * (f + j);
            }
            for (int b = 0; b < 2 * blocks; ++b) {
                truth.block_xy[static_cast<std::size_t>(b)] = -0.25 * (f + b);
            }
            for (int b = 0; b < blocks; ++b) {
                truth.visible_px[static_cast<std::size_t>(b)] = 100*f + b;
            }
            truth.contact_mask = static_cast<std::uint8_t>(f + 1);
            writer.append(rgb.data(), 5 + f, truth,
                          static_cast<std::uint32_t>(70000 + f),
                          static_cast<std::uint16_t>(600 + f));
        }

        // The crash-safety claim, checked at the one moment it is observable.
        if (std::filesystem::exists(sidecar_path)) {
            mju_error("the sidecar exists before commit, so a crashed run would "
                      "leave a shard the loader accepts as complete");
        }
        writer.commit();
    }

    const std::uintmax_t pixel_size = std::filesystem::file_size(pixel_path);
    const std::uintmax_t meta_size = std::filesystem::file_size(meta_path);
    const auto want_pixel_size = static_cast<std::uintmax_t>(frames * pixel_bytes_per_frame);
    const auto want_meta_size = static_cast<std::uintmax_t>(frames * record_bytes);
    if (pixel_size != want_pixel_size || meta_size != want_meta_size) {
        mju_error("blobs are %llu and %llu bytes, expected %llu and %llu - "
                  "something padded a record",
                  static_cast<unsigned long long>(pixel_size),
                  static_cast<unsigned long long>(meta_size),
                  static_cast<unsigned long long>(want_pixel_size),
                  static_cast<unsigned long long>(want_meta_size));
    }

    std::vector<unsigned char> pixel_bytes(static_cast<std::size_t>(pixel_size));
    std::vector<unsigned char> meta_bytes(static_cast<std::size_t>(meta_size));
    {
        std::ifstream pixel_in(pixel_path, std::ios::binary);
        std::ifstream meta_in(meta_path, std::ios::binary);
        pixel_in.read(reinterpret_cast<char*>(pixel_bytes.data()),
                      static_cast<std::streamsize>(pixel_size));
        meta_in.read(reinterpret_cast<char*>(meta_bytes.data()),
                     static_cast<std::streamsize>(meta_size));
        if (!pixel_in || !meta_in) {
            mju_error("could not read the blobs back");
        }
    }

    for (int f = 0; f < frames; ++f) {
        for (std::int64_t i = 0; i < pixel_bytes_per_frame; ++i) {
            const auto at_byte = static_cast<std::size_t>(f * pixel_bytes_per_frame + i);
            const auto want = static_cast<unsigned char>((17*f + static_cast<int>(i)) & 0xFF);
            if (pixel_bytes[at_byte] != want) {
                mju_error("pixel byte %zu of frame %d is %d, expected %d - the "
                          "blob is not frame-contiguous",
                          at_byte, f, pixel_bytes[at_byte], want);
            }
        }

        const unsigned char* record = meta_bytes.data() + f*record_bytes;
        int at = 0;
        if (record[at] != static_cast<unsigned char>(5 + f)) {
            mju_error("frame %d: action reads %d, expected %d", f, record[at], 5 + f);
        }
        at += 1;
        for (int j = 0; j < joints; ++j) {
            const float want = static_cast<float>(0.5 * (f + j));
            if (GetF32(record + at) != want) {
                mju_error("frame %d: qpos[%d] reads %g, expected %g",
                          f, j, static_cast<double>(GetF32(record + at)),
                          static_cast<double>(want));
            }
            at += 4;
        }
        for (int b = 0; b < 2 * blocks; ++b) {
            const float want = static_cast<float>(-0.25 * (f + b));
            if (GetF32(record + at) != want) {
                mju_error("frame %d: block_xy[%d] reads %g, expected %g",
                          f, b, static_cast<double>(GetF32(record + at)),
                          static_cast<double>(want));
            }
            at += 4;
        }
        for (int b = 0; b < blocks; ++b) {
            const auto want = static_cast<std::uint16_t>(100*f + b);
            if (GetU16(record + at) != want) {
                mju_error("frame %d: visible_px[%d] reads %u, expected %u",
                          f, b, GetU16(record + at), want);
            }
            at += 2;
        }
        if (record[at] != static_cast<unsigned char>(f + 1)) {
            mju_error("frame %d: contact_mask reads %d, expected %d",
                      f, record[at], f + 1);
        }
        at += 1;
        if (GetU32(record + at) != static_cast<std::uint32_t>(70000 + f)) {
            mju_error("frame %d: episode_id reads %u, expected %u",
                      f, GetU32(record + at), static_cast<std::uint32_t>(70000 + f));
        }
        at += 4;
        if (GetU16(record + at) != static_cast<std::uint16_t>(600 + f)) {
            mju_error("frame %d: step_idx reads %u, expected %u",
                      f, GetU16(record + at), static_cast<std::uint16_t>(600 + f));
        }
    }

    std::ifstream sidecar_in(sidecar_path);
    const std::string sidecar((std::istreambuf_iterator<char>(sidecar_in)),
                              std::istreambuf_iterator<char>());
    if (sidecar.find("\"frames\": 3") == std::string::npos ||
        sidecar.find("\"meta_record_bytes\": 46") == std::string::npos ||
        sidecar.find("\"data_hash\": \"deadbeef\"") == std::string::npos) {
        mju_error("sidecar is missing a field it must carry:\n%s", sidecar.c_str());
    }

    std::filesystem::remove_all(dir, ec);
    std::printf("shard_writer_self_check passed: %d frames, %lld + %d bytes each, "
                "sidecar written last\n",
                frames, static_cast<long long>(pixel_bytes_per_frame), record_bytes);
}
