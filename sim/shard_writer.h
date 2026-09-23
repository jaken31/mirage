#pragma once

#include <cstdint>
#include <fstream>
#include <string>

#include "truth.h"

// ---------------------------------------------------------------------------
// A shard is three files, written in an order that makes the last one a
// "this shard is complete" marker.
//
//   shard_NNN.pixels  raw uint8, 3*H*W per frame, no header
//   shard_NNN.meta    one fixed-width record per frame
//   shard_NNN.json    written last, after both data files close
//
// A crashed run leaves the two data files with no .json sidecar, and the
// loader skips that shard. That is the entire crash-safety scheme (no
// lockfiles, no truncation bookkeeping), and it only works if the sidecar is
// written after both data files close. commit() does that; the destructor
// deliberately does not write the sidecar.
// ---------------------------------------------------------------------------

// Provenance strings copied into the sidecar as-is. Both come from outside
// C++: data_hash from mirage/config.py, which owns hashing, and git_sha from
// the caller. Deliberately not recomputed here: a second implementation of the
// hash would have to match Python's float formatting byte for byte, and the
// day it drifted, identical shards would get different names.
//
// Not recomputed, but still checked: main.cpp's VerifyDataHash asks
// mirage/config.py what the config hashes to and aborts on a mismatch, so a
// stale hash never reaches this writer.
struct ShardProvenance {
    std::string data_hash;
    std::string git_sha;
    int seed;
};

// The stored contact_mask byte holds two fields. Bits 0..6 are
// TruthFrame::contact_mask (block i touches the arm). Bit 7 says the episode
// is a scripted reach rather than random.
//
// Packed into a spare bit instead of its own byte, which would grow every
// record from 46 to 47 bytes for one boolean. Three blocks use bits 0..2 and
// truth.cpp refuses more than seven, so bit 7 is free.
//
// Every reader must mask it off. Testing `contact_mask != 0` on the raw byte
// counts every scripted frame as a contact, taking the contact rate from
// about 17% to over 50% with no error. mirage/data.py has the Python copy of
// this constant.
constexpr std::uint8_t kScriptedBit = 0x80;

// True when written + add fits in a non-negative int64.
//
// The write site needs a bounds check, and we need proof it fires on an
// overflowing offset. An mju_error there cannot be tested because it ends the
// process, so the rule is this pure function: the write site calls it, and
// shard_writer_self_check can test it at INT64_MAX.
bool shard_offset_fits(std::int64_t written, std::int64_t add);

// Appends frames to one shard. Owns the two data-file handles and the byte
// counters; it measures nothing and decides nothing about frame contents.
class ShardWriter {
public:
    // Creates dir if needed and opens both data files, overwriting any shard
    // with the same index. Aborts via mju_error if either open fails, so you
    // never get a half-built writer.
    //
    // joints and blocks fix the meta record's width for the whole shard. The
    // reader computes the record layout from the counts in the sidecar instead
    // of hardcoding 46 bytes.
    ShardWriter(const std::string& dir, int shard_index, int height, int width,
                int joints, int blocks, ShardProvenance provenance);

    // Closes whatever is still open. Does not write the sidecar: a shard that
    // died before commit() is exactly the incomplete shard the loader must skip.
    ~ShardWriter();

    ShardWriter(const ShardWriter&) = delete;
    ShardWriter& operator=(const ShardWriter&) = delete;

    // Writes one frame. rgb must hold 3*height*width bytes from the RGB pass,
    // not the segmentation pass. truth must have the joint and block counts this
    // writer was built with; a mismatch aborts rather than writing a record the
    // reader would silently misread.
    //
    // is_scripted is Policy::is_scripted() for this frame's episode. It is a
    // separate argument, not a bit the caller ORs into truth.contact_mask, so
    // TruthFrame keeps one meaning and the bit packing stays with the record
    // layout.
    void append(const unsigned char* rgb, int action, const TruthFrame& truth,
                bool is_scripted, std::uint32_t episode_id,
                std::uint16_t step_idx);

    // Closes both data files, checks they closed cleanly, then writes the
    // sidecar. Call exactly once, after the last append. An abort between the
    // close and the sidecar leaves the shard correctly marked incomplete.
    void commit();

    std::int64_t frames() const { return frames_; }

    // Bytes per frame in each file. Public because the caller sizes its pixel
    // buffer from the first, and the self-check compares both with the file
    // sizes on disk.
    std::int64_t pixel_bytes_per_frame() const { return pixel_bytes_per_frame_; }
    int meta_record_bytes() const { return meta_record_bytes_; }

private:
    void write_sidecar();

    // ofstream rather than FILE*: with /W4 /WX, MSVC's deprecation warning for
    // fopen (C4996) is fatal, and silencing it with _CRT_SECURE_NO_WARNINGS would
    // turn it off for every file in the build.
    std::ofstream pixels_;
    std::ofstream meta_;

    std::string dir_;
    int shard_index_;
    int height_;
    int width_;
    int joints_;
    int blocks_;
    ShardProvenance provenance_;

    std::int64_t pixel_bytes_per_frame_;
    int meta_record_bytes_;

    // 64-bit on purpose. At 300k frames the pixel file is 3.7 GB, which would
    // overflow a 32-bit offset about halfway through, and the symptom would be a
    // shard that reads back as garbage instead of an error.
    std::int64_t pixel_bytes_ = 0;
    std::int64_t meta_bytes_ = 0;
    std::int64_t frames_ = 0;

    bool committed_ = false;
};

// Writes a small shard to the system temp directory, reads the bytes back and
// checks them, then deletes it. Aborts via mju_error on any mismatch.
//
// What it checks:
//   the sidecar does not exist until commit() (the crash-safety rule)
//   both data files are exactly frames * per-frame bytes, so nothing padded them
//   every meta field decodes at the offset the layout says
//   shard_offset_fits rejects an overflowing offset without ending the process
//
// What it does not check: that numpy reads the same bytes. mirage/data.py's
// self-check does that.
void shard_writer_self_check();
