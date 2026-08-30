#pragma once

#include <cstdint>
#include <fstream>
#include <string>

#include "truth.h"

// ---------------------------------------------------------------------------
// The three files a shard is made of, written in the order that makes the last
// one a commit marker.
//
//   shard_NNN.pixels  raw uint8, 3*H*W per frame, no header
//   shard_NNN.meta    one fixed-width record per frame
//   shard_NNN.json    written last, after both blobs close
//
// A crashed run leaves blobs with no sidecar and the loader skips the shard.
// That is the whole crash-safety argument - no lockfiles, no truncation
// bookkeeping - and it only holds if the sidecar is written after the blobs
// close, which is what commit() does and the destructor deliberately does not.
// ---------------------------------------------------------------------------

// Provenance strings the writer copies into the sidecar without interpreting
// them. Both come from outside C++: data_hash from mirage/config.py, which owns
// the hash tree, and git_sha from the caller. Deliberately not recomputed here -
// a second implementation of the canonical-JSON hash would have to match
// Python's float formatting byte for byte, and the day it stopped, two shards
// with identical contents would carry different names.
//
// Not recomputed is not the same as not checked. main.cpp's VerifyDataHash asks
// mirage/config.py what the config hashes to and aborts on a mismatch, so this
// writer still copies the string without interpreting it, but a stale one never
// reaches it. Until 2026-08-30 it was neither recomputed nor checked, which made
// data_hash the one field in the sidecar that could claim anything at all.
struct ShardProvenance {
    std::string data_hash;
    std::string git_sha;
    int seed;
};

// The meta record's contact_mask byte is two fields sharing one byte. Bits 0..6
// are TruthFrame::contact_mask - block i touches the arm - and bit 7 says the
// episode is the scripted-reach half rather than the random half.
//
// Packed into a spare bit rather than added as a u8, which would take the record
// 46 -> 47 bytes and every blob 3.69 -> 3.77 GB for one boolean. Three blocks
// use bits 0..2 and truth.cpp refuses a scene with more than seven, so the high
// bit cannot collide.
//
// Every reader must mask. `contact_mask != 0` on the raw byte reads every
// scripted frame as a contact and would take F-6 from 16.6% to over 50% without
// failing anything. mirage/data.py holds the Python half of this constant.
constexpr std::uint8_t kScriptedBit = 0x80;

// True when written + add is representable as a non-negative int64.
//
// E-3 wants a bounds assert at the write site and evidence that it fires on a
// deliberately overflowed offset. An mju_error at the write site cannot be
// tested - it ends the process - so the rule lives here as a pure predicate that
// the write site consults and shard_writer_self_check can probe at INT64_MAX.
bool shard_offset_fits(std::int64_t written, std::int64_t add);

// Appends frames to one shard. Owns the two blob handles and the byte counters;
// measures nothing and decides nothing about what a frame contains.
class ShardWriter {
public:
    // Creates dir if needed and opens both blobs, truncating any shard of the
    // same index. Aborts via mju_error if either open fails, so a half-built
    // writer is never handed back.
    //
    // joints and blocks fix the meta record's width for the whole shard. The
    // record is fixed-width by design, and the reader computes its dtype from
    // the counts in the sidecar rather than hardcoding 46 bytes.
    ShardWriter(const std::string& dir, int shard_index, int height, int width,
                int joints, int blocks, ShardProvenance provenance);

    // Closes whatever is still open. Does not write the sidecar: a shard that
    // died before commit is exactly the incomplete shard the loader must skip.
    ~ShardWriter();

    ShardWriter(const ShardWriter&) = delete;
    ShardWriter& operator=(const ShardWriter&) = delete;

    // One frame. rgb must hold 3*height*width bytes - the buffer the RGB pass
    // read back, not the segmentation one. truth must carry the joint and block
    // counts this writer was constructed with; a mismatch aborts rather than
    // writing a record the reader would silently misparse.
    // is_scripted is Policy::is_scripted() for the episode this frame belongs
    // to. It is a separate argument rather than a bit the caller pre-ORs into
    // truth.contact_mask so that TruthFrame keeps exactly one meaning and the
    // packing lives with the record layout it belongs to.
    void append(const unsigned char* rgb, int action, const TruthFrame& truth,
                bool is_scripted, std::uint32_t episode_id,
                std::uint16_t step_idx);

    // Closes both blobs, checks they closed cleanly, then writes the sidecar.
    // Call exactly once, after the last append. Aborting between the close and
    // the sidecar leaves the shard correctly marked incomplete.
    void commit();

    std::int64_t frames() const { return frames_; }

    // Bytes each file takes per frame. Public because the caller sizes its own
    // pixel buffer against the first, and the self-check compares both against
    // the file sizes on disk.
    std::int64_t pixel_bytes_per_frame() const { return pixel_bytes_per_frame_; }
    int meta_record_bytes() const { return meta_record_bytes_; }

private:
    void write_sidecar();

    // ofstream rather than FILE*: MSVC's /W4 /WX makes fopen's C4996
    // deprecation warning fatal, and silencing it with _CRT_SECURE_NO_WARNINGS
    // would turn the warning off for every file in the target.
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

    // 64-bit on purpose - E-3. At 300k frames the pixel blob is 3.7 GB, which
    // overflows a 32-bit offset a third of the way in, and the symptom would be
    // a shard that reads back as garbage rather than an error.
    std::int64_t pixel_bytes_ = 0;
    std::int64_t meta_bytes_ = 0;
    std::int64_t frames_ = 0;

    bool committed_ = false;
};

// Writes a small shard into the system temp directory, reads the bytes back and
// checks them, then deletes it. Aborts via mju_error on any mismatch.
//
// What it holds:
//   the sidecar does not exist until commit(), which is the crash-safety claim
//   both blobs are exactly frames * per-frame bytes, so nothing padded them
//   every meta field decodes at the offset the layout says it does
//   shard_offset_fits rejects an overflowing offset without ending the process
//
// What it does not hold: that numpy reads the same bytes. That needs the reader
// and lands with mirage/data.py, which is F-8's acceptance test.
void shard_writer_self_check();
