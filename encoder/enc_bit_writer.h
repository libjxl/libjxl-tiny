// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef LIB_JXL_ENC_BIT_WRITER_H_
#define LIB_JXL_ENC_BIT_WRITER_H_

// BitWriter class: unbuffered writes using unaligned 64-bit stores.

#include <stddef.h>
#include <stdint.h>

#include <utility>
#include <vector>

#include "encoder/base/compiler_specific.h"
#include "encoder/base/padded_bytes.h"
#include "encoder/base/span.h"
#include "encoder/base/status.h"
#include "encoder/common.h"

namespace jxl {

constexpr inline size_t RoundUpBitsToByteMultiple(size_t bits) {
  return (bits + 7) & ~size_t(7);
}

struct BitWriter {
  // Upper bound on `n_bits` in each call to Write. We shift a 64-bit word by
  // 7 bits (max already valid bits in the last byte) and at least 1 bit is
  // needed to zero-initialize the bit-stream ahead (i.e. if 7 bits are valid
  // and we write 57 bits, then the next write will access a byte that was not
  // yet zero-initialized).
  static constexpr size_t kMaxBitsPerCall = 56;

  BitWriter() : bits_written_(0) {}

  // Disallow copying - may lead to bugs.
  BitWriter(const BitWriter&) = delete;
  BitWriter& operator=(const BitWriter&) = delete;
  BitWriter(BitWriter&&) = default;
  BitWriter& operator=(BitWriter&&) = default;

  size_t BitsWritten() const { return bits_written_; }

  Span<const uint8_t> GetSpan() const {
    // Callers must ensure byte alignment to avoid uninitialized bits.
    JXL_ASSERT(bits_written_ % kBitsPerByte == 0);
    return Span<const uint8_t>(storage_.data(), bits_written_ / kBitsPerByte);
  }

  // Example usage: bytes = std::move(writer).TakeBytes(); Useful for the
  // top-level encoder which returns PaddedBytes, not a BitWriter.
  // *this must be an rvalue reference and is invalid afterwards.
  PaddedBytes&& TakeBytes() && {
    // Callers must ensure byte alignment to avoid uninitialized bits.
    JXL_ASSERT(bits_written_ % kBitsPerByte == 0);
    storage_.resize(bits_written_ / kBitsPerByte);
    return std::move(storage_);
  }

 public:
  void Append(const BitWriter& other);
  void AppendByteAligned(std::vector<BitWriter>* others);

  class Allotment {
   public:
    // Expands a BitWriter's storage. Must happen before calling Write or
    // ZeroPadToByte. Must call ReclaimUnused after writing to reclaim the
    // unused storage so that BitWriter memory use remains tightly bounded.
    Allotment(BitWriter* JXL_RESTRICT writer, size_t max_bits);
    ~Allotment();

    void PrivateReclaim(BitWriter* JXL_RESTRICT writer,
                        size_t* JXL_RESTRICT used_bits,
                        size_t* JXL_RESTRICT unused_bits);

    void Reclaim(BitWriter* writer) {
      size_t used, unused;
      PrivateReclaim(writer, &used, &unused);
    }

   private:
    size_t prev_bits_written_;
    const size_t max_bits_;
    bool called_ = false;
    Allotment* parent_;
  };

  // Writes bits into bytes in increasing addresses, and within a byte
  // least-significant-bit first.
  //
  // The function can write up to 56 bits in one go.
  void Write(size_t n_bits, uint64_t bits);

  void AllocateAndWrite(size_t n_bits, uint64_t bits) {
    Allotment allotment(this, n_bits);
    Write(n_bits, bits);
    allotment.Reclaim(this);
  }

  // This should only rarely be used - e.g. when the current location will be
  // referenced via byte offset (TOCs point to groups), or byte-aligned reading
  // is required for speed.
  void ZeroPadToByte() {
    const size_t remainder_bits =
        RoundUpBitsToByteMultiple(bits_written_) - bits_written_;
    if (remainder_bits == 0) return;
    Write(remainder_bits, 0);
    JXL_ASSERT(bits_written_ % kBitsPerByte == 0);
  }

 private:
  size_t bits_written_;
  PaddedBytes storage_;
  Allotment* current_allotment_ = nullptr;
};

}  // namespace jxl

#endif  // LIB_JXL_ENC_BIT_WRITER_H_
