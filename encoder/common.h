// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_COMMON_H_
#define ENCODER_COMMON_H_

// Shared constants and helper functions.

#include <inttypes.h>
#include <stddef.h>

#include "encoder/base/compiler_specific.h"

namespace jxl {

constexpr size_t kBitsPerByte = 8;  // more clear than CHAR_BIT

template <typename T1, typename T2>
constexpr inline T1 DivCeil(T1 a, T2 b) {
  return (a + b - 1) / b;
}

// Works for any `align`; if a power of two, compiler emits ADD+AND.
constexpr inline size_t RoundUpTo(size_t what, size_t align) {
  return DivCeil(what, align) * align;
}

// Block is the square grid of pixels to which an "energy compaction"
// transformation (e.g. DCT) is applied. Each block has its own AC quantizer.
constexpr size_t kBlockDim = 8;
constexpr size_t kDCTBlockSize = kBlockDim * kBlockDim;
constexpr size_t kGroupDim = 256;
constexpr size_t kGroupDimInBlocks = kGroupDim / kBlockDim;
constexpr size_t kDCGroupDim = kGroupDim * kBlockDim;
constexpr size_t kColorTileDim = 64;
constexpr size_t kColorTileDimInBlocks = kColorTileDim / kBlockDim;
constexpr size_t kGroupDimInColorTiles = kGroupDim / kColorTileDim;

template <typename T>
JXL_INLINE T Clamp1(T val, T low, T hi) {
  return val < low ? low : val > hi ? hi : val;
}

// Encodes non-negative (X) into (2 * X), negative (-X) into (2 * X - 1)
constexpr uint32_t PackSigned(int32_t value)
    JXL_NO_SANITIZE("unsigned-integer-overflow") {
  return (static_cast<uint32_t>(value) << 1) ^
         ((static_cast<uint32_t>(~value) >> 31) - 1);
}

}  // namespace jxl

#endif  // ENCODER_COMMON_H_
