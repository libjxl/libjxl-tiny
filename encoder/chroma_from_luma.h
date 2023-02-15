// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_CHROMA_FROM_LUMA_H_
#define ENCODER_CHROMA_FROM_LUMA_H_

// Chroma-from-luma, computed using heuristics to determine the best linear
// model for the X and B channels from the Y channel.

#include <stddef.h>
#include <stdint.h>

#include "encoder/common.h"
#include "encoder/image.h"

namespace jxl {

static constexpr float kInvColorFactor = 1.0f / 84;

static inline float YtoXRatio(int8_t x) { return x * kInvColorFactor; }
static inline float YtoBRatio(int8_t b) { return 1.0f + b * kInvColorFactor; }

}  // namespace jxl

#endif  // ENCODER_CHROMA_FROM_LUMA_H_
