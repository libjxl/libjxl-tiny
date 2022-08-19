// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_DEC_NOISE_H_
#define ENCODER_DEC_NOISE_H_

// Noise synthesis. Currently disabled.

#include <stddef.h>
#include <stdint.h>

#include "encoder/base/status.h"
#include "encoder/chroma_from_luma.h"
#include "encoder/dec_bit_reader.h"
#include "encoder/image.h"
#include "encoder/noise.h"

namespace jxl {

void Random3Planes(size_t visible_frame_index, size_t nonvisible_frame_index,
                   size_t x0, size_t y0, const std::pair<ImageF*, Rect>& plane0,
                   const std::pair<ImageF*, Rect>& plane1,
                   const std::pair<ImageF*, Rect>& plane2);

// Must only call if FrameHeader.flags.kNoise.
Status DecodeNoise(BitReader* br, NoiseParams* noise_params);

}  // namespace jxl

#endif  // ENCODER_DEC_NOISE_H_
