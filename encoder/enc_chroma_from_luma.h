// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_ENC_CHROMA_FROM_LUMA_H_
#define ENCODER_ENC_CHROMA_FROM_LUMA_H_

// Chroma-from-luma, computed using heuristics to determine the best linear
// model for the X and B channels from the Y channel.

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "encoder/base/compiler_specific.h"
#include "encoder/base/data_parallel.h"
#include "encoder/base/status.h"
#include "encoder/chroma_from_luma.h"
#include "encoder/common.h"
#include "encoder/enc_ans.h"
#include "encoder/enc_bit_writer.h"
#include "encoder/entropy_coder.h"
#include "encoder/image.h"
#include "encoder/opsin_params.h"
#include "encoder/quant_weights.h"
#include "encoder/quantizer.h"

namespace jxl {

void ColorCorrelationMapEncodeDC(ColorCorrelationMap* map, BitWriter* writer);

Status ComputeColorCorrelationMap(const Image3F& opsin,
                                  const DequantMatrices& dequant,
                                  ThreadPool* pool, ColorCorrelationMap* cmap);
}  // namespace jxl

#endif  // ENCODER_ENC_CHROMA_FROM_LUMA_H_
