// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_ENC_ADAPTIVE_QUANTIZATION_H_
#define ENCODER_ENC_ADAPTIVE_QUANTIZATION_H_

#include <stddef.h>

#include "encoder/ac_strategy.h"
#include "encoder/base/data_parallel.h"
#include "encoder/chroma_from_luma.h"
#include "encoder/common.h"
#include "encoder/enc_cache.h"
#include "encoder/frame_header.h"
#include "encoder/image.h"
#include "encoder/loop_filter.h"
#include "encoder/quant_weights.h"
#include "encoder/quantizer.h"

// Heuristics to find a good quantizer for a given image. InitialQuantField
// produces a quantization field (i.e. relative quantization amounts for each
// block) out of an opsin-space image. `InitialQuantField` uses heuristics,
// `FindBestQuantizer` (in non-fast mode) will run multiple encoding-decoding
// steps and try to improve the given quant field.

namespace jxl {

// Returns an image subsampled by kBlockDim in each direction. If the value
// at pixel (x,y) in the returned image is greater than 1.0, it means that
// more fine-grained quantization should be used in the corresponding block
// of the input image, while a value less than 1.0 indicates that less
// fine-grained quantization should be enough. Returns a mask, too, which
// can later be used to make better decisions about ac strategy.
ImageF InitialQuantField(float butteraugli_target, const Image3F& opsin,
                         const FrameDimensions& frame_dim, ThreadPool* pool,
                         float rescale, ImageF* initial_quant_mask);

float InitialQuantDC(float butteraugli_target);

void AdjustQuantField(const AcStrategyImage& ac_strategy, const Rect& rect,
                      ImageF* quant_field);

}  // namespace jxl

#endif  // ENCODER_ENC_ADAPTIVE_QUANTIZATION_H_
