// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_ENC_ADAPTIVE_QUANTIZATION_H_
#define ENCODER_ENC_ADAPTIVE_QUANTIZATION_H_

#include "encoder/image.h"

namespace jxl {

void ComputeAdaptiveQuantFieldTile(const Image3F& xyb, const Rect& rect,
                                   const Rect& block_rect, float inv_scale,
                                   float distance, ImageF* pre_erosion,
                                   float* diff_buffer, ImageF* aq_map,
                                   ImageF* mask, ImageB* raw_quant_field);

}  // namespace jxl

#endif  // ENCODER_ENC_ADAPTIVE_QUANTIZATION_H_
