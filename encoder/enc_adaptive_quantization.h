// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_ENC_ADAPTIVE_QUANTIZATION_H_
#define ENCODER_ENC_ADAPTIVE_QUANTIZATION_H_

#include "encoder/base/data_parallel.h"
#include "encoder/image.h"

namespace jxl {

// Returns an image subsampled by kBlockDim in each direction. If the value
// at pixel (x,y) in the returned image is greater than 1.0, it means that
// more fine-grained quantization should be used in the corresponding block
// of the input image, while a value less than 1.0 indicates that less
// fine-grained quantization should be enough. Returns a mask, too, which
// can later be used to make better decisions about ac strategy.
void ComputeAdaptiveQuantField(const Image3F& opsin, const float distance,
                               const float scale, ThreadPool* pool,
                               ImageF* masking, ImageF* quant_field,
                               ImageI* raw_quant_field);

}  // namespace jxl

#endif  // ENCODER_ENC_ADAPTIVE_QUANTIZATION_H_
