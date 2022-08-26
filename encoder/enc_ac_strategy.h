// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_ENC_AC_STRATEGY_H_
#define ENCODER_ENC_AC_STRATEGY_H_

#include "encoder/ac_strategy.h"
#include "encoder/base/data_parallel.h"
#include "encoder/base/status.h"
#include "encoder/chroma_from_luma.h"
#include "encoder/image.h"
#include "encoder/quant_weights.h"

namespace jxl {

Status ComputeAcStrategyImage(const Image3F& opsin, const float distance,
                              const ColorCorrelationMap& cmap,
                              const ImageF& quant_field,
                              const ImageF& masking_field, ThreadPool* pool,
                              const DequantMatrices& matrices,
                              AcStrategyImage* ac_strategy);

void AdjustQuantField(const AcStrategyImage& ac_strategy, ImageI* quant_field);

}  // namespace jxl

#endif  // ENCODER_ENC_AC_STRATEGY_H_
