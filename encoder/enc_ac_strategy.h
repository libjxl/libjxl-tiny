// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_ENC_AC_STRATEGY_H_
#define ENCODER_ENC_AC_STRATEGY_H_

#include <stdint.h>

#include "encoder/ac_strategy.h"
#include "encoder/base/data_parallel.h"
#include "encoder/base/status.h"
#include "encoder/chroma_from_luma.h"
#include "encoder/common.h"
#include "encoder/image.h"
#include "encoder/quant_weights.h"

// `FindBestAcStrategy` uses heuristics to choose which AC strategy should be
// used in each block, as well as the initial quantization field.

namespace jxl {

// AC strategy selection: utility struct.

struct ACSConfig {
  const DequantMatrices* JXL_RESTRICT dequant;
  float info_loss_multiplier;
  float info_loss_multiplier2;
  const float* JXL_RESTRICT quant_field_row;
  size_t quant_field_stride;
  const float* JXL_RESTRICT masking_field_row;
  size_t masking_field_stride;
  const float* JXL_RESTRICT src_rows[3];
  size_t src_stride;
  // Cost for 1 (-1), 2 (-2) explicitly, cost for others computed with cost1 +
  // cost2 + sqrt(q) * cost_delta.
  float cost1;
  float cost2;
  float cost_delta;
  float base_entropy;
  float zeros_mul;
  const float& Pixel(size_t c, size_t x, size_t y) const {
    return src_rows[c][y * src_stride + x];
  }
  float Masking(size_t bx, size_t by) const {
    JXL_DASSERT(masking_field_row[by * masking_field_stride + bx] > 0);
    return masking_field_row[by * masking_field_stride + bx];
  }
  float Quant(size_t bx, size_t by) const {
    JXL_DASSERT(quant_field_row[by * quant_field_stride + bx] > 0);
    return quant_field_row[by * quant_field_stride + bx];
  }
};

struct AcStrategyHeuristics {
  void Init(const Image3F& src, float distance, const ColorCorrelationMap& cmap,
            const ImageF& quant_field, const ImageF& masking_field,
            DequantMatrices* matrices, AcStrategyImage* ac_strategy);
  void ProcessRect(const Rect& rect);
  ACSConfig config;
  AcStrategyImage* ac_strategy;
  const ColorCorrelationMap* cmap;
  float butteraugli_target;
};

Status ComputeAcStrategyImage(const Image3F& opsin, const float distance,
                              const ColorCorrelationMap& cmap,
                              const ImageF& quant_field,
                              const ImageF& masking_field, ThreadPool* pool,
                              DequantMatrices* matrices,
                              AcStrategyImage* ac_strategy);

}  // namespace jxl

#endif  // ENCODER_ENC_AC_STRATEGY_H_
