// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_PASSES_STATE_H_
#define ENCODER_PASSES_STATE_H_

#include "encoder/ac_context.h"
#include "encoder/ac_strategy.h"
#include "encoder/chroma_from_luma.h"
#include "encoder/common.h"
#include "encoder/image.h"
#include "encoder/quant_weights.h"
#include "encoder/quantizer.h"

// Structures that hold the (en/de)coder state for a JPEG XL kVarDCT
// (en/de)coder.

namespace jxl {

// State common to both encoder and decoder.
// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
struct PassesSharedState {
  FrameDimensions frame_dim;

  // Control fields and parameters.
  AcStrategyImage ac_strategy;

  // Dequant matrices + quantizer.
  DequantMatrices matrices;
  Quantizer quantizer{&matrices};
  ImageI raw_quant_field;

  // Per-block side information for EPF detail preservation.
  ImageB epf_sharpness;

  ColorCorrelationMap cmap;

  // Memory area for storing coefficient orders.
  // `coeff_order_size` is the size used by *one* set of coefficient orders (at
  // most kMaxCoeffOrderSize). A set of coefficient orders is present for each
  // pass.
  size_t coeff_order_size = 0;
  std::vector<coeff_order_t> coeff_orders;

  BlockCtxMap block_ctx_map;

  Image3F dc_frames[4];

  // Number of pre-clustered set of histograms (with the same ctx map), per
  // pass. Encoded as num_histograms_ - 1.
  size_t num_histograms = 0;

  Rect GroupRect(size_t group_index) const {
    const size_t gx = group_index % frame_dim.xsize_groups;
    const size_t gy = group_index / frame_dim.xsize_groups;
    const Rect rect(gx * frame_dim.group_dim, gy * frame_dim.group_dim,
                    frame_dim.group_dim, frame_dim.group_dim, frame_dim.xsize,
                    frame_dim.ysize);
    return rect;
  }

  Rect PaddedGroupRect(size_t group_index) const {
    const size_t gx = group_index % frame_dim.xsize_groups;
    const size_t gy = group_index / frame_dim.xsize_groups;
    const Rect rect(gx * frame_dim.group_dim, gy * frame_dim.group_dim,
                    frame_dim.group_dim, frame_dim.group_dim,
                    frame_dim.xsize_padded, frame_dim.ysize_padded);
    return rect;
  }

  Rect BlockGroupRect(size_t group_index) const {
    const size_t gx = group_index % frame_dim.xsize_groups;
    const size_t gy = group_index / frame_dim.xsize_groups;
    const Rect rect(gx * (frame_dim.group_dim >> 3),
                    gy * (frame_dim.group_dim >> 3), frame_dim.group_dim >> 3,
                    frame_dim.group_dim >> 3, frame_dim.xsize_blocks,
                    frame_dim.ysize_blocks);
    return rect;
  }

  Rect DCGroupRect(size_t group_index) const {
    const size_t gx = group_index % frame_dim.xsize_dc_groups;
    const size_t gy = group_index / frame_dim.xsize_dc_groups;
    const Rect rect(gx * frame_dim.group_dim, gy * frame_dim.group_dim,
                    frame_dim.group_dim, frame_dim.group_dim,
                    frame_dim.xsize_blocks, frame_dim.ysize_blocks);
    return rect;
  }
};

}  // namespace jxl

#endif  // ENCODER_PASSES_STATE_H_
