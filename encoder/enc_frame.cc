// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/enc_frame.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <vector>

#include "encoder/ac_context.h"
#include "encoder/ac_strategy.h"
#include "encoder/ans_params.h"
#include "encoder/base/bits.h"
#include "encoder/base/compiler_specific.h"
#include "encoder/base/data_parallel.h"
#include "encoder/base/padded_bytes.h"
#include "encoder/base/status.h"
#include "encoder/chroma_from_luma.h"
#include "encoder/coeff_order.h"
#include "encoder/coeff_order_fwd.h"
#include "encoder/common.h"
#include "encoder/dct_util.h"
#include "encoder/enc_ac_strategy.h"
#include "encoder/enc_adaptive_quantization.h"
#include "encoder/enc_ans.h"
#include "encoder/enc_bit_writer.h"
#include "encoder/enc_chroma_from_luma.h"
#include "encoder/enc_coeff_order.h"
#include "encoder/enc_entropy_coder.h"
#include "encoder/enc_group.h"
#include "encoder/enc_toc.h"
#include "encoder/enc_xyb.h"
#include "encoder/gaborish.h"
#include "encoder/image.h"
#include "encoder/quant_weights.h"
#include "encoder/quantizer.h"

namespace jxl {
namespace {

uint32_t ComputeXQuantScale(float distance) {
  uint32_t x_qm_scale = 2;
  float x_qm_scale_steps[2] = {1.25f, 9.0f};
  for (float x_qm_scale_step : x_qm_scale_steps) {
    if (distance > x_qm_scale_step) {
      x_qm_scale++;
    }
  }
  if (distance < 0.299f) {
    // Favor chromacity preservation for making images appear more
    // faithful to original even with extreme (5-10x) zooming.
    x_qm_scale++;
  }
  return x_qm_scale;
}

uint32_t ComputeNumEpfIters(float distance) {
  constexpr float kEpfThresholds[3] = {0.7, 1.5, 4.0};
  uint32_t epf_iters = 0;
  for (size_t i = 0; i < 3; i++) {
    if (distance >= kEpfThresholds[i]) {
      epf_iters++;
    }
  }
  return epf_iters;
}

void WriteFrameHeader(uint32_t x_qm_scale, uint32_t epf_iters,
                      BitWriter* writer) {
  BitWriter::Allotment allotment(writer, 1024);
  writer->Write(1, 0);    // not all default
  writer->Write(2, 0);    // regular frame
  writer->Write(1, 0);    // vardct
  writer->Write(2, 2);    // flags selector bits (17 .. 272)
  writer->Write(8, 111);  // skip adaptive dc flag (128)
  writer->Write(2, 0);    // no upsampling
  writer->Write(3, x_qm_scale);
  writer->Write(3, 2);  // b_qm_scale
  writer->Write(2, 0);  // one pass
  writer->Write(1, 0);  // no custom frame size or origin
  writer->Write(2, 0);  // replace blend mode
  writer->Write(1, 1);  // last frame
  writer->Write(2, 0);  // no name
  if (epf_iters == 2) {
    writer->Write(1, 1);  // default loop filter
  } else {
    writer->Write(1, 0);  // not default loop filter
    writer->Write(1, 1);  // gaborish on
    writer->Write(1, 0);  // default gaborish
    writer->Write(2, epf_iters);
    if (epf_iters > 0) {
      writer->Write(1, 0);  // default epf sharpness
      writer->Write(1, 0);  // default epf weights
      writer->Write(1, 0);  // default epf sigma
    }
    writer->Write(2, 0);  // no loop filter extensions
  }
  writer->Write(2, 0);  // no frame header extensions
  allotment.Reclaim(writer);
}

// Clamps gradient to the min/max of n, w (and l, implicitly).
JXL_INLINE int32_t ClampedGradient(const int32_t n, const int32_t w,
                                   const int32_t l) {
  const int32_t m = std::min(n, w);
  const int32_t M = std::max(n, w);
  // The end result of this operation doesn't overflow or underflow if the
  // result is between m and M, but the intermediate value may overflow, so we
  // do the intermediate operations in uint32_t and check later if we had an
  // overflow or underflow condition comparing m, M and l directly.
  // grad = M + m - l = n + w - l
  const int32_t grad =
      static_cast<int32_t>(static_cast<uint32_t>(n) + static_cast<uint32_t>(w) -
                           static_cast<uint32_t>(l));
  // We use two sets of ternary operators to force the evaluation of them in
  // any case, allowing the compiler to avoid branches and use cmovl/cmovg in
  // x86.
  const int32_t grad_clamp_M = (l < m) ? M : grad;
  return (l > M) ? m : grad_clamp_M;
}

// Modular context tree for DC and control fields.
static constexpr size_t kNumTreeContexts = 6;
static constexpr size_t kNumContextTreeTokens = 313;
static const Token kContextTreeTokens[kNumContextTreeTokens] = {
    {1, 2},   {0, 4},  {1, 1},   {0, 2},  {1, 10},   {0, 0},  {1, 1},   {0, 4},
    {1, 1},   {0, 0},  {1, 10},  {0, 94}, {1, 10},   {0, 61}, {1, 0},   {2, 0},
    {3, 0},   {4, 0},  {5, 0},   {1, 3},  {0, 0},    {1, 0},  {2, 5},   {3, 0},
    {4, 0},   {5, 0},  {1, 0},   {2, 5},  {3, 0},    {4, 0},  {5, 0},   {1, 10},
    {0, 382}, {1, 10}, {0, 22},  {1, 10}, {0, 13},   {1, 10}, {0, 253}, {1, 8},
    {0, 10},  {1, 8},  {0, 10},  {1, 10}, {0, 784},  {1, 10}, {0, 190}, {1, 10},
    {0, 46},  {1, 10}, {0, 10},  {1, 10}, {0, 5},    {1, 10}, {0, 29},  {1, 10},
    {0, 125}, {1, 10}, {0, 509}, {1, 8},  {0, 22},   {1, 8},  {0, 6},   {1, 8},
    {0, 22},  {1, 8},  {0, 6},   {1, 10}, {0, 1000}, {1, 10}, {0, 510}, {1, 10},
    {0, 254}, {1, 10}, {0, 126}, {1, 10}, {0, 62},   {1, 10}, {0, 30},  {1, 10},
    {0, 14},  {1, 10}, {0, 6},   {1, 10}, {0, 1},    {1, 10}, {0, 7},   {1, 10},
    {0, 21},  {1, 10}, {0, 45},  {1, 10}, {0, 93},   {1, 10}, {0, 189}, {1, 10},
    {0, 381}, {1, 10}, {0, 783}, {1, 0},  {2, 1},    {3, 0},  {4, 0},   {5, 0},
    {1, 0},   {2, 1},  {3, 0},   {4, 0},  {5, 0},    {1, 0},  {2, 1},   {3, 0},
    {4, 0},   {5, 0},  {1, 0},   {2, 1},  {3, 0},    {4, 0},  {5, 0},   {1, 0},
    {2, 0},   {3, 0},  {4, 0},   {5, 0},  {1, 0},    {2, 0},  {3, 0},   {4, 0},
    {5, 0},   {1, 0},  {2, 0},   {3, 0},  {4, 0},    {5, 0},  {1, 0},   {2, 0},
    {3, 0},   {4, 0},  {5, 0},   {1, 0},  {2, 5},    {3, 0},  {4, 0},   {5, 0},
    {1, 0},   {2, 5},  {3, 0},   {4, 0},  {5, 0},    {1, 0},  {2, 5},   {3, 0},
    {4, 0},   {5, 0},  {1, 0},   {2, 5},  {3, 0},    {4, 0},  {5, 0},   {1, 0},
    {2, 5},   {3, 0},  {4, 0},   {5, 0},  {1, 0},    {2, 5},  {3, 0},   {4, 0},
    {5, 0},   {1, 0},  {2, 5},   {3, 0},  {4, 0},    {5, 0},  {1, 0},   {2, 5},
    {3, 0},   {4, 0},  {5, 0},   {1, 0},  {2, 5},    {3, 0},  {4, 0},   {5, 0},
    {1, 0},   {2, 5},  {3, 0},   {4, 0},  {5, 0},    {1, 0},  {2, 5},   {3, 0},
    {4, 0},   {5, 0},  {1, 0},   {2, 5},  {3, 0},    {4, 0},  {5, 0},   {1, 0},
    {2, 5},   {3, 0},  {4, 0},   {5, 0},  {1, 0},    {2, 5},  {3, 0},   {4, 0},
    {5, 0},   {1, 0},  {2, 5},   {3, 0},  {4, 0},    {5, 0},  {1, 10},  {0, 2},
    {1, 0},   {2, 5},  {3, 0},   {4, 0},  {5, 0},    {1, 0},  {2, 5},   {3, 0},
    {4, 0},   {5, 0},  {1, 0},   {2, 5},  {3, 0},    {4, 0},  {5, 0},   {1, 0},
    {2, 5},   {3, 0},  {4, 0},   {5, 0},  {1, 0},    {2, 5},  {3, 0},   {4, 0},
    {5, 0},   {1, 0},  {2, 5},   {3, 0},  {4, 0},    {5, 0},  {1, 0},   {2, 5},
    {3, 0},   {4, 0},  {5, 0},   {1, 0},  {2, 5},    {3, 0},  {4, 0},   {5, 0},
    {1, 0},   {2, 5},  {3, 0},   {4, 0},  {5, 0},    {1, 0},  {2, 5},   {3, 0},
    {4, 0},   {5, 0},  {1, 0},   {2, 5},  {3, 0},    {4, 0},  {5, 0},   {1, 0},
    {2, 5},   {3, 0},  {4, 0},   {5, 0},  {1, 0},    {2, 5},  {3, 0},   {4, 0},
    {5, 0},   {1, 0},  {2, 5},   {3, 0},  {4, 0},    {5, 0},  {1, 0},   {2, 5},
    {3, 0},   {4, 0},  {5, 0},   {1, 10}, {0, 999},  {1, 0},  {2, 5},   {3, 0},
    {4, 0},   {5, 0},  {1, 0},   {2, 5},  {3, 0},    {4, 0},  {5, 0},   {1, 0},
    {2, 5},   {3, 0},  {4, 0},   {5, 0},  {1, 0},    {2, 5},  {3, 0},   {4, 0},
    {5, 0},
};

// Context lookup table for DC coding. The context is given by looking up the
// "gradient" modular property (left + top - topleft) from this table.
static constexpr uint8_t kGradientContextLut[1024] = {
    44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 43, 43, 43, 43, 43, 43,
    43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
    43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
    43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
    43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
    43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
    43, 43, 43, 43, 43, 43, 43, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 39, 39, 39, 39, 39, 39, 39,
    39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39,
    39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39,
    39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 38,
    38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38,
    38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38,
    38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38,
    38, 38, 38, 38, 38, 38, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
    37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
    36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
    36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 35, 35, 35, 35, 35, 35,
    35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 34, 34, 34, 34, 34, 34, 34, 34, 34,
    34, 34, 34, 34, 34, 34, 34, 33, 33, 33, 33, 33, 33, 33, 33, 32, 32, 32, 32,
    32, 32, 32, 32, 31, 31, 31, 31, 30, 30, 30, 30, 29, 29, 29, 28, 27, 27, 26,
    42, 41, 41, 25, 25, 24, 24, 23, 23, 23, 23, 22, 22, 22, 22, 21, 21, 21, 21,
    21, 21, 21, 21, 20, 20, 20, 20, 20, 20, 20, 20, 19, 19, 19, 19, 19, 19, 19,
    19, 19, 19, 19, 19, 19, 19, 19, 19, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    18, 18, 18, 18, 18, 18, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
};
static constexpr int64_t kGradRangeMin = 0;
static constexpr int64_t kGradRangeMid = 512;
static constexpr int64_t kGradRangeMax = 1023;
static constexpr size_t kNumDCContexts = 45;

}  // namespace

Status EncodeFrame(const float distance, const Image3F& linear,
                   ThreadPool* pool, BitWriter* writer) {
  // Pre-compute image dimension-derived values.
  const size_t xsize = linear.xsize();
  const size_t ysize = linear.ysize();
  FrameDimensions frame_dim;
  frame_dim.Set(linear.xsize(), linear.ysize());
  const size_t xsize_blocks = DivCeil(xsize, kBlockDim);
  const size_t ysize_blocks = DivCeil(ysize, kBlockDim);
  const size_t xsize_groups = DivCeil(xsize, kGroupDim);
  const size_t xsize_dc_groups = DivCeil(xsize_blocks, kGroupDim);
  const size_t num_groups = frame_dim.num_groups;
  const size_t num_dc_groups = frame_dim.num_dc_groups;
  auto block_rect = [&](size_t idx, size_t n, size_t m) {
    return Rect((idx % n) * m, (idx / n) * m, m, m, xsize_blocks, ysize_blocks);
  };

  // Write frame header.
  uint32_t x_qm_scale = ComputeXQuantScale(distance);
  uint32_t epf_iters = ComputeNumEpfIters(distance);
  WriteFrameHeader(x_qm_scale, epf_iters, writer);

  // Allocate bit writers for all sections.
  size_t num_toc_entries = 2 + num_dc_groups + num_groups;
  std::vector<BitWriter> group_codes(num_toc_entries);
  const size_t global_ac_index = num_dc_groups + 1;
  const bool is_small_image = num_groups == 1;
  const auto get_output = [&](const size_t index) {
    return &group_codes[is_small_image ? 0 : index];
  };

  // Transform image to XYB colorspace.
  Image3F opsin(frame_dim.xsize_padded, frame_dim.ysize_padded);
  opsin.ShrinkTo(linear.xsize(), linear.ysize());
  ToXYB(linear, pool, &opsin);
  PadImageToBlockMultipleInPlace(&opsin);

  // Compute adaptive quantization field (relies on pre-gaborish values).
  const float quant_dc = InitialQuantDC(distance);
  ImageF masking_field;
  ImageF quant_field =
      InitialQuantField(distance, opsin, frame_dim, pool, 1.0f, &masking_field);

  // Initialize DCT8 quant weights and compute X quant matrix scale.
  DequantMatrices matrices;
  JXL_CHECK(matrices.EnsureComputed(1 << AcStrategy::DCT));
  float x_qm_multiplier = std::pow(1.25f, x_qm_scale - 2.0f);

  // Compute quant scales and raw quant field field.
  Quantizer quantizer(&matrices);
  ImageI raw_quant_field(xsize_blocks, ysize_blocks);
  quantizer.SetQuantField(quant_dc, quant_field, &raw_quant_field);

  // Apply inverse-gaborish.
  GaborishInverse(&opsin, 0.9908511000000001f, pool);

  // Flat AR field.
  ImageB epf_sharpness(xsize_blocks, ysize_blocks);
  FillPlane(static_cast<uint8_t>(4), &epf_sharpness);

  // Compute per-tile color correlation values.
  ColorCorrelationMap cmap(xsize, ysize);
  JXL_RETURN_IF_ERROR(ComputeColorCorrelationMap(opsin, matrices, pool, &cmap));

  // Compute block sizes.
  AcStrategyImage ac_strategy(xsize_blocks, ysize_blocks);
  JXL_RETURN_IF_ERROR(ComputeAcStrategyImage(opsin, distance, cmap, quant_field,
                                             masking_field, pool, &matrices,
                                             &ac_strategy));
  AdjustQuantField(ac_strategy, &raw_quant_field);

  // Compute DC image and AC coefficients.
  Image3F dc(xsize_blocks, ysize_blocks);
  ACImageT<int32_t> ac_coeffs(kGroupDim * kGroupDim, num_groups);
  JXL_RETURN_IF_ERROR(RunOnPool(
      pool, 0, num_groups, ThreadPool::NoInit,
      [&](size_t group_idx, size_t _) {
        ComputeCoefficients(group_idx, opsin, raw_quant_field, quantizer, cmap,
                            ac_strategy, x_qm_multiplier, &ac_coeffs, &dc);
      },
      "Compute coeffs"));

  // Compute DC tokens.
  std::vector<std::vector<Token>> dc_tokens(num_dc_groups);
  auto compute_dc_tokens = [&](int group_index, int /* thread */) {
    const Rect r = block_rect(group_index, xsize_dc_groups, kGroupDim);
    dc_tokens[group_index].reserve(3 * r.xsize() * r.ysize());
    Image3I quant_dc(r.xsize(), r.ysize());
    const float y_dc_step = quantizer.GetDcStep(1);
    for (size_t c : {1, 0, 2}) {
      const intptr_t onerow = quant_dc.Plane(0).PixelsPerRow();
      float inv_factor = quantizer.GetInvDcStep(c);
      float cfl_factor = c == 1 ? 0.0f : cmap.DCFactors()[c] * y_dc_step;
      for (size_t y = 0; y < r.ysize(); y++) {
        const float* row = r.ConstPlaneRow(dc, c, y);
        const int32_t* qrow_y = quant_dc.PlaneRow(1, y);
        int32_t* qrow = quant_dc.PlaneRow(c, y);
        for (size_t x = 0; x < r.xsize(); x++) {
          qrow[x] = roundf((row[x] - qrow_y[x] * cfl_factor) * inv_factor);
          int64_t left = (x ? qrow[x - 1] : y ? *(qrow + x - onerow) : 0);
          int64_t top = (y ? *(qrow + x - onerow) : left);
          int64_t topleft = (x && y ? *(qrow + x - 1 - onerow) : left);
          int32_t guess = ClampedGradient(top, left, topleft);
          uint32_t gradprop = Clamp1(kGradRangeMid + top + left - topleft,
                                     kGradRangeMin, kGradRangeMax);
          int32_t residual = qrow[x] - guess;
          uint32_t ctx_id = kGradientContextLut[gradprop];
          dc_tokens[group_index].push_back(Token(ctx_id, PackSigned(residual)));
        }
      }
    }
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, num_dc_groups, ThreadPool::NoInit,
                                compute_dc_tokens, "Compute DC tokens"));

  // Compute control fields tokens.
  std::vector<size_t> num_ac_blocks(num_dc_groups);
  std::vector<std::vector<Token>> ac_meta_tokens(num_dc_groups);
  auto compute_ac_meta_tokens = [&](int group_index, int /* thread */) {
    const Rect r = block_rect(group_index, xsize_dc_groups, kGroupDim);
    Rect cr(r.x0() >> 3, r.y0() >> 3, (r.xsize() + 7) >> 3,
            (r.ysize() + 7) >> 3);
    // YtoX and YtoB tokens.
    for (size_t c = 0; c < 2; ++c) {
      const ImageSB& cfl_map = (c == 0 ? cmap.ytox_map : cmap.ytob_map);
      ImageI cfl_imap(cr.xsize(), cr.ysize());
      ConvertPlaneAndClamp(cr, cfl_map, Rect(cfl_imap), &cfl_imap);
      const intptr_t onerow = cfl_imap.PixelsPerRow();
      for (size_t y = 0; y < cr.ysize(); y++) {
        const int32_t* row = cfl_imap.ConstRow(y);
        for (size_t x = 0; x < cr.xsize(); x++) {
          int64_t left = (x ? row[x - 1] : y ? *(row + x - onerow) : 0);
          int64_t top = (y ? *(row + x - onerow) : left);
          int64_t topleft = (x && y ? *(row + x - 1 - onerow) : left);
          int32_t guess = ClampedGradient(top, left, topleft);
          int32_t residual = row[x] - guess;
          uint32_t ctx_id = 2u - c;
          Token token(ctx_id, PackSigned(residual));
          ac_meta_tokens[group_index].push_back(token);
        }
      }
    }
    // Ac strategy tokens.
    size_t num = 0;
    int32_t left = 0;
    for (size_t y = 0; y < r.ysize(); y++) {
      AcStrategyRow row_acs = ac_strategy.ConstRow(r, y);
      for (size_t x = 0; x < r.xsize(); x++) {
        if (!row_acs[x].IsFirstBlock()) continue;
        int32_t cur = row_acs[x].RawStrategy();
        uint32_t ctx_id = (left > 11 ? 7 : left > 5 ? 8 : left > 3 ? 9 : 10);
        Token token(ctx_id, PackSigned(cur));
        ac_meta_tokens[group_index].push_back(token);
        left = cur;
        num++;
      }
    }
    num_ac_blocks[group_index] = num;
    // Quant field tokens.
    left = ac_strategy.ConstRow(r, 0)[0].RawStrategy();
    for (size_t y = 0; y < r.ysize(); y++) {
      AcStrategyRow row_acs = ac_strategy.ConstRow(r, y);
      const int32_t* row_qf = r.ConstRow(raw_quant_field, y);
      for (size_t x = 0; x < r.xsize(); x++) {
        if (!row_acs[x].IsFirstBlock()) continue;
        size_t cur = row_qf[x] - 1;
        int32_t residual = cur - left;
        uint32_t ctx_id = (left > 11 ? 3 : left > 5 ? 4 : left > 3 ? 5 : 6);
        Token token(ctx_id, PackSigned(residual));
        ac_meta_tokens[group_index].push_back(token);
        left = cur;
      }
    }
    // EPF tokens.
    for (size_t i = 0; i < r.ysize() * r.xsize(); ++i) {
      ac_meta_tokens[group_index].push_back(Token(0, PackSigned(4)));
    }
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, num_dc_groups, ThreadPool::NoInit,
                                compute_ac_meta_tokens,
                                "Compute AC Metadata tokens"));

  // Write DC global and compute DC and control fields histograms.
  HistogramBuilder dc_builder(kNumDCContexts);
  EntropyEncodingData dc_code;
  std::vector<uint8_t> dc_context_map;
  {
    BitWriter* group_writer = get_output(0);
    BitWriter::Allotment allotment(group_writer, 1024);
    group_writer->Write(1, 1);  // default quant dc
    quantizer.Encode(group_writer);
    group_writer->Write(1, 1);  // default BlockCtxMap
    ColorCorrelationMapEncodeDC(&cmap, group_writer);
    group_writer->Write(1, 1);  // not an empty tree
    allotment.Reclaim(group_writer);
    std::vector<Token> tree_tokens(kContextTreeTokens,
                                   kContextTreeTokens + kNumContextTreeTokens);
    tree_tokens[1].value = PackSigned(1 + num_dc_groups);
    WriteHistogramsAndTokens(kNumTreeContexts, tree_tokens, group_writer);
    for (const auto& t : dc_tokens) dc_builder.AddTokens(t);
    for (const auto& t : ac_meta_tokens) dc_builder.AddTokens(t);
    dc_builder.BuildAndStoreEntropyCodes(&dc_code, &dc_context_map,
                                         group_writer);
  }

  // Write DC groups and control fields.
  const auto process_dc_group = [&](const uint32_t group_index,
                                    const size_t thread) {
    BitWriter* writer = get_output(group_index + 1);
    {
      BitWriter::Allotment allotment(writer, 1024);
      writer->Write(2, 0);  // extra_dc_precision
      writer->Write(4, 3);  // use global tree, default wp, no transforms
      allotment.Reclaim(writer);
      WriteTokens(dc_tokens[group_index], dc_code, dc_context_map, writer);
    }
    {
      const Rect r = block_rect(group_index, xsize_dc_groups, kGroupDim);
      BitWriter::Allotment allotment(writer, 1024);
      size_t nb_bits = CeilLog2Nonzero(r.xsize() * r.ysize());
      if (nb_bits != 0) writer->Write(nb_bits, num_ac_blocks[group_index] - 1);
      writer->Write(4, 3);  // use global tree, default wp, no transforms
      allotment.Reclaim(writer);
      WriteTokens(ac_meta_tokens[group_index], dc_code, dc_context_map, writer);
    }
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, num_dc_groups, ThreadPool::NoInit,
                                process_dc_group, "EncodeDCGroup"));

  // Compute AC coefficient orders.
  auto used_orders_info = ComputeUsedOrders(ac_strategy, Rect(raw_quant_field));
  std::vector<coeff_order_t> coeff_orders(kCoeffOrderMaxSize);
  ComputeCoeffOrder(ac_coeffs, ac_strategy, frame_dim, used_orders_info.second,
                    used_orders_info.first, &coeff_orders[0]);

  // Compute AC tokens.
  std::vector<std::vector<Token>> ac_tokens(num_groups);
  std::vector<Image3I> num_nzeroes;
  const auto tokenize_group_init = [&](const size_t num_threads) {
    num_nzeroes.resize(num_threads);
    for (size_t t = 0; t < num_threads; ++t) {
      num_nzeroes[t] = Image3I(kGroupDimInBlocks, kGroupDimInBlocks);
    }
    return true;
  };
  const auto tokenize_group = [&](const uint32_t group_index,
                                  const size_t thread) {
    const Rect r = block_rect(group_index, xsize_groups, kGroupDimInBlocks);
    JXL_ASSERT(ac_coeffs.Type() == ACType::k32);
    const int32_t* JXL_RESTRICT ac_rows[3] = {
        ac_coeffs.PlaneRow(0, group_index, 0).ptr32,
        ac_coeffs.PlaneRow(1, group_index, 0).ptr32,
        ac_coeffs.PlaneRow(2, group_index, 0).ptr32,
    };
    TokenizeCoefficients(&coeff_orders[0], r, ac_rows, ac_strategy,
                         &num_nzeroes[thread], &ac_tokens[group_index]);
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, num_groups, tokenize_group_init,
                                tokenize_group, "TokenizeGroup"));

  // Write AC global and compute AC histograms.
  std::vector<uint8_t> context_map;
  EntropyEncodingData codes;
  {
    BitWriter* group_writer = get_output(global_ac_index);
    BitWriter::Allotment allotment(group_writer, 1024);
    group_writer->Write(1, 1);  // all default quant matrices
    size_t num_histo_bits = CeilLog2Nonzero(num_groups);
    if (num_histo_bits != 0) group_writer->Write(num_histo_bits, 0);
    group_writer->Write(2, 3);
    group_writer->Write(kNumOrders, used_orders_info.second);
    allotment.Reclaim(group_writer);
    EncodeCoeffOrders(used_orders_info.second, &coeff_orders[0], group_writer);
    BuildAndEncodeHistograms(kNumACContexts, ac_tokens, &codes, &context_map,
                             group_writer);
  }

  // Write AC groups.
  const auto process_group = [&](const uint32_t group_index,
                                 const size_t thread) {
    BitWriter* writer = get_output(2 + num_dc_groups + group_index);
    WriteTokens(ac_tokens[group_index], codes, context_map, writer);
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, num_groups, ThreadPool::NoInit,
                                process_group, "EncodeGroupCoefficients"));

  // Zero pad all sections.
  for (BitWriter& bw : group_codes) {
    BitWriter::Allotment allotment(&bw, 8);
    bw.ZeroPadToByte();  // end of group.
    allotment.Reclaim(&bw);
  }

  // Write TOC and assemble bit stream.
  JXL_RETURN_IF_ERROR(WriteGroupOffsets(group_codes, nullptr, writer));
  writer->AppendByteAligned(group_codes);

  return true;
}

}  // namespace jxl
