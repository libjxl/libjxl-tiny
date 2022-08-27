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

#include "encoder/ac_strategy.h"
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
#include "encoder/enc_cluster.h"
#include "encoder/enc_coeff_order.h"
#include "encoder/enc_entropy_coder.h"
#include "encoder/enc_group.h"
#include "encoder/enc_xyb.h"
#include "encoder/gaborish.h"
#include "encoder/image.h"
#include "encoder/quant_weights.h"

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

float QuantDC(float distance) {
  const float kDcQuantPow = 0.57f;
  const float kDcQuant = 1.12f;
  const float kDcMul = 2.9;  // Butteraugli target where non-linearity kicks in.
  float effective_dist = kDcMul * std::pow(distance / kDcMul, kDcQuantPow);
  effective_dist = Clamp1(effective_dist, 0.5f * distance, distance);
  return std::min(kDcQuant / effective_dist, 50.f);
}

struct QuantScales {
  int global_scale;
  int quant_dc;
  float scale;
  float scale_dc;
};

QuantScales ComputeQuantScales(float distance) {
  constexpr int kGlobalScaleDenom = 1 << 16;
  constexpr int kGlobalScaleNumerator = 4096;
  constexpr float kAcQuant = 0.8f;
  constexpr float kQuantFieldTarget = 5;
  float quant_dc = QuantDC(distance);
  float scale = kGlobalScaleDenom * kAcQuant / (distance * kQuantFieldTarget);
  int scaled_quant_dc =
      static_cast<int>(quant_dc * kGlobalScaleNumerator * 1.6);
  QuantScales quant;
  quant.global_scale = Clamp1(static_cast<int>(scale), 1, scaled_quant_dc);
  quant.scale = quant.global_scale * (1.0f / kGlobalScaleDenom);
  quant.quant_dc = static_cast<int>(quant_dc / quant.scale + 0.5f);
  quant.quant_dc = Clamp1(quant.quant_dc, 1, 1 << 16);
  quant.scale_dc = quant.quant_dc * quant.scale;
  return quant;
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

void WriteQuantScales(int global_scale, int quant_dc, BitWriter* writer) {
  if (global_scale < 2049) {
    writer->Write(2, 0);
    writer->Write(11, global_scale - 1);
  } else if (global_scale < 4097) {
    writer->Write(2, 1);
    writer->Write(11, global_scale - 2049);
  } else if (global_scale < 8193) {
    writer->Write(2, 2);
    writer->Write(12, global_scale - 4097);
  } else {
    writer->Write(2, 3);
    writer->Write(16, global_scale - 8193);
  }
  if (quant_dc == 16) {
    writer->Write(2, 0);
  } else if (quant_dc < 33) {
    writer->Write(2, 1);
    writer->Write(5, quant_dc - 1);
  } else if (quant_dc < 257) {
    writer->Write(2, 2);
    writer->Write(8, quant_dc - 1);
  } else {
    writer->Write(2, 3);
    writer->Write(16, quant_dc - 1);
  }
}

void WriteContextMap(const std::vector<uint8_t>& context_map,
                     BitWriter* writer) {
  if (context_map.empty()) {
    return;
  }
  if (*std::max_element(context_map.begin(), context_map.end()) == 0) {
    writer->AllocateAndWrite(3, 1);  // simple code, 0 bits per entry
    return;
  }
  writer->AllocateAndWrite(3, 0);  // no simple code, no MTF, no LZ77
  std::vector<Token> tokens;
  for (size_t i = 0; i < context_map.size(); i++) {
    tokens.emplace_back(0, context_map[i]);
  }
  EntropyEncodingData codes;
  std::vector<uint8_t> dummy_context_map(1);
  WriteHistograms(BuildHistograms(1, tokens), &codes, writer);
  WriteTokens(tokens, codes, dummy_context_map, writer);
}

void WriteContextTree(size_t num_dc_groups, BitWriter* writer) {
  std::vector<Token> tokens(kContextTreeTokens,
                            kContextTreeTokens + kNumContextTreeTokens);
  tokens[1].value = PackSigned(1 + num_dc_groups);
  EntropyEncodingData codes;
  std::vector<uint8_t> context_map;
  auto histograms = BuildHistograms(kNumTreeContexts, tokens);
  ClusterHistograms(&histograms, &context_map);
  writer->AllocateAndWrite(1, 1);  // not an empty tree
  writer->AllocateAndWrite(1, 0);  // no lz77
  WriteContextMap(context_map, writer);
  WriteHistograms(histograms, &codes, writer);
  WriteTokens(tokens, codes, context_map, writer);
}

}  // namespace

Status EncodeFrame(const float distance, const Image3F& linear,
                   ThreadPool* pool, BitWriter* writer) {
  // Pre-compute image dimension-derived values.
  const size_t xsize = linear.xsize();
  const size_t ysize = linear.ysize();
  const size_t xsize_blocks = DivCeil(xsize, kBlockDim);
  const size_t ysize_blocks = DivCeil(ysize, kBlockDim);
  const size_t xsize_groups = DivCeil(xsize, kGroupDim);
  const size_t ysize_groups = DivCeil(ysize, kGroupDim);
  const size_t xsize_dc_groups = DivCeil(xsize_blocks, kGroupDim);
  const size_t ysize_dc_groups = DivCeil(ysize_blocks, kGroupDim);
  const size_t num_groups = xsize_groups * ysize_groups;
  const size_t num_dc_groups = xsize_dc_groups * ysize_dc_groups;
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
  Image3F opsin(xsize_blocks * kBlockDim, ysize_blocks * kBlockDim);
  opsin.ShrinkTo(linear.xsize(), linear.ysize());
  ToXYB(linear, pool, &opsin);
  PadImageToBlockMultipleInPlace(&opsin);

  // Compute adaptive quantization field (relies on pre-gaborish values).
  QuantScales qscales = ComputeQuantScales(distance);
  ImageF quant_field, masking;
  ImageI raw_quant_field;
  ComputeAdaptiveQuantField(opsin, distance, qscales.scale, pool, &masking,
                            &quant_field, &raw_quant_field);

  // Initialize quant weights and compute X quant matrix scale.
  DequantMatrices matrices;
  constexpr uint32_t kAcStrategyMask = 0x30df;  // up to 16x16 DCT
  JXL_CHECK(matrices.EnsureComputed(kAcStrategyMask));
  float x_qm_multiplier = std::pow(1.25f, x_qm_scale - 2.0f);

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
                                             masking, pool, matrices,
                                             &ac_strategy));
  AdjustQuantField(ac_strategy, &raw_quant_field);

  // Compute DC image and AC coefficients.
  Image3F dc(xsize_blocks, ysize_blocks);
  ACImageT<int32_t> ac_coeffs(kGroupDim * kGroupDim, num_groups);
  JXL_RETURN_IF_ERROR(RunOnPool(
      pool, 0, num_groups, ThreadPool::NoInit,
      [&](size_t group_idx, size_t _) {
        ComputeCoefficients(group_idx, opsin, raw_quant_field, matrices,
                            qscales.scale, cmap, ac_strategy, x_qm_multiplier,
                            &ac_coeffs, &dc);
      },
      "Compute coeffs"));

  // Compute DC tokens.
  std::vector<std::vector<Token>> dc_tokens(num_dc_groups);
  auto compute_dc_tokens = [&](int group_index, int /* thread */) {
    const Rect r = block_rect(group_index, xsize_dc_groups, kGroupDim);
    dc_tokens[group_index].reserve(3 * r.xsize() * r.ysize());
    Image3I quant_dc(r.xsize(), r.ysize());
    const float y_dc_step = matrices.DCQuant(1) / qscales.scale_dc;
    for (size_t c : {1, 0, 2}) {
      const intptr_t onerow = quant_dc.Plane(0).PixelsPerRow();
      float inv_factor = matrices.InvDCQuant(c) * qscales.scale_dc;
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
  EntropyEncodingData dc_code;
  std::vector<uint8_t> dc_context_map;
  {
    BitWriter* group_writer = get_output(0);
    BitWriter::Allotment allotment(group_writer, 1024);
    group_writer->Write(1, 1);  // default dequant dc
    WriteQuantScales(qscales.global_scale, qscales.quant_dc, group_writer);
    group_writer->Write(1, 1);  // default BlockCtxMap
    int32_t ytox_dc = cmap.GetYToXDC();
    int32_t ytob_dc = cmap.GetYToBDC();
    if (ytox_dc == 0 && ytob_dc == 0) {
      group_writer->Write(1, 1);  // default DC camp
    } else {
      group_writer->Write(1, 0);        // non-default DC cmap
      group_writer->Write(2, 0);        // default color factor
      group_writer->Write(16, 0);       // base ytox 0.0
      group_writer->Write(16, 0x3c00);  // base ytob 1.0
      group_writer->Write(8, ytox_dc + 128);
      group_writer->Write(8, ytob_dc + 128);
    }
    allotment.Reclaim(group_writer);
    WriteContextTree(num_dc_groups, group_writer);
    HistogramBuilder builder(kNumDCContexts);
    builder.Add(dc_tokens);
    builder.Add(ac_meta_tokens);
    group_writer->AllocateAndWrite(1, 0);  // no lz77
    ClusterHistograms(&builder.histograms, &dc_context_map);
    WriteContextMap(dc_context_map, group_writer);
    WriteHistograms(builder.histograms, &dc_code, group_writer);
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
  std::vector<coeff_order_t> coeff_orders(kCoeffOrderMaxSize);
  ComputeCoeffOrder(kAcStrategyMask, &coeff_orders[0]);

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
    group_writer->Write(13, 0);  // all default coeff order
    allotment.Reclaim(group_writer);
    auto histograms = BuildHistograms(kNumACContexts, ac_tokens);
    group_writer->AllocateAndWrite(1, 0);  // no lz77
    ClusterHistograms(&histograms, &context_map);
    WriteContextMap(context_map, group_writer);
    WriteHistograms(histograms, &codes, group_writer);
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
  {
    size_t num_sizes = group_codes.size();
    BitWriter::Allotment allotment(writer, 1024 + 30 * num_sizes);
    writer->Write(1, 0);      // no permutation
    writer->ZeroPadToByte();  // before TOC entries
    for (size_t i = 0; i < group_codes.size(); i++) {
      JXL_ASSERT(group_codes[i].BitsWritten() % kBitsPerByte == 0);
      const size_t group_size = group_codes[i].BitsWritten() / kBitsPerByte;
      size_t offset = 0;
      bool success = false;
      static const size_t kBits[4] = {10, 14, 22, 30};
      for (size_t i = 0; i < 4; ++i) {
        if (group_size < offset + (1u << kBits[i])) {
          writer->Write(2, i);
          writer->Write(kBits[i], group_size - offset);
          success = true;
          break;
        }
        offset += (1u << kBits[i]);
      }
      JXL_RETURN_IF_ERROR(success);
    }
    writer->ZeroPadToByte();
    allotment.Reclaim(writer);
  }
  writer->AppendByteAligned(group_codes);

  return true;
}

}  // namespace jxl
