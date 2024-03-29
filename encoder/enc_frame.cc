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
#include "encoder/base/bits.h"
#include "encoder/base/compiler_specific.h"
#include "encoder/base/data_parallel.h"
#include "encoder/base/padded_bytes.h"
#include "encoder/base/status.h"
#include "encoder/chroma_from_luma.h"
#include "encoder/common.h"
#include "encoder/config.h"
#include "encoder/dc_group_data.h"
#include "encoder/enc_ac_strategy.h"
#include "encoder/enc_adaptive_quantization.h"
#include "encoder/enc_bit_writer.h"
#include "encoder/enc_chroma_from_luma.h"
#include "encoder/enc_cluster.h"
#include "encoder/enc_entropy_code.h"
#include "encoder/enc_group.h"
#include "encoder/enc_xyb.h"
#include "encoder/entropy_code.h"
#include "encoder/image.h"
#include "encoder/quant_weights.h"
#include "encoder/static_entropy_codes.h"

namespace jxl {
namespace {

struct ImageDim {
  ImageDim(size_t xs, size_t ys)
      : xsize(xs),
        ysize(ys),
        xsize_blocks(DivCeil(xsize, kBlockDim)),
        ysize_blocks(DivCeil(ysize, kBlockDim)),
        xsize_tiles(DivCeil(xsize, kTileDim)),
        ysize_tiles(DivCeil(ysize, kTileDim)),
        xsize_groups(DivCeil(xsize, kGroupDim)),
        ysize_groups(DivCeil(ysize, kGroupDim)),
        xsize_dc_groups(DivCeil(xsize, kDCGroupDim)),
        ysize_dc_groups(DivCeil(ysize, kDCGroupDim)),
        num_groups(xsize_groups * ysize_groups),
        num_dc_groups(xsize_dc_groups * ysize_dc_groups) {}

  Rect PixelRect(size_t ix, size_t iy, size_t dimx, size_t dimy) const {
    return Rect(ix * dimx, iy * dimy, dimx, dimy, xsize, ysize);
  }
  Rect PixelRect(size_t ix, size_t iy, size_t dim) const {
    return PixelRect(ix, iy, dim, dim);
  }

  Rect BlockRect(size_t ix, size_t iy, size_t dimx, size_t dimy) const {
    return Rect(ix * dimx, iy * dimy, dimx, dimy, xsize_blocks, ysize_blocks);
  }
  Rect BlockRect(size_t ix, size_t iy, size_t dim) const {
    return BlockRect(ix, iy, dim, dim);
  }

  Rect TileRect(size_t ix, size_t iy, size_t dimx, size_t dimy) const {
    return Rect(ix * dimx, iy * dimy, dimx, dimy, xsize_tiles, ysize_tiles);
  }

  const size_t xsize;
  const size_t ysize;
  const size_t xsize_blocks;
  const size_t ysize_blocks;
  const size_t xsize_tiles;
  const size_t ysize_tiles;
  const size_t xsize_groups;
  const size_t ysize_groups;
  const size_t xsize_dc_groups;
  const size_t ysize_dc_groups;
  const size_t num_groups;
  const size_t num_dc_groups;
};

float QuantDC(float distance) {
  const float kDcQuantPow = 0.57f;
  const float kDcQuant = 1.12f;
  const float kDcMul = 2.9;  // Butteraugli target where non-linearity kicks in.
  float effective_dist = kDcMul * std::pow(distance / kDcMul, kDcQuantPow);
  effective_dist = Clamp1(effective_dist, 0.5f * distance, distance);
  return std::min(kDcQuant / effective_dist, 50.f);
}

struct DistanceParams {
  float distance;
  int global_scale;
  int quant_dc;
  float scale;
  float inv_scale;
  float scale_dc;
  uint32_t x_qm_scale;
  uint32_t epf_iters;
};

DistanceParams ComputeDistanceParams(float distance) {
  DistanceParams p;
  p.distance = distance;
  // Quantization scales.
  constexpr int kGlobalScaleDenom = 1 << 16;
  constexpr int kGlobalScaleNumerator = 4096;
  constexpr float kAcQuant = 0.8f;
  constexpr float kQuantFieldTarget = 5;
  float quant_dc = QuantDC(distance);
  float scale = kGlobalScaleDenom * kAcQuant / (distance * kQuantFieldTarget);
  scale = Clamp1(scale, 1.0f, 1.0f * (1 << 15));
  int scaled_quant_dc =
      static_cast<int>(quant_dc * kGlobalScaleNumerator * 1.6);
  p.global_scale = Clamp1(static_cast<int>(scale), 1, scaled_quant_dc);
  p.scale = p.global_scale * (1.0f / kGlobalScaleDenom);
  p.inv_scale = 1.0f / p.scale;
  p.quant_dc = static_cast<int>(quant_dc / p.scale + 0.5f);
  p.quant_dc = Clamp1(p.quant_dc, 1, 1 << 16);
  p.scale_dc = p.quant_dc * p.scale;
  // X quant matrix scale.
  p.x_qm_scale = 2;
  float x_qm_scale_steps[2] = {1.25f, 9.0f};
  for (float x_qm_scale_step : x_qm_scale_steps) {
    if (distance > x_qm_scale_step) {
      p.x_qm_scale++;
    }
  }
  if (distance < 0.299f) {
    // Favor chromacity preservation for making images appear more
    // faithful to original even with extreme (5-10x) zooming.
    p.x_qm_scale++;
  }
  // Number of edge preserving filter iters that the decoder will do.
  constexpr float kEpfThresholds[3] = {0.7, 1.5, 4.0};
  p.epf_iters = 0;
  for (size_t i = 0; i < 3; i++) {
    if (distance >= kEpfThresholds[i]) {
      p.epf_iters++;
    }
  }
  return p;
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

void WriteDCTokens(const Image3S& quant_dc, const EntropyCode& dc_code,
                   BitWriter* writer) {
  size_t nblocks = quant_dc.xsize() * quant_dc.ysize();
  size_t allotment_size = kMaxBitsPerToken * nblocks;
  const intptr_t onerow = quant_dc.Plane(0).PixelsPerRow();
  for (size_t c : {1, 0, 2}) {
    BitWriter::Allotment allotment(writer, allotment_size);
    for (size_t y = 0; y < quant_dc.ysize(); y++) {
      for (size_t x = 0; x < quant_dc.xsize(); x++) {
        const int16_t* qrow = quant_dc.PlaneRow(c, y);
        int64_t left = (x ? qrow[x - 1] : y ? *(qrow + x - onerow) : 0);
        int64_t top = (y ? *(qrow + x - onerow) : left);
        int64_t topleft = (x && y ? *(qrow + x - 1 - onerow) : left);
        int32_t guess = ClampedGradient(top, left, topleft);
        uint32_t gradprop = Clamp1(kGradRangeMid + top + left - topleft,
                                   kGradRangeMin, kGradRangeMax);
        int32_t residual = qrow[x] - guess;
        uint32_t ctx_id = kGradientContextLut[gradprop];
        Token token(ctx_id, PackSigned(residual));
#if OPTIMIZE_CODE
        writer->Write(8, dc_code.context_map[token.context]);
        writer->Write(16, token.value);
#else
        WriteToken(token, dc_code, writer);
#endif
      }
    }
    allotment.Reclaim(writer);
  }
}

size_t CountACBlocks(const AcStrategyImage& ac_strategy) {
  size_t num = 0;
  for (size_t y = 0; y < ac_strategy.ysize(); y++) {
    AcStrategyRow row_acs = ac_strategy.ConstRow(y);
    for (size_t x = 0; x < ac_strategy.xsize(); x++) {
      if (row_acs[x].IsFirstBlock()) ++num;
    }
  }
  return num;
}

void WriteACMetadataTokens(const ImageSB& ytox_map, const ImageSB& ytob_map,
                           const AcStrategyImage& ac_strategy,
                           const ImageB& raw_quant_field,
                           const EntropyCode& dc_code, BitWriter* writer) {
  size_t xsize_blocks = ac_strategy.xsize();
  size_t ysize_blocks = ac_strategy.ysize();
  size_t nblocks = xsize_blocks * ysize_blocks;
  size_t allotment_size = kMaxBitsPerToken * nblocks;
  {
    // YtoX and YtoB tokens.
    BitWriter::Allotment allotment(writer, allotment_size);
    for (size_t c = 0; c < 2; ++c) {
      const ImageSB& cfl_map = (c == 0 ? ytox_map : ytob_map);
      const intptr_t onerow = cfl_map.PixelsPerRow();
      for (size_t y = 0; y < cfl_map.ysize(); y++) {
        const int8_t* row = cfl_map.ConstRow(y);
        for (size_t x = 0; x < cfl_map.xsize(); x++) {
          int64_t left = (x ? row[x - 1] : y ? *(row + x - onerow) : 0);
          int64_t top = (y ? *(row + x - onerow) : left);
          int64_t topleft = (x && y ? *(row + x - 1 - onerow) : left);
          int32_t guess = ClampedGradient(top, left, topleft);
          int32_t residual = static_cast<int32_t>(row[x]) - guess;
          uint32_t ctx_id = 2u - c;
          Token token(ctx_id, PackSigned(residual));
#if OPTIMIZE_CODE
          writer->Write(8, dc_code.context_map[token.context]);
          writer->Write(16, token.value);
#else
          WriteToken(token, dc_code, writer);
#endif
        }
      }
    }
    allotment.Reclaim(writer);
  }
  {
    // Ac strategy tokens.
    BitWriter::Allotment allotment(writer, allotment_size);
    int32_t left = 0;
    for (size_t y = 0; y < ysize_blocks; y++) {
      AcStrategyRow row_acs = ac_strategy.ConstRow(y);
      for (size_t x = 0; x < xsize_blocks; x++) {
        if (!row_acs[x].IsFirstBlock()) continue;
        int32_t cur = row_acs[x].StrategyCode();
        uint32_t ctx_id = (left > 11 ? 7 : left > 5 ? 8 : left > 3 ? 9 : 10);
        Token token(ctx_id, PackSigned(cur));
#if OPTIMIZE_CODE
        writer->Write(8, dc_code.context_map[token.context]);
        writer->Write(16, token.value);
#else
        WriteToken(token, dc_code, writer);
#endif
        left = cur;
      }
    }
    allotment.Reclaim(writer);
  }
  {
    // Quant field tokens.
    BitWriter::Allotment allotment(writer, allotment_size);
    int32_t left = ac_strategy.ConstRow(0)[0].StrategyCode();
    for (size_t y = 0; y < ysize_blocks; y++) {
      AcStrategyRow row_acs = ac_strategy.ConstRow(y);
      const uint8_t* row_qf = raw_quant_field.ConstRow(y);
      for (size_t x = 0; x < xsize_blocks; x++) {
        if (!row_acs[x].IsFirstBlock()) continue;
        size_t cur = row_qf[x] - 1;
        int32_t residual = cur - left;
        uint32_t ctx_id = (left > 11 ? 3 : left > 5 ? 4 : left > 3 ? 5 : 6);
        Token token(ctx_id, PackSigned(residual));
#if OPTIMIZE_CODE
        writer->Write(8, dc_code.context_map[token.context]);
        writer->Write(16, token.value);
#else
        WriteToken(token, dc_code, writer);
#endif
        left = cur;
      }
    }
    allotment.Reclaim(writer);
  }
  {
    // EPF tokens.
    BitWriter::Allotment allotment(writer, allotment_size);
    for (size_t i = 0; i < nblocks; ++i) {
      Token token(0, PackSigned(4));
#if OPTIMIZE_CODE
      writer->Write(8, dc_code.context_map[token.context]);
      writer->Write(16, token.value);
#else
      WriteToken(token, dc_code, writer);
#endif
    }
    allotment.Reclaim(writer);
  }
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
    writer->Write(1, 0);  // no gaborish
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

void WriteContextTree(size_t num_dc_groups, BitWriter* writer) {
  std::vector<Token> tokens(kContextTreeTokens,
                            kContextTreeTokens + kNumContextTreeTokens);
  tokens[1].value = PackSigned(1 + num_dc_groups);
  EntropyCode code(nullptr, kNumTreeContexts, nullptr, 0);
  // NOTE: this could be done offline since we have a static context tree.
  OptimizeEntropyCode(tokens, &code);
  writer->AllocateAndWrite(1, 1);  // not an empty tree
  writer->AllocateAndWrite(1, 0);  // no lz77
  WriteEntropyCode(code, writer);
  BitWriter::Allotment allotment(writer, kMaxBitsPerToken * tokens.size());
  for (const Token& t : tokens) {
    WriteToken(t, code, writer);
  }
  allotment.Reclaim(writer);
}

void WriteDCGlobal(const DistanceParams& distp, const size_t num_dc_groups,
                   const EntropyCode& dc_code, BitWriter* writer) {
  BitWriter::Allotment allotment(writer, 1024);
  writer->Write(1, 1);  // default dequant dc
  WriteQuantScales(distp.global_scale, distp.quant_dc, writer);
  writer->Write(1, 0);   // non-default BlockCtxMap
  writer->Write(16, 0);  // no dc ctx, no qft
  {
    EntropyCode code(kCompactBlockContextMap, sizeof(kCompactBlockContextMap),
                     nullptr, 0);
    WriteContextMap(code, writer);
  }
  writer->Write(1, 1);  // default DC camp
  WriteContextTree(num_dc_groups, writer);
  writer->Write(1, 0);  // no lz77
  allotment.Reclaim(writer);
  WriteEntropyCode(dc_code, writer);
}

void WriteACGlobal(size_t num_groups, const EntropyCode& ac_code,
                   BitWriter* writer) {
  BitWriter::Allotment allotment(writer, 1024);
  writer->Write(1, 1);  // all default quant matrices
  size_t num_histo_bits = CeilLog2Nonzero(num_groups);
  if (num_histo_bits != 0) writer->Write(num_histo_bits, 0);
  writer->Write(2, 3);
  writer->Write(13, 0);  // all default coeff order
  writer->Write(1, 0);   // no lz77
  allotment.Reclaim(writer);
  WriteEntropyCode(ac_code, writer);
}

void WriteDCGroup(const DCGroupData& data, const EntropyCode& dc_code,
                  BitWriter* writer) {
  {
    BitWriter::Allotment allotment(writer, 1024);
#if OPTIMIZE_CODE
    writer->Write(8, kMaxContexts + 6);
    writer->Write(16, 12);
#else
    writer->Write(2, 0);  // extra_dc_precision
    writer->Write(4, 3);  // use global tree, default wp, no transforms
#endif
    allotment.Reclaim(writer);
  }
  WriteDCTokens(data.quant_dc, dc_code, writer);
  {
    size_t num_blocks = data.ac_strategy.xsize() * data.ac_strategy.ysize();
    size_t num_ac_blocks = CountACBlocks(data.ac_strategy);
    size_t nb_bits = CeilLog2Nonzero(num_blocks);
    BitWriter::Allotment allotment(writer, 1024);
#if OPTIMIZE_CODE
    if (nb_bits != 0) {
      writer->Write(8, kMaxContexts + nb_bits);
      writer->Write(16, num_ac_blocks - 1);
    }
    writer->Write(8, kMaxContexts + 4);
    writer->Write(16, 3);
#else
    if (nb_bits != 0) writer->Write(nb_bits, num_ac_blocks - 1);
    writer->Write(4, 3);  // use global tree, default wp, no transforms
#endif
    allotment.Reclaim(writer);
  }
  WriteACMetadataTokens(data.ytox_map, data.ytob_map, data.ac_strategy,
                        data.raw_quant_field, dc_code, writer);
}

void WriteTOC(const std::vector<BitWriter>& sections, BitWriter* output) {
  BitWriter::Allotment allotment(output, 1024 + 30 * sections.size());
  output->Write(1, 0);      // no permutation
  output->ZeroPadToByte();  // before TOC entries
  for (size_t i = 0; i < sections.size(); i++) {
    size_t section_size = DivCeil(sections[i].BitsWritten(), kBitsPerByte);
    JXL_ASSERT(section_size < (1u << 22));
    size_t offset = 0;
    bool success = false;
    static const size_t kBits[4] = {10, 14, 22, 30};
    for (size_t i = 0; i < 4; ++i) {
      if (section_size < offset + (1u << kBits[i])) {
        output->Write(2, i);
        output->Write(kBits[i], section_size - offset);
        success = true;
        break;
      }
      offset += (1u << kBits[i]);
    }
    JXL_ASSERT(success);
  }
  output->ZeroPadToByte();
  allotment.Reclaim(output);
}

void CopyAndPadImage(const Image3F& from, const Rect& r, Image3F* to) {
  size_t xsize_padded = DivCeil(r.xsize(), kBlockDim) * kBlockDim;
  size_t ysize_padded = DivCeil(r.ysize(), kBlockDim) * kBlockDim;
  to->ShrinkTo(xsize_padded, ysize_padded);
  for (size_t y = 0; y < r.ysize(); ++y) {
    for (size_t c = 0; c < 3; ++c) {
      memcpy(to->PlaneRow(c, y), r.ConstPlaneRow(from, c, y),
             r.xsize() * sizeof(float));
      float last_val = to->PlaneRow(c, y)[r.xsize() - 1];
      for (size_t x = r.xsize(); x < xsize_padded; ++x) {
        to->PlaneRow(c, y)[x] = last_val;
      }
    }
  }
  for (size_t c = 0; c < 3; ++c) {
    float* last_row = to->PlaneRow(c, r.ysize() - 1);
    for (size_t y = r.ysize(); y < ysize_padded; ++y) {
      memcpy(to->PlaneRow(c, y), last_row, xsize_padded * sizeof(float));
    }
  }
}

// These are temporary structures needed to process one tile.
struct TileProcessorMemory {
  TileProcessorMemory()
      : quant_field(kTileDimInBlocks, kTileDimInBlocks),
        masking(kTileDimInBlocks, kTileDimInBlocks),
        pre_erosion(kTileDimInBlocks * 2 + 2, kTileDimInBlocks * 2 + 2),
        diff_buffer(kTileDim + 8, 1) {
#if (OPTIMIZE_BLOCK_SIZES || OPTIMIZE_CHROMA_FROM_LUMA)
    mem_dct = hwy::AllocateAligned<float>(kMaxCoeffArea * 4);
#endif
#if OPTIMIZE_CHROMA_FROM_LUMA
    mem_cmap = hwy::AllocateAligned<float>(kTileDim * kTileDim * 4);
#endif
  }
  ImageF quant_field;
  ImageF masking;
  ImageF pre_erosion;
  ImageF diff_buffer;
#if (OPTIMIZE_BLOCK_SIZES || OPTIMIZE_CHROMA_FROM_LUMA)
  hwy::AlignedFreeUniquePtr<float[]> mem_dct;
  float* block_storage() { return mem_dct.get(); }
  float* scratch_space() { return mem_dct.get() + 3 * kMaxCoeffArea; }
#endif
#if OPTIMIZE_CHROMA_FROM_LUMA
  float* coeff_storage() { return mem_cmap.get(); }
  hwy::AlignedFreeUniquePtr<float[]> mem_cmap;
#endif
};

void ProcessTile(const Image3F& group, const Rect& tile_brect,
                 const Rect& group_brect, const Rect& group_trect,
                 const DistanceParams& distp, const DequantMatrices& matrices,
                 DCGroupData* dc_data, TileProcessorMemory* tmem) {
  ComputeAdaptiveQuantFieldTile(group, tile_brect, group_brect, distp.distance,
                                distp.inv_scale, &tmem->pre_erosion,
                                tmem->diff_buffer.Row(0), &tmem->quant_field,
                                &tmem->masking, &dc_data->raw_quant_field);
  int8_t ytox = 0, ytob = 0;
  (void)ytox;
  (void)ytob;
#if OPTIMIZE_CHROMA_FROM_LUMA
  ComputeCmapTile(group, tile_brect, matrices, &ytox, &ytob,
                  tmem->block_storage(), tmem->scratch_space(),
                  tmem->coeff_storage());
  const size_t tx = tile_brect.x0() / kTileDimInBlocks;
  const size_t ty = tile_brect.y0() / kTileDimInBlocks;
  group_trect.Row(&dc_data->ytox_map, ty)[tx] = ytox;
  group_trect.Row(&dc_data->ytob_map, ty)[tx] = ytob;
#endif
#if OPTIMIZE_BLOCK_SIZES
  for (size_t cy = 0; cy + 1 < tile_brect.ysize(); cy += 2) {
    for (size_t cx = 0; cx + 1 < tile_brect.xsize(); cx += 2) {
      FindBest16x16Transform(group, group_brect, tile_brect.x0(),
                             tile_brect.y0(), cx, cy, distp.distance, matrices,
                             tmem->quant_field, tmem->masking, ytox, ytob,
                             &dc_data->ac_strategy, tmem->block_storage(),
                             tmem->scratch_space());
    }
  }
  Rect rect(group_brect.x0() + tile_brect.x0(),
            group_brect.y0() + tile_brect.y0(), tile_brect.xsize(),
            tile_brect.ysize());
  AdjustQuantField(dc_data->ac_strategy, rect, &dc_data->raw_quant_field);
#endif
}

Status ProcessDCGroup(const Image3F& linear, size_t dc_gx, size_t dc_gy,
                      const DistanceParams& distp,
                      const DequantMatrices& matrices,
                      const EntropyCode& dc_code, const EntropyCode& ac_code,
                      ThreadPool* pool, std::vector<BitWriter>* output) {
  // Dimensions of the whole image.
  ImageDim dim(linear.xsize(), linear.ysize());
  // Rectangle of the current DC group within the image.
  Rect dc_group_rect = dim.PixelRect(dc_gx, dc_gy, kDCGroupDim);
  // Dimensions of the current DC group.
  ImageDim dc_group_dim(dc_group_rect.xsize(), dc_group_rect.ysize());

  // 514 kB total memory for DC group data (384 kB quantized DC, 64 kB AQ field
  // 64 kB AC strategy, 2 kB Chroma from luma).
  DCGroupData dc_data(dc_group_dim.xsize_blocks, dc_group_dim.ysize_blocks);
  // 192 kB for holding the XYB image for one AC stripe, can be reduced to 48 kB
  // if OPTIMIZE_CHROMA_FROM_LUMA is disabled and 16x16 tiles are used. One per
  // group processor thread.
  Image3F stripe(kGroupDim, kTileDim);
  // 3.5 kB of temporary data for DCT and holding the quantized coefficients.
  // One per group processor thread.
  GroupProcessorMemory gmem;
  // 3 kB for the number of nonzeros per block, needed for context calculation.
  // One per group processor thread.
  Image3B num_nzeros(kGroupDimInBlocks, kGroupDimInBlocks);
  // 68 kB temporary data per tile processor thread, this can be reduced to less
  // than 4 kB if OPTIMIZE_CHROMA_FROM_LUMA is disabled.
  TileProcessorMemory tmem;

  // Process AC groups, can be done in parallel, each thread fills in 1/64th of
  // the dc_data.
  for (size_t gix = 0; gix < dc_group_dim.num_groups; ++gix) {
    size_t gx = gix % dc_group_dim.xsize_groups;
    size_t gy = gix / dc_group_dim.xsize_groups;
    size_t image_gx = dc_gx * kBlockDim + gx;
    size_t image_gy = dc_gy * kBlockDim + gy;
    const size_t ac_group_idx =
        2 + dim.num_dc_groups + image_gy * dim.xsize_groups + image_gx;
    // Rectangle of the current AC group within the image.
    Rect group_rect = dim.PixelRect(image_gx, image_gy, kGroupDim);
    // Dimensions of the current AC group.
    ImageDim group_dim(group_rect.xsize(), group_rect.ysize());
    // Process AC group one 256 x kTileDim stripe at a time. These must be done
    // sequentially, because there is context dependence between the stripes.
    for (size_t ty = 0; ty < group_dim.ysize_tiles; ++ty) {
      size_t dc_ty = gy * kGroupDimInTiles + ty;
      size_t image_ty = image_gy * kGroupDimInTiles + ty;
      // Rectangle of the current AC stripe within the image.
      Rect stripe_rect = dim.PixelRect(image_gx, image_ty, kGroupDim, kTileDim);
      // Dimensions of the current AC stripe.
      ImageDim stripe_dim(stripe_rect.xsize(), stripe_rect.ysize());
      // Block-rectangle of the current AC stripe within the DC group.
      Rect stripe_brect = dc_group_dim.BlockRect(gx, dc_ty, kGroupDimInBlocks,
                                                 kTileDimInBlocks);
      // Tile-rectangle of the current AC stripe within the DC group.
      Rect stripe_trect = dc_group_dim.TileRect(gx, dc_ty, kGroupDimInTiles, 1);
      // Convert current AC stripe to XYB, pad to whole blocks if necessary.
      CopyAndPadImage(linear, stripe_rect, &stripe);
      ToXYB(&stripe);
      // Compute heuristics data one kTileDim x kTileDim tile at a time. These
      // can be done in parallel.
      for (size_t tx = 0; tx < group_dim.xsize_tiles; ++tx) {
        // Block-rectangle of the current tile within the AC stripe.
        Rect tile_brect = stripe_dim.BlockRect(tx, 0, kTileDimInBlocks);
        ProcessTile(stripe, tile_brect, stripe_brect, stripe_trect, distp,
                    matrices, &dc_data, &tmem);
      }
      // Write AC stripe to bitstream and fill in dc_data->quant_dc.
      WriteACGroup(stripe, stripe_brect, matrices, distp.scale, distp.scale_dc,
                   distp.x_qm_scale, &dc_data, ac_code, &num_nzeros, &gmem,
                   &(*output)[ac_group_idx]);
    }
  }

  // Write DC group to bitstream.
  const size_t dc_group_idx = 1 + dc_gy * dim.xsize_dc_groups + dc_gx;
  WriteDCGroup(dc_data, dc_code, &(*output)[dc_group_idx]);
  return true;
}

#if OPTIMIZE_CODE
void OptimizeSections(EntropyCode* code, BitWriter* sections, size_t num) {
  std::vector<Histogram> histograms(code->num_prefix_codes);
  for (size_t i = 0; i < num; ++i) {
    Span<const uint8_t> bytes = sections[i].GetSpan();
    JXL_ASSERT(bytes.size() % 3 == 0);
    for (size_t j = 0; j < bytes.size(); j += 3) {
      uint8_t context = bytes[j];
      uint16_t value = (bytes[j + 2] << 8) + bytes[j + 1];
      if (context < kMaxContexts) {
        JXL_ASSERT(context < histograms.size());
        uint32_t tok, nbits, bits;
        UintCoder().Encode(value, &tok, &nbits, &bits);
        JXL_ASSERT(tok < kAlphabetSize);
        histograms[context].Add(tok);
      }
    }
  }
  OptimizeEntropyCode(&histograms, code);
  for (size_t i = 0; i < num; ++i) {
    Span<const uint8_t> bytes = sections[i].GetSpan();
    BitWriter writer;
    size_t num_tokens = bytes.size() / 3;
    BitWriter::Allotment allotment(&writer, kMaxBitsPerToken * num_tokens);
    for (size_t j = 0; j < bytes.size(); j += 3) {
      uint8_t context = bytes[j];
      uint16_t value = (bytes[j + 2] << 8) + bytes[j + 1];
      if (context >= kMaxContexts) {
        writer.Write(context - kMaxContexts, value);
      } else {
        WriteToken(Token(context, value), *code, &writer);
      }
    }
    allotment.Reclaim(&writer);
    std::swap(sections[i], writer);
  }
}
#endif

void CombineSections(std::vector<BitWriter>* sections, BitWriter* writer) {
  if (sections->size() == 4) {
    // If we have only one AC group, everything must be put into one section.
    for (size_t i = 1; i < 4; ++i) {
      (*sections)[0].Append((*sections)[i]);
    }
    sections->resize(1);
  }
  WriteTOC(*sections, writer);
  writer->AppendByteAligned(sections);
}

}  // namespace

Status EncodeFrame(const float distance, const Image3F& linear,
                   ThreadPool* pool, BitWriter* writer) {
  // Pre-compute image dimension-derived values.
  ImageDim dim(linear.xsize(), linear.ysize());

  // Compute distance dependent parameters.
  DistanceParams distp = ComputeDistanceParams(distance);

  // Initialize dequantization matrices and static entropy codes.
  DequantMatrices matrices;
  EntropyCode dc_code(kDCContextMap, kNumDCContexts, kDCPrefixCodes,
                      kNumDCPrefixCodes);
  EntropyCode ac_code(kACContextMap, kNumACContexts, kACPrefixCodes,
                      kNumACPrefixCodes);

  // Allocate section writers.
  const size_t num_sections = 2 + dim.num_dc_groups + dim.num_groups;
  std::vector<BitWriter> sections(num_sections);

  // Generate DC group and AC group sections per 2048x2048 tile.
  // These can be done in parallel.
  for (size_t i = 0; i < dim.num_dc_groups; ++i) {
    size_t dc_gx = i % dim.xsize_dc_groups;
    size_t dc_gy = i / dim.xsize_dc_groups;
    JXL_RETURN_IF_ERROR(ProcessDCGroup(linear, dc_gx, dc_gy, distp, matrices,
                                       dc_code, ac_code, pool, &sections));
  }

#if OPTIMIZE_CODE
  OptimizeSections(&dc_code, &sections[1], dim.num_dc_groups);
  size_t ac_group_start = 2 + dim.num_dc_groups;
  OptimizeSections(&ac_code, &sections[ac_group_start], dim.num_groups);
#endif

  // Generate DC and AC global sections.
  WriteDCGlobal(distp, dim.num_dc_groups, dc_code, &sections[0]);
  WriteACGlobal(dim.num_groups, ac_code, &sections[1 + dim.num_dc_groups]);

  // Assemble final bitstream.
  WriteFrameHeader(distp.x_qm_scale, distp.epf_iters, writer);
  CombineSections(&sections, writer);
  return true;
}

}  // namespace jxl
