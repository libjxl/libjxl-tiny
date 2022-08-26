// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/enc_chroma_from_luma.h"

#include <float.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "encoder/enc_chroma_from_luma.cc"
#include <hwy/aligned_allocator.h>
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "encoder/base/data_parallel.h"
#include "encoder/base/status.h"
#include "encoder/common.h"
#include "encoder/enc_transforms-inl.h"
#include "encoder/image_ops.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Abs;
using hwy::HWY_NAMESPACE::Ge;
using hwy::HWY_NAMESPACE::GetLane;
using hwy::HWY_NAMESPACE::IfThenElse;
using hwy::HWY_NAMESPACE::Lt;

static HWY_FULL(float) df;

int32_t FindBestMultiplier(const float* values_m, const float* values_s,
                           size_t num, float base, float distance_mul) {
  if (num == 0) {
    return 0;
  }
  float x;
  auto ca = Zero(df);
  auto cb = Zero(df);
  const auto inv_color_factor = Set(df, kInvColorFactor);
  const auto base_v = Set(df, base);
  for (size_t i = 0; i < num; i += Lanes(df)) {
    // color residual = ax + b
    const auto a = Mul(inv_color_factor, Load(df, values_m + i));
    const auto b =
        Sub(Mul(base_v, Load(df, values_m + i)), Load(df, values_s + i));
    ca = MulAdd(a, a, ca);
    cb = MulAdd(a, b, cb);
  }
  // + distance_mul * x^2 * num
  x = -GetLane(SumOfLanes(df, cb)) /
      (GetLane(SumOfLanes(df, ca)) + num * distance_mul * 0.5f);
  return std::max(-128.0f, std::min(127.0f, roundf(x)));
}

void InitDCStorage(size_t num_blocks, ImageF* dc_values) {
  // First row: Y channel
  // Second row: X channel
  // Third row: Y channel
  // Fourth row: B channel
  *dc_values = ImageF(RoundUpTo(num_blocks, Lanes(df)), 4);

  JXL_ASSERT(dc_values->xsize() != 0);
  // Zero-fill the last lanes
  for (size_t y = 0; y < 4; y++) {
    for (size_t x = dc_values->xsize() - Lanes(df); x < dc_values->xsize();
         x++) {
      dc_values->Row(y)[x] = 0;
    }
  }
}

void ComputeDC(const ImageF& dc_values, int32_t* dc_x, int32_t* dc_b) {
  constexpr float kDistanceMultiplierDC = 1e-5f;
  const float* JXL_RESTRICT dc_values_yx = dc_values.Row(0);
  const float* JXL_RESTRICT dc_values_x = dc_values.Row(1);
  const float* JXL_RESTRICT dc_values_yb = dc_values.Row(2);
  const float* JXL_RESTRICT dc_values_b = dc_values.Row(3);
  *dc_x = FindBestMultiplier(dc_values_yx, dc_values_x, dc_values.xsize(), 0.0f,
                             kDistanceMultiplierDC);
  *dc_b = FindBestMultiplier(dc_values_yb, dc_values_b, dc_values.xsize(), 1.0f,
                             kDistanceMultiplierDC);
}

void ComputeTile(const Image3F& opsin, const DequantMatrices& dequant,
                 const Rect& r, ImageSB* map_x, ImageSB* map_b,
                 ImageF* dc_values, float* mem) {
  size_t xsize_blocks = opsin.xsize() / kBlockDim;
  constexpr float kDistanceMultiplierAC = 1e-3f;

  const size_t y0 = r.y0();
  const size_t x0 = r.x0();
  const size_t x1 = r.x0() + r.xsize();
  const size_t y1 = r.y0() + r.ysize();

  int ty = y0 / kColorTileDimInBlocks;
  int tx = x0 / kColorTileDimInBlocks;

  int8_t* JXL_RESTRICT row_out_x = map_x->Row(ty);
  int8_t* JXL_RESTRICT row_out_b = map_b->Row(ty);

  float* JXL_RESTRICT dc_values_yx = dc_values->Row(0);
  float* JXL_RESTRICT dc_values_x = dc_values->Row(1);
  float* JXL_RESTRICT dc_values_yb = dc_values->Row(2);
  float* JXL_RESTRICT dc_values_b = dc_values->Row(3);

  // All are aligned.
  float* HWY_RESTRICT block_y = mem;
  float* HWY_RESTRICT block_x = block_y + AcStrategy::kMaxCoeffArea;
  float* HWY_RESTRICT block_b = block_x + AcStrategy::kMaxCoeffArea;
  float* HWY_RESTRICT coeffs_yx = block_b + AcStrategy::kMaxCoeffArea;
  float* HWY_RESTRICT coeffs_x = coeffs_yx + kColorTileDim * kColorTileDim;
  float* HWY_RESTRICT coeffs_yb = coeffs_x + kColorTileDim * kColorTileDim;
  float* HWY_RESTRICT coeffs_b = coeffs_yb + kColorTileDim * kColorTileDim;
  float* HWY_RESTRICT scratch_space = coeffs_b + kColorTileDim * kColorTileDim;

  // Small (~256 bytes each)
  HWY_ALIGN_MAX float
      dc_y[AcStrategy::kMaxCoeffBlocks * AcStrategy::kMaxCoeffBlocks] = {};
  HWY_ALIGN_MAX float
      dc_x[AcStrategy::kMaxCoeffBlocks * AcStrategy::kMaxCoeffBlocks] = {};
  HWY_ALIGN_MAX float
      dc_b[AcStrategy::kMaxCoeffBlocks * AcStrategy::kMaxCoeffBlocks] = {};
  size_t num_ac = 0;

  for (size_t y = y0; y < y1; ++y) {
    const float* JXL_RESTRICT row_y = opsin.ConstPlaneRow(1, y * kBlockDim);
    const float* JXL_RESTRICT row_x = opsin.ConstPlaneRow(0, y * kBlockDim);
    const float* JXL_RESTRICT row_b = opsin.ConstPlaneRow(2, y * kBlockDim);
    size_t stride = opsin.PixelsPerRow();

    for (size_t x = x0; x < x1; x++) {
      AcStrategy acs = AcStrategy::FromRawStrategy(AcStrategy::Type::DCT);
      TransformFromPixels(acs.Strategy(), row_y + x * kBlockDim, stride,
                          block_y, scratch_space);
      DCFromLowestFrequencies(acs.Strategy(), block_y, dc_y, 1);
      TransformFromPixels(acs.Strategy(), row_x + x * kBlockDim, stride,
                          block_x, scratch_space);
      DCFromLowestFrequencies(acs.Strategy(), block_x, dc_x, 1);
      TransformFromPixels(acs.Strategy(), row_b + x * kBlockDim, stride,
                          block_b, scratch_space);
      DCFromLowestFrequencies(acs.Strategy(), block_b, dc_b, 1);
      const float* const JXL_RESTRICT qm_x =
          dequant.InvMatrix(acs.Strategy(), 0);
      const float* const JXL_RESTRICT qm_b =
          dequant.InvMatrix(acs.Strategy(), 2);

      // Copy DCs in dc_values.
      dc_values_yx[y * xsize_blocks + x] = dc_y[0];
      dc_values_x[y * xsize_blocks + x] = dc_x[0];
      dc_values_yb[y * xsize_blocks + x] = dc_y[0];
      dc_values_b[y * xsize_blocks + x] = dc_b[0];

      // Zero out DCs. This introduces terms in the optimization loop that
      // don't affect the result, as they are all 0, but allow for simpler
      // SIMDfication.
      block_y[0] = 0;
      block_x[0] = 0;
      block_b[0] = 0;
      for (size_t i = 0; i < 64; i += Lanes(df)) {
        const auto b_y = Load(df, block_y + i);
        const auto b_x = Load(df, block_x + i);
        const auto b_b = Load(df, block_b + i);
        const auto qqm_x = Load(df, qm_x + i);
        const auto qqm_b = Load(df, qm_b + i);
        Store(Mul(b_y, qqm_x), df, coeffs_yx + num_ac);
        Store(Mul(b_x, qqm_x), df, coeffs_x + num_ac);
        Store(Mul(b_y, qqm_b), df, coeffs_yb + num_ac);
        Store(Mul(b_b, qqm_b), df, coeffs_b + num_ac);
        num_ac += Lanes(df);
      }
    }
  }
  JXL_CHECK(num_ac % Lanes(df) == 0);
  row_out_x[tx] = FindBestMultiplier(coeffs_yx, coeffs_x, num_ac, 0.0f,
                                     kDistanceMultiplierAC);
  row_out_b[tx] = FindBestMultiplier(coeffs_yb, coeffs_b, num_ac, 1.0f,
                                     kDistanceMultiplierAC);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(InitDCStorage);
HWY_EXPORT(ComputeDC);
HWY_EXPORT(ComputeTile);

struct CfLHeuristics {
  void Init(const Image3F& opsin);

  void PrepareForThreads(size_t num_threads) {
    mem = hwy::AllocateAligned<float>(num_threads * kItemsPerThread);
  }

  void ComputeTile(const Rect& r, const Image3F& opsin,
                   const DequantMatrices& dequant, size_t thread,
                   ColorCorrelationMap* cmap);

  void ComputeDC(ColorCorrelationMap* cmap);

  ImageF dc_values;
  hwy::AlignedFreeUniquePtr<float[]> mem;

  // Working set is too large for stack; allocate dynamically.
  constexpr static size_t kItemsPerThread =
      AcStrategy::kMaxCoeffArea * 3        // Blocks
      + kColorTileDim * kColorTileDim * 4  // AC coeff storage
      + AcStrategy::kMaxCoeffArea * 2;     // Scratch space
};

void CfLHeuristics::Init(const Image3F& opsin) {
  size_t xsize_blocks = opsin.xsize() / kBlockDim;
  size_t ysize_blocks = opsin.ysize() / kBlockDim;
  HWY_DYNAMIC_DISPATCH(InitDCStorage)
  (xsize_blocks * ysize_blocks, &dc_values);
}

void CfLHeuristics::ComputeTile(const Rect& r, const Image3F& opsin,
                                const DequantMatrices& dequant, size_t thread,
                                ColorCorrelationMap* cmap) {
  HWY_DYNAMIC_DISPATCH(ComputeTile)
  (opsin, dequant, r, &cmap->ytox_map, &cmap->ytob_map, &dc_values,
   mem.get() + thread * kItemsPerThread);
}

void CfLHeuristics::ComputeDC(ColorCorrelationMap* cmap) {
  int32_t ytob_dc = 0;
  int32_t ytox_dc = 0;
  HWY_DYNAMIC_DISPATCH(ComputeDC)(dc_values, &ytox_dc, &ytob_dc);
  cmap->SetYToBDC(ytob_dc);
  cmap->SetYToXDC(ytox_dc);
}

Status ComputeColorCorrelationMap(const Image3F& opsin,
                                  const DequantMatrices& dequant,
                                  ThreadPool* pool, ColorCorrelationMap* cmap) {
  size_t xsize_blocks = DivCeil(opsin.xsize(), kBlockDim);
  size_t ysize_blocks = DivCeil(opsin.ysize(), kBlockDim);
  size_t xsize_tiles = DivCeil(xsize_blocks, kColorTileDimInBlocks);
  size_t ysize_tiles = DivCeil(ysize_blocks, kColorTileDimInBlocks);
  CfLHeuristics cfl_heuristics;
  cfl_heuristics.Init(opsin);
  auto process_tile_cfl = [&](const uint32_t tid, const size_t thread) {
    size_t tx = tid % xsize_tiles;
    size_t ty = tid / xsize_tiles;
    size_t by0 = ty * kColorTileDimInBlocks;
    size_t by1 = std::min((ty + 1) * kColorTileDimInBlocks, ysize_blocks);
    size_t bx0 = tx * kColorTileDimInBlocks;
    size_t bx1 = std::min((tx + 1) * kColorTileDimInBlocks, xsize_blocks);
    Rect r(bx0, by0, bx1 - bx0, by1 - by0);
    cfl_heuristics.ComputeTile(r, opsin, dequant, thread, cmap);
  };
  JXL_RETURN_IF_ERROR(RunOnPool(
      pool, 0, xsize_tiles * ysize_tiles,
      [&](const size_t num_threads) {
        cfl_heuristics.PrepareForThreads(num_threads);
        return true;
      },
      process_tile_cfl, "Cfl Heuristics"));
  cfl_heuristics.ComputeDC(cmap);
  return true;
}

}  // namespace jxl
#endif  // HWY_ONCE
