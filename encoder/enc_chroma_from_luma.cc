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

#include "encoder/base/status.h"
#include "encoder/chroma_from_luma.h"
#include "encoder/common.h"
#include "encoder/enc_transforms-inl.h"
#include "encoder/image.h"

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

void ComputeCmapTile(const Image3F& opsin, const Rect& tile_brect,
                     const DequantMatrices& dequant, int8_t* ytox, int8_t* ytob,
                     float* block_storage, float* scratch,
                     float* coeff_storage) {
  constexpr float kDistanceMultiplierAC = 1e-3f;

  const size_t y0 = tile_brect.y0();
  const size_t x0 = tile_brect.x0();
  const size_t x1 = tile_brect.x0() + tile_brect.xsize();
  const size_t y1 = tile_brect.y0() + tile_brect.ysize();

  // All are aligned.
  float* HWY_RESTRICT block_y = block_storage;
  float* HWY_RESTRICT block_x = block_y + kDCTBlockSize;
  float* HWY_RESTRICT block_b = block_x + kDCTBlockSize;
  float* HWY_RESTRICT coeffs_yx = coeff_storage;
  float* HWY_RESTRICT coeffs_x = coeffs_yx + kColorTileDim * kColorTileDim;
  float* HWY_RESTRICT coeffs_yb = coeffs_x + kColorTileDim * kColorTileDim;
  float* HWY_RESTRICT coeffs_b = coeffs_yb + kColorTileDim * kColorTileDim;
  float* HWY_RESTRICT scratch_space = scratch;

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
      TransformFromPixels(acs.Strategy(), row_x + x * kBlockDim, stride,
                          block_x, scratch_space);
      TransformFromPixels(acs.Strategy(), row_b + x * kBlockDim, stride,
                          block_b, scratch_space);
      const float* const JXL_RESTRICT qm_x =
          dequant.InvMatrix(acs.Strategy(), 0);
      const float* const JXL_RESTRICT qm_b =
          dequant.InvMatrix(acs.Strategy(), 2);

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
  *ytox = FindBestMultiplier(coeffs_yx, coeffs_x, num_ac, 0.0f,
                             kDistanceMultiplierAC);
  *ytob = FindBestMultiplier(coeffs_yb, coeffs_b, num_ac, 1.0f,
                             kDistanceMultiplierAC);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(ComputeCmapTile);

void ComputeCmapTile(const Image3F& opsin, const Rect& tile_brect,
                     const DequantMatrices& dequant, int8_t* ytox, int8_t* ytob,
                     float* block_storage, float* scratch_space,
                     float* coeff_storage) {
  HWY_DYNAMIC_DISPATCH(ComputeCmapTile)
  (opsin, tile_brect, dequant, ytox, ytob, block_storage, scratch_space,
   coeff_storage);
}

}  // namespace jxl
#endif  // HWY_ONCE
