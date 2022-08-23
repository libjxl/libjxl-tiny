// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/enc_adaptive_quantization.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "encoder/enc_adaptive_quantization.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "encoder/ac_strategy.h"
#include "encoder/base/compiler_specific.h"
#include "encoder/base/data_parallel.h"
#include "encoder/base/status.h"
#include "encoder/coeff_order_fwd.h"
#include "encoder/common.h"
#include "encoder/enc_group.h"
#include "encoder/fast_math-inl.h"
#include "encoder/image.h"
#include "encoder/image_ops.h"
#include "encoder/opsin_params.h"
#include "encoder/quant_weights.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::AbsDiff;
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::And;
using hwy::HWY_NAMESPACE::Max;
using hwy::HWY_NAMESPACE::Rebind;
using hwy::HWY_NAMESPACE::Sqrt;
using hwy::HWY_NAMESPACE::ZeroIfNegative;

// The following functions modulate an exponent (out_val) and return the updated
// value. Their descriptor is limited to 8 lanes for 8x8 blocks.

// Hack for mask estimation. Eventually replace this code with butteraugli's
// masking.
float ComputeMaskForAcStrategyUse(const float out_val) {
  const float kMul = 1.0f;
  const float kOffset = 0.001f;
  return kMul / (out_val + kOffset);
}

template <class D, class V>
V ComputeMask(const D d, const V out_val) {
  const auto kBase = Set(d, -0.74174993f);
  const auto kMul4 = Set(d, 3.2353257320940401f);
  const auto kMul2 = Set(d, 12.906028311180409f);
  const auto kOffset2 = Set(d, 305.04035728311436f);
  const auto kMul3 = Set(d, 5.0220313103171232f);
  const auto kOffset3 = Set(d, 2.1925739705298404f);
  const auto kOffset4 = Mul(Set(d, 0.25f), kOffset3);
  const auto kMul0 = Set(d, 0.74760422233706747f);
  const auto k1 = Set(d, 1.0f);

  // Avoid division by zero.
  const auto v1 = Max(Mul(out_val, kMul0), Set(d, 1e-3f));
  const auto v2 = Div(k1, Add(v1, kOffset2));
  const auto v3 = Div(k1, MulAdd(v1, v1, kOffset3));
  const auto v4 = Div(k1, MulAdd(v1, v1, kOffset4));
  // TODO(jyrki):
  // A log or two here could make sense. In butteraugli we have effectively
  // log(log(x + C)) for this kind of use, as a single log is used in
  // saturating visual masking and here the modulation values are exponential,
  // another log would counter that.
  return Add(kBase, MulAdd(kMul4, v4, MulAdd(kMul2, v2, Mul(kMul3, v3))));
}

// mul and mul2 represent a scaling difference between jxl and butteraugli.
static const float kSGmul = 226.0480446705883f;
static const float kSGmul2 = 1.0f / 73.377132366608819f;
static const float kLog2 = 0.693147181f;
// Includes correction factor for std::log -> log2.
static const float kSGRetMul = kSGmul2 * 18.6580932135f * kLog2;
static const float kSGVOffset = 7.14672470003f;

template <bool invert, typename D, typename V>
V RatioOfDerivativesOfCubicRootToSimpleGamma(const D d, V v) {
  // The opsin space in jxl is the cubic root of photons, i.e., v * v * v
  // is related to the number of photons.
  //
  // SimpleGamma(v * v * v) is the psychovisual space in butteraugli.
  // This ratio allows quantization to move from jxl's opsin space to
  // butteraugli's log-gamma space.
  float kEpsilon = 1e-2;
  v = ZeroIfNegative(v);
  const auto kNumMul = Set(d, kSGRetMul * 3 * kSGmul);
  const auto kVOffset = Set(d, kSGVOffset * kLog2 + kEpsilon);
  const auto kDenMul = Set(d, kLog2 * kSGmul);

  const auto v2 = Mul(v, v);

  const auto num = MulAdd(kNumMul, v2, Set(d, kEpsilon));
  const auto den = MulAdd(Mul(kDenMul, v), v2, kVOffset);
  return invert ? Div(num, den) : Div(den, num);
}

template <bool invert = false>
static float RatioOfDerivativesOfCubicRootToSimpleGamma(float v) {
  using DScalar = HWY_CAPPED(float, 1);
  auto vscalar = Load(DScalar(), &v);
  return GetLane(
      RatioOfDerivativesOfCubicRootToSimpleGamma<invert>(DScalar(), vscalar));
}

// TODO(veluca): this function computes an approximation of the derivative of
// SimpleGamma with (f(x+eps)-f(x))/eps. Consider two-sided approximation or
// exact derivatives. For reference, SimpleGamma was:
/*
template <typename D, typename V>
V SimpleGamma(const D d, V v) {
  // A simple HDR compatible gamma function.
  const auto mul = Set(d, kSGmul);
  const auto kRetMul = Set(d, kSGRetMul);
  const auto kRetAdd = Set(d, kSGmul2 * -20.2789020414f);
  const auto kVOffset = Set(d, kSGVOffset);

  v *= mul;

  // This should happen rarely, but may lead to a NaN, which is rather
  // undesirable. Since negative photons don't exist we solve the NaNs by
  // clamping here.
  // TODO(veluca): with FastLog2f, this no longer leads to NaNs.
  v = ZeroIfNegative(v);
  return kRetMul * FastLog2f(d, v + kVOffset) + kRetAdd;
}
*/

template <class D, class V>
V GammaModulation(const D d, const size_t x, const size_t y,
                  const ImageF& xyb_x, const ImageF& xyb_y, const V out_val) {
  const float kBias = 0.16f;
  JXL_DASSERT(kBias > kOpsinAbsorbanceBias[0]);
  JXL_DASSERT(kBias > kOpsinAbsorbanceBias[1]);
  JXL_DASSERT(kBias > kOpsinAbsorbanceBias[2]);
  auto overall_ratio = Zero(d);
  auto bias = Set(d, kBias);
  auto half = Set(d, 0.5f);
  for (size_t dy = 0; dy < 8; ++dy) {
    const float* const JXL_RESTRICT row_in_x = xyb_x.Row(y + dy);
    const float* const JXL_RESTRICT row_in_y = xyb_y.Row(y + dy);
    for (size_t dx = 0; dx < 8; dx += Lanes(d)) {
      const auto iny = Add(Load(d, row_in_y + x + dx), bias);
      const auto inx = Load(d, row_in_x + x + dx);
      const auto r = Sub(iny, inx);
      const auto g = Add(iny, inx);
      const auto ratio_r =
          RatioOfDerivativesOfCubicRootToSimpleGamma</*invert=*/true>(d, r);
      const auto ratio_g =
          RatioOfDerivativesOfCubicRootToSimpleGamma</*invert=*/true>(d, g);
      const auto avg_ratio = Mul(half, Add(ratio_r, ratio_g));

      overall_ratio = Add(overall_ratio, avg_ratio);
    }
  }
  overall_ratio = Mul(SumOfLanes(d, overall_ratio), Set(d, 1.0f / 64));
  // ideally -1.0, but likely optimal correction adds some entropy, so slightly
  // less than that.
  // ln(2) constant folded in because we want std::log but have FastLog2f.
  const auto kGam = Set(d, -0.15526878023684174f * 0.693147180559945f);
  return MulAdd(kGam, FastLog2f(d, overall_ratio), out_val);
}

template <class D, class V>
V ColorModulation(const D d, const size_t x, const size_t y,
                  const ImageF& xyb_x, const ImageF& xyb_y, const ImageF& xyb_b,
                  const double butteraugli_target, V out_val) {
  static const float kStrengthMul = 2.177823400325309;
  static const float kRedRampStart = 0.0073200141118951231;
  static const float kRedRampLength = 0.019421555948474039;
  static const float kBlueRampLength = 0.086890611400405895;
  static const float kBlueRampStart = 0.26973418507870539;
  const float strength = kStrengthMul * (1.0f - 0.25f * butteraugli_target);
  if (strength < 0) {
    return out_val;
  }
  // x values are smaller than y and b values, need to take the difference into
  // account.
  const float red_strength = strength * 5.992297772961519f;
  const float blue_strength = strength;
  {
    // Reduce some bits from areas not blue or red.
    const float offset = strength * -0.009174542291185913f;
    out_val = Add(out_val, Set(d, offset));
  }
  // Calculate how much of the 8x8 block is covered with blue or red.
  auto blue_coverage = Zero(d);
  auto red_coverage = Zero(d);
  for (size_t dy = 0; dy < 8; ++dy) {
    const float* const JXL_RESTRICT row_in_x = xyb_x.Row(y + dy);
    const float* const JXL_RESTRICT row_in_y = xyb_y.Row(y + dy);
    const float* const JXL_RESTRICT row_in_b = xyb_b.Row(y + dy);
    for (size_t dx = 0; dx < 8; dx += Lanes(d)) {
      const auto pixel_x = Max(
          Set(d, 0.0f), Sub(Load(d, row_in_x + x + dx), Set(d, kRedRampStart)));
      const auto pixel_y = Load(d, row_in_y + x + dx);
      const auto pixel_b =
          Max(Set(d, 0.0f), Sub(Load(d, row_in_b + x + dx),
                                Add(pixel_y, Set(d, kBlueRampStart))));
      const auto blue_slope = Min(pixel_b, Set(d, kBlueRampLength));
      const auto red_slope = Min(pixel_x, Set(d, kRedRampLength));
      red_coverage = Add(red_coverage, red_slope);
      blue_coverage = Add(blue_coverage, blue_slope);
    }
  }

  // Saturate when the high red or high blue coverage is above a level.
  // The idea here is that if a certain fraction of the block is red or
  // blue we consider as if it was fully red or blue.
  static const float ratio = 30.610615782142737f;  // out of 64 pixels.

  auto overall_red_coverage = SumOfLanes(d, red_coverage);
  overall_red_coverage =
      Min(overall_red_coverage, Set(d, ratio * kRedRampLength));
  overall_red_coverage =
      Mul(overall_red_coverage, Set(d, red_strength / ratio));

  auto overall_blue_coverage = SumOfLanes(d, blue_coverage);
  overall_blue_coverage =
      Min(overall_blue_coverage, Set(d, ratio * kBlueRampLength));
  overall_blue_coverage =
      Mul(overall_blue_coverage, Set(d, blue_strength / ratio));

  return Add(overall_red_coverage, Add(overall_blue_coverage, out_val));
}

// Change precision in 8x8 blocks that have high frequency content.
template <class D, class V>
V HfModulation(const D d, const size_t x, const size_t y, const ImageF& xyb,
               const V out_val) {
  // Zero out the invalid differences for the rightmost value per row.
  const Rebind<uint32_t, D> du;
  HWY_ALIGN constexpr uint32_t kMaskRight[kBlockDim] = {~0u, ~0u, ~0u, ~0u,
                                                        ~0u, ~0u, ~0u, 0};

  auto sum = Zero(d);  // sum of absolute differences with right and below

  for (size_t dy = 0; dy < 8; ++dy) {
    const float* JXL_RESTRICT row_in = xyb.Row(y + dy) + x;
    const float* JXL_RESTRICT row_in_next =
        dy == 7 ? row_in : xyb.Row(y + dy + 1) + x;

    // In SCALAR, there is no guarantee of having extra row padding.
    // Hence, we need to ensure we don't access pixels outside the row itself.
    // In SIMD modes, however, rows are padded, so it's safe to access one
    // garbage value after the row. The vector then gets masked with kMaskRight
    // to remove the influence of that value.
#if HWY_TARGET != HWY_SCALAR
    for (size_t dx = 0; dx < 8; dx += Lanes(d)) {
#else
    for (size_t dx = 0; dx < 7; dx += Lanes(d)) {
#endif
      const auto p = Load(d, row_in + dx);
      const auto pr = LoadU(d, row_in + dx + 1);
      const auto mask = BitCast(d, Load(du, kMaskRight + dx));
      sum = Add(sum, And(mask, AbsDiff(p, pr)));

      const auto pd = Load(d, row_in_next + dx);
      sum = Add(sum, AbsDiff(p, pd));
    }
  }

  sum = SumOfLanes(d, sum);
  return MulAdd(sum, Set(d, -2.0052193233688884f / 112), out_val);
}

void PerBlockModulations(const float butteraugli_target, const ImageF& xyb_x,
                         const ImageF& xyb_y, const ImageF& xyb_b,
                         const float scale, const Rect& rect, ImageF* out) {
  JXL_ASSERT(SameSize(xyb_x, xyb_y));
  JXL_ASSERT(DivCeil(xyb_x.xsize(), kBlockDim) == out->xsize());
  JXL_ASSERT(DivCeil(xyb_x.ysize(), kBlockDim) == out->ysize());

  float base_level = 0.5f * scale;
  float kDampenRampStart = 7.0f;
  float kDampenRampEnd = 14.0f;
  float dampen = 1.0f;
  if (butteraugli_target >= kDampenRampStart) {
    dampen = 1.0f - ((butteraugli_target - kDampenRampStart) /
                     (kDampenRampEnd - kDampenRampStart));
    if (dampen < 0) {
      dampen = 0;
    }
  }
  const float mul = scale * dampen;
  const float add = (1.0f - dampen) * base_level;
  for (size_t iy = rect.y0(); iy < rect.y0() + rect.ysize(); iy++) {
    const size_t y = iy * 8;
    float* const JXL_RESTRICT row_out = out->Row(iy);
    const HWY_CAPPED(float, kBlockDim) df;
    for (size_t ix = rect.x0(); ix < rect.x0() + rect.xsize(); ix++) {
      size_t x = ix * 8;
      auto out_val = Set(df, row_out[ix]);
      out_val = ComputeMask(df, out_val);
      out_val = HfModulation(df, x, y, xyb_y, out_val);
      out_val = ColorModulation(df, x, y, xyb_x, xyb_y, xyb_b,
                                butteraugli_target, out_val);
      out_val = GammaModulation(df, x, y, xyb_x, xyb_y, out_val);
      // We want multiplicative quantization field, so everything
      // until this point has been modulating the exponent.
      row_out[ix] = FastPow2f(GetLane(out_val) * 1.442695041f) * mul + add;
    }
  }
}

template <typename D, typename V>
V MaskingSqrt(const D d, V v) {
  static const float kLogOffset = 26.481471032459346f;
  static const float kMul = 211.50759899638012f;
  const auto mul_v = Set(d, kMul * 1e8);
  const auto offset_v = Set(d, kLogOffset);
  return Mul(Set(d, 0.25f), Sqrt(MulAdd(v, Sqrt(mul_v), offset_v)));
}

float MaskingSqrt(const float v) {
  using DScalar = HWY_CAPPED(float, 1);
  auto vscalar = Load(DScalar(), &v);
  return GetLane(MaskingSqrt(DScalar(), vscalar));
}

void StoreMin4(const float v, float& min0, float& min1, float& min2,
               float& min3) {
  if (v < min3) {
    if (v < min0) {
      min3 = min2;
      min2 = min1;
      min1 = min0;
      min0 = v;
    } else if (v < min1) {
      min3 = min2;
      min2 = min1;
      min1 = v;
    } else if (v < min2) {
      min3 = min2;
      min2 = v;
    } else {
      min3 = v;
    }
  }
}

// Look for smooth areas near the area of degradation.
// If the areas are generally smooth, don't do masking.
// Output is downsampled 2x.
void FuzzyErosion(const Rect& from_rect, const ImageF& from,
                  const Rect& to_rect, ImageF* to) {
  const size_t xsize = from.xsize();
  const size_t ysize = from.ysize();
  constexpr int kStep = 1;
  static_assert(kStep == 1, "Step must be 1");
  JXL_ASSERT(to_rect.xsize() * 2 == from_rect.xsize());
  JXL_ASSERT(to_rect.ysize() * 2 == from_rect.ysize());
  for (size_t fy = 0; fy < from_rect.ysize(); ++fy) {
    size_t y = fy + from_rect.y0();
    size_t ym1 = y >= kStep ? y - kStep : y;
    size_t yp1 = y + kStep < ysize ? y + kStep : y;
    const float* rowt = from.Row(ym1);
    const float* row = from.Row(y);
    const float* rowb = from.Row(yp1);
    float* row_out = to_rect.Row(to, fy / 2);
    for (size_t fx = 0; fx < from_rect.xsize(); ++fx) {
      size_t x = fx + from_rect.x0();
      size_t xm1 = x >= kStep ? x - kStep : x;
      size_t xp1 = x + kStep < xsize ? x + kStep : x;
      float min0 = row[x];
      float min1 = row[xm1];
      float min2 = row[xp1];
      float min3 = rowt[xm1];
      // Sort the first four values.
      if (min0 > min1) std::swap(min0, min1);
      if (min0 > min2) std::swap(min0, min2);
      if (min0 > min3) std::swap(min0, min3);
      if (min1 > min2) std::swap(min1, min2);
      if (min1 > min3) std::swap(min1, min3);
      if (min2 > min3) std::swap(min2, min3);
      // The remaining five values of a 3x3 neighbourhood.
      StoreMin4(rowt[x], min0, min1, min2, min3);
      StoreMin4(rowt[xp1], min0, min1, min2, min3);
      StoreMin4(rowb[xm1], min0, min1, min2, min3);
      StoreMin4(rowb[x], min0, min1, min2, min3);
      StoreMin4(rowb[xp1], min0, min1, min2, min3);
      static const float kMulC = 0.05f;
      static const float kMul0 = 0.05f;
      static const float kMul1 = 0.05f;
      static const float kMul2 = 0.05f;
      static const float kMul3 = 0.05f;
      float v = kMulC * row[x] + kMul0 * min0 + kMul1 * min1 + kMul2 * min2 +
                kMul3 * min3;
      if (fx % 2 == 0 && fy % 2 == 0) {
        row_out[fx / 2] = v;
      } else {
        row_out[fx / 2] += v;
      }
    }
  }
}

struct AdaptiveQuantizationImpl {
  void Init(const Image3F& xyb) {
    JXL_DASSERT(xyb.xsize() % kBlockDim == 0);
    JXL_DASSERT(xyb.ysize() % kBlockDim == 0);
    const size_t xsize = xyb.xsize();
    const size_t ysize = xyb.ysize();
    aq_map = ImageF(xsize / kBlockDim, ysize / kBlockDim);
  }
  void PrepareBuffers(size_t num_threads) {
    diff_buffer = ImageF(kColorTileDim + 8, num_threads);
    for (size_t i = pre_erosion.size(); i < num_threads; i++) {
      pre_erosion.emplace_back(kColorTileDimInBlocks * 2 + 2,
                               kColorTileDimInBlocks * 2 + 2);
    }
  }

  void ComputeTile(float butteraugli_target, float scale, const Image3F& xyb,
                   const Rect& rect, const int thread, ImageF* mask) {
    const size_t xsize = xyb.xsize();
    const size_t ysize = xyb.ysize();

    // The XYB gamma is 3.0 to be able to decode faster with two muls.
    // Butteraugli's gamma is matching the gamma of human eye, around 2.6.
    // We approximate the gamma difference by adding one cubic root into
    // the adaptive quantization. This gives us a total gamma of 2.6666
    // for quantization uses.
    const float match_gamma_offset = 0.019;

    const HWY_FULL(float) df;
    const float kXMul = 23.426802998210313f;
    const auto kXMulv = Set(df, kXMul);

    size_t y_start = rect.y0() * 8;
    size_t y_end = y_start + rect.ysize() * 8;

    size_t x0 = rect.x0() * 8;
    size_t x1 = x0 + rect.xsize() * 8;
    if (x0 != 0) x0 -= 4;
    if (x1 != xyb.xsize()) x1 += 4;
    if (y_start != 0) y_start -= 4;
    if (y_end != xyb.ysize()) y_end += 4;
    pre_erosion[thread].ShrinkTo((x1 - x0) / 4, (y_end - y_start) / 4);

    // Computes image (padded to multiple of 8x8) of local pixel differences.
    // Subsample both directions by 4.
    for (size_t y = y_start; y < y_end; ++y) {
      size_t y2 = y + 1 < ysize ? y + 1 : y;
      size_t y1 = y > 0 ? y - 1 : y;

      const float* row_in = xyb.PlaneRow(1, y);
      const float* row_in1 = xyb.PlaneRow(1, y1);
      const float* row_in2 = xyb.PlaneRow(1, y2);
      const float* row_x_in = xyb.PlaneRow(0, y);
      const float* row_x_in1 = xyb.PlaneRow(0, y1);
      const float* row_x_in2 = xyb.PlaneRow(0, y2);
      float* JXL_RESTRICT row_out = diff_buffer.Row(thread);

      auto scalar_pixel = [&](size_t x) {
        const size_t x2 = x + 1 < xsize ? x + 1 : x;
        const size_t x1 = x > 0 ? x - 1 : x;
        const float base =
            0.25f * (row_in2[x] + row_in1[x] + row_in[x1] + row_in[x2]);
        const float gammac = RatioOfDerivativesOfCubicRootToSimpleGamma(
            row_in[x] + match_gamma_offset);
        float diff = gammac * (row_in[x] - base);
        diff *= diff;
        const float base_x =
            0.25f * (row_x_in2[x] + row_x_in1[x] + row_x_in[x1] + row_x_in[x2]);
        float diff_x = gammac * (row_x_in[x] - base_x);
        diff_x *= diff_x;
        diff += kXMul * diff_x;
        diff = MaskingSqrt(diff);
        if ((y % 4) != 0) {
          row_out[x - x0] += diff;
        } else {
          row_out[x - x0] = diff;
        }
      };

      size_t x = x0;
      // First pixel of the row.
      if (x0 == 0) {
        scalar_pixel(x0);
        ++x;
      }
      // SIMD
      const auto match_gamma_offset_v = Set(df, match_gamma_offset);
      const auto quarter = Set(df, 0.25f);
      for (; x + 1 + Lanes(df) < x1; x += Lanes(df)) {
        const auto in = LoadU(df, row_in + x);
        const auto in_r = LoadU(df, row_in + x + 1);
        const auto in_l = LoadU(df, row_in + x - 1);
        const auto in_t = LoadU(df, row_in2 + x);
        const auto in_b = LoadU(df, row_in1 + x);
        auto base = Mul(quarter, Add(Add(in_r, in_l), Add(in_t, in_b)));
        auto gammacv =
            RatioOfDerivativesOfCubicRootToSimpleGamma</*invert=*/false>(
                df, Add(in, match_gamma_offset_v));
        auto diff = Mul(gammacv, Sub(in, base));
        diff = Mul(diff, diff);

        const auto in_x = LoadU(df, row_x_in + x);
        const auto in_x_r = LoadU(df, row_x_in + x + 1);
        const auto in_x_l = LoadU(df, row_x_in + x - 1);
        const auto in_x_t = LoadU(df, row_x_in2 + x);
        const auto in_x_b = LoadU(df, row_x_in1 + x);
        auto base_x =
            Mul(quarter, Add(Add(in_x_r, in_x_l), Add(in_x_t, in_x_b)));
        auto diff_x = Mul(gammacv, Sub(in_x, base_x));
        diff_x = Mul(diff_x, diff_x);
        diff = MulAdd(kXMulv, diff_x, diff);
        diff = MaskingSqrt(df, diff);
        if ((y & 3) != 0) {
          diff = Add(diff, LoadU(df, row_out + x - x0));
        }
        StoreU(diff, df, row_out + x - x0);
      }
      // Scalar
      for (; x < x1; ++x) {
        scalar_pixel(x);
      }
      if (y % 4 == 3) {
        float* row_dout = pre_erosion[thread].Row((y - y_start) / 4);
        for (size_t x = 0; x < (x1 - x0) / 4; x++) {
          row_dout[x] = (row_out[x * 4] + row_out[x * 4 + 1] +
                         row_out[x * 4 + 2] + row_out[x * 4 + 3]) *
                        0.25f;
        }
      }
    }
    Rect from_rect(x0 % 8 == 0 ? 0 : 1, y_start % 8 == 0 ? 0 : 1,
                   rect.xsize() * 2, rect.ysize() * 2);
    FuzzyErosion(from_rect, pre_erosion[thread], rect, &aq_map);
    for (size_t y = 0; y < rect.ysize(); ++y) {
      const float* aq_map_row = rect.ConstRow(aq_map, y);
      float* mask_row = rect.Row(mask, y);
      for (size_t x = 0; x < rect.xsize(); ++x) {
        mask_row[x] = ComputeMaskForAcStrategyUse(aq_map_row[x]);
      }
    }
    PerBlockModulations(butteraugli_target, xyb.Plane(0), xyb.Plane(1),
                        xyb.Plane(2), scale, rect, &aq_map);
  }
  std::vector<ImageF> pre_erosion;
  ImageF aq_map;
  ImageF diff_buffer;
};

ImageF AdaptiveQuantizationMap(const float butteraugli_target,
                               const Image3F& xyb,
                               const FrameDimensions& frame_dim, float scale,
                               ThreadPool* pool, ImageF* mask) {
  AdaptiveQuantizationImpl impl;
  impl.Init(xyb);
  *mask = ImageF(frame_dim.xsize_blocks, frame_dim.ysize_blocks);
  JXL_CHECK(RunOnPool(
      pool, 0,
      DivCeil(frame_dim.xsize_blocks, kColorTileDimInBlocks) *
          DivCeil(frame_dim.ysize_blocks, kColorTileDimInBlocks),
      [&](const size_t num_threads) {
        impl.PrepareBuffers(num_threads);
        return true;
      },
      [&](const uint32_t tid, const size_t thread) {
        size_t n_enc_tiles =
            DivCeil(frame_dim.xsize_blocks, kColorTileDimInBlocks);
        size_t tx = tid % n_enc_tiles;
        size_t ty = tid / n_enc_tiles;
        size_t by0 = ty * kColorTileDimInBlocks;
        size_t by1 =
            std::min((ty + 1) * kColorTileDimInBlocks, frame_dim.ysize_blocks);
        size_t bx0 = tx * kColorTileDimInBlocks;
        size_t bx1 =
            std::min((tx + 1) * kColorTileDimInBlocks, frame_dim.xsize_blocks);
        Rect r(bx0, by0, bx1 - bx0, by1 - by0);
        impl.ComputeTile(butteraugli_target, scale, xyb, r, thread, mask);
      },
      "AQ DiffPrecompute"));

  return std::move(impl).aq_map;
}

}  // namespace

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {
HWY_EXPORT(AdaptiveQuantizationMap);

namespace {

constexpr float kDcQuantPow = 0.57f;
static const float kDcQuant = 1.12f;
static const float kAcQuant = 0.8294f;

}  // namespace

void AdjustQuantField(const AcStrategyImage& ac_strategy, const Rect& rect,
                      ImageF* quant_field) {
  // Replace the whole quant_field in non-8x8 blocks with the maximum of each
  // 8x8 block.
  size_t stride = quant_field->PixelsPerRow();
  for (size_t y = 0; y < rect.ysize(); ++y) {
    AcStrategyRow ac_strategy_row = ac_strategy.ConstRow(rect, y);
    float* JXL_RESTRICT quant_row = rect.Row(quant_field, y);
    for (size_t x = 0; x < rect.xsize(); ++x) {
      AcStrategy acs = ac_strategy_row[x];
      if (!acs.IsFirstBlock()) continue;
      JXL_ASSERT(x + acs.covered_blocks_x() <= quant_field->xsize());
      JXL_ASSERT(y + acs.covered_blocks_y() <= quant_field->ysize());
      float max = quant_row[x];
      for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
        for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
          max = std::max(quant_row[x + ix + iy * stride], max);
        }
      }
      for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
        for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
          quant_row[x + ix + iy * stride] = max;
        }
      }
    }
  }
}

float InitialQuantDC(float butteraugli_target) {
  const float kDcMul = 2.9;  // Butteraugli target where non-linearity kicks in.
  const float butteraugli_target_dc = std::max<float>(
      0.5f * butteraugli_target,
      std::min<float>(butteraugli_target,
                      kDcMul * std::pow((1.0f / kDcMul) * butteraugli_target,
                                        kDcQuantPow)));
  // We want the maximum DC value to be at most 2**15 * kInvDCQuant / quant_dc.
  // The maximum DC value might not be in the kXybRange because of inverse
  // gaborish, so we add some slack to the maximum theoretical quant obtained
  // this way (64).
  return std::min(kDcQuant / butteraugli_target_dc, 50.f);
}

ImageF InitialQuantField(const float butteraugli_target, const Image3F& opsin,
                         const FrameDimensions& frame_dim, ThreadPool* pool,
                         float rescale, ImageF* mask) {
  const float quant_ac = kAcQuant / butteraugli_target;
  return HWY_DYNAMIC_DISPATCH(AdaptiveQuantizationMap)(
      butteraugli_target, opsin, frame_dim, quant_ac * rescale, pool, mask);
}

}  // namespace jxl
#endif  // HWY_ONCE
