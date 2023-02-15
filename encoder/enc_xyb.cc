// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/enc_xyb.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "encoder/enc_xyb.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "encoder/base/compiler_specific.h"
#include "encoder/fast_math-inl.h"
#include "encoder/image.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::MulAdd;
using hwy::HWY_NAMESPACE::Sub;
using hwy::HWY_NAMESPACE::ZeroIfNegative;

// Parameters for opsin absorbance.
static const float kM02 = 0.078f;
static const float kM00 = 0.30f;
static const float kM01 = 1.0f - kM02 - kM00;
static const float kM12 = 0.078f;
static const float kM10 = 0.23f;
static const float kM11 = 1.0f - kM12 - kM10;
static const float kM20 = 0.24342268924547819f;
static const float kM21 = 0.20476744424496821f;
static const float kM22 = 1.0f - kM20 - kM21;
static const float kOpsinAbsorbanceBias = 0.0037930732552754493f;
static constexpr float kNegBiasCbrt = -0.15595420054f;

// This is different from Butteraugli's OpsinDynamicsImage() in the sense that
// it does not contain a sensitivity multiplier based on the blurred image.
void ToXYB(Image3F* image) {
  const HWY_FULL(float) d;
  // Pre-broadcasted constants
  const auto half = Set(d, 0.5f);
  const auto bias = Set(d, kOpsinAbsorbanceBias);
  const auto neg_bias_cbrt = Set(d, kNegBiasCbrt);
  const auto m00 = Set(d, kM00);
  const auto m01 = Set(d, kM01);
  const auto m02 = Set(d, kM02);
  const auto m10 = Set(d, kM10);
  const auto m11 = Set(d, kM11);
  const auto m12 = Set(d, kM12);
  const auto m20 = Set(d, kM20);
  const auto m21 = Set(d, kM21);
  const auto m22 = Set(d, kM22);
  const size_t xsize = image->xsize();
  const size_t ysize = image->ysize();
  for (size_t y = 0; y < ysize; ++y) {
    float* JXL_RESTRICT row0 = image->PlaneRow(0, y);
    float* JXL_RESTRICT row1 = image->PlaneRow(1, y);
    float* JXL_RESTRICT row2 = image->PlaneRow(2, y);
    for (size_t x = 0; x < xsize; x += Lanes(d)) {
      const auto r = Load(d, row0 + x);
      const auto g = Load(d, row1 + x);
      const auto b = Load(d, row2 + x);
      const auto mixed0 = MulAdd(m00, r, MulAdd(m01, g, MulAdd(m02, b, bias)));
      const auto mixed1 = MulAdd(m10, r, MulAdd(m11, g, MulAdd(m12, b, bias)));
      const auto mixed2 = MulAdd(m20, r, MulAdd(m21, g, MulAdd(m22, b, bias)));
      // mixed* should be non-negative even for wide-gamut, so clamp to zero.
      const auto tm0 = CubeRootAndAdd(ZeroIfNegative(mixed0), neg_bias_cbrt);
      const auto tm1 = CubeRootAndAdd(ZeroIfNegative(mixed1), neg_bias_cbrt);
      const auto tm2 = CubeRootAndAdd(ZeroIfNegative(mixed2), neg_bias_cbrt);
      Store(Mul(half, Sub(tm0, tm1)), d, row0 + x);
      Store(Mul(half, Add(tm0, tm1)), d, row1 + x);
      Store(tm2, d, row2 + x);
    }
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {
HWY_EXPORT(ToXYB);
void ToXYB(Image3F* image) { return HWY_DYNAMIC_DISPATCH(ToXYB)(image); }
}  // namespace jxl
#endif  // HWY_ONCE
