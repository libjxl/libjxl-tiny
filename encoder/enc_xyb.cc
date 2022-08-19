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

#include "encoder/base/data_parallel.h"
#include "encoder/base/status.h"
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

static const float kB0 = 0.0037930732552754493f;
static const float kB1 = kB0;
static const float kB2 = kB0;

// Opsin absorbance matrix is now frozen.
static const float kOpsinAbsorbanceMatrix[9] = {
    kM00, kM01, kM02, kM10, kM11, kM12, kM20, kM21, kM22,
};

static const float kOpsinAbsorbanceBias[3] = {
    kB0,
    kB1,
    kB2,
};

// 4x3 matrix * 3x1 SIMD vectors
template <class V>
JXL_INLINE void OpsinAbsorbance(const V r, const V g, const V b,
                                const float* JXL_RESTRICT premul_absorb,
                                V* JXL_RESTRICT mixed0, V* JXL_RESTRICT mixed1,
                                V* JXL_RESTRICT mixed2) {
  const float* bias = &kOpsinAbsorbanceBias[0];
  const HWY_FULL(float) d;
  const size_t N = Lanes(d);
  const auto m0 = Load(d, premul_absorb + 0 * N);
  const auto m1 = Load(d, premul_absorb + 1 * N);
  const auto m2 = Load(d, premul_absorb + 2 * N);
  const auto m3 = Load(d, premul_absorb + 3 * N);
  const auto m4 = Load(d, premul_absorb + 4 * N);
  const auto m5 = Load(d, premul_absorb + 5 * N);
  const auto m6 = Load(d, premul_absorb + 6 * N);
  const auto m7 = Load(d, premul_absorb + 7 * N);
  const auto m8 = Load(d, premul_absorb + 8 * N);
  *mixed0 = MulAdd(m0, r, MulAdd(m1, g, MulAdd(m2, b, Set(d, bias[0]))));
  *mixed1 = MulAdd(m3, r, MulAdd(m4, g, MulAdd(m5, b, Set(d, bias[1]))));
  *mixed2 = MulAdd(m6, r, MulAdd(m7, g, MulAdd(m8, b, Set(d, bias[2]))));
}

// Converts one RGB vector to XYB.
template <class V>
void LinearRGBToXYB(const V r, const V g, const V b,
                    const float* JXL_RESTRICT premul_absorb,
                    float* JXL_RESTRICT valx, float* JXL_RESTRICT valy,
                    float* JXL_RESTRICT valz) {
  V mixed0, mixed1, mixed2;
  OpsinAbsorbance(r, g, b, premul_absorb, &mixed0, &mixed1, &mixed2);

  // mixed* should be non-negative even for wide-gamut, so clamp to zero.
  mixed0 = ZeroIfNegative(mixed0);
  mixed1 = ZeroIfNegative(mixed1);
  mixed2 = ZeroIfNegative(mixed2);

  const HWY_FULL(float) d;
  const size_t N = Lanes(d);
  mixed0 = CubeRootAndAdd(mixed0, Load(d, premul_absorb + 9 * N));
  mixed1 = CubeRootAndAdd(mixed1, Load(d, premul_absorb + 10 * N));
  mixed2 = CubeRootAndAdd(mixed2, Load(d, premul_absorb + 11 * N));
  const V half = Set(d, 0.5f);
  Store(Mul(half, Sub(mixed0, mixed1)), d, valx);
  Store(Mul(half, Add(mixed0, mixed1)), d, valy);
  Store(mixed2, d, valz);
}

// This is different from Butteraugli's OpsinDynamicsImage() in the sense that
// it does not contain a sensitivity multiplier based on the blurred image.
void ToXYB(const Image3F& linear, ThreadPool* pool, Image3F* JXL_RESTRICT xyb) {
  JXL_ASSERT(SameSize(linear, *xyb));

  const HWY_FULL(float) d;
  // Pre-broadcasted constants
  HWY_ALIGN float premul_absorb[MaxLanes(d) * 12];
  const size_t N = Lanes(d);
  for (size_t i = 0; i < 9; ++i) {
    const auto absorb = Set(d, kOpsinAbsorbanceMatrix[i]);
    Store(absorb, d, premul_absorb + i * N);
  }
  for (size_t i = 0; i < 3; ++i) {
    const auto neg_bias_cbrt = Set(d, -cbrtf(kOpsinAbsorbanceBias[i]));
    Store(neg_bias_cbrt, d, premul_absorb + (9 + i) * N);
  }

  const size_t xsize = linear.xsize();
  JXL_CHECK(RunOnPool(
      pool, 0, static_cast<uint32_t>(linear.ysize()), ThreadPool::NoInit,
      [&](const uint32_t task, size_t /*thread*/) {
        const size_t y = static_cast<size_t>(task);
        const float* JXL_RESTRICT row_in0 = linear.ConstPlaneRow(0, y);
        const float* JXL_RESTRICT row_in1 = linear.ConstPlaneRow(1, y);
        const float* JXL_RESTRICT row_in2 = linear.ConstPlaneRow(2, y);
        float* JXL_RESTRICT row_xyb0 = xyb->PlaneRow(0, y);
        float* JXL_RESTRICT row_xyb1 = xyb->PlaneRow(1, y);
        float* JXL_RESTRICT row_xyb2 = xyb->PlaneRow(2, y);

        for (size_t x = 0; x < xsize; x += Lanes(d)) {
          const auto in_r = Load(d, row_in0 + x);
          const auto in_g = Load(d, row_in1 + x);
          const auto in_b = Load(d, row_in2 + x);
          LinearRGBToXYB(in_r, in_g, in_b, premul_absorb, row_xyb0 + x,
                         row_xyb1 + x, row_xyb2 + x);
        }
      },
      "LinearToXYB"));
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {
HWY_EXPORT(ToXYB);
void ToXYB(const Image3F& linear, ThreadPool* pool, Image3F* JXL_RESTRICT xyb) {
  return HWY_DYNAMIC_DISPATCH(ToXYB)(linear, pool, xyb);
}
}  // namespace jxl
#endif  // HWY_ONCE
