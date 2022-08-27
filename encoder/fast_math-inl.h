// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

// Fast SIMD math ops (log2, encoder only, cos, erf for splines)

#if defined(ENCODER_FAST_MATH_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef ENCODER_FAST_MATH_INL_H_
#undef ENCODER_FAST_MATH_INL_H_
#else
#define ENCODER_FAST_MATH_INL_H_
#endif

#include <hwy/highway.h>

#include "encoder/common.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Abs;
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::Eq;
using hwy::HWY_NAMESPACE::Floor;
using hwy::HWY_NAMESPACE::Ge;
using hwy::HWY_NAMESPACE::GetLane;
using hwy::HWY_NAMESPACE::IfThenElse;
using hwy::HWY_NAMESPACE::IfThenZeroElse;
using hwy::HWY_NAMESPACE::Le;
using hwy::HWY_NAMESPACE::Min;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::MulAdd;
using hwy::HWY_NAMESPACE::NegMulAdd;
using hwy::HWY_NAMESPACE::Rebind;
using hwy::HWY_NAMESPACE::ShiftLeft;
using hwy::HWY_NAMESPACE::ShiftRight;
using hwy::HWY_NAMESPACE::Sub;
using hwy::HWY_NAMESPACE::Xor;

// Primary template: default to actual division.
template <typename T, class V>
struct FastDivision {
  HWY_INLINE V operator()(const V n, const V d) const { return n / d; }
};
// Partial specialization for float vectors.
template <class V>
struct FastDivision<float, V> {
  // One Newton-Raphson iteration.
  static HWY_INLINE V ReciprocalNR(const V x) {
    const auto rcp = ApproximateReciprocal(x);
    const auto sum = Add(rcp, rcp);
    const auto x_rcp = Mul(x, rcp);
    return NegMulAdd(x_rcp, rcp, sum);
  }

  V operator()(const V n, const V d) const {
#if 1  // Faster on SKX
    return Div(n, d);
#else
    return n * ReciprocalNR(d);
#endif
  }
};

// Approximates smooth functions via rational polynomials (i.e. dividing two
// polynomials). Evaluates polynomials via Horner's scheme, which is faster than
// Clenshaw recurrence for Chebyshev polynomials. LoadDup128 allows us to
// specify constants (replicated 4x) independently of the lane count.
template <size_t NP, size_t NQ, class D, class V, typename T>
HWY_INLINE HWY_MAYBE_UNUSED V EvalRationalPolynomial(const D d, const V x,
                                                     const T (&p)[NP],
                                                     const T (&q)[NQ]) {
  constexpr size_t kDegP = NP / 4 - 1;
  constexpr size_t kDegQ = NQ / 4 - 1;
  auto yp = LoadDup128(d, &p[kDegP * 4]);
  auto yq = LoadDup128(d, &q[kDegQ * 4]);
  // We use pointer arithmetic to refer to &p[(kDegP - n) * 4] to avoid a
  // compiler warning that the index is out of bounds since we are already
  // checking that it is not out of bounds with (kDegP >= n) and the access
  // will be optimized away. Similarly with q and kDegQ.
  HWY_FENCE;
  if (kDegP >= 1) yp = MulAdd(yp, x, LoadDup128(d, p + ((kDegP - 1) * 4)));
  if (kDegQ >= 1) yq = MulAdd(yq, x, LoadDup128(d, q + ((kDegQ - 1) * 4)));
  HWY_FENCE;
  if (kDegP >= 2) yp = MulAdd(yp, x, LoadDup128(d, p + ((kDegP - 2) * 4)));
  if (kDegQ >= 2) yq = MulAdd(yq, x, LoadDup128(d, q + ((kDegQ - 2) * 4)));
  HWY_FENCE;
  if (kDegP >= 3) yp = MulAdd(yp, x, LoadDup128(d, p + ((kDegP - 3) * 4)));
  if (kDegQ >= 3) yq = MulAdd(yq, x, LoadDup128(d, q + ((kDegQ - 3) * 4)));
  HWY_FENCE;
  if (kDegP >= 4) yp = MulAdd(yp, x, LoadDup128(d, p + ((kDegP - 4) * 4)));
  if (kDegQ >= 4) yq = MulAdd(yq, x, LoadDup128(d, q + ((kDegQ - 4) * 4)));
  HWY_FENCE;
  if (kDegP >= 5) yp = MulAdd(yp, x, LoadDup128(d, p + ((kDegP - 5) * 4)));
  if (kDegQ >= 5) yq = MulAdd(yq, x, LoadDup128(d, q + ((kDegQ - 5) * 4)));
  HWY_FENCE;
  if (kDegP >= 6) yp = MulAdd(yp, x, LoadDup128(d, p + ((kDegP - 6) * 4)));
  if (kDegQ >= 6) yq = MulAdd(yq, x, LoadDup128(d, q + ((kDegQ - 6) * 4)));
  HWY_FENCE;
  if (kDegP >= 7) yp = MulAdd(yp, x, LoadDup128(d, p + ((kDegP - 7) * 4)));
  if (kDegQ >= 7) yq = MulAdd(yq, x, LoadDup128(d, q + ((kDegQ - 7) * 4)));

  return FastDivision<T, V>()(yp, yq);
}

// Computes base-2 logarithm like std::log2. Undefined if negative / NaN.
// L1 error ~3.9E-6
template <class DF, class V>
V FastLog2f(const DF df, V x) {
  // 2,2 rational polynomial approximation of std::log1p(x) / std::log(2).
  HWY_ALIGN const float p[4 * (2 + 1)] = {HWY_REP4(-1.8503833400518310E-06f),
                                          HWY_REP4(1.4287160470083755E+00f),
                                          HWY_REP4(7.4245873327820566E-01f)};
  HWY_ALIGN const float q[4 * (2 + 1)] = {HWY_REP4(9.9032814277590719E-01f),
                                          HWY_REP4(1.0096718572241148E+00f),
                                          HWY_REP4(1.7409343003366853E-01f)};

  const Rebind<int32_t, DF> di;
  const auto x_bits = BitCast(di, x);

  // Range reduction to [-1/3, 1/3] - 3 integer, 2 float ops
  const auto exp_bits = Sub(x_bits, Set(di, 0x3f2aaaab));  // = 2/3
  // Shifted exponent = log2; also used to clear mantissa.
  const auto exp_shifted = ShiftRight<23>(exp_bits);
  const auto mantissa = BitCast(df, Sub(x_bits, ShiftLeft<23>(exp_shifted)));
  const auto exp_val = ConvertTo(df, exp_shifted);
  return Add(EvalRationalPolynomial(df, Sub(mantissa, Set(df, 1.0f)), p, q),
             exp_val);
}

// max relative error ~3e-7
template <class DF, class V>
V FastPow2f(const DF df, V x) {
  const Rebind<int32_t, DF> di;
  auto floorx = Floor(x);
  auto exp =
      BitCast(df, ShiftLeft<23>(Add(ConvertTo(di, floorx), Set(di, 127))));
  auto frac = Sub(x, floorx);
  auto num = Add(frac, Set(df, 1.01749063e+01));
  num = MulAdd(num, frac, Set(df, 4.88687798e+01));
  num = MulAdd(num, frac, Set(df, 9.85506591e+01));
  num = Mul(num, exp);
  auto den = MulAdd(frac, Set(df, 2.10242958e-01), Set(df, -2.22328856e-02));
  den = MulAdd(den, frac, Set(df, -1.94414990e+01));
  den = MulAdd(den, frac, Set(df, 9.85506633e+01));
  return Div(num, den);
}

// max relative error ~3e-5
template <class DF, class V>
V FastPowf(const DF df, V base, V exponent) {
  return FastPow2f(df, Mul(FastLog2f(df, base), exponent));
}

inline float FastLog2f(float f) {
  HWY_CAPPED(float, 1) D;
  return GetLane(FastLog2f(D, Set(D, f)));
}

inline float FastPow2f(float f) {
  HWY_CAPPED(float, 1) D;
  return GetLane(FastPow2f(D, Set(D, f)));
}

inline float FastPowf(float b, float e) {
  HWY_CAPPED(float, 1) D;
  return GetLane(FastPowf(D, Set(D, b), Set(D, e)));
}

// Returns cbrt(x) + add with 6 ulp max error.
// Modified from vectormath_exp.h, Apache 2 license.
// https://www.agner.org/optimize/vectorclass.zip
template <class V>
V CubeRootAndAdd(const V x, const V add) {
  const HWY_FULL(float) df;
  const HWY_FULL(int32_t) di;

  const auto kExpBias = Set(di, 0x54800000);  // cast(1.) + cast(1.) / 3
  const auto kExpMul = Set(di, 0x002AAAAA);   // shifted 1/3
  const auto k1_3 = Set(df, 1.0f / 3);
  const auto k4_3 = Set(df, 4.0f / 3);

  const auto xa = x;  // assume inputs never negative
  const auto xa_3 = Mul(k1_3, xa);

  // Multiply exponent by -1/3
  const auto m1 = BitCast(di, xa);
  // Special case for 0. 0 is represented with an exponent of 0, so the
  // "kExpBias - 1/3 * exp" below gives the wrong result. The IfThenZeroElse()
  // sets those values as 0, which prevents having NaNs in the computations
  // below.
  // TODO(eustas): use fused op
  const auto m2 = IfThenZeroElse(
      Eq(m1, Zero(di)), Sub(kExpBias, Mul((ShiftRight<23>(m1)), kExpMul)));
  auto r = BitCast(df, m2);

  // Newton-Raphson iterations
  for (int i = 0; i < 3; i++) {
    const auto r2 = Mul(r, r);
    r = NegMulAdd(xa_3, Mul(r2, r2), Mul(k4_3, r));
  }
  // Final iteration
  auto r2 = Mul(r, r);
  r = MulAdd(k1_3, NegMulAdd(xa, Mul(r2, r2), r), r);
  r2 = Mul(r, r);
  r = MulAdd(r2, x, add);

  return r;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#endif  // ENCODER_FAST_MATH_INL_H_

#if HWY_ONCE
#ifndef FAST_MATH_ONCE
#define FAST_MATH_ONCE

namespace jxl {
inline float FastLog2f(float f) { return HWY_STATIC_DISPATCH(FastLog2f)(f); }
inline float FastPow2f(float f) { return HWY_STATIC_DISPATCH(FastPow2f)(f); }
inline float FastPowf(float b, float e) {
  return HWY_STATIC_DISPATCH(FastPowf)(b, e);
}
}  // namespace jxl

#endif  // FAST_MATH_ONCE
#endif  // HWY_ONCE
