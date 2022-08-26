// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#if defined(ENCODER_ENC_TRANSFORMS_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef ENCODER_ENC_TRANSFORMS_INL_H_
#undef ENCODER_ENC_TRANSFORMS_INL_H_
#else
#define ENCODER_ENC_TRANSFORMS_INL_H_
#endif

#include <stddef.h>

#include <hwy/highway.h>

#include "encoder/ac_strategy.h"
#include "encoder/coeff_order_fwd.h"
#include "encoder/dct_scales.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::MulAdd;
using hwy::HWY_NAMESPACE::NegMulAdd;
using hwy::HWY_NAMESPACE::Sub;
using hwy::HWY_NAMESPACE::Vec;

#ifndef JXL_INLINE_TRANSPOSE
// Workaround for issue #42 - (excessive?) inlining causes invalid codegen.
#if defined(__arm__)
#define JXL_INLINE_TRANSPOSE HWY_NOINLINE
#else
#define JXL_INLINE_TRANSPOSE HWY_INLINE
#endif
#endif  // JXL_INLINE_TRANSPOSE

// Simple wrapper that ensures that a function will not be inlined.
template <typename T, typename... Args>
JXL_NOINLINE void NoInlineWrapper(const T& f, const Args&... args) {
  return f(args...);
}

template <size_t N>
using BlockDesc = HWY_CAPPED(float, N);

template <bool enabled>
struct TransposeSimdTag {};

// TODO(veluca): it's not super useful to have this in the SIMD namespace.
template <size_t ROWS_or_0, size_t COLS_or_0, class From, class To>
JXL_INLINE_TRANSPOSE void GenericTransposeBlock(TransposeSimdTag<false>,
                                                const From& from, const To& to,
                                                size_t ROWSp, size_t COLSp) {
  size_t ROWS = ROWS_or_0 == 0 ? ROWSp : ROWS_or_0;
  size_t COLS = COLS_or_0 == 0 ? COLSp : COLS_or_0;
  for (size_t n = 0; n < ROWS; ++n) {
    for (size_t m = 0; m < COLS; ++m) {
      to.Write(from.Read(n, m), m, n);
    }
  }
}

// TODO(veluca): AVX3?
#if HWY_CAP_GE256
constexpr bool TransposeUseSimd(size_t ROWS, size_t COLS) {
  return ROWS % 8 == 0 && COLS % 8 == 0;
}

template <size_t ROWS_or_0, size_t COLS_or_0, class From, class To>
JXL_INLINE_TRANSPOSE void GenericTransposeBlock(TransposeSimdTag<true>,
                                                const From& from, const To& to,
                                                size_t ROWSp, size_t COLSp) {
  size_t ROWS = ROWS_or_0 == 0 ? ROWSp : ROWS_or_0;
  size_t COLS = COLS_or_0 == 0 ? COLSp : COLS_or_0;
  static_assert(MaxLanes(BlockDesc<8>()) == 8, "Invalid descriptor size");
  static_assert(ROWS_or_0 % 8 == 0, "Invalid number of rows");
  static_assert(COLS_or_0 % 8 == 0, "Invalid number of columns");
  for (size_t n = 0; n < ROWS; n += 8) {
    for (size_t m = 0; m < COLS; m += 8) {
      const BlockDesc<8> d;
      auto i0 = from.LoadPart(d, n + 0, m + 0);
      auto i1 = from.LoadPart(d, n + 1, m + 0);
      auto i2 = from.LoadPart(d, n + 2, m + 0);
      auto i3 = from.LoadPart(d, n + 3, m + 0);
      auto i4 = from.LoadPart(d, n + 4, m + 0);
      auto i5 = from.LoadPart(d, n + 5, m + 0);
      auto i6 = from.LoadPart(d, n + 6, m + 0);
      auto i7 = from.LoadPart(d, n + 7, m + 0);
      // Surprisingly, this straightforward implementation (24 cycles on port5)
      // is faster than load128+insert and LoadDup128+ConcatUpperLower+blend.
      const auto q0 = InterleaveLower(d, i0, i2);
      const auto q1 = InterleaveLower(d, i1, i3);
      const auto q2 = InterleaveUpper(d, i0, i2);
      const auto q3 = InterleaveUpper(d, i1, i3);
      const auto q4 = InterleaveLower(d, i4, i6);
      const auto q5 = InterleaveLower(d, i5, i7);
      const auto q6 = InterleaveUpper(d, i4, i6);
      const auto q7 = InterleaveUpper(d, i5, i7);

      const auto r0 = InterleaveLower(d, q0, q1);
      const auto r1 = InterleaveUpper(d, q0, q1);
      const auto r2 = InterleaveLower(d, q2, q3);
      const auto r3 = InterleaveUpper(d, q2, q3);
      const auto r4 = InterleaveLower(d, q4, q5);
      const auto r5 = InterleaveUpper(d, q4, q5);
      const auto r6 = InterleaveLower(d, q6, q7);
      const auto r7 = InterleaveUpper(d, q6, q7);

      i0 = ConcatLowerLower(d, r4, r0);
      i1 = ConcatLowerLower(d, r5, r1);
      i2 = ConcatLowerLower(d, r6, r2);
      i3 = ConcatLowerLower(d, r7, r3);
      i4 = ConcatUpperUpper(d, r4, r0);
      i5 = ConcatUpperUpper(d, r5, r1);
      i6 = ConcatUpperUpper(d, r6, r2);
      i7 = ConcatUpperUpper(d, r7, r3);
      to.StorePart(d, i0, m + 0, n + 0);
      to.StorePart(d, i1, m + 1, n + 0);
      to.StorePart(d, i2, m + 2, n + 0);
      to.StorePart(d, i3, m + 3, n + 0);
      to.StorePart(d, i4, m + 4, n + 0);
      to.StorePart(d, i5, m + 5, n + 0);
      to.StorePart(d, i6, m + 6, n + 0);
      to.StorePart(d, i7, m + 7, n + 0);
    }
  }
}
#elif HWY_TARGET != HWY_SCALAR
constexpr bool TransposeUseSimd(size_t ROWS, size_t COLS) {
  return ROWS % 4 == 0 && COLS % 4 == 0;
}

template <size_t ROWS_or_0, size_t COLS_or_0, class From, class To>
JXL_INLINE_TRANSPOSE void GenericTransposeBlock(TransposeSimdTag<true>,
                                                const From& from, const To& to,
                                                size_t ROWSp, size_t COLSp) {
  size_t ROWS = ROWS_or_0 == 0 ? ROWSp : ROWS_or_0;
  size_t COLS = COLS_or_0 == 0 ? COLSp : COLS_or_0;
  static_assert(MaxLanes(BlockDesc<4>()) == 4, "Invalid descriptor size");
  static_assert(ROWS_or_0 % 4 == 0, "Invalid number of rows");
  static_assert(COLS_or_0 % 4 == 0, "Invalid number of columns");
  for (size_t n = 0; n < ROWS; n += 4) {
    for (size_t m = 0; m < COLS; m += 4) {
      const BlockDesc<4> d;
      const auto p0 = from.LoadPart(d, n + 0, m + 0);
      const auto p1 = from.LoadPart(d, n + 1, m + 0);
      const auto p2 = from.LoadPart(d, n + 2, m + 0);
      const auto p3 = from.LoadPart(d, n + 3, m + 0);

      const auto q0 = InterleaveLower(d, p0, p2);
      const auto q1 = InterleaveLower(d, p1, p3);
      const auto q2 = InterleaveUpper(d, p0, p2);
      const auto q3 = InterleaveUpper(d, p1, p3);

      const auto r0 = InterleaveLower(d, q0, q1);
      const auto r1 = InterleaveUpper(d, q0, q1);
      const auto r2 = InterleaveLower(d, q2, q3);
      const auto r3 = InterleaveUpper(d, q2, q3);

      to.StorePart(d, r0, m + 0, n + 0);
      to.StorePart(d, r1, m + 1, n + 0);
      to.StorePart(d, r2, m + 2, n + 0);
      to.StorePart(d, r3, m + 3, n + 0);
    }
  }
}
#else
constexpr bool TransposeUseSimd(size_t ROWS, size_t COLS) { return false; }
#endif

template <size_t N, size_t M, typename = void>
struct Transpose {
  template <typename From, typename To>
  static void Run(const From& from, const To& to) {
    // This does not guarantee anything, just saves from the most stupid
    // mistakes.
    JXL_DASSERT(from.Address(0, 0) != to.Address(0, 0));
    TransposeSimdTag<TransposeUseSimd(N, M)> tag;
    GenericTransposeBlock<N, M>(tag, from, to, N, M);
  }
};

// Avoid inlining and unrolling transposes for large blocks.
template <size_t N, size_t M>
struct Transpose<
    N, M, typename std::enable_if<(N >= 8 && M >= 8 && N * M >= 512)>::type> {
  template <typename From, typename To>
  static void Run(const From& from, const To& to) {
    // This does not guarantee anything, just saves from the most stupid
    // mistakes.
    JXL_DASSERT(from.Address(0, 0) != to.Address(0, 0));
    TransposeSimdTag<TransposeUseSimd(N, M)> tag;
    constexpr void (*transpose)(TransposeSimdTag<TransposeUseSimd(N, M)>,
                                const From&, const To&, size_t, size_t) =
        GenericTransposeBlock<0, 0, From, To>;
    NoInlineWrapper(transpose, tag, from, to, N, M);
  }
};

// Block: (x, y) <-> (N * y + x)
// Lines: (x, y) <-> (stride * y + x)
//
// I.e. Block is a specialization of Lines with fixed stride.
//
// FromXXX should implement Read and Load (Read vector).
// ToXXX should implement Write and Store (Write vector).

// Here and in the following, the SZ template parameter specifies the number of
// values to load/store. Needed because we want to handle 4x4 sub-blocks of
// 16x16 blocks.
class DCTFrom {
 public:
  DCTFrom(const float* data, size_t stride) : stride_(stride), data_(data) {}

  template <typename D>
  HWY_INLINE Vec<D> LoadPart(D, const size_t row, size_t i) const {
    JXL_DASSERT(Lanes(D()) <= stride_);
    // Since these functions are used also for DC, no alignment at all is
    // guaranteed in the case of floating blocks.
    // TODO(veluca): consider using a different class for DC-to-LF and
    // DC-from-LF, or copying DC values to/from a temporary aligned location.
    return LoadU(D(), Address(row, i));
  }

  HWY_INLINE float Read(const size_t row, const size_t i) const {
    return *Address(row, i);
  }

  constexpr HWY_INLINE const float* Address(const size_t row,
                                            const size_t i) const {
    return data_ + row * stride_ + i;
  }

  size_t Stride() const { return stride_; }

 private:
  size_t stride_;
  const float* JXL_RESTRICT data_;
};

class DCTTo {
 public:
  DCTTo(float* data, size_t stride) : stride_(stride), data_(data) {}

  template <typename D>
  HWY_INLINE void StorePart(D, const Vec<D>& v, const size_t row,
                            size_t i) const {
    JXL_DASSERT(Lanes(D()) <= stride_);
    // Since these functions are used also for DC, no alignment at all is
    // guaranteed in the case of floating blocks.
    // TODO(veluca): consider using a different class for DC-to-LF and
    // DC-from-LF, or copying DC values to/from a temporary aligned location.
    StoreU(v, D(), Address(row, i));
  }

  HWY_INLINE void Write(float v, const size_t row, const size_t i) const {
    *Address(row, i) = v;
  }

  constexpr HWY_INLINE float* Address(const size_t row, const size_t i) const {
    return data_ + row * stride_ + i;
  }

  size_t Stride() const { return stride_; }

 private:
  size_t stride_;
  float* JXL_RESTRICT data_;
};

template <size_t SZ>
struct FVImpl {
  using type = HWY_CAPPED(float, SZ);
};

template <>
struct FVImpl<0> {
  using type = HWY_FULL(float);
};

template <size_t SZ>
using FV = typename FVImpl<SZ>::type;

// Implementation of Lowest Complexity Self Recursive Radix-2 DCT II/III
// Algorithms, by Siriani M. Perera and Jianhua Liu.

template <size_t N, size_t SZ>
struct CoeffBundle {
  static void AddReverse(const float* JXL_RESTRICT ain1,
                         const float* JXL_RESTRICT ain2,
                         float* JXL_RESTRICT aout) {
    for (size_t i = 0; i < N; i++) {
      auto in1 = Load(FV<SZ>(), ain1 + i * SZ);
      auto in2 = Load(FV<SZ>(), ain2 + (N - i - 1) * SZ);
      Store(Add(in1, in2), FV<SZ>(), aout + i * SZ);
    }
  }
  static void SubReverse(const float* JXL_RESTRICT ain1,
                         const float* JXL_RESTRICT ain2,
                         float* JXL_RESTRICT aout) {
    for (size_t i = 0; i < N; i++) {
      auto in1 = Load(FV<SZ>(), ain1 + i * SZ);
      auto in2 = Load(FV<SZ>(), ain2 + (N - i - 1) * SZ);
      Store(Sub(in1, in2), FV<SZ>(), aout + i * SZ);
    }
  }
  static void B(float* JXL_RESTRICT coeff) {
    auto sqrt2 = Set(FV<SZ>(), kSqrt2);
    auto in1 = Load(FV<SZ>(), coeff);
    auto in2 = Load(FV<SZ>(), coeff + SZ);
    Store(MulAdd(in1, sqrt2, in2), FV<SZ>(), coeff);
    for (size_t i = 1; i + 1 < N; i++) {
      auto in1 = Load(FV<SZ>(), coeff + i * SZ);
      auto in2 = Load(FV<SZ>(), coeff + (i + 1) * SZ);
      Store(Add(in1, in2), FV<SZ>(), coeff + i * SZ);
    }
  }
  static void BTranspose(float* JXL_RESTRICT coeff) {
    for (size_t i = N - 1; i > 0; i--) {
      auto in1 = Load(FV<SZ>(), coeff + i * SZ);
      auto in2 = Load(FV<SZ>(), coeff + (i - 1) * SZ);
      Store(Add(in1, in2), FV<SZ>(), coeff + i * SZ);
    }
    auto sqrt2 = Set(FV<SZ>(), kSqrt2);
    auto in1 = Load(FV<SZ>(), coeff);
    Store(Mul(in1, sqrt2), FV<SZ>(), coeff);
  }
  // Ideally optimized away by compiler (except the multiply).
  static void InverseEvenOdd(const float* JXL_RESTRICT ain,
                             float* JXL_RESTRICT aout) {
    for (size_t i = 0; i < N / 2; i++) {
      auto in1 = Load(FV<SZ>(), ain + i * SZ);
      Store(in1, FV<SZ>(), aout + 2 * i * SZ);
    }
    for (size_t i = N / 2; i < N; i++) {
      auto in1 = Load(FV<SZ>(), ain + i * SZ);
      Store(in1, FV<SZ>(), aout + (2 * (i - N / 2) + 1) * SZ);
    }
  }
  // Ideally optimized away by compiler.
  static void ForwardEvenOdd(const float* JXL_RESTRICT ain, size_t ain_stride,
                             float* JXL_RESTRICT aout) {
    for (size_t i = 0; i < N / 2; i++) {
      auto in1 = LoadU(FV<SZ>(), ain + 2 * i * ain_stride);
      Store(in1, FV<SZ>(), aout + i * SZ);
    }
    for (size_t i = N / 2; i < N; i++) {
      auto in1 = LoadU(FV<SZ>(), ain + (2 * (i - N / 2) + 1) * ain_stride);
      Store(in1, FV<SZ>(), aout + i * SZ);
    }
  }
  // Invoked on full vector.
  static void Multiply(float* JXL_RESTRICT coeff) {
    for (size_t i = 0; i < N / 2; i++) {
      auto in1 = Load(FV<SZ>(), coeff + (N / 2 + i) * SZ);
      auto mul = Set(FV<SZ>(), WcMultipliers<N>::kMultipliers[i]);
      Store(Mul(in1, mul), FV<SZ>(), coeff + (N / 2 + i) * SZ);
    }
  }
  static void MultiplyAndAdd(const float* JXL_RESTRICT coeff,
                             float* JXL_RESTRICT out, size_t out_stride) {
    for (size_t i = 0; i < N / 2; i++) {
      auto mul = Set(FV<SZ>(), WcMultipliers<N>::kMultipliers[i]);
      auto in1 = Load(FV<SZ>(), coeff + i * SZ);
      auto in2 = Load(FV<SZ>(), coeff + (N / 2 + i) * SZ);
      auto out1 = MulAdd(mul, in2, in1);
      auto out2 = NegMulAdd(mul, in2, in1);
      StoreU(out1, FV<SZ>(), out + i * out_stride);
      StoreU(out2, FV<SZ>(), out + (N - i - 1) * out_stride);
    }
  }
  template <typename Block>
  static void LoadFromBlock(const Block& in, size_t off,
                            float* JXL_RESTRICT coeff) {
    for (size_t i = 0; i < N; i++) {
      Store(in.LoadPart(FV<SZ>(), i, off), FV<SZ>(), coeff + i * SZ);
    }
  }
  template <typename Block>
  static void StoreToBlockAndScale(const float* JXL_RESTRICT coeff,
                                   const Block& out, size_t off) {
    auto mul = Set(FV<SZ>(), 1.0f / N);
    for (size_t i = 0; i < N; i++) {
      out.StorePart(FV<SZ>(), Mul(mul, Load(FV<SZ>(), coeff + i * SZ)), i, off);
    }
  }
};

template <size_t N, size_t SZ>
struct DCT1DImpl;

template <size_t SZ>
struct DCT1DImpl<1, SZ> {
  JXL_INLINE void operator()(float* JXL_RESTRICT mem) {}
};

template <size_t SZ>
struct DCT1DImpl<2, SZ> {
  JXL_INLINE void operator()(float* JXL_RESTRICT mem) {
    auto in1 = Load(FV<SZ>(), mem);
    auto in2 = Load(FV<SZ>(), mem + SZ);
    Store(Add(in1, in2), FV<SZ>(), mem);
    Store(Sub(in1, in2), FV<SZ>(), mem + SZ);
  }
};

template <size_t N, size_t SZ>
struct DCT1DImpl {
  void operator()(float* JXL_RESTRICT mem) {
    // This is relatively small (4kB with 64-DCT and AVX-512)
    HWY_ALIGN float tmp[N * SZ];
    CoeffBundle<N / 2, SZ>::AddReverse(mem, mem + N / 2 * SZ, tmp);
    DCT1DImpl<N / 2, SZ>()(tmp);
    CoeffBundle<N / 2, SZ>::SubReverse(mem, mem + N / 2 * SZ, tmp + N / 2 * SZ);
    CoeffBundle<N, SZ>::Multiply(tmp);
    DCT1DImpl<N / 2, SZ>()(tmp + N / 2 * SZ);
    CoeffBundle<N / 2, SZ>::B(tmp + N / 2 * SZ);
    CoeffBundle<N, SZ>::InverseEvenOdd(tmp, mem);
  }
};

template <size_t N, size_t SZ>
struct IDCT1DImpl;

template <size_t SZ>
struct IDCT1DImpl<1, SZ> {
  JXL_INLINE void operator()(const float* from, size_t from_stride, float* to,
                             size_t to_stride) {
    StoreU(LoadU(FV<SZ>(), from), FV<SZ>(), to);
  }
};

template <size_t SZ>
struct IDCT1DImpl<2, SZ> {
  JXL_INLINE void operator()(const float* from, size_t from_stride, float* to,
                             size_t to_stride) {
    JXL_DASSERT(from_stride >= SZ);
    JXL_DASSERT(to_stride >= SZ);
    auto in1 = LoadU(FV<SZ>(), from);
    auto in2 = LoadU(FV<SZ>(), from + from_stride);
    StoreU(Add(in1, in2), FV<SZ>(), to);
    StoreU(Sub(in1, in2), FV<SZ>(), to + to_stride);
  }
};

template <size_t N, size_t SZ>
struct IDCT1DImpl {
  void operator()(const float* from, size_t from_stride, float* to,
                  size_t to_stride) {
    JXL_DASSERT(from_stride >= SZ);
    JXL_DASSERT(to_stride >= SZ);
    // This is relatively small (4kB with 64-DCT and AVX-512)
    HWY_ALIGN float tmp[N * SZ];
    CoeffBundle<N, SZ>::ForwardEvenOdd(from, from_stride, tmp);
    IDCT1DImpl<N / 2, SZ>()(tmp, SZ, tmp, SZ);
    CoeffBundle<N / 2, SZ>::BTranspose(tmp + N / 2 * SZ);
    IDCT1DImpl<N / 2, SZ>()(tmp + N / 2 * SZ, SZ, tmp + N / 2 * SZ, SZ);
    CoeffBundle<N, SZ>::MultiplyAndAdd(tmp, to, to_stride);
  }
};

template <size_t N, size_t M_or_0, typename FromBlock, typename ToBlock>
void DCT1DWrapper(const FromBlock& from, const ToBlock& to, size_t Mp) {
  size_t M = M_or_0 != 0 ? M_or_0 : Mp;
  constexpr size_t SZ = MaxLanes(FV<M_or_0>());
  HWY_ALIGN float tmp[N * SZ];
  for (size_t i = 0; i < M; i += Lanes(FV<M_or_0>())) {
    // TODO(veluca): consider removing the temporary memory here (as is done in
    // IDCT), if it turns out that some compilers don't optimize away the loads
    // and this is performance-critical.
    CoeffBundle<N, SZ>::LoadFromBlock(from, i, tmp);
    DCT1DImpl<N, SZ>()(tmp);
    CoeffBundle<N, SZ>::StoreToBlockAndScale(tmp, to, i);
  }
}

template <size_t N, size_t M_or_0, typename FromBlock, typename ToBlock>
void IDCT1DWrapper(const FromBlock& from, const ToBlock& to, size_t Mp) {
  size_t M = M_or_0 != 0 ? M_or_0 : Mp;
  constexpr size_t SZ = MaxLanes(FV<M_or_0>());
  for (size_t i = 0; i < M; i += Lanes(FV<M_or_0>())) {
    IDCT1DImpl<N, SZ>()(from.Address(0, i), from.Stride(), to.Address(0, i),
                        to.Stride());
  }
}

template <size_t N, size_t M, typename = void>
struct DCT1D {
  template <typename FromBlock, typename ToBlock>
  void operator()(const FromBlock& from, const ToBlock& to) {
    return DCT1DWrapper<N, M>(from, to, M);
  }
};

template <size_t N, size_t M>
struct DCT1D<N, M, typename std::enable_if<(M > MaxLanes(FV<0>()))>::type> {
  template <typename FromBlock, typename ToBlock>
  void operator()(const FromBlock& from, const ToBlock& to) {
    return NoInlineWrapper(DCT1DWrapper<N, 0, FromBlock, ToBlock>, from, to, M);
  }
};

template <size_t N, size_t M, typename = void>
struct IDCT1D {
  template <typename FromBlock, typename ToBlock>
  void operator()(const FromBlock& from, const ToBlock& to) {
    return IDCT1DWrapper<N, M>(from, to, M);
  }
};

template <size_t N, size_t M>
struct IDCT1D<N, M, typename std::enable_if<(M > MaxLanes(FV<0>()))>::type> {
  template <typename FromBlock, typename ToBlock>
  void operator()(const FromBlock& from, const ToBlock& to) {
    return NoInlineWrapper(IDCT1DWrapper<N, 0, FromBlock, ToBlock>, from, to,
                           M);
  }
};

// Computes the maybe-transposed, scaled DCT of a block, that needs to be
// HWY_ALIGN'ed.
template <size_t ROWS, size_t COLS>
struct ComputeScaledDCT {
  // scratch_space must be aligned, and should have space for ROWS*COLS
  // floats.
  template <class From>
  HWY_MAYBE_UNUSED void operator()(const From& from, float* to,
                                   float* JXL_RESTRICT scratch_space) {
    float* JXL_RESTRICT block = scratch_space;
    if (ROWS < COLS) {
      DCT1D<ROWS, COLS>()(from, DCTTo(block, COLS));
      Transpose<ROWS, COLS>::Run(DCTFrom(block, COLS), DCTTo(to, ROWS));
      DCT1D<COLS, ROWS>()(DCTFrom(to, ROWS), DCTTo(block, ROWS));
      Transpose<COLS, ROWS>::Run(DCTFrom(block, ROWS), DCTTo(to, COLS));
    } else {
      DCT1D<ROWS, COLS>()(from, DCTTo(to, COLS));
      Transpose<ROWS, COLS>::Run(DCTFrom(to, COLS), DCTTo(block, ROWS));
      DCT1D<COLS, ROWS>()(DCTFrom(block, ROWS), DCTTo(to, ROWS));
    }
  }
};
// Computes the maybe-transposed, scaled IDCT of a block, that needs to be
// HWY_ALIGN'ed.
template <size_t ROWS, size_t COLS>
struct ComputeScaledIDCT {
  // scratch_space must be aligned, and should have space for ROWS*COLS
  // floats.
  template <class To>
  HWY_MAYBE_UNUSED void operator()(float* JXL_RESTRICT from, const To& to,
                                   float* JXL_RESTRICT scratch_space) {
    float* JXL_RESTRICT block = scratch_space;
    // Reverse the steps done in ComputeScaledDCT.
    if (ROWS < COLS) {
      Transpose<ROWS, COLS>::Run(DCTFrom(from, COLS), DCTTo(block, ROWS));
      IDCT1D<COLS, ROWS>()(DCTFrom(block, ROWS), DCTTo(from, ROWS));
      Transpose<COLS, ROWS>::Run(DCTFrom(from, ROWS), DCTTo(block, COLS));
      IDCT1D<ROWS, COLS>()(DCTFrom(block, COLS), to);
    } else {
      IDCT1D<COLS, ROWS>()(DCTFrom(from, ROWS), DCTTo(block, ROWS));
      Transpose<COLS, ROWS>::Run(DCTFrom(block, ROWS), DCTTo(from, COLS));
      IDCT1D<ROWS, COLS>()(DCTFrom(from, COLS), to);
    }
  }
};

// Inverse of ReinterpretingDCT.
template <size_t DCT_ROWS, size_t DCT_COLS, size_t LF_ROWS, size_t LF_COLS,
          size_t ROWS, size_t COLS>
HWY_INLINE void ReinterpretingIDCT(const float* input,
                                   const size_t input_stride, float* output,
                                   const size_t output_stride) {
  HWY_ALIGN float block[ROWS * COLS] = {};
  if (ROWS < COLS) {
    for (size_t y = 0; y < LF_ROWS; y++) {
      for (size_t x = 0; x < LF_COLS; x++) {
        block[y * COLS + x] = input[y * input_stride + x] *
                              DCTTotalResampleScale<DCT_ROWS, ROWS>(y) *
                              DCTTotalResampleScale<DCT_COLS, COLS>(x);
      }
    }
  } else {
    for (size_t y = 0; y < LF_COLS; y++) {
      for (size_t x = 0; x < LF_ROWS; x++) {
        block[y * ROWS + x] = input[y * input_stride + x] *
                              DCTTotalResampleScale<DCT_COLS, COLS>(y) *
                              DCTTotalResampleScale<DCT_ROWS, ROWS>(x);
      }
    }
  }

  // ROWS, COLS <= 8, so we can put scratch space on the stack.
  HWY_ALIGN float scratch_space[ROWS * COLS];
  ComputeScaledIDCT<ROWS, COLS>()(block, DCTTo(output, output_stride),
                                  scratch_space);
}

template <size_t S>
void DCT2TopBlock(const float* block, size_t stride, float* out) {
  static_assert(kBlockDim % S == 0, "S should be a divisor of kBlockDim");
  static_assert(S % 2 == 0, "S should be even");
  float temp[kDCTBlockSize];
  constexpr size_t num_2x2 = S / 2;
  for (size_t y = 0; y < num_2x2; y++) {
    for (size_t x = 0; x < num_2x2; x++) {
      float c00 = block[y * 2 * stride + x * 2];
      float c01 = block[y * 2 * stride + x * 2 + 1];
      float c10 = block[(y * 2 + 1) * stride + x * 2];
      float c11 = block[(y * 2 + 1) * stride + x * 2 + 1];
      float r00 = c00 + c01 + c10 + c11;
      float r01 = c00 + c01 - c10 - c11;
      float r10 = c00 - c01 + c10 - c11;
      float r11 = c00 - c01 - c10 + c11;
      r00 *= 0.25f;
      r01 *= 0.25f;
      r10 *= 0.25f;
      r11 *= 0.25f;
      temp[y * kBlockDim + x] = r00;
      temp[y * kBlockDim + num_2x2 + x] = r01;
      temp[(y + num_2x2) * kBlockDim + x] = r10;
      temp[(y + num_2x2) * kBlockDim + num_2x2 + x] = r11;
    }
  }
  for (size_t y = 0; y < S; y++) {
    for (size_t x = 0; x < S; x++) {
      out[y * kBlockDim + x] = temp[y * kBlockDim + x];
    }
  }
}

HWY_MAYBE_UNUSED void TransformFromPixels(const AcStrategy::Type strategy,
                                          const float* JXL_RESTRICT pixels,
                                          size_t pixels_stride,
                                          float* JXL_RESTRICT coefficients,
                                          float* JXL_RESTRICT scratch_space) {
  using Type = AcStrategy::Type;
  switch (strategy) {
    case Type::IDENTITY: {
      for (size_t y = 0; y < 2; y++) {
        for (size_t x = 0; x < 2; x++) {
          float block_dc = 0;
          for (size_t iy = 0; iy < 4; iy++) {
            for (size_t ix = 0; ix < 4; ix++) {
              block_dc += pixels[(y * 4 + iy) * pixels_stride + x * 4 + ix];
            }
          }
          block_dc *= 1.0f / 16;
          for (size_t iy = 0; iy < 4; iy++) {
            for (size_t ix = 0; ix < 4; ix++) {
              if (ix == 1 && iy == 1) continue;
              coefficients[(y + iy * 2) * 8 + x + ix * 2] =
                  pixels[(y * 4 + iy) * pixels_stride + x * 4 + ix] -
                  pixels[(y * 4 + 1) * pixels_stride + x * 4 + 1];
            }
          }
          coefficients[(y + 2) * 8 + x + 2] = coefficients[y * 8 + x];
          coefficients[y * 8 + x] = block_dc;
        }
      }
      float block00 = coefficients[0];
      float block01 = coefficients[1];
      float block10 = coefficients[8];
      float block11 = coefficients[9];
      coefficients[0] = (block00 + block01 + block10 + block11) * 0.25f;
      coefficients[1] = (block00 + block01 - block10 - block11) * 0.25f;
      coefficients[8] = (block00 - block01 + block10 - block11) * 0.25f;
      coefficients[9] = (block00 - block01 - block10 + block11) * 0.25f;
      break;
    }
    case Type::DCT8X4: {
      for (size_t x = 0; x < 2; x++) {
        HWY_ALIGN float block[4 * 8];
        ComputeScaledDCT<8, 4>()(DCTFrom(pixels + x * 4, pixels_stride), block,
                                 scratch_space);
        for (size_t iy = 0; iy < 4; iy++) {
          for (size_t ix = 0; ix < 8; ix++) {
            // Store transposed.
            coefficients[(x + iy * 2) * 8 + ix] = block[iy * 8 + ix];
          }
        }
      }
      float block0 = coefficients[0];
      float block1 = coefficients[8];
      coefficients[0] = (block0 + block1) * 0.5f;
      coefficients[8] = (block0 - block1) * 0.5f;
      break;
    }
    case Type::DCT4X8: {
      for (size_t y = 0; y < 2; y++) {
        HWY_ALIGN float block[4 * 8];
        ComputeScaledDCT<4, 8>()(
            DCTFrom(pixels + y * 4 * pixels_stride, pixels_stride), block,
            scratch_space);
        for (size_t iy = 0; iy < 4; iy++) {
          for (size_t ix = 0; ix < 8; ix++) {
            coefficients[(y + iy * 2) * 8 + ix] = block[iy * 8 + ix];
          }
        }
      }
      float block0 = coefficients[0];
      float block1 = coefficients[8];
      coefficients[0] = (block0 + block1) * 0.5f;
      coefficients[8] = (block0 - block1) * 0.5f;
      break;
    }
    case Type::DCT4X4: {
      for (size_t y = 0; y < 2; y++) {
        for (size_t x = 0; x < 2; x++) {
          HWY_ALIGN float block[4 * 4];
          ComputeScaledDCT<4, 4>()(
              DCTFrom(pixels + y * 4 * pixels_stride + x * 4, pixels_stride),
              block, scratch_space);
          for (size_t iy = 0; iy < 4; iy++) {
            for (size_t ix = 0; ix < 4; ix++) {
              coefficients[(y + iy * 2) * 8 + x + ix * 2] = block[iy * 4 + ix];
            }
          }
        }
      }
      float block00 = coefficients[0];
      float block01 = coefficients[1];
      float block10 = coefficients[8];
      float block11 = coefficients[9];
      coefficients[0] = (block00 + block01 + block10 + block11) * 0.25f;
      coefficients[1] = (block00 + block01 - block10 - block11) * 0.25f;
      coefficients[8] = (block00 - block01 + block10 - block11) * 0.25f;
      coefficients[9] = (block00 - block01 - block10 + block11) * 0.25f;
      break;
    }
    case Type::DCT2X2: {
      DCT2TopBlock<8>(pixels, pixels_stride, coefficients);
      DCT2TopBlock<4>(coefficients, kBlockDim, coefficients);
      DCT2TopBlock<2>(coefficients, kBlockDim, coefficients);
      break;
    }
    case Type::DCT16X16: {
      ComputeScaledDCT<16, 16>()(DCTFrom(pixels, pixels_stride), coefficients,
                                 scratch_space);
      break;
    }
    case Type::DCT16X8: {
      ComputeScaledDCT<16, 8>()(DCTFrom(pixels, pixels_stride), coefficients,
                                scratch_space);
      break;
    }
    case Type::DCT8X16: {
      ComputeScaledDCT<8, 16>()(DCTFrom(pixels, pixels_stride), coefficients,
                                scratch_space);
      break;
    }
    case Type::DCT: {
      ComputeScaledDCT<8, 8>()(DCTFrom(pixels, pixels_stride), coefficients,
                               scratch_space);
      break;
    }
    case Type::AFV0:
    case Type::AFV1:
    case Type::AFV2:
    case Type::AFV3:
    case Type::DCT32X8:
    case Type::DCT8X32:
    case Type::DCT32X16:
    case Type::DCT16X32:
    case Type::DCT32X32:
    case Type::DCT64X32:
    case Type::DCT32X64:
    case Type::DCT64X64:
    case Type::DCT64X128:
    case Type::DCT128X64:
    case Type::DCT128X128:
    case Type::DCT256X128:
    case Type::DCT128X256:
    case Type::DCT256X256:
    case Type::kNumValidStrategies:
      JXL_ABORT("Invalid strategy");
  }
}

HWY_MAYBE_UNUSED void DCFromLowestFrequencies(const AcStrategy::Type strategy,
                                              const float* block, float* dc,
                                              size_t dc_stride) {
  using Type = AcStrategy::Type;
  switch (strategy) {
    case Type::DCT16X8: {
      ReinterpretingIDCT</*DCT_ROWS=*/2 * kBlockDim, /*DCT_COLS=*/kBlockDim,
                         /*LF_ROWS=*/2, /*LF_COLS=*/1, /*ROWS=*/2,
                         /*COLS=*/1>(block, 2 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT8X16: {
      ReinterpretingIDCT</*DCT_ROWS=*/kBlockDim, /*DCT_COLS=*/2 * kBlockDim,
                         /*LF_ROWS=*/1, /*LF_COLS=*/2, /*ROWS=*/1,
                         /*COLS=*/2>(block, 2 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT16X16: {
      ReinterpretingIDCT<
          /*DCT_ROWS=*/2 * kBlockDim, /*DCT_COLS=*/2 * kBlockDim,
          /*LF_ROWS=*/2, /*LF_COLS=*/2, /*ROWS=*/2, /*COLS=*/2>(
          block, 2 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT32X8:
    case Type::DCT8X32:
    case Type::DCT32X16:
    case Type::DCT16X32:
    case Type::DCT32X32:
    case Type::DCT64X32:
    case Type::DCT32X64:
    case Type::DCT64X64:
    case Type::DCT64X128:
    case Type::DCT128X64:
    case Type::DCT128X128:
    case Type::DCT256X128:
    case Type::DCT128X256:
    case Type::DCT256X256:
      break;
    case Type::DCT:
    case Type::DCT2X2:
    case Type::DCT4X4:
    case Type::DCT4X8:
    case Type::DCT8X4:
    case Type::AFV0:
    case Type::AFV1:
    case Type::AFV2:
    case Type::AFV3:
    case Type::IDENTITY:
      dc[0] = block[0];
      break;
    case Type::kNumValidStrategies:
      JXL_ABORT("Invalid strategy");
  }
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#endif  // ENCODER_ENC_TRANSFORMS_INL_H_
