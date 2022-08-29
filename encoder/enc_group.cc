// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/enc_group.h"

#include <utility>

#include "hwy/aligned_allocator.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "encoder/enc_group.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "encoder/ac_context.h"
#include "encoder/ac_strategy.h"
#include "encoder/base/bits.h"
#include "encoder/base/compiler_specific.h"
#include "encoder/common.h"
#include "encoder/enc_transforms-inl.h"
#include "encoder/image.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Abs;
using hwy::HWY_NAMESPACE::And;
using hwy::HWY_NAMESPACE::AndNot;
using hwy::HWY_NAMESPACE::ApproximateReciprocal;
using hwy::HWY_NAMESPACE::Ge;
using hwy::HWY_NAMESPACE::Gt;
using hwy::HWY_NAMESPACE::IfThenElse;
using hwy::HWY_NAMESPACE::IfThenElseZero;
using hwy::HWY_NAMESPACE::Lt;
using hwy::HWY_NAMESPACE::MaskFromVec;
using hwy::HWY_NAMESPACE::Rebind;
using hwy::HWY_NAMESPACE::Round;
using hwy::HWY_NAMESPACE::Vec;
using hwy::HWY_NAMESPACE::Xor;

// Returns number of non-zero coefficients (but skip LLF).
// We cannot rely on block[] being all-zero bits, so first truncate to integer.
// Also writes the per-8x8 block nzeros starting at nzeros_pos.
int32_t NumNonZeroExceptLLF(const size_t cx, const size_t cy,
                            const AcStrategy acs, const size_t covered_blocks,
                            const size_t log2_covered_blocks,
                            const int32_t* JXL_RESTRICT block,
                            const size_t nzeros_stride,
                            int32_t* JXL_RESTRICT nzeros_pos) {
  const HWY_CAPPED(int32_t, kBlockDim) di;

  const auto zero = Zero(di);
  // Add FF..FF for every zero coefficient, negate to get #zeros.
  auto neg_sum_zero = zero;

  {
    // Mask sufficient for one row of coefficients.
    HWY_ALIGN const int32_t
        llf_mask_lanes[AcStrategy::kMaxCoeffBlocks * (1 + kBlockDim)] = {
            -1, -1, -1, -1};
    // First cx=1,2,4 elements are FF..FF, others 0.
    const int32_t* llf_mask_pos =
        llf_mask_lanes + AcStrategy::kMaxCoeffBlocks - cx;

    // Rows with LLF: mask out the LLF
    for (size_t y = 0; y < cy; y++) {
      for (size_t x = 0; x < cx * kBlockDim; x += Lanes(di)) {
        const auto llf_mask = LoadU(di, llf_mask_pos + x);

        // LLF counts as zero so we don't include it in nzeros.
        const auto coef =
            AndNot(llf_mask, Load(di, &block[y * cx * kBlockDim + x]));

        neg_sum_zero = Add(neg_sum_zero, VecFromMask(di, Eq(coef, zero)));
      }
    }
  }

  // Remaining rows: no mask
  for (size_t y = cy; y < cy * kBlockDim; y++) {
    for (size_t x = 0; x < cx * kBlockDim; x += Lanes(di)) {
      const auto coef = Load(di, &block[y * cx * kBlockDim + x]);
      neg_sum_zero = Add(neg_sum_zero, VecFromMask(di, Eq(coef, zero)));
    }
  }

  // We want area - sum_zero, add because neg_sum_zero is already negated.
  const int32_t nzeros =
      int32_t(cx * cy * kDCTBlockSize) + GetLane(SumOfLanes(di, neg_sum_zero));

  const int32_t shifted_nzeros = static_cast<int32_t>(
      (nzeros + covered_blocks - 1) >> log2_covered_blocks);
  // Need non-canonicalized dimensions!
  for (size_t y = 0; y < acs.covered_blocks_y(); y++) {
    for (size_t x = 0; x < acs.covered_blocks_x(); x++) {
      nzeros_pos[x + y * nzeros_stride] = shifted_nzeros;
    }
  }

  return nzeros;
}

// Specialization for 8x8, where only top-left is LLF/DC.
// About 1% overall speedup vs. NumNonZeroExceptLLF.
int32_t NumNonZero8x8ExceptDC(const int32_t* JXL_RESTRICT block,
                              int32_t* JXL_RESTRICT nzeros_pos) {
  const HWY_CAPPED(int32_t, kBlockDim) di;

  const auto zero = Zero(di);
  // Add FF..FF for every zero coefficient, negate to get #zeros.
  auto neg_sum_zero = zero;

  {
    // First row has DC, so mask
    const size_t y = 0;
    HWY_ALIGN const int32_t dc_mask_lanes[kBlockDim] = {-1};

    for (size_t x = 0; x < kBlockDim; x += Lanes(di)) {
      const auto dc_mask = Load(di, dc_mask_lanes + x);

      // DC counts as zero so we don't include it in nzeros.
      const auto coef = AndNot(dc_mask, Load(di, &block[y * kBlockDim + x]));

      neg_sum_zero = Add(neg_sum_zero, VecFromMask(di, Eq(coef, zero)));
    }
  }

  // Remaining rows: no mask
  for (size_t y = 1; y < kBlockDim; y++) {
    for (size_t x = 0; x < kBlockDim; x += Lanes(di)) {
      const auto coef = Load(di, &block[y * kBlockDim + x]);
      neg_sum_zero = Add(neg_sum_zero, VecFromMask(di, Eq(coef, zero)));
    }
  }

  // We want 64 - sum_zero, add because neg_sum_zero is already negated.
  const int32_t nzeros =
      int32_t(kDCTBlockSize) + GetLane(SumOfLanes(di, neg_sum_zero));

  *nzeros_pos = nzeros;

  return nzeros;
}

JXL_INLINE int32_t PredictFromTopAndLeft(
    const int32_t* const JXL_RESTRICT row_top,
    const int32_t* const JXL_RESTRICT row, size_t x, int32_t default_val) {
  if (x == 0) {
    return row_top == nullptr ? default_val : row_top[x];
  }
  if (row_top == nullptr) {
    return row[x - 1];
  }
  return (row_top[x] + row[x - 1] + 1) / 2;
}

// Needs at least 16 bits. A 32-bit type speeds up DecodeAC by 2% at the cost of
// more memory.
using coeff_order_t = uint32_t;

constexpr coeff_order_t kCoeffOrders[] = {
    0,   1,   8,   16, 9,   2,   3,   10,  17,  24,  32,  25,  18,  11,  4,
    5,   12,  19,  26, 33,  40,  48,  41,  34,  27,  20,  13,  6,   7,   14,
    21,  28,  35,  42, 49,  56,  57,  50,  43,  36,  29,  22,  15,  23,  30,
    37,  44,  51,  58, 59,  52,  45,  38,  31,  39,  46,  53,  60,  61,  54,
    47,  55,  62,  63, 0,   1,   16,  2,   3,   17,  32,  18,  4,   5,   19,
    33,  48,  34,  20, 6,   7,   21,  35,  49,  64,  50,  36,  22,  8,   9,
    23,  37,  51,  65, 80,  66,  52,  38,  24,  10,  11,  25,  39,  53,  67,
    81,  96,  82,  68, 54,  40,  26,  12,  13,  27,  41,  55,  69,  83,  97,
    112, 98,  84,  70, 56,  42,  28,  14,  15,  29,  43,  57,  71,  85,  99,
    113, 114, 100, 86, 72,  58,  44,  30,  31,  45,  59,  73,  87,  101, 115,
    116, 102, 88,  74, 60,  46,  47,  61,  75,  89,  103, 117, 118, 104, 90,
    76,  62,  63,  77, 91,  105, 119, 120, 106, 92,  78,  79,  93,  107, 121,
    122, 108, 94,  95, 109, 123, 124, 110, 111, 125, 126, 127,
};

// Maps from ac strategy to offset in kCoeffOrders[]
static constexpr size_t kCoeffOrderOffset[] = {0, kDCTBlockSize, kDCTBlockSize};

template <class DI>
HWY_INLINE HWY_MAYBE_UNUSED Vec<Rebind<float, DI>> AdjustQuantBias(
    DI di, const size_t c, const Vec<DI> quant_i,
    const float* HWY_RESTRICT biases) {
  const Rebind<float, DI> df;

  const auto quant = ConvertTo(df, quant_i);

  // Compare |quant|, keep sign bit for negating result.
  const auto kSign = BitCast(df, Set(di, INT32_MIN));
  const auto sign = And(quant, kSign);  // TODO(janwas): = abs ^ orig
  const auto abs_quant = AndNot(kSign, quant);

  // If |x| is 1, kZeroBias creates a different bias for each channel.
  // We're implementing the following:
  // if (quant == 0) return 0;
  // if (quant == 1) return biases[c];
  // if (quant == -1) return -biases[c];
  // return quant - biases[3] / quant;

  // Integer comparison is not helpful because Clang incurs bypass penalties
  // from unnecessarily mixing integer and float.
  const auto is_01 = Lt(abs_quant, Set(df, 1.125f));
  const auto not_0 = Gt(abs_quant, Zero(df));

  // Bitwise logic is faster than quant * biases[c].
  const auto one_bias = IfThenElseZero(not_0, Xor(Set(df, biases[c]), sign));

  // About 2E-5 worse than ReciprocalNR or division.
  const auto bias =
      NegMulAdd(Set(df, biases[3]), ApproximateReciprocal(quant), quant);

  return IfThenElse(is_01, one_bias, bias);
}

// NOTE: caller takes care of extracting quant from rect of RawQuantField.
void QuantizeBlockAC(const float* JXL_RESTRICT block_in, size_t c,
                     const float* JXL_RESTRICT qm, int32_t quant, float scale,
                     float qm_multiplier, size_t xsize, size_t ysize,
                     int32_t* JXL_RESTRICT block_out) {
  const float qac = scale * quant;
  // Not SIMD-fied for now.
  float thres[4] = {0.58f, 0.635f, 0.66f, 0.7f};
  if (c == 0) {
    for (int i = 1; i < 4; ++i) {
      thres[i] += 0.08f;
    }
  }
  if (c == 2) {
    for (int i = 1; i < 4; ++i) {
      thres[i] = 0.75f;
    }
  }
  if (xsize > 1 || ysize > 1) {
    for (int i = 0; i < 4; ++i) {
      thres[i] -= Clamp1(0.003f * xsize * ysize, 0.f, (c > 0 ? 0.08f : 0.12f));
    }
  }

  {
    HWY_CAPPED(float, kBlockDim) df;
    HWY_CAPPED(int32_t, kBlockDim) di;
    HWY_CAPPED(uint32_t, kBlockDim) du;
    const auto quant = Set(df, qac * qm_multiplier);

    for (size_t y = 0; y < ysize * kBlockDim; y++) {
      size_t yfix = static_cast<size_t>(y >= ysize * kBlockDim / 2) * 2;
      const size_t off = y * kBlockDim * xsize;
      for (size_t x = 0; x < xsize * kBlockDim; x += Lanes(df)) {
        auto thr = Zero(df);
        if (xsize == 1) {
          HWY_ALIGN uint32_t kMask[kBlockDim] = {0,   0,   0,   0,
                                                 ~0u, ~0u, ~0u, ~0u};
          const auto mask = MaskFromVec(BitCast(df, Load(du, kMask + x)));
          thr =
              IfThenElse(mask, Set(df, thres[yfix + 1]), Set(df, thres[yfix]));
        } else {
          // Same for all lanes in the vector.
          thr = Set(
              df,
              thres[yfix + static_cast<size_t>(x >= xsize * kBlockDim / 2)]);
        }

        const auto q = Mul(Load(df, qm + off + x), quant);
        const auto in = Load(df, block_in + off + x);
        const auto val = Mul(q, in);
        const auto nzero_mask = Ge(Abs(val), thr);
        const auto v = ConvertTo(di, IfThenElseZero(nzero_mask, Round(val)));
        Store(v, di, block_out + off + x);
      }
    }
    return;
  }
}

// NOTE: caller takes care of extracting quant from rect of RawQuantField.
void QuantizeRoundtripYBlockAC(const float* JXL_RESTRICT qm,
                               const float* JXL_RESTRICT dqm, float scale,
                               int32_t quant, size_t xsize, size_t ysize,
                               float* JXL_RESTRICT inout,
                               int32_t* JXL_RESTRICT quantized) {
  QuantizeBlockAC(inout, 1, qm, quant, scale, 1.0f, xsize, ysize, quantized);
  HWY_CAPPED(float, kDCTBlockSize) df;
  HWY_CAPPED(int32_t, kDCTBlockSize) di;
  const auto inv_qac = Set(df, 1.0 / (scale * quant));
  constexpr float kDefaultQuantBias[4] = {
      1.0f - 0.05465007330715401f,
      1.0f - 0.07005449891748593f,
      1.0f - 0.049935103337343655f,
      0.145f,
  };
  for (size_t k = 0; k < kDCTBlockSize * xsize * ysize; k += Lanes(df)) {
    const auto quant = Load(di, quantized + k);
    const auto adj_quant = AdjustQuantBias(di, 1, quant, kDefaultQuantBias);
    const auto dequantm = Load(df, dqm + k);
    Store(Mul(Mul(adj_quant, dequantm), inv_qac), df, inout + k);
  }
}

void ComputeCoefficients(size_t group_idx, const Image3F& opsin,
                         const ImageI& raw_quant_field,
                         const DequantMatrices& matrices, const float scale,
                         const ColorCorrelationMap& cmap,
                         const AcStrategyImage& ac_strategy,
                         const float x_qm_mul, Image3I* tmp_num_nzeroes,
                         Image3F* dc, std::vector<Token>* output) {
  const size_t xsize_groups = DivCeil(opsin.xsize(), kGroupDim);
  const size_t gx = group_idx % xsize_groups;
  const size_t gy = group_idx / xsize_groups;
  const Rect block_group_rect(gx * kGroupDimInBlocks, gy * kGroupDimInBlocks,
                              kGroupDimInBlocks, kGroupDimInBlocks,
                              DivCeil(opsin.xsize(), kBlockDim),
                              DivCeil(opsin.ysize(), kBlockDim));
  const Rect group_rect(gx * kGroupDim, gy * kGroupDim, kGroupDim, kGroupDim,
                        opsin.xsize(), opsin.ysize());
  const Rect cmap_rect(
      block_group_rect.x0() / kColorTileDimInBlocks,
      block_group_rect.y0() / kColorTileDimInBlocks,
      DivCeil(block_group_rect.xsize(), kColorTileDimInBlocks),
      DivCeil(block_group_rect.ysize(), kColorTileDimInBlocks));

  const size_t xsize_blocks = block_group_rect.xsize();
  const size_t ysize_blocks = block_group_rect.ysize();

  const size_t dc_stride = static_cast<size_t>(dc->PixelsPerRow());
  const size_t opsin_stride = static_cast<size_t>(opsin.PixelsPerRow());

  // TODO(veluca): consider strategies to reduce this memory.
  auto mem = hwy::AllocateAligned<int32_t>(3 * AcStrategy::kMaxCoeffArea);
  auto fmem = hwy::AllocateAligned<float>(5 * AcStrategy::kMaxCoeffArea);
  float* JXL_RESTRICT scratch_space =
      fmem.get() + 3 * AcStrategy::kMaxCoeffArea;
  constexpr HWY_CAPPED(float, kDCTBlockSize) d;
  HWY_ALIGN float* coeffs_in = fmem.get();
  HWY_ALIGN int32_t* quantized = mem.get();

  output->reserve(output->size() +
                  3 * xsize_blocks * ysize_blocks * kDCTBlockSize);

  const size_t nzeros_stride = tmp_num_nzeroes->PixelsPerRow();

  for (size_t by = 0; by < ysize_blocks; ++by) {
    const int32_t* JXL_RESTRICT row_quant_ac =
        block_group_rect.ConstRow(raw_quant_field, by);
    size_t ty = by / kColorTileDimInBlocks;
    const int8_t* JXL_RESTRICT row_cmap[3] = {
        cmap_rect.ConstRow(cmap.ytox_map, ty),
        nullptr,
        cmap_rect.ConstRow(cmap.ytob_map, ty),
    };
    const float* JXL_RESTRICT opsin_rows[3] = {
        group_rect.ConstPlaneRow(opsin, 0, by * kBlockDim),
        group_rect.ConstPlaneRow(opsin, 1, by * kBlockDim),
        group_rect.ConstPlaneRow(opsin, 2, by * kBlockDim),
    };
    float* JXL_RESTRICT dc_rows[3] = {
        block_group_rect.PlaneRow(dc, 0, by),
        block_group_rect.PlaneRow(dc, 1, by),
        block_group_rect.PlaneRow(dc, 2, by),
    };
    AcStrategyRow ac_strategy_row = ac_strategy.ConstRow(block_group_rect, by);
    int32_t* JXL_RESTRICT row_nzeros[3] = {
        tmp_num_nzeroes->PlaneRow(0, by),
        tmp_num_nzeroes->PlaneRow(1, by),
        tmp_num_nzeroes->PlaneRow(2, by),
    };
    const int32_t* JXL_RESTRICT row_nzeros_top[3] = {
        by == 0 ? nullptr : tmp_num_nzeroes->ConstPlaneRow(0, by - 1),
        by == 0 ? nullptr : tmp_num_nzeroes->ConstPlaneRow(1, by - 1),
        by == 0 ? nullptr : tmp_num_nzeroes->ConstPlaneRow(2, by - 1),
    };
    for (size_t bx = 0; bx < xsize_blocks; ++bx) {
      size_t tx = bx / kColorTileDimInBlocks;
      const auto x_factor = Set(d, cmap.YtoXRatio(row_cmap[0][tx]));
      const auto b_factor = Set(d, cmap.YtoBRatio(row_cmap[2][tx]));
      const AcStrategy acs = ac_strategy_row[bx];
      if (!acs.IsFirstBlock()) continue;

      size_t cx = acs.covered_blocks_x();
      size_t cy = acs.covered_blocks_y();
      if (cy > cx) std::swap(cx, cy);
      const size_t covered_blocks = cx * cy;  // = #LLF coefficients
      const size_t size = kDCTBlockSize * covered_blocks;

      // DCT Y channel, roundtrip-quantize it and set DC.
      const int32_t quant_ac = row_quant_ac[bx];
      TransformFromPixels(acs.Strategy(), opsin_rows[1] + bx * kBlockDim,
                          opsin_stride, coeffs_in + size, scratch_space);
      DCFromLowestFrequencies(acs.Strategy(), coeffs_in + size, dc_rows[1] + bx,
                              dc_stride);
      int kind = acs.RawStrategy();
      const float* JXL_RESTRICT yqm = matrices.InvMatrix(kind, 1);
      const float* JXL_RESTRICT ydqm = matrices.Matrix(kind, 1);
      QuantizeRoundtripYBlockAC(yqm, ydqm, scale, quant_ac, cx, cy,
                                coeffs_in + size, quantized + size);

      // DCT X and B channels
      for (size_t c : {0, 2}) {
        TransformFromPixels(acs.Strategy(), opsin_rows[c] + bx * kBlockDim,
                            opsin_stride, coeffs_in + c * size, scratch_space);
      }

      // Unapply color correlation
      for (size_t k = 0; k < size; k += Lanes(d)) {
        const auto in_x = Load(d, coeffs_in + k);
        const auto in_y = Load(d, coeffs_in + size + k);
        const auto in_b = Load(d, coeffs_in + 2 * size + k);
        const auto out_x = NegMulAdd(x_factor, in_y, in_x);
        const auto out_b = NegMulAdd(b_factor, in_y, in_b);
        Store(out_x, d, coeffs_in + k);
        Store(out_b, d, coeffs_in + 2 * size + k);
      }

      // Quantize X and B channels and set DC.
      for (size_t c : {0, 2}) {
        const float* JXL_RESTRICT qm = matrices.InvMatrix(kind, c);
        QuantizeBlockAC(coeffs_in + c * size, c, qm, quant_ac, scale,
                        c == 0 ? x_qm_mul : 1.0, cx, cy, quantized + c * size);
        DCFromLowestFrequencies(acs.Strategy(), coeffs_in + c * size,
                                dc_rows[c] + bx, dc_stride);
      }

      // Tokenize coefficients
      const size_t log2_covered_blocks =
          Num0BitsBelowLS1Bit_Nonzero(covered_blocks);
      for (int c : {1, 0, 2}) {
        const int32_t* JXL_RESTRICT block = quantized + c * size;

        int32_t nzeros =
            (covered_blocks == 1)
                ? NumNonZero8x8ExceptDC(block, row_nzeros[c] + bx)
                : NumNonZeroExceptLLF(cx, cy, acs, covered_blocks,
                                      log2_covered_blocks, block, nzeros_stride,
                                      row_nzeros[c] + bx);

        const coeff_order_t* JXL_RESTRICT order =
            &kCoeffOrders[kCoeffOrderOffset[acs.RawStrategy()]];

        int32_t predicted_nzeros =
            PredictFromTopAndLeft(row_nzeros_top[c], row_nzeros[c], bx, 32);
        const size_t block_ctx = BlockContext(c, acs.StrategyCode());
        const size_t nzero_ctx = NonZeroContext(predicted_nzeros, block_ctx);
        const size_t histo_offset = ZeroDensityContextsOffset(block_ctx);

        output->emplace_back(nzero_ctx, nzeros);
        // Skip LLF.
        size_t prev = (nzeros > static_cast<ssize_t>(size / 16) ? 0 : 1);
        for (size_t k = covered_blocks; k < size && nzeros != 0; ++k) {
          int32_t coeff = block[order[k]];
          size_t ctx =
              histo_offset + ZeroDensityContext(nzeros, k, covered_blocks,
                                                log2_covered_blocks, prev);
          uint32_t u_coeff = PackSigned(coeff);
          output->emplace_back(ctx, u_coeff);
          prev = coeff != 0;
          nzeros -= prev;
        }
        JXL_DASSERT(nzeros == 0);
      }
    }
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {
HWY_EXPORT(ComputeCoefficients);
Status ComputeCoefficients(const Image3F& opsin, const ImageI& raw_quant_field,
                           const DequantMatrices& matrices, const float scale,
                           const ColorCorrelationMap& cmap,
                           const AcStrategyImage& ac_strategy,
                           const float x_qm_mul, ThreadPool* pool, Image3F* dc,
                           std::vector<std::vector<Token>>* ac_tokens) {
  const size_t xsize_groups = DivCeil(opsin.xsize(), kGroupDim);
  const size_t ysize_groups = DivCeil(opsin.ysize(), kGroupDim);
  std::vector<Image3I> num_nzeroes;
  const auto tokenize_group_init = [&](const size_t num_threads) {
    num_nzeroes.resize(num_threads);
    for (size_t t = 0; t < num_threads; ++t) {
      num_nzeroes[t] = Image3I(kGroupDimInBlocks, kGroupDimInBlocks);
    }
    return true;
  };
  return RunOnPool(
      pool, 0, xsize_groups * ysize_groups, tokenize_group_init,
      [&](size_t group_idx, size_t thread) {
        HWY_DYNAMIC_DISPATCH(ComputeCoefficients)
        (group_idx, opsin, raw_quant_field, matrices, scale, cmap, ac_strategy,
         x_qm_mul, &num_nzeroes[thread], dc, &(*ac_tokens)[group_idx]);
      },
      "Compute coeffs");
}

}  // namespace jxl
#endif  // HWY_ONCE
