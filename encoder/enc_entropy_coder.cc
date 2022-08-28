// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/enc_entropy_coder.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <utility>
#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "encoder/enc_entropy_coder.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "encoder/ac_context.h"
#include "encoder/ac_strategy.h"
#include "encoder/base/bits.h"
#include "encoder/base/compiler_specific.h"
#include "encoder/base/status.h"
#include "encoder/common.h"
#include "encoder/image.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::AndNot;
using hwy::HWY_NAMESPACE::Eq;
using hwy::HWY_NAMESPACE::GetLane;

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

// The number of nonzeros of each block is predicted from the top and the left
// blocks, with opportune scaling to take into account the number of blocks of
// each strategy.  The predicted number of nonzeros divided by two is used as a
// context; if this number is above 63, a specific context is used.  If the
// number of nonzeros of a strategy is above 63, it is written directly using a
// fixed number of bits (that depends on the size of the strategy).
void TokenizeCoefficients(const Rect& rect,
                          const int32_t* JXL_RESTRICT* JXL_RESTRICT ac_rows,
                          const AcStrategyImage& ac_strategy,
                          Image3I* JXL_RESTRICT tmp_num_nzeroes,
                          std::vector<Token>* JXL_RESTRICT output) {
  const size_t xsize_blocks = rect.xsize();
  const size_t ysize_blocks = rect.ysize();

  // TODO(user): update the estimate: usually less coefficients are used.
  output->reserve(output->size() +
                  3 * xsize_blocks * ysize_blocks * kDCTBlockSize);

  size_t offset[3] = {};
  const size_t nzeros_stride = tmp_num_nzeroes->PixelsPerRow();
  for (size_t by = 0; by < ysize_blocks; ++by) {
    size_t sby[3] = {by, by, by};
    int32_t* JXL_RESTRICT row_nzeros[3] = {
        tmp_num_nzeroes->PlaneRow(0, sby[0]),
        tmp_num_nzeroes->PlaneRow(1, sby[1]),
        tmp_num_nzeroes->PlaneRow(2, sby[2]),
    };
    const int32_t* JXL_RESTRICT row_nzeros_top[3] = {
        sby[0] == 0 ? nullptr : tmp_num_nzeroes->ConstPlaneRow(0, sby[0] - 1),
        sby[1] == 0 ? nullptr : tmp_num_nzeroes->ConstPlaneRow(1, sby[1] - 1),
        sby[2] == 0 ? nullptr : tmp_num_nzeroes->ConstPlaneRow(2, sby[2] - 1),
    };
    AcStrategyRow acs_row = ac_strategy.ConstRow(rect, by);
    for (size_t bx = 0; bx < xsize_blocks; ++bx) {
      AcStrategy acs = acs_row[bx];
      if (!acs.IsFirstBlock()) continue;
      size_t sbx[3] = {bx, bx, bx};
      size_t cx = acs.covered_blocks_x();
      size_t cy = acs.covered_blocks_y();
      const size_t covered_blocks = cx * cy;  // = #LLF coefficients
      const size_t log2_covered_blocks =
          Num0BitsBelowLS1Bit_Nonzero(covered_blocks);
      const size_t size = covered_blocks * kDCTBlockSize;

      if (cy > cx) std::swap(cx, cy);

      for (int c : {1, 0, 2}) {
        const int32_t* JXL_RESTRICT block = ac_rows[c] + offset[c];

        int32_t nzeros =
            (covered_blocks == 1)
                ? NumNonZero8x8ExceptDC(block, row_nzeros[c] + sbx[c])
                : NumNonZeroExceptLLF(cx, cy, acs, covered_blocks,
                                      log2_covered_blocks, block, nzeros_stride,
                                      row_nzeros[c] + sbx[c]);

        const coeff_order_t* JXL_RESTRICT order =
            &kCoeffOrders[kCoeffOrderOffset[acs.RawStrategy()]];

        int32_t predicted_nzeros =
            PredictFromTopAndLeft(row_nzeros_top[c], row_nzeros[c], sbx[c], 32);
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
        offset[c] += size;
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
HWY_EXPORT(TokenizeCoefficients);
void TokenizeCoefficients(const Rect& rect,
                          const int32_t* JXL_RESTRICT* JXL_RESTRICT ac_rows,
                          const AcStrategyImage& ac_strategy,
                          Image3I* JXL_RESTRICT tmp_num_nzeroes,
                          std::vector<Token>* JXL_RESTRICT output) {
  return HWY_DYNAMIC_DISPATCH(TokenizeCoefficients)(rect, ac_rows, ac_strategy,
                                                    tmp_num_nzeroes, output);
}

}  // namespace jxl
#endif  // HWY_ONCE
