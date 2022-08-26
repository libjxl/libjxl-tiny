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

#include "encoder/ac_strategy.h"
#include "encoder/base/bits.h"
#include "encoder/base/compiler_specific.h"
#include "encoder/common.h"
#include "encoder/dct_util.h"
#include "encoder/enc_bit_writer.h"
#include "encoder/enc_transforms-inl.h"
#include "encoder/image.h"
#include "encoder/quantizer.h"
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
void QuantizeBlockAC(const Quantizer& quantizer, size_t c, int32_t quant,
                     float qm_multiplier, size_t quant_kind, size_t xsize,
                     size_t ysize, const float* JXL_RESTRICT block_in,
                     int32_t* JXL_RESTRICT block_out) {
  const float* JXL_RESTRICT qm = quantizer.InvDequantMatrix(quant_kind, c);
  const float qac = quantizer.Scale() * quant;
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
void QuantizeRoundtripYBlockAC(const Quantizer& quantizer, int32_t quant,
                               size_t quant_kind, size_t xsize, size_t ysize,
                               const float* JXL_RESTRICT biases,
                               float* JXL_RESTRICT inout,
                               int32_t* JXL_RESTRICT quantized) {
  QuantizeBlockAC(quantizer, 1, quant, 1.0f, quant_kind, xsize, ysize, inout,
                  quantized);

  const float* JXL_RESTRICT dequant_matrix =
      quantizer.DequantMatrix(quant_kind, 1);

  HWY_CAPPED(float, kDCTBlockSize) df;
  HWY_CAPPED(int32_t, kDCTBlockSize) di;
  const auto inv_qac = Set(df, quantizer.inv_quant_ac(quant));
  for (size_t k = 0; k < kDCTBlockSize * xsize * ysize; k += Lanes(df)) {
    const auto quant = Load(di, quantized + k);
    const auto adj_quant = AdjustQuantBias(di, 1, quant, biases);
    const auto dequantm = Load(df, dequant_matrix + k);
    Store(Mul(Mul(adj_quant, dequantm), inv_qac), df, inout + k);
  }
}

void ComputeCoefficients(size_t group_idx, const Image3F& opsin,
                         const ImageI& raw_quant_field,
                         const Quantizer& quantizer,
                         const ColorCorrelationMap& cmap,
                         const AcStrategyImage& ac_strategy,
                         const float x_qm_mul, ACImage* coeffs_out,
                         Image3F* dc) {
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
  {
    constexpr HWY_CAPPED(float, kDCTBlockSize) d;

    int32_t* JXL_RESTRICT coeffs[3] = {};
    // TODO(veluca): 16-bit quantized coeffs are not implemented yet.
    JXL_ASSERT(coeffs_out->Type() == ACType::k32);
    for (size_t c = 0; c < 3; c++) {
      coeffs[c] = coeffs_out->PlaneRow(c, group_idx, 0).ptr32;
    }

    HWY_ALIGN float* coeffs_in = fmem.get();
    HWY_ALIGN int32_t* quantized = mem.get();

    size_t offset = 0;

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
      AcStrategyRow ac_strategy_row =
          ac_strategy.ConstRow(block_group_rect, by);
      for (size_t tx = 0; tx < DivCeil(xsize_blocks, kColorTileDimInBlocks);
           tx++) {
        const auto x_factor = Set(d, cmap.YtoXRatio(row_cmap[0][tx]));
        const auto b_factor = Set(d, cmap.YtoBRatio(row_cmap[2][tx]));
        for (size_t bx = tx * kColorTileDimInBlocks;
             bx < xsize_blocks && bx < (tx + 1) * kColorTileDimInBlocks; ++bx) {
          const AcStrategy acs = ac_strategy_row[bx];
          if (!acs.IsFirstBlock()) continue;

          size_t xblocks = acs.covered_blocks_x();
          size_t yblocks = acs.covered_blocks_y();

          CoefficientLayout(&yblocks, &xblocks);

          size_t size = kDCTBlockSize * xblocks * yblocks;

          // DCT Y channel, roundtrip-quantize it and set DC.
          const int32_t quant_ac = row_quant_ac[bx];
          TransformFromPixels(acs.Strategy(), opsin_rows[1] + bx * kBlockDim,
                              opsin_stride, coeffs_in + size, scratch_space);
          DCFromLowestFrequencies(acs.Strategy(), coeffs_in + size,
                                  dc_rows[1] + bx, dc_stride);
          QuantizeRoundtripYBlockAC(quantizer, quant_ac, acs.RawStrategy(),
                                    xblocks, yblocks, kDefaultQuantBias,
                                    coeffs_in + size, quantized + size);

          // DCT X and B channels
          for (size_t c : {0, 2}) {
            TransformFromPixels(acs.Strategy(), opsin_rows[c] + bx * kBlockDim,
                                opsin_stride, coeffs_in + c * size,
                                scratch_space);
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
            QuantizeBlockAC(quantizer, c, quant_ac, c == 0 ? x_qm_mul : 1.0,
                            acs.RawStrategy(), xblocks, yblocks,
                            coeffs_in + c * size, quantized + c * size);
            DCFromLowestFrequencies(acs.Strategy(), coeffs_in + c * size,
                                    dc_rows[c] + bx, dc_stride);
          }
          for (size_t c = 0; c < 3; c++) {
            memcpy(coeffs[c] + offset, quantized + c * size,
                   sizeof(quantized[0]) * size);
          }
          offset += size;
        }
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
void ComputeCoefficients(size_t group_idx, const Image3F& opsin,
                         const ImageI& raw_quant_field,
                         const Quantizer& quantizer,
                         const ColorCorrelationMap& cmap,
                         const AcStrategyImage& ac_strategy,
                         const float x_qm_mul, ACImage* coeffs, Image3F* dc) {
  return HWY_DYNAMIC_DISPATCH(ComputeCoefficients)(
      group_idx, opsin, raw_quant_field, quantizer, cmap, ac_strategy, x_qm_mul,
      coeffs, dc);
}

}  // namespace jxl
#endif  // HWY_ONCE
