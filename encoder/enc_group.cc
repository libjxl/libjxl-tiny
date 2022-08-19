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
#include "encoder/base/profiler.h"
#include "encoder/common.h"
#include "encoder/dct_util.h"
#include "encoder/dec_transforms-inl.h"
#include "encoder/enc_bit_writer.h"
#include "encoder/enc_transforms-inl.h"
#include "encoder/image.h"
#include "encoder/quantizer-inl.h"
#include "encoder/quantizer.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Abs;
using hwy::HWY_NAMESPACE::Ge;
using hwy::HWY_NAMESPACE::IfThenElse;
using hwy::HWY_NAMESPACE::IfThenElseZero;
using hwy::HWY_NAMESPACE::MaskFromVec;
using hwy::HWY_NAMESPACE::Round;

// NOTE: caller takes care of extracting quant from rect of RawQuantField.
void QuantizeBlockAC(const Quantizer& quantizer, size_t c, int32_t quant,
                     float qm_multiplier, size_t quant_kind, size_t xsize,
                     size_t ysize, const float* JXL_RESTRICT block_in,
                     int32_t* JXL_RESTRICT block_out) {
  PROFILER_FUNC;
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

  PROFILER_ZONE("enc quant adjust bias");
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

void ComputeCoefficients(size_t group_idx, PassesEncoderState* enc_state,
                         const Image3F& opsin, Image3F* dc) {
  PROFILER_FUNC;
  const Rect block_group_rect = enc_state->shared.BlockGroupRect(group_idx);
  const Rect group_rect = enc_state->shared.GroupRect(group_idx);
  const Rect cmap_rect(
      block_group_rect.x0() / kColorTileDimInBlocks,
      block_group_rect.y0() / kColorTileDimInBlocks,
      DivCeil(block_group_rect.xsize(), kColorTileDimInBlocks),
      DivCeil(block_group_rect.ysize(), kColorTileDimInBlocks));

  const size_t xsize_blocks = block_group_rect.xsize();
  const size_t ysize_blocks = block_group_rect.ysize();

  const size_t dc_stride = static_cast<size_t>(dc->PixelsPerRow());
  const size_t opsin_stride = static_cast<size_t>(opsin.PixelsPerRow());

  const ImageI& full_quant_field = enc_state->shared.raw_quant_field;

  // TODO(veluca): consider strategies to reduce this memory.
  auto mem = hwy::AllocateAligned<int32_t>(3 * AcStrategy::kMaxCoeffArea);
  auto fmem = hwy::AllocateAligned<float>(5 * AcStrategy::kMaxCoeffArea);
  float* JXL_RESTRICT scratch_space =
      fmem.get() + 3 * AcStrategy::kMaxCoeffArea;
  {
    constexpr HWY_CAPPED(float, kDCTBlockSize) d;

    int32_t* JXL_RESTRICT coeffs[kMaxNumPasses][3] = {};
    // TODO(veluca): 16-bit quantized coeffs are not implemented yet.
    JXL_ASSERT(enc_state->coeffs[0]->Type() == ACType::k32);
    for (size_t c = 0; c < 3; c++) {
      coeffs[0][c] = enc_state->coeffs[0]->PlaneRow(c, group_idx, 0).ptr32;
    }

    HWY_ALIGN float* coeffs_in = fmem.get();
    HWY_ALIGN int32_t* quantized = mem.get();

    size_t offset = 0;

    for (size_t by = 0; by < ysize_blocks; ++by) {
      const int32_t* JXL_RESTRICT row_quant_ac =
          block_group_rect.ConstRow(full_quant_field, by);
      size_t ty = by / kColorTileDimInBlocks;
      const int8_t* JXL_RESTRICT row_cmap[3] = {
          cmap_rect.ConstRow(enc_state->shared.cmap.ytox_map, ty),
          nullptr,
          cmap_rect.ConstRow(enc_state->shared.cmap.ytob_map, ty),
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
          enc_state->shared.ac_strategy.ConstRow(block_group_rect, by);
      for (size_t tx = 0; tx < DivCeil(xsize_blocks, kColorTileDimInBlocks);
           tx++) {
        const auto x_factor =
            Set(d, enc_state->shared.cmap.YtoXRatio(row_cmap[0][tx]));
        const auto b_factor =
            Set(d, enc_state->shared.cmap.YtoBRatio(row_cmap[2][tx]));
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
          QuantizeRoundtripYBlockAC(
              enc_state->shared.quantizer, quant_ac, acs.RawStrategy(), xblocks,
              yblocks, kDefaultQuantBias, coeffs_in + size, quantized + size);

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
            QuantizeBlockAC(enc_state->shared.quantizer, c, quant_ac,
                            c == 0 ? enc_state->x_qm_multiplier
                                   : enc_state->b_qm_multiplier,
                            acs.RawStrategy(), xblocks, yblocks,
                            coeffs_in + c * size, quantized + c * size);
            DCFromLowestFrequencies(acs.Strategy(), coeffs_in + c * size,
                                    dc_rows[c] + bx, dc_stride);
          }
          for (size_t c = 0; c < 3; c++) {
            memcpy(coeffs[0][c] + offset, quantized + c * size,
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
void ComputeCoefficients(size_t group_idx, PassesEncoderState* enc_state,
                         const Image3F& opsin, Image3F* dc) {
  return HWY_DYNAMIC_DISPATCH(ComputeCoefficients)(group_idx, enc_state, opsin,
                                                   dc);
}

Status EncodeGroupTokenizedCoefficients(size_t group_idx, size_t pass_idx,
                                        size_t histogram_idx,
                                        const PassesEncoderState& enc_state,
                                        BitWriter* writer) {
  // Select which histogram to use among those of the current pass.
  const size_t num_histograms = enc_state.shared.num_histograms;
  // num_histograms is 0 only for lossless.
  JXL_ASSERT(num_histograms == 0 || histogram_idx < num_histograms);
  size_t histo_selector_bits = CeilLog2Nonzero(num_histograms);

  if (histo_selector_bits != 0) {
    BitWriter::Allotment allotment(writer, histo_selector_bits);
    writer->Write(histo_selector_bits, histogram_idx);
    allotment.Reclaim(writer);
  }
  WriteTokens(enc_state.passes[pass_idx].ac_tokens[group_idx],
              enc_state.passes[pass_idx].codes,
              enc_state.passes[pass_idx].context_map, writer);

  return true;
}

}  // namespace jxl
#endif  // HWY_ONCE