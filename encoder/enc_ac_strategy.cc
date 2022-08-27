// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/enc_ac_strategy.h"

#include <stdint.h>

#include <algorithm>
#include <cmath>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "encoder/enc_ac_strategy.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "encoder/ac_strategy.h"
#include "encoder/base/compiler_specific.h"
#include "encoder/base/status.h"
#include "encoder/enc_transforms-inl.h"

// Some of the floating point constants in this file and in other
// files in the libjxl project have been obtained using the
// tools/optimizer/simplex_fork.py tool. It is a variation of
// Nelder-Mead optimization, and we generally try to minimize
// BPP * pnorm aggregate as reported by the benchmark_xl tool,
// but occasionally the values are optimized by using additional
// constraints such as maintaining a certain density, or ratio of
// popularity of integral transforms. Jyrki visually reviews all
// such changes and often makes manual changes to maintain good
// visual quality to changes where butteraugli was not sufficiently
// sensitive to some kind of degradation. Unfortunately image quality
// is still more of an art than science.

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::AbsDiff;
using hwy::HWY_NAMESPACE::Eq;
using hwy::HWY_NAMESPACE::IfThenElseZero;
using hwy::HWY_NAMESPACE::IfThenZeroElse;
using hwy::HWY_NAMESPACE::Round;
using hwy::HWY_NAMESPACE::Sqrt;

float EstimateEntropy(const AcStrategy& acs, const Image3F& opsin, size_t bx,
                      size_t by, const float distance,
                      const DequantMatrices& matrices, const ImageF& qf,
                      const ImageF& maskf, const float* cmap_factors,
                      float* block, float* scratch_space) {
  const size_t num_blocks = acs.covered_blocks_x() * acs.covered_blocks_y();
  const size_t size = num_blocks * kDCTBlockSize;

  // Apply transform.
  for (size_t c = 0; c < 3; c++) {
    float* JXL_RESTRICT block_c = block + size * c;
    TransformFromPixels(acs.Strategy(), &opsin.ConstPlaneRow(c, by * 8)[bx * 8],
                        opsin.PixelsPerRow(), block_c, scratch_space);
  }

  // Load QF value, calculate empirical heuristic on masking field
  // for weighting the information loss. Information loss manifests
  // itself as ringing, and masking could hide it.
  float quant = 0;
  float masking = 0;
  for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
    for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
      quant = std::max(quant, qf.ConstRow(by + iy)[bx + ix]);
      masking = std::max(masking, maskf.ConstRow(by + iy)[bx + ix]);
    }
  }

  HWY_FULL(float) df;
  const auto q = Set(df, quant);

  // Entropy estimate is composed of two factors:
  //  - estimate of the number of bits that will be used by the block
  //  - information loss due to quantization
  // The following constant controls the relative weights of these components.
  constexpr float kInfoLossMultiplier = 138.0f;
  constexpr float kInfoLossMultiplier2 = 50.46839691767866;
  float entropy = 0.0f;
  auto info_loss = Zero(df);
  auto info_loss2 = Zero(df);

  for (size_t c = 0; c < 3; c++) {
    const float* inv_matrix = matrices.InvMatrix(acs.RawStrategy(), c);
    const auto cmap_factor = Set(df, cmap_factors[c]);
    auto entropy_v = Zero(df);
    auto nzeros_v = Zero(df);
    // Lots of +1 and -1 coefficients at high quality, it is
    // beneficial to favor them. At low qualities zeros matter more
    // and +1 / -1 coefficients are already quite harmful.
    float slope = std::min<float>(1.0f, distance * (1.0f / 3));
    float cost_of_1 = 1 + slope * 8.8703248061477744f;
    constexpr float kCost2 = 4.4628149885273363f;
    auto cost1 = Set(df, cost_of_1);
    auto cost2 = Set(df, kCost2);
    constexpr float kCostDelta = 5.3359184934516337f;
    auto cost_delta = Set(df, kCostDelta);
    for (size_t i = 0; i < num_blocks * kDCTBlockSize; i += Lanes(df)) {
      const auto in = Load(df, block + c * size + i);
      const auto in_y = Mul(Load(df, block + size + i), cmap_factor);
      const auto im = Load(df, inv_matrix + i);
      const auto val = Mul(Sub(in, in_y), Mul(im, q));
      const auto rval = Round(val);
      const auto diff = AbsDiff(val, rval);
      info_loss = Add(info_loss, diff);
      info_loss2 = MulAdd(diff, diff, info_loss2);
      const auto q = Abs(rval);
      const auto q_is_zero = Eq(q, Zero(df));
      entropy_v = Add(entropy_v, IfThenElseZero(Ge(q, Set(df, 1.5f)), cost2));
      // We used to have q * C here, but that cost model seems to
      // be punishing large values more than necessary. Sqrt tries
      // to avoid large values less aggressively. Having high accuracy
      // around zero is most important at low qualities, and there
      // we have directly specified costs for 0, 1, and 2.
      entropy_v = MulAdd(Sqrt(q), cost_delta, entropy_v);
      nzeros_v = Add(nzeros_v, IfThenZeroElse(q_is_zero, Set(df, 1.0f)));
    }
    entropy_v = MulAdd(nzeros_v, cost1, entropy_v);

    entropy += GetLane(SumOfLanes(df, entropy_v));
    size_t num_nzeros = GetLane(SumOfLanes(df, nzeros_v));
    // Add #bit of num_nonzeros, as an estimate of the cost for encoding the
    // number of non-zeros of the block.
    size_t nbits = CeilLog2Nonzero(num_nzeros + 1) + 1;
    // Also add #bit of #bit of num_nonzeros, to estimate the ANS cost, with a
    // bias.
    constexpr float kZerosMul = 7.565053364251793f;
    entropy += kZerosMul * (CeilLog2Nonzero(nbits + 17) + nbits);
  }
  float infoloss = GetLane(SumOfLanes(df, info_loss));
  float infoloss2 = sqrt(num_blocks * GetLane(SumOfLanes(df, info_loss2)));
  float info_loss_score =
      (kInfoLossMultiplier * infoloss + kInfoLossMultiplier2 * infoloss2);
  return entropy + masking * info_loss_score;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {
HWY_EXPORT(EstimateEntropy);

float EstimateEntropy(const AcStrategy& acs, const Image3F& opsin, size_t bx,
                      size_t by, const float distance,
                      const DequantMatrices& matrices, const ImageF& qf,
                      const ImageF& maskf, const float* cmap_factors,
                      float* block, float* scratch_space) {
  return HWY_DYNAMIC_DISPATCH(EstimateEntropy)(
      acs, opsin, bx, by, distance, matrices, qf, maskf, cmap_factors, block,
      scratch_space);
}

void FindBest16x16Transform(const Image3F& opsin, size_t bx, size_t by,
                            size_t cx, size_t cy, float distance,
                            const DequantMatrices& matrices, const ImageF& qf,
                            const ImageF& maskf,
                            const float* JXL_RESTRICT cmap_factors,
                            AcStrategyImage* JXL_RESTRICT ac_strategy,
                            float* block, float* scratch_space) {
  const AcStrategy acs8X8 = AcStrategy::FromRawStrategy(AcStrategy::DCT);
  const AcStrategy acs16X8 = AcStrategy::FromRawStrategy(AcStrategy::DCT16X8);
  const AcStrategy acs8X16 = AcStrategy::FromRawStrategy(AcStrategy::DCT8X16);
  // Favor all 8x8 transforms 16x8 at low butteraugli_target distances.
  static const float k8x8mul1 = -0.55 * 0.75f;
  static const float k8x8mul2 = 1.0735757687292623f * 0.75f;
  static const float k8x8base = 1.4;
  static const float mul8x8 = k8x8mul2 + k8x8mul1 / (distance + k8x8base);
  static const float k8X16mul1 = -0.55;
  static const float k8X16mul2 = 0.9019587899705066;
  static const float k8X16base = 1.6;
  static const float mul16x8 = k8X16mul2 + k8X16mul1 / (distance + k8X16base);
  float entropy[2][2] = {};
  for (size_t dy = 0; dy < 2; ++dy) {
    for (size_t dx = 0; dx < 2; ++dx) {
      float entropy8x8 = 3.0f * mul8x8;
      entropy8x8 +=
          mul8x8 * EstimateEntropy(acs8X8, opsin, bx + cx + dx, by + cy + dy,
                                   distance, matrices, qf, maskf, cmap_factors,
                                   block, scratch_space);
      entropy[dy][dx] = entropy8x8;
    }
  }
  float entropy_16X8_left =
      mul16x8 * EstimateEntropy(acs16X8, opsin, bx + cx, by + cy, distance,
                                matrices, qf, maskf, cmap_factors, block,
                                scratch_space);
  float entropy_16X8_right =
      mul16x8 * EstimateEntropy(acs16X8, opsin, bx + cx + 1, by + cy, distance,
                                matrices, qf, maskf, cmap_factors, block,
                                scratch_space);
  float entropy_8X16_top =
      mul16x8 * EstimateEntropy(acs8X16, opsin, bx + cx, by + cy, distance,
                                matrices, qf, maskf, cmap_factors, block,
                                scratch_space);
  float entropy_8X16_bottom =
      mul16x8 * EstimateEntropy(acs8X16, opsin, bx + cx, by + cy + 1, distance,
                                matrices, qf, maskf, cmap_factors, block,
                                scratch_space);
  // Test if this 16x16 block should have 16x8 or 8x16 transforms,
  // because it can have only one or the other.
  float cost16x8 = std::min(entropy_16X8_left, entropy[0][0] + entropy[1][0]) +
                   std::min(entropy_16X8_right, entropy[0][1] + entropy[1][1]);
  float cost8x16 = std::min(entropy_8X16_top, entropy[0][0] + entropy[0][1]) +
                   std::min(entropy_8X16_bottom, entropy[1][0] + entropy[1][1]);
  if (cost16x8 < cost8x16) {
    if (entropy_16X8_left < entropy[0][0] + entropy[1][0]) {
      ac_strategy->Set(bx + cx, by + cy, AcStrategy::DCT16X8);
    }
    if (entropy_16X8_right < entropy[0][1] + entropy[1][1]) {
      ac_strategy->Set(bx + cx + 1, by + cy, AcStrategy::DCT16X8);
    }
  } else {
    if (entropy_8X16_top < entropy[0][0] + entropy[0][1]) {
      ac_strategy->Set(bx + cx, by + cy, AcStrategy::DCT8X16);
    }
    if (entropy_8X16_bottom < entropy[1][0] + entropy[1][1]) {
      ac_strategy->Set(bx + cx, by + cy + 1, AcStrategy::DCT8X16);
    }
  }
}

Status ComputeAcStrategyImage(const Image3F& opsin, const float distance,
                              const ColorCorrelationMap& cmap,
                              const ImageF& quant_field,
                              const ImageF& masking_field, ThreadPool* pool,
                              const DequantMatrices& matrices,
                              AcStrategyImage* ac_strategy) {
  ac_strategy->FillDCT8();
  size_t xsize_blocks = DivCeil(opsin.xsize(), kBlockDim);
  size_t ysize_blocks = DivCeil(opsin.ysize(), kBlockDim);
  size_t xsize_tiles = DivCeil(xsize_blocks, kColorTileDimInBlocks);
  size_t ysize_tiles = DivCeil(ysize_blocks, kColorTileDimInBlocks);
  auto process_tile_acs = [&](const uint32_t tid, const size_t thread) {
    size_t tx = tid % xsize_tiles;
    size_t ty = tid / xsize_tiles;
    size_t by0 = ty * kColorTileDimInBlocks;
    size_t by1 = std::min((ty + 1) * kColorTileDimInBlocks, ysize_blocks);
    size_t bx0 = tx * kColorTileDimInBlocks;
    size_t bx1 = std::min((tx + 1) * kColorTileDimInBlocks, xsize_blocks);
    Rect rect(bx0, by0, bx1 - bx0, by1 - by0);
    auto mem = hwy::AllocateAligned<float>(5 * AcStrategy::kMaxCoeffArea);
    float* block = mem.get();
    float* scratch_space = mem.get() + 3 * AcStrategy::kMaxCoeffArea;
    const float cmap_factors[3] = {
        cmap.YtoXRatio(cmap.ytox_map.ConstRow(ty)[tx]),
        0.0f,
        cmap.YtoBRatio(cmap.ytob_map.ConstRow(ty)[tx]),
    };
    for (size_t cy = 0; cy + 1 < rect.ysize(); cy += 2) {
      for (size_t cx = 0; cx + 1 < rect.xsize(); cx += 2) {
        FindBest16x16Transform(opsin, bx0, by0, cx, cy, distance, matrices,
                               quant_field, masking_field, cmap_factors,
                               ac_strategy, block, scratch_space);
      }
    }
  };
  return RunOnPool(pool, 0, xsize_tiles * ysize_tiles, ThreadPool::NoInit,
                   process_tile_acs, "Acs Heuristics");
}

void AdjustQuantField(const AcStrategyImage& ac_strategy, ImageI* quant_field) {
  // Replace the whole quant_field in non-8x8 blocks with the maximum of each
  // 8x8 block.
  size_t stride = quant_field->PixelsPerRow();
  for (size_t y = 0; y < quant_field->ysize(); ++y) {
    AcStrategyRow ac_strategy_row = ac_strategy.ConstRow(y);
    int* JXL_RESTRICT quant_row = quant_field->Row(y);
    for (size_t x = 0; x < quant_field->xsize(); ++x) {
      AcStrategy acs = ac_strategy_row[x];
      if (!acs.IsFirstBlock()) continue;
      JXL_ASSERT(x + acs.covered_blocks_x() <= quant_field->xsize());
      JXL_ASSERT(y + acs.covered_blocks_y() <= quant_field->ysize());
      int max = quant_row[x];
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

}  // namespace jxl
#endif  // HWY_ONCE
