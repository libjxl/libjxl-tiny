// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd
#include "encoder/quant_weights.h"

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "encoder/base/bits.h"
#include "encoder/base/printf_macros.h"
#include "encoder/base/status.h"
#include "encoder/common.h"
#include "encoder/image.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "encoder/quant_weights.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "encoder/fast_math-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Lt;
using hwy::HWY_NAMESPACE::MulAdd;
using hwy::HWY_NAMESPACE::Sqrt;

// kQuantWeights[N * N * c + N * y + x] is the relative weight of the (x, y)
// coefficient in component c. Higher weights correspond to finer quantization
// intervals and more bits spent in encoding.

static constexpr const float kAlmostZero = 1e-8f;

void GetQuantWeightsDCT2(const QuantEncoding::DCT2Weights& dct2weights,
                         float* weights) {
  for (size_t c = 0; c < 3; c++) {
    size_t start = c * 64;
    weights[start] = 0xBAD;
    weights[start + 1] = weights[start + 8] = dct2weights[c][0];
    weights[start + 9] = dct2weights[c][1];
    for (size_t y = 0; y < 2; y++) {
      for (size_t x = 0; x < 2; x++) {
        weights[start + y * 8 + x + 2] = dct2weights[c][2];
        weights[start + (y + 2) * 8 + x] = dct2weights[c][2];
      }
    }
    for (size_t y = 0; y < 2; y++) {
      for (size_t x = 0; x < 2; x++) {
        weights[start + (y + 2) * 8 + x + 2] = dct2weights[c][3];
      }
    }
    for (size_t y = 0; y < 4; y++) {
      for (size_t x = 0; x < 4; x++) {
        weights[start + y * 8 + x + 4] = dct2weights[c][4];
        weights[start + (y + 4) * 8 + x] = dct2weights[c][4];
      }
    }
    for (size_t y = 0; y < 4; y++) {
      for (size_t x = 0; x < 4; x++) {
        weights[start + (y + 4) * 8 + x + 4] = dct2weights[c][5];
      }
    }
  }
}

void GetQuantWeightsIdentity(const QuantEncoding::IdWeights& idweights,
                             float* weights) {
  for (size_t c = 0; c < 3; c++) {
    for (int i = 0; i < 64; i++) {
      weights[64 * c + i] = idweights[c][0];
    }
    weights[64 * c + 1] = idweights[c][1];
    weights[64 * c + 8] = idweights[c][1];
    weights[64 * c + 9] = idweights[c][2];
  }
}

float Mult(float v) {
  if (v > 0.0f) return 1.0f + v;
  return 1.0f / (1.0f - v);
}

using DF4 = HWY_CAPPED(float, 4);

hwy::HWY_NAMESPACE::Vec<DF4> InterpolateVec(
    hwy::HWY_NAMESPACE::Vec<DF4> scaled_pos, const float* array) {
  HWY_CAPPED(int32_t, 4) di;

  auto idx = ConvertTo(di, scaled_pos);

  auto frac = Sub(scaled_pos, ConvertTo(DF4(), idx));

  // TODO(veluca): in theory, this could be done with 8 TableLookupBytes, but
  // it's probably slower.
  auto a = GatherIndex(DF4(), array, idx);
  auto b = GatherIndex(DF4(), array + 1, idx);

  return Mul(a, FastPowf(DF4(), Div(b, a), frac));
}

// Computes quant weights for a COLS*ROWS-sized transform, using num_bands
// eccentricity bands and num_ebands eccentricity bands. If print_mode is 1,
// prints the resulting matrix; if print_mode is 2, prints the matrix in a
// format suitable for a 3d plot with gnuplot.
Status GetQuantWeights(
    size_t ROWS, size_t COLS,
    const DctQuantWeightParams::DistanceBandsArray& distance_bands,
    size_t num_bands, float* out) {
  for (size_t c = 0; c < 3; c++) {
    float bands[DctQuantWeightParams::kMaxDistanceBands] = {
        distance_bands[c][0]};
    if (bands[0] < kAlmostZero) return JXL_FAILURE("Invalid distance bands");
    for (size_t i = 1; i < num_bands; i++) {
      bands[i] = bands[i - 1] * Mult(distance_bands[c][i]);
      if (bands[i] < kAlmostZero) return JXL_FAILURE("Invalid distance bands");
    }
    static constexpr float kSqrt2 = 1.41421356237f;
    float scale = (num_bands - 1) / (kSqrt2 + 1e-6f);
    float rcpcol = scale / (COLS - 1);
    float rcprow = scale / (ROWS - 1);
    JXL_ASSERT(COLS >= Lanes(DF4()));
    HWY_ALIGN float l0123[4] = {0, 1, 2, 3};
    for (uint32_t y = 0; y < ROWS; y++) {
      float dy = y * rcprow;
      float dy2 = dy * dy;
      for (uint32_t x = 0; x < COLS; x += Lanes(DF4())) {
        auto dx =
            Mul(Add(Set(DF4(), x), Load(DF4(), l0123)), Set(DF4(), rcpcol));
        auto scaled_distance = Sqrt(MulAdd(dx, dx, Set(DF4(), dy2)));
        auto weight = num_bands == 1 ? Set(DF4(), bands[0])
                                     : InterpolateVec(scaled_distance, bands);
        StoreU(weight, DF4(), out + c * COLS * ROWS + y * COLS + x);
      }
    }
  }
  return true;
}

// TODO(veluca): SIMD-fy. With 256x256, this is actually slow.
Status ComputeQuantTable(const QuantEncoding& encoding,
                         float* JXL_RESTRICT table,
                         float* JXL_RESTRICT inv_table, size_t table_num,
                         DequantMatrices::QuantTable kind, size_t* pos) {
  constexpr size_t N = kBlockDim;
  size_t wrows = 8 * DequantMatrices::required_size_x[kind],
         wcols = 8 * DequantMatrices::required_size_y[kind];
  size_t num = wrows * wcols;

  std::vector<float> weights(3 * num);

  switch (encoding.mode) {
    case QuantEncoding::kQuantModeLibrary: {
      // Library and copy quant encoding should get replaced by the actual
      // parameters by the caller.
      JXL_ASSERT(false);
      break;
    }
    case QuantEncoding::kQuantModeID: {
      JXL_ASSERT(num == kDCTBlockSize);
      GetQuantWeightsIdentity(encoding.idweights, weights.data());
      break;
    }
    case QuantEncoding::kQuantModeDCT2: {
      JXL_ASSERT(num == kDCTBlockSize);
      GetQuantWeightsDCT2(encoding.dct2weights, weights.data());
      break;
    }
    case QuantEncoding::kQuantModeDCT4: {
      JXL_ASSERT(num == kDCTBlockSize);
      float weights4x4[3 * 4 * 4];
      // Always use 4x4 GetQuantWeights for DCT4 quantization tables.
      JXL_RETURN_IF_ERROR(
          GetQuantWeights(4, 4, encoding.dct_params.distance_bands,
                          encoding.dct_params.num_distance_bands, weights4x4));
      for (size_t c = 0; c < 3; c++) {
        for (size_t y = 0; y < kBlockDim; y++) {
          for (size_t x = 0; x < kBlockDim; x++) {
            weights[c * num + y * kBlockDim + x] =
                weights4x4[c * 16 + (y / 2) * 4 + (x / 2)];
          }
        }
        weights[c * num + 1] /= encoding.dct4multipliers[c][0];
        weights[c * num + N] /= encoding.dct4multipliers[c][0];
        weights[c * num + N + 1] /= encoding.dct4multipliers[c][1];
      }
      break;
    }
    case QuantEncoding::kQuantModeDCT4X8: {
      JXL_ASSERT(num == kDCTBlockSize);
      float weights4x8[3 * 4 * 8];
      // Always use 4x8 GetQuantWeights for DCT4X8 quantization tables.
      JXL_RETURN_IF_ERROR(
          GetQuantWeights(4, 8, encoding.dct_params.distance_bands,
                          encoding.dct_params.num_distance_bands, weights4x8));
      for (size_t c = 0; c < 3; c++) {
        for (size_t y = 0; y < kBlockDim; y++) {
          for (size_t x = 0; x < kBlockDim; x++) {
            weights[c * num + y * kBlockDim + x] =
                weights4x8[c * 32 + (y / 2) * 8 + x];
          }
        }
        weights[c * num + N] /= encoding.dct4x8multipliers[c];
      }
      break;
    }
    case QuantEncoding::kQuantModeDCT: {
      JXL_RETURN_IF_ERROR(GetQuantWeights(
          wrows, wcols, encoding.dct_params.distance_bands,
          encoding.dct_params.num_distance_bands, weights.data()));
      break;
    }
  }
  size_t prev_pos = *pos;
  HWY_CAPPED(float, 64) d;
  for (size_t i = 0; i < num * 3; i += Lanes(d)) {
    auto inv_val = LoadU(d, weights.data() + i);
    if (JXL_UNLIKELY(!AllFalse(d, Ge(inv_val, Set(d, 1.0f / kAlmostZero))) ||
                     !AllFalse(d, Lt(inv_val, Set(d, kAlmostZero))))) {
      return JXL_FAILURE("Invalid quantization table");
    }
    auto val = Div(Set(d, 1.0f), inv_val);
    StoreU(val, d, table + *pos + i);
    StoreU(inv_val, d, inv_table + *pos + i);
  }
  (*pos) += 3 * num;

  // Ensure that the lowest frequencies have a 0 inverse table.
  // This does not affect en/decoding, but allows AC strategy selection to be
  // slightly simpler.
  size_t xs = DequantMatrices::required_size_x[kind];
  size_t ys = DequantMatrices::required_size_y[kind];
  CoefficientLayout(&ys, &xs);
  for (size_t c = 0; c < 3; c++) {
    for (size_t y = 0; y < ys; y++) {
      for (size_t x = 0; x < xs; x++) {
        inv_table[prev_pos + c * ys * xs * kDCTBlockSize + y * kBlockDim * xs +
                  x] = 0;
      }
    }
  }
  return true;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace jxl {
namespace {

HWY_EXPORT(ComputeQuantTable);

}  // namespace

// These definitions are needed before C++17.
constexpr size_t DequantMatrices::required_size_[];
constexpr size_t DequantMatrices::required_size_x[];
constexpr size_t DequantMatrices::required_size_y[];
constexpr DequantMatrices::QuantTable DequantMatrices::kQuantTable[];

constexpr float V(float v) { return static_cast<float>(v); }

namespace {
struct DequantMatricesLibraryDef {
  // DCT8
  static constexpr const QuantEncodingInternal DCT() {
    return QuantEncodingInternal::DCT(DctQuantWeightParams({{{{
                                                                 V(3150.0),
                                                                 V(0.0),
                                                                 V(-0.4),
                                                                 V(-0.4),
                                                                 V(-0.4),
                                                                 V(-2.0),
                                                             }},
                                                             {{
                                                                 V(560.0),
                                                                 V(0.0),
                                                                 V(-0.3),
                                                                 V(-0.3),
                                                                 V(-0.3),
                                                                 V(-0.3),
                                                             }},
                                                             {{
                                                                 V(512.0),
                                                                 V(-2.0),
                                                                 V(-1.0),
                                                                 V(0.0),
                                                                 V(-1.0),
                                                                 V(-2.0),
                                                             }}}},
                                                           6));
  }

  // Identity
  static constexpr const QuantEncodingInternal IDENTITY() {
    return QuantEncodingInternal::Identity({{{{
                                                 V(280.0),
                                                 V(3160.0),
                                                 V(3160.0),
                                             }},
                                             {{
                                                 V(60.0),
                                                 V(864.0),
                                                 V(864.0),
                                             }},
                                             {{
                                                 V(18.0),
                                                 V(200.0),
                                                 V(200.0),
                                             }}}});
  }

  // DCT2
  static constexpr const QuantEncodingInternal DCT2X2() {
    return QuantEncodingInternal::DCT2({{{{
                                             V(3840.0),
                                             V(2560.0),
                                             V(1280.0),
                                             V(640.0),
                                             V(480.0),
                                             V(300.0),
                                         }},
                                         {{
                                             V(960.0),
                                             V(640.0),
                                             V(320.0),
                                             V(180.0),
                                             V(140.0),
                                             V(120.0),
                                         }},
                                         {{
                                             V(640.0),
                                             V(320.0),
                                             V(128.0),
                                             V(64.0),
                                             V(32.0),
                                             V(16.0),
                                         }}}});
  }

  // DCT4 (quant_kind 3)
  static constexpr const QuantEncodingInternal DCT4X4() {
    return QuantEncodingInternal::DCT4(DctQuantWeightParams({{{{
                                                                  V(2200.0),
                                                                  V(0.0),
                                                                  V(0.0),
                                                                  V(0.0),
                                                              }},
                                                              {{
                                                                  V(392.0),
                                                                  V(0.0),
                                                                  V(0.0),
                                                                  V(0.0),
                                                              }},
                                                              {{
                                                                  V(112.0),
                                                                  V(-0.25),
                                                                  V(-0.25),
                                                                  V(-0.5),
                                                              }}}},
                                                            4),
                                       /* kMul */
                                       {{{{
                                             V(1.0),
                                             V(1.0),
                                         }},
                                         {{
                                             V(1.0),
                                             V(1.0),
                                         }},
                                         {{
                                             V(1.0),
                                             V(1.0),
                                         }}}});
  }

  // DCT16
  static constexpr const QuantEncodingInternal DCT16X16() {
    return QuantEncodingInternal::DCT(
        DctQuantWeightParams({{{{
                                   V(8996.8725711814115328),
                                   V(-1.3000777393353804),
                                   V(-0.49424529824571225),
                                   V(-0.439093774457103443),
                                   V(-0.6350101832695744),
                                   V(-0.90177264050827612),
                                   V(-1.6162099239887414),
                               }},
                               {{
                                   V(3191.48366296844234752),
                                   V(-0.67424582104194355),
                                   V(-0.80745813428471001),
                                   V(-0.44925837484843441),
                                   V(-0.35865440981033403),
                                   V(-0.31322389111877305),
                                   V(-0.37615025315725483),
                               }},
                               {{
                                   V(1157.50408145487200256),
                                   V(-2.0531423165804414),
                                   V(-1.4),
                                   V(-0.50687130033378396),
                                   V(-0.42708730624733904),
                                   V(-1.4856834539296244),
                                   V(-4.9209142884401604),
                               }}}},
                             7));
  }

  // DCT16X8
  static constexpr const QuantEncodingInternal DCT8X16() {
    return QuantEncodingInternal::DCT(
        DctQuantWeightParams({{{{
                                   V(7240.7734393502),
                                   V(-0.7),
                                   V(-0.7),
                                   V(-0.2),
                                   V(-0.2),
                                   V(-0.2),
                                   V(-0.5),
                               }},
                               {{
                                   V(1448.15468787004),
                                   V(-0.5),
                                   V(-0.5),
                                   V(-0.5),
                                   V(-0.2),
                                   V(-0.2),
                                   V(-0.2),
                               }},
                               {{
                                   V(506.854140754517),
                                   V(-1.4),
                                   V(-0.2),
                                   V(-0.5),
                                   V(-0.5),
                                   V(-1.5),
                                   V(-3.6),
                               }}}},
                             7));
  }

  // DCT4X8 and 8x4
  static constexpr const QuantEncodingInternal DCT4X8() {
    return QuantEncodingInternal::DCT4X8(
        DctQuantWeightParams({{
                                 {{
                                     V(2198.050556016380522),
                                     V(-0.96269623020744692),
                                     V(-0.76194253026666783),
                                     V(-0.6551140670773547),
                                 }},
                                 {{
                                     V(764.3655248643528689),
                                     V(-0.92630200888366945),
                                     V(-0.9675229603596517),
                                     V(-0.27845290869168118),
                                 }},
                                 {{
                                     V(527.107573587542228),
                                     V(-1.4594385811273854),
                                     V(-1.450082094097871593),
                                     V(-1.5843722511996204),
                                 }},
                             }},
                             4),
        /* kMuls */
        {{
            V(1.0),
            V(1.0),
            V(1.0),
        }});
  }

};
}  // namespace

const DequantMatrices::DequantLibraryInternal DequantMatrices::LibraryInit() {
  static_assert(kNum == 7,
                "Update this function when adding new quantization kinds.");
  static_assert(kNumPredefinedTables == 1,
                "Update this function when adding new quantization matrices to "
                "the library.");

  // The library and the indices need to be kept in sync manually.
  static_assert(0 == DCT, "Update the DequantLibrary array below.");
  static_assert(1 == IDENTITY, "Update the DequantLibrary array below.");
  static_assert(2 == DCT2X2, "Update the DequantLibrary array below.");
  static_assert(3 == DCT4X4, "Update the DequantLibrary array below.");
  static_assert(4 == DCT16X16, "Update the DequantLibrary array below.");
  static_assert(5 == DCT8X16, "Update the DequantLibrary array below.");
  static_assert(6 == DCT4X8, "Update the DequantLibrary array below.");
  return DequantMatrices::DequantLibraryInternal{{
      DequantMatricesLibraryDef::DCT(),
      DequantMatricesLibraryDef::IDENTITY(),
      DequantMatricesLibraryDef::DCT2X2(),
      DequantMatricesLibraryDef::DCT4X4(),
      DequantMatricesLibraryDef::DCT16X16(),
      DequantMatricesLibraryDef::DCT8X16(),
      DequantMatricesLibraryDef::DCT4X8(),
  }};
}

const QuantEncoding* DequantMatrices::Library() {
  static const DequantMatrices::DequantLibraryInternal kDequantLibrary =
      DequantMatrices::LibraryInit();
  // Downcast the result to a const QuantEncoding* from QuantEncodingInternal*
  // since the subclass (QuantEncoding) doesn't add any new members and users
  // will need to upcast to QuantEncodingInternal to access the members of that
  // class.
  return reinterpret_cast<const QuantEncoding*>(kDequantLibrary.data());
}

DequantMatrices::DequantMatrices() {
  size_t pos = 0;
  size_t offsets[kNum * 3];
  for (size_t i = 0; i < size_t(QuantTable::kNum); i++) {
    size_t num = required_size_[i] * kDCTBlockSize;
    for (size_t c = 0; c < 3; c++) {
      offsets[3 * i + c] = pos + c * num;
    }
    pos += 3 * num;
  }
  for (size_t i = 0; i < AcStrategy::kNumValidStrategies; i++) {
    for (size_t c = 0; c < 3; c++) {
      table_offsets_[i * 3 + c] = offsets[kQuantTable[i] * 3 + c];
    }
  }
}

Status DequantMatrices::EnsureComputed(uint32_t acs_mask) {
  const QuantEncoding* library = Library();

  if (!table_storage_) {
    table_storage_ = hwy::AllocateAligned<float>(2 * kTotalTableSize);
    table_ = table_storage_.get();
    inv_table_ = table_storage_.get() + kTotalTableSize;
  }

  size_t offsets[kNum * 3 + 1];
  size_t pos = 0;
  for (size_t i = 0; i < kNum; i++) {
    size_t num = required_size_[i] * kDCTBlockSize;
    for (size_t c = 0; c < 3; c++) {
      offsets[3 * i + c] = pos + c * num;
    }
    pos += 3 * num;
  }
  offsets[kNum * 3] = pos;
  JXL_ASSERT(pos == kTotalTableSize);

  uint32_t kind_mask = 0;
  for (size_t i = 0; i < AcStrategy::kNumValidStrategies; i++) {
    if (acs_mask & (1u << i)) {
      kind_mask |= 1u << kQuantTable[i];
    }
  }
  uint32_t computed_kind_mask = 0;
  for (size_t i = 0; i < AcStrategy::kNumValidStrategies; i++) {
    if (computed_mask_ & (1u << i)) {
      computed_kind_mask |= 1u << kQuantTable[i];
    }
  }
  for (size_t table = 0; table < kNum; table++) {
    if ((1 << table) & computed_kind_mask) continue;
    if ((1 << table) & ~kind_mask) continue;
    size_t pos = offsets[table * 3];
    JXL_CHECK(HWY_DYNAMIC_DISPATCH(ComputeQuantTable)(
        library[table], table_storage_.get(),
        table_storage_.get() + kTotalTableSize, table, QuantTable(table),
        &pos));
    JXL_ASSERT(pos == offsets[table * 3 + 3]);
  }
  computed_mask_ |= acs_mask;

  return true;
}

}  // namespace jxl
#endif
