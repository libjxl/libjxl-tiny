// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_QUANT_WEIGHTS_H_
#define ENCODER_QUANT_WEIGHTS_H_

#include <stdint.h>
#include <string.h>

#include <array>
#include <hwy/aligned_allocator.h>
#include <utility>
#include <vector>

#include "encoder/ac_strategy.h"
#include "encoder/base/cache_aligned.h"
#include "encoder/base/compiler_specific.h"
#include "encoder/base/span.h"
#include "encoder/base/status.h"
#include "encoder/common.h"
#include "encoder/image.h"

namespace jxl {

template <typename T, size_t N>
constexpr T ArraySum(T (&a)[N], size_t i = N - 1) {
  static_assert(N > 0, "Trying to compute the sum of an empty array");
  return i == 0 ? a[0] : a[i] + ArraySum(a, i - 1);
}

static constexpr size_t kMaxQuantTableSize = AcStrategy::kMaxCoeffArea;

struct DctQuantWeightParams {
  static constexpr size_t kLog2MaxDistanceBands = 4;
  static constexpr size_t kMaxDistanceBands = 1 + (1 << kLog2MaxDistanceBands);
  typedef std::array<std::array<float, kMaxDistanceBands>, 3>
      DistanceBandsArray;

  size_t num_distance_bands = 0;
  DistanceBandsArray distance_bands = {};

  constexpr DctQuantWeightParams() : num_distance_bands(0) {}

  constexpr DctQuantWeightParams(const DistanceBandsArray& dist_bands,
                                 size_t num_dist_bands)
      : num_distance_bands(num_dist_bands), distance_bands(dist_bands) {}

  template <size_t num_dist_bands>
  explicit DctQuantWeightParams(const float dist_bands[3][num_dist_bands]) {
    num_distance_bands = num_dist_bands;
    for (size_t c = 0; c < 3; c++) {
      memcpy(distance_bands[c].data(), dist_bands[c],
             sizeof(float) * num_dist_bands);
    }
  }
};

// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
struct QuantEncodingInternal {
  enum Mode {
    kQuantModeLibrary,
    kQuantModeDCT,
  };

  template <Mode mode>
  struct Tag {};

  static constexpr QuantEncodingInternal Library() {
    return (QuantEncodingInternal(Tag<kQuantModeLibrary>()));
  }
  constexpr QuantEncodingInternal(Tag<kQuantModeLibrary> /* tag */)
      : mode(kQuantModeLibrary) {}

  // DCT
  static constexpr QuantEncodingInternal DCT(
      const DctQuantWeightParams& params) {
    return QuantEncodingInternal(Tag<kQuantModeDCT>(), params);
  }
  constexpr QuantEncodingInternal(Tag<kQuantModeDCT> /* tag */,
                                  const DctQuantWeightParams& params)
      : mode(kQuantModeDCT), dct_params(params) {}

  // This constructor is not constexpr so it can't be used in any of the
  // constexpr cases above.
  explicit QuantEncodingInternal(Mode mode) : mode(mode) {}

  Mode mode;
  DctQuantWeightParams dct_params;
};

class QuantEncoding final : public QuantEncodingInternal {
 public:
  QuantEncoding(const QuantEncoding& other)
      : QuantEncodingInternal(
            static_cast<const QuantEncodingInternal&>(other)) {
  }
  QuantEncoding(QuantEncoding&& other) noexcept
      : QuantEncodingInternal(
            static_cast<const QuantEncodingInternal&>(other)) {
  }
  QuantEncoding& operator=(const QuantEncoding& other) {
    *static_cast<QuantEncodingInternal*>(this) =
        QuantEncodingInternal(static_cast<const QuantEncodingInternal&>(other));
    return *this;
  }

  // Wrappers of the QuantEncodingInternal:: static functions that return a
  // QuantEncoding instead. This is using the explicit and private cast from
  // QuantEncodingInternal to QuantEncoding, which would be inlined anyway.
  // In general, you should use this wrappers. The only reason to directly
  // create a QuantEncodingInternal instance is if you need a constexpr version
  // of this class.
  static QuantEncoding Library() {
    return QuantEncoding(QuantEncodingInternal::Library());
  }
  static QuantEncoding DCT(const DctQuantWeightParams& params) {
    return QuantEncoding(QuantEncodingInternal::DCT(params));
  }

 private:
  explicit QuantEncoding(const QuantEncodingInternal& other)
      : QuantEncodingInternal(other) {}

  explicit QuantEncoding(QuantEncodingInternal::Mode mode)
      : QuantEncodingInternal(mode) {}
};

// A constexpr QuantEncodingInternal instance is often downcasted to the
// QuantEncoding subclass even if the instance wasn't an instance of the
// subclass. This is safe because user will upcast to QuantEncodingInternal to
// access any of its members.
static_assert(sizeof(QuantEncoding) == sizeof(QuantEncodingInternal),
              "Don't add any members to QuantEncoding");

// Let's try to keep these 2**N for possible future simplicity.
const float kInvDCQuant[3] = {
    4096.0f,
    512.0f,
    256.0f,
};

const float kDCQuant[3] = {
    1.0f / kInvDCQuant[0],
    1.0f / kInvDCQuant[1],
    1.0f / kInvDCQuant[2],
};

class ModularFrameDecoder;

class DequantMatrices {
 public:
  enum QuantTable : size_t {
    DCT = 0,
    DCT8X16,
    kNum
  };

  static constexpr QuantTable kQuantTable[] = {
      QuantTable::DCT,     QuantTable::DCT,     QuantTable::DCT,
      QuantTable::DCT,     QuantTable::DCT,     QuantTable::DCT,
      QuantTable::DCT8X16, QuantTable::DCT8X16, QuantTable::DCT,
      QuantTable::DCT,     QuantTable::DCT,     QuantTable::DCT,
      QuantTable::DCT,     QuantTable::DCT,     QuantTable::DCT,
      QuantTable::DCT,     QuantTable::DCT,     QuantTable::DCT,
      QuantTable::DCT,     QuantTable::DCT,     QuantTable::DCT,
      QuantTable::DCT,     QuantTable::DCT,     QuantTable::DCT,
      QuantTable::DCT,     QuantTable::DCT,     QuantTable::DCT,
  };
  static_assert(AcStrategy::kNumValidStrategies ==
                    sizeof(kQuantTable) / sizeof *kQuantTable,
                "Update this array when adding or removing AC strategies.");

  DequantMatrices();

  static const QuantEncoding* Library();

  typedef std::array<QuantEncodingInternal, kNum> DequantLibraryInternal;
  // Return the array of library QuantEncoding entries as
  // a constexpr array. Use Library() to obtain a pointer to the copy in the
  // .cc file.
  static const DequantLibraryInternal LibraryInit();

  // Returns aligned memory.
  JXL_INLINE const float* Matrix(size_t quant_kind, size_t c) const {
    JXL_DASSERT(quant_kind < AcStrategy::kNumValidStrategies);
    JXL_DASSERT((1 << quant_kind) & computed_mask_);
    return &table_[table_offsets_[quant_kind * 3 + c]];
  }

  JXL_INLINE const float* InvMatrix(size_t quant_kind, size_t c) const {
    JXL_DASSERT(quant_kind < AcStrategy::kNumValidStrategies);
    JXL_DASSERT((1 << quant_kind) & computed_mask_);
    return &inv_table_[table_offsets_[quant_kind * 3 + c]];
  }

  // DC quants are used in modular mode for XYB multipliers.
  JXL_INLINE float DCQuant(size_t c) const { return dc_quant_[c]; }
  JXL_INLINE const float* DCQuants() const { return dc_quant_; }

  JXL_INLINE float InvDCQuant(size_t c) const { return inv_dc_quant_[c]; }

  static constexpr size_t required_size_x[] = {
      1,
      1,
  };
  static_assert(kNum == sizeof(required_size_x) / sizeof(*required_size_x),
                "Update this array when adding or removing quant tables.");

  static constexpr size_t required_size_y[] = {
      1,
      2,
  };
  static_assert(kNum == sizeof(required_size_y) / sizeof(*required_size_y),
                "Update this array when adding or removing quant tables.");

  Status EnsureComputed(uint32_t acs_mask);

 private:
  static constexpr size_t required_size_[] = {
      1,
      2,
  };
  static_assert(kNum == sizeof(required_size_) / sizeof(*required_size_),
                "Update this array when adding or removing quant tables.");
  static constexpr size_t kTotalTableSize =
      ArraySum(required_size_) * kDCTBlockSize * 3;

  uint32_t computed_mask_ = 0;
  // kTotalTableSize entries followed by kTotalTableSize for inv_table
  hwy::AlignedFreeUniquePtr<float[]> table_storage_;
  const float* table_;
  const float* inv_table_;
  float dc_quant_[3] = {kDCQuant[0], kDCQuant[1], kDCQuant[2]};
  float inv_dc_quant_[3] = {kInvDCQuant[0], kInvDCQuant[1], kInvDCQuant[2]};
  size_t table_offsets_[AcStrategy::kNumValidStrategies * 3];
};

}  // namespace jxl

#endif  // ENCODER_QUANT_WEIGHTS_H_
