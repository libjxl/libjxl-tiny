// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_AC_STRATEGY_H_
#define ENCODER_AC_STRATEGY_H_

#include <stddef.h>
#include <stdint.h>

#include <hwy/base.h>  // kMaxVectorSize

#include "encoder/base/status.h"
#include "encoder/common.h"
#include "encoder/image.h"

// Defines the different kinds of transforms, and heuristics to choose between
// them.
// `AcStrategy` represents what transform should be used, and which sub-block of
// that transform we are currently in. Note that DCT4x4 is applied on all four
// 4x4 sub-blocks of an 8x8 block.
// `AcStrategyImage` defines which strategy should be used for each 8x8 block
// of the image. The highest 4 bits represent the strategy to be used, the
// lowest 4 represent the index of the block inside that strategy.

namespace jxl {

class AcStrategy {
 public:
  // Extremal values for the number of blocks/coefficients of a single strategy.
  static constexpr size_t kMaxCoeffBlocks = 32;
  static constexpr size_t kMaxBlockDim = kBlockDim * kMaxCoeffBlocks;
  // Maximum number of coefficients in a block. Guaranteed to be a multiple of
  // the vector size.
  static constexpr size_t kMaxCoeffArea = kMaxBlockDim * kMaxBlockDim;
  static_assert((kMaxCoeffArea * sizeof(float)) % hwy::kMaxVectorSize == 0,
                "Coefficient area is not a multiple of vector size");

  // Raw strategy types.
  enum Type : uint32_t {
    // Regular block size DCT
    DCT = 0,
    DCT16X8 = 1,
    DCT8X16 = 2,
    kNumValidStrategies
  };

  static constexpr uint32_t TypeBit(const Type type) {
    return 1u << static_cast<uint32_t>(type);
  }

  // Returns true if this block is the first 8x8 block (i.e. top-left) of a
  // possibly multi-block strategy.
  JXL_INLINE bool IsFirstBlock() const { return is_first_; }

  // Returns the raw strategy value. Should only be used for tokenization.
  JXL_INLINE uint8_t RawStrategy() const {
    return static_cast<uint8_t>(strategy_);
  }

  JXL_INLINE uint8_t StrategyCode() const {
    constexpr uint8_t kLut[] = {0, 6, 7};
    return kLut[RawStrategy()];
  }

  JXL_INLINE Type Strategy() const { return strategy_; }

  // Inverse check
  static JXL_INLINE constexpr bool IsRawStrategyValid(int raw_strategy) {
    return raw_strategy < static_cast<int32_t>(kNumValidStrategies) &&
           raw_strategy >= 0;
  }
  static JXL_INLINE AcStrategy FromRawStrategy(uint8_t raw_strategy) {
    return FromRawStrategy(static_cast<Type>(raw_strategy));
  }
  static JXL_INLINE AcStrategy FromRawStrategy(Type raw_strategy) {
    JXL_DASSERT(IsRawStrategyValid(static_cast<uint32_t>(raw_strategy)));
    return AcStrategy(raw_strategy, /*is_first=*/true);
  }

  // Number of 8x8 blocks that this strategy will cover. 0 for non-top-left
  // blocks inside a multi-block transform.
  JXL_INLINE size_t covered_blocks_x() const {
    static constexpr uint8_t kLut[] = {1, 1, 2};
    static_assert(sizeof(kLut) / sizeof(*kLut) == kNumValidStrategies,
                  "Update LUT");
    return kLut[size_t(strategy_)];
  }

  JXL_INLINE size_t covered_blocks_y() const {
    static constexpr uint8_t kLut[] = {1, 2, 1};
    static_assert(sizeof(kLut) / sizeof(*kLut) == kNumValidStrategies,
                  "Update LUT");
    return kLut[size_t(strategy_)];
  }

  JXL_INLINE size_t log2_covered_blocks() const {
    static constexpr uint8_t kLut[] = {0, 1, 1};
    static_assert(sizeof(kLut) / sizeof(*kLut) == kNumValidStrategies,
                  "Update LUT");
    return kLut[size_t(strategy_)];
  }

 private:
  friend class AcStrategyRow;
  JXL_INLINE AcStrategy(Type strategy, bool is_first)
      : strategy_(strategy), is_first_(is_first) {
  }

  Type strategy_;
  bool is_first_;
};

// Class to use a certain row of the AC strategy.
class AcStrategyRow {
 public:
  explicit AcStrategyRow(const uint8_t* row) : row_(row) {}
  AcStrategy operator[](size_t x) const {
    return AcStrategy(static_cast<AcStrategy::Type>(row_[x] >> 1), row_[x] & 1);
  }

 private:
  const uint8_t* JXL_RESTRICT row_;
};

class AcStrategyImage {
 public:
  AcStrategyImage() = default;
  AcStrategyImage(size_t xsize, size_t ysize) : layers_(xsize, ysize) {
    row_ = layers_.Row(0);
    stride_ = layers_.PixelsPerRow();
  }
  AcStrategyImage(AcStrategyImage&&) = default;
  AcStrategyImage& operator=(AcStrategyImage&&) = default;

  void FillDCT8(const Rect& rect) {
    FillPlane<uint8_t>((static_cast<uint8_t>(AcStrategy::Type::DCT) << 1) | 1,
                       &layers_, rect);
  }
  void FillDCT8() { FillDCT8(Rect(layers_)); }

  void FillInvalid() { FillImage(INVALID, &layers_); }

  void Set(size_t x, size_t y, AcStrategy::Type type) {
#if JXL_ENABLE_ASSERT
    AcStrategy acs = AcStrategy::FromRawStrategy(type);
#endif  // JXL_ENABLE_ASSERT
    JXL_ASSERT(y + acs.covered_blocks_y() <= layers_.ysize());
    JXL_ASSERT(x + acs.covered_blocks_x() <= layers_.xsize());
    JXL_CHECK(SetNoBoundsCheck(x, y, type, /*check=*/false));
  }

  Status SetNoBoundsCheck(size_t x, size_t y, AcStrategy::Type type,
                          bool check = true) {
    AcStrategy acs = AcStrategy::FromRawStrategy(type);
    for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
      for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
        size_t pos = (y + iy) * stride_ + x + ix;
        if (check && row_[pos] != INVALID) {
          return JXL_FAILURE("Invalid AC strategy: block overlap");
        }
        row_[pos] =
            (static_cast<uint8_t>(type) << 1) | ((iy | ix) == 0 ? 1 : 0);
      }
    }
    return true;
  }

  bool IsValid(size_t x, size_t y) { return row_[y * stride_ + x] != INVALID; }

  AcStrategyRow ConstRow(size_t y, size_t x_prefix = 0) const {
    return AcStrategyRow(layers_.ConstRow(y) + x_prefix);
  }

  AcStrategyRow ConstRow(const Rect& rect, size_t y) const {
    return ConstRow(rect.y0() + y, rect.x0());
  }

  size_t PixelsPerRow() const { return layers_.PixelsPerRow(); }

  size_t xsize() const { return layers_.xsize(); }
  size_t ysize() const { return layers_.ysize(); }

 private:
  ImageB layers_;
  uint8_t* JXL_RESTRICT row_;
  size_t stride_;

  // A value that does not represent a valid combined AC strategy
  // value. Used as a sentinel.
  static constexpr uint8_t INVALID = 0xFF;
};

}  // namespace jxl

#endif  // ENCODER_AC_STRATEGY_H_
