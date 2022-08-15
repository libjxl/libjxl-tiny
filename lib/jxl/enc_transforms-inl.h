// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#if defined(LIB_JXL_ENC_TRANSFORMS_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef LIB_JXL_ENC_TRANSFORMS_INL_H_
#undef LIB_JXL_ENC_TRANSFORMS_INL_H_
#else
#define LIB_JXL_ENC_TRANSFORMS_INL_H_
#endif

#include <stddef.h>

#include <hwy/highway.h>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/dct-inl.h"
#include "lib/jxl/dct_scales.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

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
      PROFILER_ZONE("DCT Identity");
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
      PROFILER_ZONE("DCT 8x4");
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
      PROFILER_ZONE("DCT 4x8");
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
      PROFILER_ZONE("DCT 4");
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
      PROFILER_ZONE("DCT 2");
      DCT2TopBlock<8>(pixels, pixels_stride, coefficients);
      DCT2TopBlock<4>(coefficients, kBlockDim, coefficients);
      DCT2TopBlock<2>(coefficients, kBlockDim, coefficients);
      break;
    }
    case Type::DCT16X16: {
      PROFILER_ZONE("DCT 16");
      ComputeScaledDCT<16, 16>()(DCTFrom(pixels, pixels_stride), coefficients,
                                 scratch_space);
      break;
    }
    case Type::DCT16X8: {
      PROFILER_ZONE("DCT 16x8");
      ComputeScaledDCT<16, 8>()(DCTFrom(pixels, pixels_stride), coefficients,
                                scratch_space);
      break;
    }
    case Type::DCT8X16: {
      PROFILER_ZONE("DCT 8x16");
      ComputeScaledDCT<8, 16>()(DCTFrom(pixels, pixels_stride), coefficients,
                                scratch_space);
      break;
    }
    case Type::DCT32X8: {
      PROFILER_ZONE("DCT 32x8");
      ComputeScaledDCT<32, 8>()(DCTFrom(pixels, pixels_stride), coefficients,
                                scratch_space);
      break;
    }
    case Type::DCT8X32: {
      PROFILER_ZONE("DCT 8x32");
      ComputeScaledDCT<8, 32>()(DCTFrom(pixels, pixels_stride), coefficients,
                                scratch_space);
      break;
    }
    case Type::DCT32X16: {
      PROFILER_ZONE("DCT 32x16");
      ComputeScaledDCT<32, 16>()(DCTFrom(pixels, pixels_stride), coefficients,
                                 scratch_space);
      break;
    }
    case Type::DCT16X32: {
      PROFILER_ZONE("DCT 16x32");
      ComputeScaledDCT<16, 32>()(DCTFrom(pixels, pixels_stride), coefficients,
                                 scratch_space);
      break;
    }
    case Type::DCT32X32: {
      PROFILER_ZONE("DCT 32");
      ComputeScaledDCT<32, 32>()(DCTFrom(pixels, pixels_stride), coefficients,
                                 scratch_space);
      break;
    }
    case Type::DCT: {
      PROFILER_ZONE("DCT 8");
      ComputeScaledDCT<8, 8>()(DCTFrom(pixels, pixels_stride), coefficients,
                               scratch_space);
      break;
    }
    case Type::AFV0: {
      JXL_ABORT("Invalid strategy");
      break;
    }
    case Type::AFV1: {
      JXL_ABORT("Invalid strategy");
      break;
    }
    case Type::AFV2: {
      JXL_ABORT("Invalid strategy");
      break;
    }
    case Type::AFV3: {
      JXL_ABORT("Invalid strategy");
      break;
    }
    case Type::DCT64X64: {
      PROFILER_ZONE("DCT 64x64");
      ComputeScaledDCT<64, 64>()(DCTFrom(pixels, pixels_stride), coefficients,
                                 scratch_space);
      break;
    }
    case Type::DCT64X32: {
      PROFILER_ZONE("DCT 64x32");
      ComputeScaledDCT<64, 32>()(DCTFrom(pixels, pixels_stride), coefficients,
                                 scratch_space);
      break;
    }
    case Type::DCT32X64: {
      PROFILER_ZONE("DCT 32x64");
      ComputeScaledDCT<32, 64>()(DCTFrom(pixels, pixels_stride), coefficients,
                                 scratch_space);
      break;
    }
    case Type::DCT128X128: {
      PROFILER_ZONE("DCT 128x128");
      ComputeScaledDCT<128, 128>()(DCTFrom(pixels, pixels_stride), coefficients,
                                   scratch_space);
      break;
    }
    case Type::DCT128X64: {
      PROFILER_ZONE("DCT 128x64");
      ComputeScaledDCT<128, 64>()(DCTFrom(pixels, pixels_stride), coefficients,
                                  scratch_space);
      break;
    }
    case Type::DCT64X128: {
      PROFILER_ZONE("DCT 64x128");
      ComputeScaledDCT<64, 128>()(DCTFrom(pixels, pixels_stride), coefficients,
                                  scratch_space);
      break;
    }
    case Type::DCT256X256: {
      PROFILER_ZONE("DCT 256x256");
      ComputeScaledDCT<256, 256>()(DCTFrom(pixels, pixels_stride), coefficients,
                                   scratch_space);
      break;
    }
    case Type::DCT256X128: {
      PROFILER_ZONE("DCT 256x128");
      ComputeScaledDCT<256, 128>()(DCTFrom(pixels, pixels_stride), coefficients,
                                   scratch_space);
      break;
    }
    case Type::DCT128X256: {
      PROFILER_ZONE("DCT 128x256");
      ComputeScaledDCT<128, 256>()(DCTFrom(pixels, pixels_stride), coefficients,
                                   scratch_space);
      break;
    }
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
                         /*LF_ROWS=*/2, /*LF_COLS=*/1, /*ROWS=*/2, /*COLS=*/1>(
          block, 2 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT8X16: {
      ReinterpretingIDCT</*DCT_ROWS=*/kBlockDim, /*DCT_COLS=*/2 * kBlockDim,
                         /*LF_ROWS=*/1, /*LF_COLS=*/2, /*ROWS=*/1, /*COLS=*/2>(
          block, 2 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT16X16: {
      ReinterpretingIDCT</*DCT_ROWS=*/2 * kBlockDim, /*DCT_COLS=*/2 * kBlockDim,
                         /*LF_ROWS=*/2, /*LF_COLS=*/2, /*ROWS=*/2, /*COLS=*/2>(
          block, 2 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT32X8: {
      ReinterpretingIDCT</*DCT_ROWS=*/4 * kBlockDim, /*DCT_COLS=*/kBlockDim,
                         /*LF_ROWS=*/4, /*LF_COLS=*/1, /*ROWS=*/4, /*COLS=*/1>(
          block, 4 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT8X32: {
      ReinterpretingIDCT</*DCT_ROWS=*/kBlockDim, /*DCT_COLS=*/4 * kBlockDim,
                         /*LF_ROWS=*/1, /*LF_COLS=*/4, /*ROWS=*/1, /*COLS=*/4>(
          block, 4 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT32X16: {
      ReinterpretingIDCT</*DCT_ROWS=*/4 * kBlockDim, /*DCT_COLS=*/2 * kBlockDim,
                         /*LF_ROWS=*/4, /*LF_COLS=*/2, /*ROWS=*/4, /*COLS=*/2>(
          block, 4 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT16X32: {
      ReinterpretingIDCT</*DCT_ROWS=*/2 * kBlockDim, /*DCT_COLS=*/4 * kBlockDim,
                         /*LF_ROWS=*/2, /*LF_COLS=*/4, /*ROWS=*/2, /*COLS=*/4>(
          block, 4 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT32X32: {
      ReinterpretingIDCT</*DCT_ROWS=*/4 * kBlockDim, /*DCT_COLS=*/4 * kBlockDim,
                         /*LF_ROWS=*/4, /*LF_COLS=*/4, /*ROWS=*/4, /*COLS=*/4>(
          block, 4 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT64X32: {
      ReinterpretingIDCT</*DCT_ROWS=*/8 * kBlockDim, /*DCT_COLS=*/4 * kBlockDim,
                         /*LF_ROWS=*/8, /*LF_COLS=*/4, /*ROWS=*/8, /*COLS=*/4>(
          block, 8 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT32X64: {
      ReinterpretingIDCT</*DCT_ROWS=*/4 * kBlockDim, /*DCT_COLS=*/8 * kBlockDim,
                         /*LF_ROWS=*/4, /*LF_COLS=*/8, /*ROWS=*/4, /*COLS=*/8>(
          block, 8 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT64X64: {
      ReinterpretingIDCT</*DCT_ROWS=*/8 * kBlockDim, /*DCT_COLS=*/8 * kBlockDim,
                         /*LF_ROWS=*/8, /*LF_COLS=*/8, /*ROWS=*/8, /*COLS=*/8>(
          block, 8 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT128X64: {
      ReinterpretingIDCT<
          /*DCT_ROWS=*/16 * kBlockDim, /*DCT_COLS=*/8 * kBlockDim,
          /*LF_ROWS=*/16, /*LF_COLS=*/8, /*ROWS=*/16, /*COLS=*/8>(
          block, 16 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT64X128: {
      ReinterpretingIDCT<
          /*DCT_ROWS=*/8 * kBlockDim, /*DCT_COLS=*/16 * kBlockDim,
          /*LF_ROWS=*/8, /*LF_COLS=*/16, /*ROWS=*/8, /*COLS=*/16>(
          block, 16 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT128X128: {
      ReinterpretingIDCT<
          /*DCT_ROWS=*/16 * kBlockDim, /*DCT_COLS=*/16 * kBlockDim,
          /*LF_ROWS=*/16, /*LF_COLS=*/16, /*ROWS=*/16, /*COLS=*/16>(
          block, 16 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT256X128: {
      ReinterpretingIDCT<
          /*DCT_ROWS=*/32 * kBlockDim, /*DCT_COLS=*/16 * kBlockDim,
          /*LF_ROWS=*/32, /*LF_COLS=*/16, /*ROWS=*/32, /*COLS=*/16>(
          block, 32 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT128X256: {
      ReinterpretingIDCT<
          /*DCT_ROWS=*/16 * kBlockDim, /*DCT_COLS=*/32 * kBlockDim,
          /*LF_ROWS=*/16, /*LF_COLS=*/32, /*ROWS=*/16, /*COLS=*/32>(
          block, 32 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT256X256: {
      ReinterpretingIDCT<
          /*DCT_ROWS=*/32 * kBlockDim, /*DCT_COLS=*/32 * kBlockDim,
          /*LF_ROWS=*/32, /*LF_COLS=*/32, /*ROWS=*/32, /*COLS=*/32>(
          block, 32 * kBlockDim, dc, dc_stride);
      break;
    }
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

#endif  // LIB_JXL_ENC_TRANSFORMS_INL_H_
