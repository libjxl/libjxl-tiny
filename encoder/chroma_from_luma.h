// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_CHROMA_FROM_LUMA_H_
#define ENCODER_CHROMA_FROM_LUMA_H_

// Chroma-from-luma, computed using heuristics to determine the best linear
// model for the X and B channels from the Y channel.

#include <stddef.h>
#include <stdint.h>

#include "encoder/common.h"
#include "encoder/image.h"
#include "encoder/image_ops.h"

namespace jxl {

static constexpr float kInvColorFactor = 1.0f / 84;

struct ColorCorrelationMap {
  ColorCorrelationMap() = default;
  // xsize/ysize are in pixels
  ColorCorrelationMap(size_t xsize, size_t ysize)
      : ytox_map(DivCeil(xsize, kColorTileDim), DivCeil(ysize, kColorTileDim)),
        ytob_map(DivCeil(xsize, kColorTileDim), DivCeil(ysize, kColorTileDim)) {
    ZeroFillImage(&ytox_map);
    ZeroFillImage(&ytob_map);
    RecomputeDCFactors();
  }

  float YtoXRatio(int32_t x_factor) const { return x_factor * kInvColorFactor; }
  float YtoBRatio(int32_t b_factor) const {
    return 1.0f + b_factor * kInvColorFactor;
  }
  void SetYToBDC(int32_t ytob_dc) {
    ytob_dc_ = ytob_dc;
    RecomputeDCFactors();
  }
  void SetYToXDC(int32_t ytox_dc) {
    ytox_dc_ = ytox_dc;
    RecomputeDCFactors();
  }

  int32_t GetYToXDC() const { return ytox_dc_; }
  int32_t GetYToBDC() const { return ytob_dc_; }
  const float* DCFactors() const { return dc_factors_; }

  ImageSB ytox_map;
  ImageSB ytob_map;

 private:
  void RecomputeDCFactors() {
    dc_factors_[0] = YtoXRatio(ytox_dc_);
    dc_factors_[2] = YtoBRatio(ytob_dc_);
  }
  float dc_factors_[4] = {};
  int32_t ytox_dc_ = 0;
  int32_t ytob_dc_ = 0;
};

}  // namespace jxl

#endif  // ENCODER_CHROMA_FROM_LUMA_H_
