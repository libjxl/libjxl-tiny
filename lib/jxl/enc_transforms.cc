// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "lib/jxl/enc_transforms.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_transforms.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/dct_scales.h"
#include "lib/jxl/enc_transforms-inl.h"

namespace jxl {

#if HWY_ONCE
HWY_EXPORT(TransformFromPixels);
void TransformFromPixels(const AcStrategy::Type strategy,
                         const float* JXL_RESTRICT pixels, size_t pixels_stride,
                         float* JXL_RESTRICT coefficients,
                         float* scratch_space) {
  return HWY_DYNAMIC_DISPATCH(TransformFromPixels)(
      strategy, pixels, pixels_stride, coefficients, scratch_space);
}

HWY_EXPORT(DCFromLowestFrequencies);
void DCFromLowestFrequencies(AcStrategy::Type strategy, const float* block,
                             float* dc, size_t dc_stride) {
  return HWY_DYNAMIC_DISPATCH(DCFromLowestFrequencies)(strategy, block, dc,
                                                       dc_stride);
}

#endif  // HWY_ONCE
}  // namespace jxl
