// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_CONVOLVE_H_
#define ENCODER_CONVOLVE_H_

// 2D convolution.

#include <stddef.h>
#include <stdint.h>

#include "encoder/base/data_parallel.h"
#include "encoder/image.h"

namespace jxl {

// Weights must already be normalized.

struct WeightsSymmetric5 {
  // The lower-right quadrant is: c r R  (each replicated 4x)
  //                              r d L
  //                              R L D
  float c[4];
  float r[4];
  float R[4];
  float d[4];
  float D[4];
  float L[4];
};

void Symmetric5(const ImageF& in, const Rect& rect,
                const WeightsSymmetric5& weights, ThreadPool* pool,
                ImageF* JXL_RESTRICT out);

}  // namespace jxl

#endif  // ENCODER_CONVOLVE_H_
