// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_GABORISH_H_
#define ENCODER_GABORISH_H_

// Linear smoothing (3x3 convolution) for deblocking without too much blur.

#include <stdint.h>

#include "encoder/base/compiler_specific.h"
#include "encoder/base/data_parallel.h"
#include "encoder/image.h"

namespace jxl {

// Used in encoder to reduce the impact of the decoder's smoothing.
// This is not exact. Works in-place to reduce memory use.
// The input is typically in XYB space.
void GaborishInverse(Image3F* in_out, float mul, ThreadPool* pool);

}  // namespace jxl

#endif  // ENCODER_GABORISH_H_
