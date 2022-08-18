// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_ENC_XYB_H_
#define ENCODER_ENC_XYB_H_

// Converts to XYB color space.

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/image.h"

namespace jxl {

// Converts linear SRGB to XYB.
void ToXYB(const Image3F& linear, ThreadPool* pool, Image3F* JXL_RESTRICT xyb);

}  // namespace jxl

#endif  // ENCODER_ENC_XYB_H_
