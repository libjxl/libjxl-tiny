// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_ENC_XYB_H_
#define ENCODER_ENC_XYB_H_

#include "encoder/image.h"

namespace jxl {

// Converts linear SRGB to XYB in place.
void ToXYB(Image3F* image);

}  // namespace jxl

#endif  // ENCODER_ENC_XYB_H_
