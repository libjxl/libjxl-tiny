// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_READ_PFM_H_
#define ENCODER_READ_PFM_H_

#include "encoder/image.h"

namespace jxl {

bool ReadPFM(const char* fn, jxl::Image3F* image);

}  // namespace jxl

#endif  // ENCODER_READ_PFM_H_
