// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_ENC_FILE_H_
#define ENCODER_ENC_FILE_H_

#include <stdint.h>

#include <vector>

#include "encoder/image.h"

namespace jxl {

bool EncodeFile(const Image3F& input, float distance,
                std::vector<uint8_t>* output);

}  // namespace jxl

#endif  // ENCODER_ENC_FILE_H_
