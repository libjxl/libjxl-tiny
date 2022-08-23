// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_ENC_MODULAR_H_
#define ENCODER_ENC_MODULAR_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "encoder/enc_ans.h"
#include "encoder/enc_bit_writer.h"
#include "encoder/image.h"
#include "encoder/modular.h"

namespace jxl {

Status ModularGenericCompress(Image &image, size_t group_id, const Tree &tree,
                              std::vector<Token> *tokens, size_t *widths);
}  // namespace jxl

#endif  // ENCODER_ENC_MODULAR_H_
