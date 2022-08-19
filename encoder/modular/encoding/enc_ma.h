// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_MODULAR_ENCODING_ENC_MA_H_
#define ENCODER_MODULAR_ENCODING_ENC_MA_H_

#include <numeric>

#include "encoder/enc_ans.h"
#include "encoder/entropy_coder.h"
#include "encoder/modular/encoding/dec_ma.h"
#include "encoder/modular/modular_image.h"
#include "encoder/modular/options.h"

namespace jxl {

void TokenizeTree(const Tree &tree, std::vector<Token> *tokens,
                  Tree *decoder_tree);

}  // namespace jxl
#endif  // ENCODER_MODULAR_ENCODING_ENC_MA_H_
