// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_MODULAR_ENCODING_ENC_MA_H_
#define ENCODER_MODULAR_ENCODING_ENC_MA_H_

#include <numeric>

#include "encoder/enc_ans.h"
#include "lib/jxl/entropy_coder.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"

namespace jxl {

void TokenizeTree(const Tree &tree, std::vector<Token> *tokens,
                  Tree *decoder_tree);

}  // namespace jxl
#endif  // ENCODER_MODULAR_ENCODING_ENC_MA_H_
