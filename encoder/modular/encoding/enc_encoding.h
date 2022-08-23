// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_MODULAR_ENCODING_ENC_ENCODING_H_
#define ENCODER_MODULAR_ENCODING_ENC_ENCODING_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "encoder/base/compiler_specific.h"
#include "encoder/base/padded_bytes.h"
#include "encoder/base/span.h"
#include "encoder/enc_ans.h"
#include "encoder/enc_bit_writer.h"
#include "encoder/image.h"
#include "encoder/modular/encoding/context_predict.h"
#include "encoder/modular/encoding/encoding.h"
#include "encoder/modular/modular_image.h"
#include "encoder/modular/options.h"

namespace jxl {

Status ModularGenericCompress(Image &image, size_t group_id, const Tree &tree,
                              std::vector<Token> *tokens, size_t *widths);
}  // namespace jxl

#endif  // ENCODER_MODULAR_ENCODING_ENC_ENCODING_H_
