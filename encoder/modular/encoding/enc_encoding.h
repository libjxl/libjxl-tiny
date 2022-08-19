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

#include "encoder/enc_ans.h"
#include "encoder/enc_bit_writer.h"
#include "encoder/modular/encoding/enc_ma.h"
#include "lib/jxl/aux_out_fwd.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/image.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/encoding/encoding.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/modular/transform/transform.h"

namespace jxl {

Status ModularGenericCompress(Image &image, const ModularOptions &opts,
                              BitWriter *writer, AuxOut *aux_out = nullptr,
                              size_t layer = 0, size_t group_id = 0,
                              size_t *total_pixels = nullptr,
                              const Tree *tree = nullptr,
                              GroupHeader *header = nullptr,
                              std::vector<Token> *tokens = nullptr,
                              size_t *widths = nullptr);
}  // namespace jxl

#endif  // ENCODER_MODULAR_ENCODING_ENC_ENCODING_H_
