// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_DEC_CONTEXT_MAP_H_
#define ENCODER_DEC_CONTEXT_MAP_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "encoder/dec_bit_reader.h"

namespace jxl {

// Context map uses uint8_t.
constexpr size_t kMaxClusters = 256;

// Reads the context map from the bit stream. On calling this function,
// context_map->size() must be the number of possible context ids.
// Sets *num_htrees to the number of different histogram ids in
// *context_map.
Status DecodeContextMap(std::vector<uint8_t>* context_map, size_t* num_htrees,
                        BitReader* input);

}  // namespace jxl

#endif  // ENCODER_DEC_CONTEXT_MAP_H_