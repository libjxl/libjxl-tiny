// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_ENTROPY_CODE_H_
#define ENCODER_ENTROPY_CODE_H_

#include <stdint.h>

#include <vector>

namespace jxl {

static constexpr size_t kAlphabetSize = 64;
static constexpr uint8_t kMaxContexts = 128;

struct PrefixCode {
  uint8_t depths[kAlphabetSize];
  uint16_t bits[kAlphabetSize];
};

struct EntropyCode {
  EntropyCode(const uint8_t* static_context_map, size_t num_c,
              const PrefixCode* static_prefix_codes, size_t num_p)
      : context_map(static_context_map),
        num_contexts(num_c),
        prefix_codes(static_prefix_codes),
        num_prefix_codes(num_p) {}
  const uint8_t* context_map;
  size_t num_contexts;
  const PrefixCode* prefix_codes;
  size_t num_prefix_codes;
  // Data storage for the optimized entropy codes.
  std::vector<uint8_t> context_map_storage;
  std::vector<PrefixCode> prefix_code_storage;
  // Original context map, in case the contexts were clustered.
  const uint8_t* orig_context_map = nullptr;
  size_t orig_num_contexts = 0;
};

}  // namespace jxl
#endif  // ENCODER_ENTROPY_CODE_H_
