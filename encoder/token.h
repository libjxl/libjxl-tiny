// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_TOKEN_H_
#define ENCODER_TOKEN_H_

#include <stdint.h>

#include "encoder/base/bits.h"

namespace jxl {

// Integer to be encoded by an entropy coder, either ANS or Huffman.
struct Token {
  Token() {}
  Token(uint32_t c, uint32_t value) : context(c), value(value) {}
  uint32_t context;
  uint32_t value;
};

// N = 0 - 15:          (token=N, nbits=0, bits='')
// N = 16 (10000):      (token=16, nbits=2, bits='00')
// N = 17 (10001):      (token=16, nbits=2, bits='01')
// N = 20 (10100):      (token=17, nbits=2, bits='00')
// N = 24 (11000):      (token=18, nbits=2, bits='00')
// N = 28 (11100):      (token=19, nbits=2, bits='00')
// N = 32 (100000):     (token=20, nbits=3, bits='000')
// N = 65535:           (token=63, nbits=13, bits='1111111111111')
struct UintCoder {
  JXL_INLINE void Encode(uint32_t value, uint32_t* JXL_RESTRICT token,
                         uint32_t* JXL_RESTRICT nbits,
                         uint32_t* JXL_RESTRICT bits) const {
    if (value < 16) {
      *token = value;
      *nbits = 0;
      *bits = 0;
    } else {
      uint32_t n = FloorLog2Nonzero(value);
      uint32_t m = value - (1 << n);
      *token = (n << 2) + (m >> (n - 2));
      *nbits = n - 2;
      *bits = value & ((1UL << *nbits) - 1);
    }
  }
};

}  // namespace jxl
#endif  // ENCODER_TOKEN_H_
