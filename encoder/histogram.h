// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_HISTOGRAM_H_
#define ENCODER_HISTOGRAM_H_

#include <stdint.h>
#include <string.h>

#include "encoder/entropy_code.h"

namespace jxl {

struct Histogram {
  Histogram() { Clear(); }
  void Clear() {
    memset(counts, 0, sizeof(counts));
    total_count = 0;
  }
  void Add(uint32_t symbol) {
    ++counts[symbol];
    ++total_count;
  }
  void AddHistogram(const Histogram& other) {
    for (size_t i = 0; i < kAlphabetSize; ++i) {
      counts[i] += other.counts[i];
    }
    total_count += other.total_count;
  }
  uint32_t counts[kAlphabetSize];
  size_t total_count;
  mutable size_t bit_cost;  // WARNING: not kept up-to-date.
};

}  // namespace jxl
#endif  // ENCODER_HISTOGRAM_H_
