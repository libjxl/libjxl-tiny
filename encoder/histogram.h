// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_HISTOGRAM_H_
#define ENCODER_HISTOGRAM_H_

#include <stdint.h>

#include <vector>

#include "encoder/common.h"
#include "encoder/token.h"

namespace jxl {

struct Histogram {
  Histogram() { total_count_ = 0; }
  void Clear() {
    data_.clear();
    total_count_ = 0;
  }
  void Add(size_t symbol) {
    if (data_.size() <= symbol) {
      data_.resize(DivCeil(symbol + 1, kRounding) * kRounding);
    }
    ++data_[symbol];
    ++total_count_;
  }
  void AddHistogram(const Histogram& other) {
    if (other.data_.size() > data_.size()) {
      data_.resize(other.data_.size());
    }
    for (size_t i = 0; i < other.data_.size(); ++i) {
      data_[i] += other.data_[i];
    }
    total_count_ += other.total_count_;
  }

  std::vector<int32_t> data_;
  size_t total_count_;
  mutable float entropy_;  // WARNING: not kept up-to-date.
  static constexpr size_t kRounding = 8;
};

struct HistogramBuilder {
  explicit HistogramBuilder(const size_t num_contexts)
      : histograms(num_contexts) {}

  void Add(int symbol, size_t context) { histograms[context].Add(symbol); }
  void Add(const Token& token) {
    uint32_t tok, nbits, bits;
    UintCoder().Encode(token.value, &tok, &nbits, &bits);
    Add(tok, token.context);
  }
  template <typename T>
  void Add(const std::vector<T>& v) {
    for (const auto& i : v) Add(i);
  }

  std::vector<Histogram> histograms;
};

template <typename T>
std::vector<Histogram> BuildHistograms(size_t num_contexts, std::vector<T>& v) {
  HistogramBuilder builder(num_contexts);
  builder.Add(v);
  return builder.histograms;
}

}  // namespace jxl
#endif  // ENCODER_HISTOGRAM_H_
