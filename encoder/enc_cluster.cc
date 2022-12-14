// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/enc_cluster.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <tuple>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "encoder/enc_cluster.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "encoder/fast_math-inl.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Eq;
using hwy::HWY_NAMESPACE::IfThenZeroElse;

template <class V>
V Entropy(V count, V inv_total, V total) {
  const HWY_CAPPED(float, Histogram::kRounding) d;
  const auto zero = Set(d, 0.0f);
  // TODO(eustas): why (0 - x) instead of Neg(x)?
  return IfThenZeroElse(
      Eq(count, total),
      Sub(zero, Mul(count, FastLog2f(d, Mul(inv_total, count)))));
}

void HistogramEntropy(const Histogram& a) {
  a.entropy_ = 0.0f;
  if (a.total_count_ == 0) return;

  const HWY_CAPPED(float, Histogram::kRounding) df;
  const HWY_CAPPED(int32_t, Histogram::kRounding) di;

  const auto inv_tot = Set(df, 1.0f / a.total_count_);
  auto entropy_lanes = Zero(df);
  auto total = Set(df, a.total_count_);

  for (size_t i = 0; i < a.data_.size(); i += Lanes(di)) {
    const auto counts = LoadU(di, &a.data_[i]);
    entropy_lanes =
        Add(entropy_lanes, Entropy(ConvertTo(df, counts), inv_tot, total));
  }
  a.entropy_ += GetLane(SumOfLanes(df, entropy_lanes));
}

float HistogramDistance(const Histogram& a, const Histogram& b) {
  if (a.total_count_ == 0 || b.total_count_ == 0) return 0;

  const HWY_CAPPED(float, Histogram::kRounding) df;
  const HWY_CAPPED(int32_t, Histogram::kRounding) di;

  const auto inv_tot = Set(df, 1.0f / (a.total_count_ + b.total_count_));
  auto distance_lanes = Zero(df);
  auto total = Set(df, a.total_count_ + b.total_count_);

  for (size_t i = 0; i < std::max(a.data_.size(), b.data_.size());
       i += Lanes(di)) {
    const auto a_counts =
        a.data_.size() > i ? LoadU(di, &a.data_[i]) : Zero(di);
    const auto b_counts =
        b.data_.size() > i ? LoadU(di, &b.data_[i]) : Zero(di);
    const auto counts = ConvertTo(df, Add(a_counts, b_counts));
    distance_lanes = Add(distance_lanes, Entropy(counts, inv_tot, total));
  }
  const float total_distance = GetLane(SumOfLanes(df, distance_lanes));
  return total_distance - a.entropy_ - b.entropy_;
}

// First step of a k-means clustering with a fancy distance metric.
void FastClusterHistograms(const std::vector<Histogram>& in,
                           size_t max_histograms, std::vector<Histogram>* out,
                           std::vector<uint32_t>* histogram_symbols) {
  out->clear();
  out->reserve(max_histograms);
  histogram_symbols->clear();
  histogram_symbols->resize(in.size(), max_histograms);

  std::vector<float> dists(in.size(), std::numeric_limits<float>::max());
  size_t largest_idx = 0;
  for (size_t i = 0; i < in.size(); i++) {
    if (in[i].total_count_ == 0) {
      (*histogram_symbols)[i] = 0;
      dists[i] = 0.0f;
      continue;
    }
    HistogramEntropy(in[i]);
    if (in[i].total_count_ > in[largest_idx].total_count_) {
      largest_idx = i;
    }
  }

  constexpr float kMinDistanceForDistinct = 64.0f;
  while (out->size() < max_histograms) {
    (*histogram_symbols)[largest_idx] = out->size();
    out->push_back(in[largest_idx]);
    dists[largest_idx] = 0.0f;
    largest_idx = 0;
    for (size_t i = 0; i < in.size(); i++) {
      if (dists[i] == 0.0f) continue;
      dists[i] = std::min(HistogramDistance(in[i], out->back()), dists[i]);
      if (dists[i] > dists[largest_idx]) largest_idx = i;
    }
    if (dists[largest_idx] < kMinDistanceForDistinct) break;
  }

  for (size_t i = 0; i < in.size(); i++) {
    if ((*histogram_symbols)[i] != max_histograms) continue;
    size_t best = 0;
    float best_dist = HistogramDistance(in[i], (*out)[best]);
    for (size_t j = 1; j < out->size(); j++) {
      float dist = HistogramDistance(in[i], (*out)[j]);
      if (dist < best_dist) {
        best = j;
        best_dist = dist;
      }
    }
    (*out)[best].AddHistogram(in[i]);
    HistogramEntropy((*out)[best]);
    (*histogram_symbols)[i] = best;
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {
HWY_EXPORT(FastClusterHistograms);  // Local function

namespace {
// -----------------------------------------------------------------------------
// Histogram refinement

// Reorder histograms in *out so that the new symbols in *symbols come in
// increasing order.
void HistogramReindex(const std::vector<uint32_t>& symbols,
                      std::vector<Histogram>* out,
                      std::vector<uint8_t>* context_map) {
  std::vector<Histogram> tmp(*out);
  std::map<int, int> new_index;
  int next_index = 0;
  for (uint32_t symbol : symbols) {
    if (new_index.find(symbol) == new_index.end()) {
      new_index[symbol] = next_index;
      (*out)[next_index] = tmp[symbol];
      ++next_index;
    }
  }
  out->resize(next_index);
  context_map->resize(symbols.size());
  for (size_t i = 0; i < symbols.size(); ++i) {
    (*context_map)[i] = static_cast<uint8_t>(new_index[symbols[i]]);
  }
}

}  // namespace

void ClusterHistograms(std::vector<Histogram>* histograms,
                       std::vector<uint8_t>* context_map) {
  if (histograms->size() <= 1) return;
  static const size_t kClustersLimit = 128;
  size_t max_histograms = std::min(kClustersLimit, histograms->size());

  std::vector<Histogram> in(*histograms);
  std::vector<uint32_t> histogram_symbols;
  HWY_DYNAMIC_DISPATCH(FastClusterHistograms)
  (in, max_histograms, histograms, &histogram_symbols);

  // Convert the context map to a canonical form.
  HistogramReindex(histogram_symbols, histograms, context_map);
}

}  // namespace jxl
#endif  // HWY_ONCE
