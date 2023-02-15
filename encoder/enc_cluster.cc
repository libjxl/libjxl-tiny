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

#include "encoder/enc_huffman_tree.h"

namespace jxl {
namespace {

void HistogramBitCost(const Histogram& a) {
  a.bit_cost = 0;
  if (a.total_count == 0) return;
  uint8_t depths[kAlphabetSize];
  CreateHuffmanTree(a.counts, kAlphabetSize, 15, depths);
  for (size_t i = 0; i < kAlphabetSize; ++i) {
    a.bit_cost += a.counts[i] * depths[i];
  }
}

float HistogramDistance(const Histogram& a, const Histogram& b) {
  if (a.total_count == 0 || b.total_count == 0) return 0;
  Histogram combined;
  combined.AddHistogram(a);
  combined.AddHistogram(b);
  HistogramBitCost(combined);
  return combined.bit_cost - a.bit_cost - b.bit_cost;
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
    if (in[i].total_count == 0) {
      (*histogram_symbols)[i] = 0;
      dists[i] = 0.0f;
      continue;
    }
    HistogramBitCost(in[i]);
    if (in[i].total_count > in[largest_idx].total_count) {
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
    HistogramBitCost((*out)[best]);
    (*histogram_symbols)[i] = best;
  }
}

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
  static const size_t kClustersLimit = 8;
  size_t max_histograms = std::min(kClustersLimit, histograms->size());

  std::vector<Histogram> in(*histograms);
  std::vector<uint32_t> histogram_symbols;
  FastClusterHistograms(in, max_histograms, histograms, &histogram_symbols);

  // Convert the context map to a canonical form.
  HistogramReindex(histogram_symbols, histograms, context_map);
}

}  // namespace jxl
