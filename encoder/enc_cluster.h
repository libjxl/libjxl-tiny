// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

// Functions for clustering similar histograms together.

#ifndef ENCODER_ENC_CLUSTER_H_
#define ENCODER_ENC_CLUSTER_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "encoder/histogram.h"

namespace jxl {

void ClusterHistograms(std::vector<Histogram>* histograms,
                       std::vector<uint8_t>* context_map);
}  // namespace jxl

#endif  // ENCODER_ENC_CLUSTER_H_
