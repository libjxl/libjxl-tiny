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
#include <string.h>

#include <vector>

#include "encoder/ans_params.h"
#include "encoder/enc_ans.h"

namespace jxl {

void ClusterHistograms(const std::vector<Histogram>& in, size_t max_histograms,
                       std::vector<Histogram>* out,
                       std::vector<uint32_t>* histogram_symbols);
}  // namespace jxl

#endif  // ENCODER_ENC_CLUSTER_H_
