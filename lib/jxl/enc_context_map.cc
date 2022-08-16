// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

// Library to encode the context map.

#include "lib/jxl/enc_context_map.h"

#include <stdint.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/entropy_coder.h"

namespace jxl {

void EncodeContextMap(const std::vector<uint8_t>& context_map,
                      size_t num_histograms, BitWriter* writer, size_t layer,
                      AuxOut* aux_out) {
  if (num_histograms == 1) {
    // Simple code
    writer->Write(1, 1);
    // 0 bits per entry.
    writer->Write(2, 0);
    return;
  }

  std::vector<std::vector<Token>> tokens(1);
  EntropyEncodingData codes;
  std::vector<uint8_t> dummy_context_map;
  for (size_t i = 0; i < context_map.size(); i++) {
    tokens[0].emplace_back(0, context_map[i]);
  }
  HistogramParams params;
  writer->Write(1, 0);
  writer->Write(1, 0);  // Don't use MTF.
  BuildAndEncodeHistograms(params, 1, tokens, &codes, &dummy_context_map,
                           writer, layer, aux_out);
  WriteTokens(tokens[0], codes, dummy_context_map, writer);
}

void EncodeBlockCtxMap(const BlockCtxMap& block_ctx_map, BitWriter* writer,
                       AuxOut* aux_out) {
  auto& dct = block_ctx_map.dc_thresholds;
  auto& qft = block_ctx_map.qf_thresholds;
  auto& ctx_map = block_ctx_map.ctx_map;
  BitWriter::Allotment allotment(
      writer,
      (dct[0].size() + dct[1].size() + dct[2].size() + qft.size()) * 34 + 1 +
          4 + 4 + ctx_map.size() * 10 + 1024);
  if (dct[0].empty() && dct[1].empty() && dct[2].empty() && qft.empty() &&
      ctx_map.size() == 21 &&
      std::equal(ctx_map.begin(), ctx_map.end(), BlockCtxMap::kDefaultCtxMap)) {
    writer->Write(1, 1);  // default
    ReclaimAndCharge(writer, &allotment, kLayerAC, aux_out);
    return;
  }
  writer->Write(1, 0);
  for (int j : {0, 1, 2}) {
    writer->Write(4, dct[j].size());
    for (int i : dct[j]) {
      JXL_CHECK(U32Coder::Write(kDCThresholdDist, PackSigned(i), writer));
    }
  }
  writer->Write(4, qft.size());
  for (uint32_t i : qft) {
    JXL_CHECK(U32Coder::Write(kQFThresholdDist, i - 1, writer));
  }
  EncodeContextMap(ctx_map, block_ctx_map.num_ctxs, writer, kLayerAC, aux_out);
  ReclaimAndCharge(writer, &allotment, kLayerAC, aux_out);
}

}  // namespace jxl
