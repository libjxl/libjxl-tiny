// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/modular/encoding/enc_ma.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "encoder/enc_ans.h"
#include "lib/jxl/base/random.h"
#include "lib/jxl/fast_math-inl.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/encoding/ma_common.h"
#include "lib/jxl/modular/options.h"

namespace jxl {

void TokenizeTree(const Tree &tree, std::vector<Token> *tokens,
                  Tree *decoder_tree) {
  JXL_ASSERT(tree.size() <= kMaxTreeSize);
  std::queue<int> q;
  q.push(0);
  size_t leaf_id = 0;
  decoder_tree->clear();
  while (!q.empty()) {
    int cur = q.front();
    q.pop();
    JXL_ASSERT(tree[cur].property >= -1);
    tokens->emplace_back(kPropertyContext, tree[cur].property + 1);
    if (tree[cur].property == -1) {
      tokens->emplace_back(kPredictorContext,
                           static_cast<int>(tree[cur].predictor));
      tokens->emplace_back(kOffsetContext,
                           PackSigned(tree[cur].predictor_offset));
      uint32_t mul_log = Num0BitsBelowLS1Bit_Nonzero(tree[cur].multiplier);
      uint32_t mul_bits = (tree[cur].multiplier >> mul_log) - 1;
      tokens->emplace_back(kMultiplierLogContext, mul_log);
      tokens->emplace_back(kMultiplierBitsContext, mul_bits);
      JXL_ASSERT(tree[cur].predictor < Predictor::Best);
      decoder_tree->emplace_back(-1, 0, leaf_id++, 0, tree[cur].predictor,
                                 tree[cur].predictor_offset,
                                 tree[cur].multiplier);
      continue;
    }
    decoder_tree->emplace_back(tree[cur].property, tree[cur].splitval,
                               decoder_tree->size() + q.size() + 1,
                               decoder_tree->size() + q.size() + 2,
                               Predictor::Zero, 0, 1);
    q.push(tree[cur].lchild);
    q.push(tree[cur].rchild);
    tokens->emplace_back(kSplitValContext, PackSigned(tree[cur].splitval));
  }
}

}  // namespace jxl
