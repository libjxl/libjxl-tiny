// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_MODULAR_ENCODING_ENCODING_H_
#define ENCODER_MODULAR_ENCODING_ENCODING_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "encoder/enc_ans.h"
#include "encoder/image.h"
#include "encoder/modular/encoding/context_predict.h"
#include "encoder/modular/modular_image.h"
#include "encoder/modular/options.h"

namespace jxl {

enum MATreeContext : size_t {
  kSplitValContext = 0,
  kPropertyContext = 1,
  kPredictorContext = 2,
  kOffsetContext = 3,
  kMultiplierLogContext = 4,
  kMultiplierBitsContext = 5,

  kNumTreeContexts = 6,
};

static constexpr size_t kMaxTreeSize = 1 << 22;

// Valid range of properties for using lookup tables instead of trees.
constexpr int32_t kPropRangeFast = 512;

// inner nodes
struct PropertyDecisionNode {
  PropertyVal splitval;
  int16_t property;  // -1: leaf node, lchild points to leaf node
  uint32_t lchild;
  uint32_t rchild;
  Predictor predictor;
  int64_t predictor_offset;
  uint32_t multiplier;

  PropertyDecisionNode(int p, int split_val, int lchild, int rchild,
                       Predictor predictor, int64_t predictor_offset,
                       uint32_t multiplier)
      : splitval(split_val),
        property(p),
        lchild(lchild),
        rchild(rchild),
        predictor(predictor),
        predictor_offset(predictor_offset),
        multiplier(multiplier) {}
  PropertyDecisionNode()
      : splitval(0),
        property(-1),
        lchild(0),
        rchild(0),
        predictor(Predictor::Zero),
        predictor_offset(0),
        multiplier(1) {}
  static PropertyDecisionNode Leaf(Predictor predictor, int64_t offset = 0,
                                   uint32_t multiplier = 1) {
    return PropertyDecisionNode(-1, 0, 0, 0, predictor, offset, multiplier);
  }
  static PropertyDecisionNode Split(int p, int split_val, int lchild,
                                    int rchild = -1) {
    if (rchild == -1) rchild = lchild + 1;
    return PropertyDecisionNode(p, split_val, lchild, rchild, Predictor::Zero,
                                0, 1);
  }
};

using Tree = std::vector<PropertyDecisionNode>;

FlatTree FilterTree(const Tree &global_tree,
                    std::array<pixel_type, kNumStaticProperties> &static_props,
                    size_t *num_props, bool *gradient_only);

void TokenizeTree(const Tree &tree, std::vector<Token> *tokens,
                  Tree *decoder_tree);

template <typename T>
bool TreeToLookupTable(const FlatTree &tree,
                       T context_lookup[2 * kPropRangeFast],
                       int8_t offsets[2 * kPropRangeFast],
                       int8_t multipliers[2 * kPropRangeFast] = nullptr) {
  struct TreeRange {
    // Begin *excluded*, end *included*. This works best with > vs <= decision
    // nodes.
    int begin, end;
    size_t pos;
  };
  std::vector<TreeRange> ranges;
  ranges.push_back(TreeRange{-kPropRangeFast - 1, kPropRangeFast - 1, 0});
  while (!ranges.empty()) {
    TreeRange cur = ranges.back();
    ranges.pop_back();
    if (cur.begin < -kPropRangeFast - 1 || cur.begin >= kPropRangeFast - 1 ||
        cur.end > kPropRangeFast - 1) {
      // Tree is outside the allowed range, exit.
      return false;
    }
    auto &node = tree[cur.pos];
    // Leaf.
    if (node.property0 == -1) {
      if (node.predictor_offset < std::numeric_limits<int8_t>::min() ||
          node.predictor_offset > std::numeric_limits<int8_t>::max()) {
        return false;
      }
      if (node.multiplier < std::numeric_limits<int8_t>::min() ||
          node.multiplier > std::numeric_limits<int8_t>::max()) {
        return false;
      }
      if (multipliers == nullptr && node.multiplier != 1) {
        return false;
      }
      for (int i = cur.begin + 1; i < cur.end + 1; i++) {
        context_lookup[i + kPropRangeFast] = node.childID;
        if (multipliers) multipliers[i + kPropRangeFast] = node.multiplier;
        offsets[i + kPropRangeFast] = node.predictor_offset;
      }
      continue;
    }
    // > side of top node.
    if (node.properties[0] >= kNumStaticProperties) {
      ranges.push_back(TreeRange({node.splitvals[0], cur.end, node.childID}));
      ranges.push_back(
          TreeRange({node.splitval0, node.splitvals[0], node.childID + 1}));
    } else {
      ranges.push_back(TreeRange({node.splitval0, cur.end, node.childID}));
    }
    // <= side
    if (node.properties[1] >= kNumStaticProperties) {
      ranges.push_back(
          TreeRange({node.splitvals[1], node.splitval0, node.childID + 2}));
      ranges.push_back(
          TreeRange({cur.begin, node.splitvals[1], node.childID + 3}));
    } else {
      ranges.push_back(
          TreeRange({cur.begin, node.splitval0, node.childID + 2}));
    }
  }
  return true;
}

}  // namespace jxl

#endif  // ENCODER_MODULAR_ENCODING_ENCODING_H_
