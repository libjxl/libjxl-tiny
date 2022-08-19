// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/modular/encoding/encoding.h"

#include <stdint.h>
#include <stdlib.h>

#include <queue>

#include "encoder/base/printf_macros.h"
#include "encoder/modular/encoding/context_predict.h"
#include "encoder/modular/options.h"

namespace jxl {

// Removes all nodes that use a static property (i.e. channel or group ID) from
// the tree and collapses each node on even levels with its two children to
// produce a flatter tree. Also computes whether the resulting tree requires
// using the weighted predictor.
FlatTree FilterTree(const Tree &global_tree,
                    std::array<pixel_type, kNumStaticProperties> &static_props,
                    size_t *num_props, bool *use_wp, bool *wp_only,
                    bool *gradient_only) {
  *num_props = 0;
  bool has_wp = false;
  bool has_non_wp = false;
  *gradient_only = true;
  const auto mark_property = [&](int32_t p) {
    if (p == kWPProp) {
      has_wp = true;
    } else if (p >= kNumStaticProperties) {
      has_non_wp = true;
    }
    if (p >= kNumStaticProperties && p != kGradientProp) {
      *gradient_only = false;
    }
  };
  FlatTree output;
  std::queue<size_t> nodes;
  nodes.push(0);
  // Produces a trimmed and flattened tree by doing a BFS visit of the original
  // tree, ignoring branches that are known to be false and proceeding two
  // levels at a time to collapse nodes in a flatter tree; if an inner parent
  // node has a leaf as a child, the leaf is duplicated and an implicit fake
  // node is added. This allows to reduce the number of branches when traversing
  // the resulting flat tree.
  while (!nodes.empty()) {
    size_t cur = nodes.front();
    nodes.pop();
    // Skip nodes that we can decide now, by jumping directly to their children.
    while (global_tree[cur].property < kNumStaticProperties &&
           global_tree[cur].property != -1) {
      if (static_props[global_tree[cur].property] > global_tree[cur].splitval) {
        cur = global_tree[cur].lchild;
      } else {
        cur = global_tree[cur].rchild;
      }
    }
    FlatDecisionNode flat;
    if (global_tree[cur].property == -1) {
      flat.property0 = -1;
      flat.childID = global_tree[cur].lchild;
      flat.predictor = global_tree[cur].predictor;
      flat.predictor_offset = global_tree[cur].predictor_offset;
      flat.multiplier = global_tree[cur].multiplier;
      *gradient_only &= flat.predictor == Predictor::Gradient;
      has_wp |= flat.predictor == Predictor::Weighted;
      has_non_wp |= flat.predictor != Predictor::Weighted;
      output.push_back(flat);
      continue;
    }
    flat.childID = output.size() + nodes.size() + 1;

    flat.property0 = global_tree[cur].property;
    *num_props = std::max<size_t>(flat.property0 + 1, *num_props);
    flat.splitval0 = global_tree[cur].splitval;

    for (size_t i = 0; i < 2; i++) {
      size_t cur_child =
          i == 0 ? global_tree[cur].lchild : global_tree[cur].rchild;
      // Skip nodes that we can decide now.
      while (global_tree[cur_child].property < kNumStaticProperties &&
             global_tree[cur_child].property != -1) {
        if (static_props[global_tree[cur_child].property] >
            global_tree[cur_child].splitval) {
          cur_child = global_tree[cur_child].lchild;
        } else {
          cur_child = global_tree[cur_child].rchild;
        }
      }
      // We ended up in a leaf, add a dummy decision and two copies of the leaf.
      if (global_tree[cur_child].property == -1) {
        flat.properties[i] = 0;
        flat.splitvals[i] = 0;
        nodes.push(cur_child);
        nodes.push(cur_child);
      } else {
        flat.properties[i] = global_tree[cur_child].property;
        flat.splitvals[i] = global_tree[cur_child].splitval;
        nodes.push(global_tree[cur_child].lchild);
        nodes.push(global_tree[cur_child].rchild);
        *num_props = std::max<size_t>(flat.properties[i] + 1, *num_props);
      }
    }

    for (size_t j = 0; j < 2; j++) mark_property(flat.properties[j]);
    mark_property(flat.property0);
    output.push_back(flat);
  }
  if (*num_props > kNumNonrefProperties) {
    *num_props =
        DivCeil(*num_props - kNumNonrefProperties, kExtraPropsPerChannel) *
            kExtraPropsPerChannel +
        kNumNonrefProperties;
  } else {
    *num_props = kNumNonrefProperties;
  }
  *use_wp = has_wp;
  *wp_only = has_wp && !has_non_wp;

  return output;
}

Status ValidateChannelDimensions(const Image &image,
                                 const ModularOptions &options) {
  size_t nb_channels = image.channel.size();
  for (bool is_dc : {true, false}) {
    size_t group_dim = options.group_dim * (is_dc ? kBlockDim : 1);
    size_t c = image.nb_meta_channels;
    for (; c < nb_channels; c++) {
      const Channel &ch = image.channel[c];
      if (ch.w > options.group_dim || ch.h > options.group_dim) break;
    }
    for (; c < nb_channels; c++) {
      const Channel &ch = image.channel[c];
      if (ch.w == 0 || ch.h == 0) continue;  // skip empty
      bool is_dc_channel = std::min(ch.hshift, ch.vshift) >= 3;
      if (is_dc_channel != is_dc) continue;
      size_t tile_dim = group_dim >> std::max(ch.hshift, ch.vshift);
      if (tile_dim == 0) {
        return JXL_FAILURE("Inconsistent transforms");
      }
    }
  }
  return true;
}

}  // namespace jxl
