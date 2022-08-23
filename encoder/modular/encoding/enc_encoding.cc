// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include <stdint.h>
#include <stdlib.h>

#include <cinttypes>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "encoder/base/printf_macros.h"
#include "encoder/base/status.h"
#include "encoder/common.h"
#include "encoder/enc_ans.h"
#include "encoder/entropy_coder.h"
#include "encoder/image_ops.h"
#include "encoder/modular/encoding/context_predict.h"
#include "encoder/modular/encoding/encoding.h"
#include "encoder/modular/options.h"

namespace jxl {

Status EncodeModularChannelMAANS(const Image &image, pixel_type chan,
                                 const Tree &global_tree, Token **tokenpp,
                                 size_t group_id) {
  const Channel &channel = image.channel[chan];
  Token *tokenp = *tokenpp;
  JXL_ASSERT(channel.w != 0 && channel.h != 0);

  JXL_DEBUG_V(6,
              "Encoding %" PRIuS "x%" PRIuS
              " channel %d, "
              "(shift=%i,%i)",
              channel.w, channel.h, chan, channel.hshift, channel.vshift);

  std::array<pixel_type, kNumStaticProperties> static_props = {
      {chan, (int)group_id}};
  bool is_gradient_only;
  size_t num_props;
  FlatTree tree =
      FilterTree(global_tree, static_props, &num_props, &is_gradient_only);
  Properties properties(num_props);
  MATreeLookup tree_lookup(tree);
  JXL_DEBUG_V(3, "Encoding using a MA tree with %" PRIuS " nodes", tree.size());

  uint16_t context_lookup[2 * kPropRangeFast] = {};
  int8_t offsets[2 * kPropRangeFast] = {};
  if (is_gradient_only) {
    is_gradient_only = TreeToLookupTable(tree, context_lookup, offsets);
  }

  if (tree.size() == 1 && tree[0].predictor == Predictor::Gradient &&
      tree[0].multiplier == 1 && tree[0].predictor_offset == 0) {
    const intptr_t onerow = channel.plane.PixelsPerRow();
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT r = channel.Row(y);
      for (size_t x = 0; x < channel.w; x++) {
        pixel_type_w left = (x ? r[x - 1] : y ? *(r + x - onerow) : 0);
        pixel_type_w top = (y ? *(r + x - onerow) : left);
        pixel_type_w topleft = (x && y ? *(r + x - 1 - onerow) : left);
        int32_t guess = ClampedGradient(top, left, topleft);
        int32_t residual = r[x] - guess;
        *tokenp++ = Token(tree[0].childID, PackSigned(residual));
      }
    }
  } else if (is_gradient_only) {
    const intptr_t onerow = channel.plane.PixelsPerRow();
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT r = channel.Row(y);
      for (size_t x = 0; x < channel.w; x++) {
        pixel_type_w left = (x ? r[x - 1] : y ? *(r + x - onerow) : 0);
        pixel_type_w top = (y ? *(r + x - onerow) : left);
        pixel_type_w topleft = (x && y ? *(r + x - 1 - onerow) : left);
        int32_t guess = ClampedGradient(top, left, topleft);
        uint32_t pos =
            kPropRangeFast +
            std::min<pixel_type_w>(
                std::max<pixel_type_w>(-kPropRangeFast, top + left - topleft),
                kPropRangeFast - 1);
        uint32_t ctx_id = context_lookup[pos];
        int32_t residual = r[x] - guess - offsets[pos];
        *tokenp++ = Token(ctx_id, PackSigned(residual));
      }
    }
  } else if (tree.size() == 1 && tree[0].predictor == Predictor::Zero &&
             tree[0].multiplier == 1 && tree[0].predictor_offset == 0) {
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT p = channel.Row(y);
      for (size_t x = 0; x < channel.w; x++) {
        *tokenp++ = Token(tree[0].childID, PackSigned(p[x]));
      }
    }
  } else if (tree.size() == 1 &&
             (tree[0].multiplier & (tree[0].multiplier - 1)) == 0 &&
             tree[0].predictor_offset == 0) {
    uint32_t mul_shift = FloorLog2Nonzero((uint32_t)tree[0].multiplier);
    const intptr_t onerow = channel.plane.PixelsPerRow();
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT r = channel.Row(y);
      for (size_t x = 0; x < channel.w; x++) {
        PredictionResult pred = PredictNoTreeNoWP(channel.w, r + x, onerow, x,
                                                  y, tree[0].predictor);
        pixel_type_w residual = r[x] - pred.guess;
        JXL_DASSERT((residual >> mul_shift) * tree[0].multiplier == residual);
        *tokenp++ = Token(tree[0].childID, PackSigned(residual >> mul_shift));
      }
    }

  } else {
    const intptr_t onerow = channel.plane.PixelsPerRow();
    Channel references(properties.size() - kNumNonrefProperties, channel.w);
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT p = channel.Row(y);
      PrecomputeReferences(channel, y, image, chan, &references);
      InitPropsRow(&properties, static_props, y);
      for (size_t x = 0; x < channel.w; x++) {
        PredictionResult res =
            PredictTreeNoWP(&properties, channel.w, p + x, onerow, x, y,
                            tree_lookup, references);
        pixel_type_w residual = p[x] - res.guess;
        JXL_ASSERT(residual % res.multiplier == 0);
        *tokenp++ = Token(res.context, PackSigned(residual / res.multiplier));
      }
    }
  }
  *tokenpp = tokenp;
  return true;
}

Status ModularEncode(const Image &image, size_t group_id, const Tree &tree,
                     std::vector<Token> *tokens, size_t *width) {
  if (image.error) return JXL_FAILURE("Invalid image");
  size_t nb_channels = image.channel.size();
  JXL_DEBUG_V(
      2, "Encoding %" PRIuS "-channel, %i-bit, %" PRIuS "x%" PRIuS " image.",
      nb_channels, image.bitdepth, image.w, image.h);

  if (nb_channels < 1) {
    return true;  // is there any use for a zero-channel image?
  }

  std::vector<std::vector<Token>> tokens_storage(1);

  size_t image_width = 0;
  size_t total_tokens = 0;
  for (size_t i = 0; i < nb_channels; i++) {
    if (image.channel[i].w > image_width) image_width = image.channel[i].w;
    total_tokens += image.channel[i].w * image.channel[i].h;
  }
  // Do one big allocation for all the tokens we'll need,
  // to avoid reallocs that might require copying.
  size_t pos = tokens->size();
  tokens->resize(pos + total_tokens);
  Token *tokenp = tokens->data() + pos;
  for (size_t i = 0; i < nb_channels; i++) {
    if (!image.channel[i].w || !image.channel[i].h) {
      continue;  // skip empty channels
    }
    JXL_RETURN_IF_ERROR(
        EncodeModularChannelMAANS(image, i, tree, &tokenp, group_id));
  }
  // Make sure we actually wrote all tokens
  JXL_CHECK(tokenp == tokens->data() + tokens->size());

  *width = image_width;
  return true;
}

Status ModularGenericCompress(Image &image, size_t group_id, const Tree &tree,
                              std::vector<Token> *tokens, size_t *width) {
  if (image.w == 0 || image.h == 0) return true;

  JXL_RETURN_IF_ERROR(ModularEncode(image, group_id, tree, tokens, width));
  return true;
}

}  // namespace jxl
