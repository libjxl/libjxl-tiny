// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "lib/jxl/enc_frame.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <vector>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/ans_params.h"
#include "lib/jxl/aux_out.h"
#include "lib/jxl/aux_out_fwd.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/profiler.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/coeff_order.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/color_management.h"
#include "lib/jxl/common.h"
#include "lib/jxl/compressed_dc.h"
#include "lib/jxl/dct_util.h"
#include "lib/jxl/dec_modular.h"
#include "lib/jxl/enc_adaptive_quantization.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_chroma_from_luma.h"
#include "lib/jxl/enc_coeff_order.h"
#include "lib/jxl/enc_entropy_coder.h"
#include "lib/jxl/enc_group.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_toc.h"
#include "lib/jxl/enc_xyb.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/gaborish.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/loop_filter.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/encoding/enc_debug_tree.h"
#include "lib/jxl/modular/encoding/enc_encoding.h"
#include "lib/jxl/modular/encoding/encoding.h"
#include "lib/jxl/modular/encoding/ma_common.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/quant_weights.h"
#include "lib/jxl/quantizer.h"
#include "lib/jxl/splines.h"
#include "lib/jxl/toc.h"

namespace jxl {
namespace {

Status LoopFilterFromParams(const CompressParams& cparams,
                            FrameHeader* JXL_RESTRICT frame_header) {
  LoopFilter* loop_filter = &frame_header->loop_filter;

  // Gaborish defaults to enabled in Hare or slower.
  loop_filter->gab = ApplyOverride(
      cparams.gaborish, cparams.speed_tier <= SpeedTier::kHare &&
                            cparams.decoding_speed_tier < 4);

  if (cparams.epf != -1) {
    loop_filter->epf_iters = cparams.epf;
  } else {
    constexpr float kThresholds[3] = {0.7, 1.5, 4.0};
    loop_filter->epf_iters = 0;
    if (cparams.decoding_speed_tier < 3) {
      for (size_t i = cparams.decoding_speed_tier == 2 ? 1 : 0; i < 3; i++) {
        if (cparams.butteraugli_distance >= kThresholds[i]) {
          loop_filter->epf_iters++;
        }
      }
    }
  }

  return true;
}

Status MakeFrameHeader(const CompressParams& cparams,
                       const FrameInfo& frame_info, const ImageBundle& ib,
                       FrameHeader* JXL_RESTRICT frame_header) {
  frame_header->nonserialized_is_preview = frame_info.is_preview;
  frame_header->is_last = frame_info.is_last;
  frame_header->save_before_color_transform =
      frame_info.save_before_color_transform;
  frame_header->frame_type = frame_info.frame_type;
  frame_header->name = ib.name;
  frame_header->passes.num_passes = 1;
  frame_header->passes.num_downsample = 0;
  frame_header->passes.shift[0] = 0;
  frame_header->encoding = FrameEncoding::kVarDCT;
  frame_header->color_transform = ColorTransform::kXYB;
  frame_header->chroma_subsampling = ib.chroma_subsampling;
  frame_header->flags = 0;

  JXL_RETURN_IF_ERROR(LoopFilterFromParams(cparams, frame_header));

  frame_header->dc_level = 0;
  frame_header->flags |= FrameHeader::kSkipAdaptiveDCSmoothing;
  frame_header->frame_origin = ib.origin;
  frame_header->frame_size.xsize = ib.xsize();
  frame_header->frame_size.ysize = ib.ysize();
  if (ib.origin.x0 != 0 || ib.origin.y0 != 0 ||
      frame_header->frame_size.xsize != frame_header->default_xsize() ||
      frame_header->frame_size.ysize != frame_header->default_ysize()) {
    frame_header->custom_size_or_origin = true;
  }
  // Upsampling.
  frame_header->upsampling = 1;
  frame_header->save_as_reference = frame_info.save_as_reference;
  frame_header->animation_frame.duration = ib.duration;
  frame_header->animation_frame.timecode = ib.timecode;

  return true;
}

// `cutoffs` must be sorted.
Tree MakeFixedTree(int property, const std::vector<int32_t>& cutoffs,
                   Predictor pred, size_t num_pixels) {
  size_t log_px = CeilLog2Nonzero(num_pixels);
  size_t min_gap = 0;
  // Reduce fixed tree height when encoding small images.
  if (log_px < 14) {
    min_gap = 8 * (14 - log_px);
  }
  Tree tree;
  struct NodeInfo {
    size_t begin, end, pos;
  };
  std::queue<NodeInfo> q;
  // Leaf IDs will be set by roundtrip decoding the tree.
  tree.push_back(PropertyDecisionNode::Leaf(pred));
  q.push(NodeInfo{0, cutoffs.size(), 0});
  while (!q.empty()) {
    NodeInfo info = q.front();
    q.pop();
    if (info.begin + min_gap >= info.end) continue;
    uint32_t split = (info.begin + info.end) / 2;
    tree[info.pos] =
        PropertyDecisionNode::Split(property, cutoffs[split], tree.size());
    q.push(NodeInfo{split + 1, info.end, tree.size()});
    tree.push_back(PropertyDecisionNode::Leaf(pred));
    q.push(NodeInfo{info.begin, split, tree.size()});
    tree.push_back(PropertyDecisionNode::Leaf(pred));
  }
  return tree;
}

Tree PredefinedTree(ModularOptions::TreeKind tree_kind, size_t total_pixels) {
  if (tree_kind == ModularOptions::TreeKind::kFalconACMeta) {
    // All the data is 0 except the quant field. TODO(veluca): make that 0 too.
    return {PropertyDecisionNode::Leaf(Predictor::Left)};
  }
  if (tree_kind == ModularOptions::TreeKind::kACMeta) {
    // Small image.
    if (total_pixels < 1024) {
      return {PropertyDecisionNode::Leaf(Predictor::Left)};
    }
    Tree tree;
    // 0: c > 1
    tree.push_back(PropertyDecisionNode::Split(0, 1, 1));
    // 1: c > 2
    tree.push_back(PropertyDecisionNode::Split(0, 2, 3));
    // 2: c > 0
    tree.push_back(PropertyDecisionNode::Split(0, 0, 5));
    // 3: EPF control field (all 0 or 4), top > 0
    tree.push_back(PropertyDecisionNode::Split(6, 0, 21));
    // 4: ACS+QF, y > 0
    tree.push_back(PropertyDecisionNode::Split(2, 0, 7));
    // 5: CfL x
    tree.push_back(PropertyDecisionNode::Leaf(Predictor::Gradient));
    // 6: CfL b
    tree.push_back(PropertyDecisionNode::Leaf(Predictor::Gradient));
    // 7: QF: split according to the left quant value.
    tree.push_back(PropertyDecisionNode::Split(7, 5, 9));
    // 8: ACS: split in 4 segments (8x8 from 0 to 3, large square 4-5, large
    // rectangular 6-11, 8x8 12+), according to previous ACS value.
    tree.push_back(PropertyDecisionNode::Split(7, 5, 15));
    // QF
    tree.push_back(PropertyDecisionNode::Split(7, 11, 11));
    tree.push_back(PropertyDecisionNode::Split(7, 3, 13));
    tree.push_back(PropertyDecisionNode::Leaf(Predictor::Left));
    tree.push_back(PropertyDecisionNode::Leaf(Predictor::Left));
    tree.push_back(PropertyDecisionNode::Leaf(Predictor::Left));
    tree.push_back(PropertyDecisionNode::Leaf(Predictor::Left));
    // ACS
    tree.push_back(PropertyDecisionNode::Split(7, 11, 17));
    tree.push_back(PropertyDecisionNode::Split(7, 3, 19));
    tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
    tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
    tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
    tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
    // EPF, left > 0
    tree.push_back(PropertyDecisionNode::Split(7, 0, 23));
    tree.push_back(PropertyDecisionNode::Split(7, 0, 25));
    tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
    tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
    tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
    tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
    return tree;
  }
  if (tree_kind == ModularOptions::TreeKind::kGradientFixedDC) {
    std::vector<int32_t> cutoffs = {
        -500, -392, -255, -191, -127, -95, -63, -47, -31, -23, -15,
        -11,  -7,   -4,   -3,   -1,   0,   1,   3,   5,   7,   11,
        15,   23,   31,   47,   63,   95,  127, 191, 255, 392, 500};
    return MakeFixedTree(kGradientProp, cutoffs, Predictor::Gradient,
                         total_pixels);
  }
  JXL_ABORT("Unreachable");
  return {};
}

// Merges the trees in `trees` using nodes that decide on stream_id, as defined
// by `tree_splits`.
void MergeTrees(const std::vector<Tree>& trees,
                const std::vector<size_t>& tree_splits, size_t begin,
                size_t end, Tree* tree) {
  JXL_ASSERT(trees.size() + 1 == tree_splits.size());
  JXL_ASSERT(end > begin);
  JXL_ASSERT(end <= trees.size());
  if (end == begin + 1) {
    // Insert the tree, adding the opportune offset to all child nodes.
    // This will make the leaf IDs wrong, but subsequent roundtripping will fix
    // them.
    size_t sz = tree->size();
    tree->insert(tree->end(), trees[begin].begin(), trees[begin].end());
    for (size_t i = sz; i < tree->size(); i++) {
      (*tree)[i].lchild += sz;
      (*tree)[i].rchild += sz;
    }
    return;
  }
  size_t mid = (begin + end) / 2;
  size_t splitval = tree_splits[mid] - 1;
  size_t cur = tree->size();
  tree->emplace_back(1 /*stream_id*/, splitval, 0, 0, Predictor::Zero, 0, 1);
  (*tree)[cur].lchild = tree->size();
  MergeTrees(trees, tree_splits, mid, end, tree);
  (*tree)[cur].rchild = tree->size();
  MergeTrees(trees, tree_splits, begin, mid, tree);
}

struct EncCache {
  // Allocates memory when first called, shrinks images to current group size.
  void InitOnce() {
    if (num_nzeroes.xsize() == 0) {
      num_nzeroes = Image3I(kGroupDimInBlocks, kGroupDimInBlocks);
    }
  }
  // TokenizeCoefficients
  Image3I num_nzeroes;
};

class ModularFrameEncoder {
 public:
  ModularFrameEncoder(const FrameHeader& frame_header,
                      const CompressParams& cparams_orig)
      : frame_dim_(frame_header.ToFrameDimensions()), cparams_(cparams_orig) {
    size_t num_streams =
        ModularStreamId::Num(frame_dim_, frame_header.passes.num_passes);
    if (cparams_.IsLossless()) {
      switch (cparams_.decoding_speed_tier) {
        case 0:
          break;
        case 1:
          cparams_.options.wp_tree_mode = ModularOptions::TreeMode::kWPOnly;
          break;
        case 2: {
          cparams_.options.wp_tree_mode =
              ModularOptions::TreeMode::kGradientOnly;
          cparams_.options.predictor = Predictor::Gradient;
          break;
        }
        case 3: {  // LZ77, no Gradient.
          cparams_.options.nb_repeats = 0;
          cparams_.options.predictor = Predictor::Gradient;
          break;
        }
        default: {  // LZ77, no predictor.
          cparams_.options.nb_repeats = 0;
          cparams_.options.predictor = Predictor::Zero;
          break;
        }
      }
    }
    stream_images_.resize(num_streams);

    cparams_.responsive = 0;

    if (cparams_.speed_tier > SpeedTier::kWombat) {
      cparams_.options.splitting_heuristics_node_threshold = 192;
    } else {
      cparams_.options.splitting_heuristics_node_threshold = 96;
    }
    {
      // Set properties.
      std::vector<uint32_t> prop_order;
      prop_order = {0, 1, 15, 9, 10, 11, 12, 13, 14, 2, 3, 4, 5, 6, 7, 8};
      switch (cparams_.speed_tier) {
        case SpeedTier::kSquirrel:
        case SpeedTier::kKitten:
        case SpeedTier::kTortoise:
          cparams_.options.splitting_heuristics_properties.assign(
              prop_order.begin(), prop_order.begin() + 8);
          cparams_.options.max_property_values = 32;
          break;
        default:
          cparams_.options.splitting_heuristics_properties.assign(
              prop_order.begin(), prop_order.begin() + 6);
          cparams_.options.max_property_values = 16;
          break;
      }
      // Gradient in previous channels.
      for (int i = 0; i < cparams_.options.max_properties; i++) {
        cparams_.options.splitting_heuristics_properties.push_back(
            kNumNonrefProperties + i * 4 + 3);
      }
    }

    if (cparams_.options.predictor == static_cast<Predictor>(-1)) {
      // no explicit predictor(s) given, set a good default
      if (cparams_.modular_mode == false && cparams_.IsLossless() &&
          cparams_.responsive == false) {
        // TODO(veluca): allow all predictors that don't break residual
        // multipliers in lossy mode.
        cparams_.options.predictor = Predictor::Variable;
      } else if (!cparams_.IsLossless()) {
        // If not responsive and lossy. TODO(veluca): use near_lossless instead?
        cparams_.options.predictor = Predictor::Gradient;
      } else if (cparams_.speed_tier < SpeedTier::kFalcon) {
        // try median and weighted predictor for anything else
        cparams_.options.predictor = Predictor::Best;
      } else if (cparams_.speed_tier == SpeedTier::kFalcon) {
        // just weighted predictor in falcon mode
        cparams_.options.predictor = Predictor::Weighted;
      } else if (cparams_.speed_tier > SpeedTier::kFalcon) {
        // just gradient predictor in thunder mode
        cparams_.options.predictor = Predictor::Gradient;
      }
    } else {
      delta_pred_ = cparams_.options.predictor;
    }
    if (!cparams_.IsLossless()) {
      if (cparams_.options.predictor == Predictor::Weighted ||
          cparams_.options.predictor == Predictor::Variable ||
          cparams_.options.predictor == Predictor::Best)
        cparams_.options.predictor = Predictor::Zero;
    }
    tree_splits_.push_back(0);
    if (cparams_.modular_mode == false) {
      cparams_.options.fast_decode_multiplier = 1.0f;
      tree_splits_.push_back(ModularStreamId::VarDCTDC(0).ID(frame_dim_));
      tree_splits_.push_back(ModularStreamId::ModularDC(0).ID(frame_dim_));
      tree_splits_.push_back(ModularStreamId::ACMetadata(0).ID(frame_dim_));
      tree_splits_.push_back(ModularStreamId::QuantTable(0).ID(frame_dim_));
      tree_splits_.push_back(ModularStreamId::ModularAC(0, 0).ID(frame_dim_));
      ac_metadata_size.resize(frame_dim_.num_dc_groups);
      extra_dc_precision.resize(frame_dim_.num_dc_groups);
    }
    tree_splits_.push_back(num_streams);
    cparams_.options.max_chan_size = frame_dim_.group_dim;
    cparams_.options.group_dim = frame_dim_.group_dim;

    // TODO(veluca): figure out how to use different predictor sets per channel.
    stream_options_.resize(num_streams, cparams_.options);
  }
  Status ComputeEncodingData(const FrameHeader& frame_header,
                             const ImageMetadata& metadata, ThreadPool* pool,
                             AuxOut* aux_out) {
    JXL_DEBUG_V(6, "Computing modular encoding data for frame %s",
                frame_header.DebugString().c_str());

    stream_options_[0] = cparams_.options;

    if (!tree_.empty()) return true;

    // Compute tree.
    size_t num_streams = stream_images_.size();
    stream_headers_.resize(num_streams);
    tokens_.resize(num_streams);

    // Avoid creating a tree with leaves that don't correspond to any pixels.
    std::vector<size_t> useful_splits;
    useful_splits.reserve(tree_splits_.size());
    for (size_t chunk = 0; chunk < tree_splits_.size() - 1; chunk++) {
      bool has_pixels = false;
      size_t start = tree_splits_[chunk];
      size_t stop = tree_splits_[chunk + 1];
      for (size_t i = start; i < stop; i++) {
        if (!stream_images_[i].empty()) has_pixels = true;
      }
      if (has_pixels) {
        useful_splits.push_back(tree_splits_[chunk]);
      }
    }
    // Don't do anything if modular mode does not have any pixels in this image
    if (useful_splits.empty()) return true;
    useful_splits.push_back(tree_splits_.back());

    std::vector<Tree> trees(useful_splits.size() - 1);
    for (uint32_t chunk = 0; chunk + 1 < useful_splits.size(); ++chunk) {
      size_t total_pixels = 0;
      uint32_t start = useful_splits[chunk];
      uint32_t stop = useful_splits[chunk + 1];
      while (start < stop && stream_images_[start].empty()) ++start;
      while (start < stop && stream_images_[stop - 1].empty()) --stop;
      for (size_t i = start; i < stop; i++) {
        for (const Channel& ch : stream_images_[i].channel) {
          total_pixels += ch.w * ch.h;
        }
      }
      trees[chunk] =
          PredefinedTree(stream_options_[start].tree_kind, total_pixels);
    }
    tree_.clear();
    MergeTrees(trees, useful_splits, 0, useful_splits.size() - 1, &tree_);
    tree_tokens_.resize(1);
    tree_tokens_[0].clear();
    Tree decoded_tree;
    TokenizeTree(tree_, &tree_tokens_[0], &decoded_tree);
    JXL_ASSERT(tree_.size() == decoded_tree.size());
    tree_ = std::move(decoded_tree);

    if (WantDebugOutput(aux_out)) {
      if (frame_header.dc_level > 0) {
        PrintTree(tree_, aux_out->debug_prefix + "/dc_frame_level" +
                             std::to_string(frame_header.dc_level) + "_tree");
      } else {
        PrintTree(tree_, aux_out->debug_prefix + "/global_tree");
      }
    }

    image_widths_.resize(num_streams);
    JXL_RETURN_IF_ERROR(RunOnPool(
        pool, 0, num_streams, ThreadPool::NoInit,
        [&](const uint32_t stream_id, size_t /* thread */) {
          AuxOut my_aux_out;
          if (aux_out) {
            my_aux_out.dump_image = aux_out->dump_image;
            my_aux_out.debug_prefix = aux_out->debug_prefix;
          }
          tokens_[stream_id].clear();
          JXL_CHECK(ModularGenericCompress(
              stream_images_[stream_id], stream_options_[stream_id],
              /*writer=*/nullptr, &my_aux_out, 0, stream_id,
              /*total_pixels=*/nullptr,
              /*tree=*/&tree_, /*header=*/&stream_headers_[stream_id],
              /*tokens=*/&tokens_[stream_id],
              /*widths=*/&image_widths_[stream_id]));
        },
        "ComputeTokens"));
    return true;
  }
  // Encodes global info (tree + histograms) in the `writer`.
  Status EncodeGlobalInfo(BitWriter* writer, AuxOut* aux_out) {
    BitWriter::Allotment allotment(writer, 1);
    // If we are using brotli, or not using modular mode.
    if (tree_tokens_.empty() || tree_tokens_[0].empty()) {
      writer->Write(1, 0);
      ReclaimAndCharge(writer, &allotment, kLayerModularTree, aux_out);
      return true;
    }
    writer->Write(1, 1);
    ReclaimAndCharge(writer, &allotment, kLayerModularTree, aux_out);

    // Write tree
    HistogramParams params;
    params.clustering = HistogramParams::ClusteringType::kFast;
    params.lz77_method =
        cparams_.decoding_speed_tier >= 3 && cparams_.modular_mode
            ? HistogramParams::LZ77Method::kRLE
            : HistogramParams::LZ77Method::kNone;
    if (cparams_.decoding_speed_tier >= 1) {
      params.max_histograms = 12;
    }
    if (cparams_.decoding_speed_tier >= 1 && cparams_.responsive) {
      params.lz77_method = HistogramParams::LZ77Method::kRLE;
    }
    BuildAndEncodeHistograms(params, kNumTreeContexts, tree_tokens_, &code_,
                             &context_map_, writer, kLayerModularTree, aux_out);
    WriteTokens(tree_tokens_[0], code_, context_map_, writer, kLayerModularTree,
                aux_out);
    params.image_widths = image_widths_;
    // Write histograms.
    BuildAndEncodeHistograms(params, (tree_.size() + 1) / 2, tokens_, &code_,
                             &context_map_, writer, kLayerModularGlobal,
                             aux_out);
    return true;
  }
  // Encodes a specific modular image (identified by `stream`) in the `writer`,
  // assigning bits to the provided `layer`.
  Status EncodeStream(BitWriter* writer, AuxOut* aux_out, size_t layer,
                      const ModularStreamId& stream) {
    size_t stream_id = stream.ID(frame_dim_);
    if (stream_images_[stream_id].channel.empty()) {
      return true;  // Image with no channels, header never gets decoded.
    }
    JXL_RETURN_IF_ERROR(
        Bundle::Write(stream_headers_[stream_id], writer, layer, aux_out));
    WriteTokens(tokens_[stream_id], code_, context_map_, writer, layer,
                aux_out);
    return true;
  }
  // Creates a modular image for a given DC group of VarDCT mode. `dc` is the
  // input DC image, not quantized; the group is specified by `group_index`, and
  // `nl_dc` decides whether to apply a near-lossless processing to the DC or
  // not.
  void AddVarDCTDC(const Image3F& dc, size_t group_index,
                   PassesSharedState* shared) {
    const Rect r = shared->DCGroupRect(group_index);
    extra_dc_precision[group_index] = 0;

    size_t stream_id = ModularStreamId::VarDCTDC(group_index).ID(frame_dim_);
    stream_options_[stream_id].max_chan_size = 0xFFFFFF;
    stream_options_[stream_id].predictor = Predictor::Gradient;
    stream_options_[stream_id].tree_kind =
        ModularOptions::TreeKind::kGradientFixedDC;

    stream_images_[stream_id] = Image(r.xsize(), r.ysize(), 8, 3);
    if (shared->frame_header.chroma_subsampling.Is444()) {
      for (size_t c : {1, 0, 2}) {
        float inv_factor = shared->quantizer.GetInvDcStep(c);
        float y_factor = shared->quantizer.GetDcStep(1);
        float cfl_factor = shared->cmap.DCFactors()[c];
        for (size_t y = 0; y < r.ysize(); y++) {
          int32_t* quant_row =
              stream_images_[stream_id].channel[c < 2 ? c ^ 1 : c].plane.Row(y);
          const float* row = r.ConstPlaneRow(dc, c, y);
          if (c == 1) {
            for (size_t x = 0; x < r.xsize(); x++) {
              quant_row[x] = roundf(row[x] * inv_factor);
            }
          } else {
            int32_t* quant_row_y =
                stream_images_[stream_id].channel[0].plane.Row(y);
            for (size_t x = 0; x < r.xsize(); x++) {
              quant_row[x] =
                  roundf((row[x] - quant_row_y[x] * (y_factor * cfl_factor)) *
                         inv_factor);
            }
          }
        }
      }
    } else {
      for (size_t c : {1, 0, 2}) {
        Rect rect(
            r.x0() >> shared->frame_header.chroma_subsampling.HShift(c),
            r.y0() >> shared->frame_header.chroma_subsampling.VShift(c),
            r.xsize() >> shared->frame_header.chroma_subsampling.HShift(c),
            r.ysize() >> shared->frame_header.chroma_subsampling.VShift(c));
        float inv_factor = shared->quantizer.GetInvDcStep(c);
        size_t ys = rect.ysize();
        size_t xs = rect.xsize();
        Channel& ch = stream_images_[stream_id].channel[c < 2 ? c ^ 1 : c];
        ch.w = xs;
        ch.h = ys;
        ch.shrink();
        for (size_t y = 0; y < ys; y++) {
          int32_t* quant_row = ch.plane.Row(y);
          const float* row = rect.ConstPlaneRow(dc, c, y);
          for (size_t x = 0; x < xs; x++) {
            quant_row[x] = roundf(row[x] * inv_factor);
          }
        }
      }
    }
  }
  // Creates a modular image for the AC metadata of the given group
  // (`group_index`).
  void AddACMetadata(size_t group_index, PassesSharedState* shared) {
    const Rect r = shared->DCGroupRect(group_index);
    size_t stream_id = ModularStreamId::ACMetadata(group_index).ID(frame_dim_);
    stream_options_[stream_id].max_chan_size = 0xFFFFFF;
    stream_options_[stream_id].wp_tree_mode = ModularOptions::TreeMode::kNoWP;
    if (cparams_.speed_tier >= SpeedTier::kFalcon) {
      stream_options_[stream_id].tree_kind =
          ModularOptions::TreeKind::kFalconACMeta;
    } else {
      stream_options_[stream_id].tree_kind = ModularOptions::TreeKind::kACMeta;
    }
    // YToX, YToB, ACS + QF, EPF
    Image& image = stream_images_[stream_id];
    image = Image(r.xsize(), r.ysize(), 8, 4);
    static_assert(kColorTileDimInBlocks == 8, "Color tile size changed");
    Rect cr(r.x0() >> 3, r.y0() >> 3, (r.xsize() + 7) >> 3,
            (r.ysize() + 7) >> 3);
    image.channel[0] = Channel(cr.xsize(), cr.ysize(), 3, 3);
    image.channel[1] = Channel(cr.xsize(), cr.ysize(), 3, 3);
    image.channel[2] = Channel(r.xsize() * r.ysize(), 2, 0, 0);
    ConvertPlaneAndClamp(cr, shared->cmap.ytox_map,
                         Rect(image.channel[0].plane), &image.channel[0].plane);
    ConvertPlaneAndClamp(cr, shared->cmap.ytob_map,
                         Rect(image.channel[1].plane), &image.channel[1].plane);
    size_t num = 0;
    for (size_t y = 0; y < r.ysize(); y++) {
      AcStrategyRow row_acs = shared->ac_strategy.ConstRow(r, y);
      const int32_t* row_qf = r.ConstRow(shared->raw_quant_field, y);
      const uint8_t* row_epf = r.ConstRow(shared->epf_sharpness, y);
      int32_t* out_acs = image.channel[2].plane.Row(0);
      int32_t* out_qf = image.channel[2].plane.Row(1);
      int32_t* row_out_epf = image.channel[3].plane.Row(y);
      for (size_t x = 0; x < r.xsize(); x++) {
        row_out_epf[x] = row_epf[x];
        if (!row_acs[x].IsFirstBlock()) continue;
        out_acs[num] = row_acs[x].RawStrategy();
        out_qf[num] = row_qf[x] - 1;
        num++;
      }
    }
    image.channel[2].w = num;
    ac_metadata_size[group_index] = num;
  }
  std::vector<size_t> ac_metadata_size;
  std::vector<uint8_t> extra_dc_precision;

 private:
  std::vector<Image> stream_images_;
  std::vector<ModularOptions> stream_options_;

  Tree tree_;
  std::vector<std::vector<Token>> tree_tokens_;
  std::vector<GroupHeader> stream_headers_;
  std::vector<std::vector<Token>> tokens_;
  EntropyEncodingData code_;
  std::vector<uint8_t> context_map_;
  FrameDimensions frame_dim_;
  CompressParams cparams_;
  std::vector<size_t> tree_splits_;
  std::vector<ModularMultiplierInfo> multiplier_info_;
  std::vector<std::vector<uint32_t>> gi_channel_;
  std::vector<size_t> image_widths_;
  Predictor delta_pred_ = Predictor::Average4;
};

}  // namespace

Status EncodeFrame(const CompressParams& cparams, const FrameInfo& frame_info,
                   const CodecMetadata* metadata, const ImageBundle& ib,
                   const JxlCmsInterface& cms, ThreadPool* pool,
                   BitWriter* writer, AuxOut* aux_out) {
  ib.VerifyMetadata();

  if (cparams.IsLossless() || !metadata->m.xyb_encoded) {
    return JXL_FAILURE("Lossless not implemented");
  }

  if (ib.extra_channels().size() > 0) {
    return JXL_FAILURE("Extra channels not implemented.");
  }
  if (frame_info.dc_level > 0) {
    return JXL_FAILURE("Too many levels of progressive DC");
  }

  if (cparams.butteraugli_distance != 0 &&
      cparams.butteraugli_distance < kMinButteraugliDistance) {
    return JXL_FAILURE("Butteraugli distance is too low (%f)",
                       cparams.butteraugli_distance);
  }

  if (ib.IsJPEG()) {
    return JXL_FAILURE("JPEG transcodeing not implemented");
  }

  if (ib.xsize() == 0 || ib.ysize() == 0) return JXL_FAILURE("Empty image");

  std::unique_ptr<FrameHeader> frame_header =
      jxl::make_unique<FrameHeader>(metadata);
  JXL_RETURN_IF_ERROR(MakeFrameHeader(cparams,
                                      frame_info, ib, frame_header.get()));

  FrameDimensions frame_dim = frame_header->ToFrameDimensions();

  const size_t num_groups = frame_dim.num_groups;

  Image3F opsin;

  std::vector<AuxOut> aux_outs;
  // LossyFrameEncoder stores a reference to a std::function<Status(size_t)>
  // so we need to keep the std::function<Status(size_t)> being referenced
  // alive while lossy_frame_encoder is used. We could make resize_aux_outs a
  // lambda type by making LossyFrameEncoder a template instead, but this is
  // simpler.
  const std::function<Status(size_t)> resize_aux_outs =
      [&aux_outs, aux_out](const size_t num_threads) -> Status {
    if (aux_out != nullptr) {
      size_t old_size = aux_outs.size();
      for (size_t i = num_threads; i < old_size; i++) {
        aux_out->Assimilate(aux_outs[i]);
      }
      aux_outs.resize(num_threads);
      // Each thread needs these INPUTS. Don't copy the entire AuxOut
      // because it may contain stats which would be Assimilated multiple
      // times below.
      for (size_t i = old_size; i < aux_outs.size(); i++) {
        aux_outs[i].dump_image = aux_out->dump_image;
        aux_outs[i].debug_prefix = aux_out->debug_prefix;
      }
    }
    return true;
  };

  PassesEncoderState enc_state;
  PassesSharedState& shared = enc_state.shared;
  JXL_CHECK(InitializePassesSharedState(*frame_header, &shared,
                                        /*encoder=*/true));
  enc_state.cparams = cparams;
  enc_state.passes.clear();

  std::unique_ptr<ModularFrameEncoder> modular_frame_encoder =
      jxl::make_unique<ModularFrameEncoder>(*frame_header, cparams);

  if (ib.IsJPEG()) {
    return JXL_FAILURE("JPEG transcoding not implemented");
  }

  float x_qm_scale_steps[2] = {1.25f, 9.0f};
  shared.frame_header.x_qm_scale = 2;
  for (float x_qm_scale_step : x_qm_scale_steps) {
    if (enc_state.cparams.butteraugli_distance > x_qm_scale_step) {
      shared.frame_header.x_qm_scale++;
    }
  }
  if (enc_state.cparams.butteraugli_distance < 0.299f) {
    // Favor chromacity preservation for making images appear more
    // faithful to original even with extreme (5-10x) zooming.
    shared.frame_header.x_qm_scale++;
  }
  DefaultEncoderHeuristics heuristics;
  JXL_RETURN_IF_ERROR(heuristics.LossyFrameHeuristics(&enc_state, &ib, &opsin,
                                                      cms, pool, aux_out));

  enc_state.histogram_idx.resize(shared.frame_dim.num_groups);

  enc_state.x_qm_multiplier =
      std::pow(1.25f, shared.frame_header.x_qm_scale - 2.0f);
  enc_state.b_qm_multiplier =
      std::pow(1.25f, shared.frame_header.b_qm_scale - 2.0f);

  if (enc_state.coeffs.size() < shared.frame_header.passes.num_passes) {
    enc_state.coeffs.reserve(shared.frame_header.passes.num_passes);
    for (size_t i = enc_state.coeffs.size();
         i < shared.frame_header.passes.num_passes; i++) {
      // Allocate enough coefficients for each group on every row.
      enc_state.coeffs.emplace_back(make_unique<ACImageT<int32_t>>(
          kGroupDim * kGroupDim, shared.frame_dim.num_groups));
    }
  }
  while (enc_state.coeffs.size() > shared.frame_header.passes.num_passes) {
    enc_state.coeffs.pop_back();
  }

  Image3F dc(shared.frame_dim.xsize_blocks, shared.frame_dim.ysize_blocks);
  JXL_RETURN_IF_ERROR(RunOnPool(
      pool, 0, shared.frame_dim.num_groups, ThreadPool::NoInit,
      [&](size_t group_idx, size_t _) {
        ComputeCoefficients(group_idx, &enc_state, opsin, &dc);
      },
      "Compute coeffs"));

  auto compute_dc_coeffs = [&](int group_index, int /* thread */) {
    modular_frame_encoder->AddVarDCTDC(dc, group_index, &shared);
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, shared.frame_dim.num_dc_groups,
                                ThreadPool::NoInit, compute_dc_coeffs,
                                "Compute DC coeffs"));
  auto compute_ac_meta = [&](int group_index, int /* thread */) {
    modular_frame_encoder->AddACMetadata(group_index, &shared);
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, shared.frame_dim.num_dc_groups,
                                ThreadPool::NoInit, compute_ac_meta,
                                "Compute AC Metadata"));

  if (aux_out != nullptr) {
    aux_out->InspectImage3F("compressed_image:InitializeFrameEncCache:dc_dec",
                            shared.dc_storage);
  }

  enc_state.passes.resize(1);
  for (PassesEncoderState::PassData& pass : enc_state.passes) {
    pass.ac_tokens.resize(shared.frame_dim.num_groups);
  }

  auto used_orders_info = ComputeUsedOrders(
      enc_state.cparams.speed_tier, enc_state.shared.ac_strategy,
      Rect(enc_state.shared.raw_quant_field));
  enc_state.used_orders.clear();
  enc_state.used_orders.resize(1, used_orders_info.second);
  ComputeCoeffOrder(enc_state.cparams.speed_tier, *enc_state.coeffs[0],
                    enc_state.shared.ac_strategy, shared.frame_dim,
                    enc_state.used_orders[0], used_orders_info.first,
                    &enc_state.shared.coeff_orders[0]);
  shared.num_histograms = 1;

  std::vector<EncCache> group_caches;
  const auto tokenize_group_init = [&](const size_t num_threads) {
    group_caches.resize(num_threads);
    return true;
  };
  const auto tokenize_group = [&](const uint32_t group_index,
                                  const size_t thread) {
    // Tokenize coefficients.
    const Rect rect = shared.BlockGroupRect(group_index);
    for (size_t idx_pass = 0; idx_pass < enc_state.passes.size(); idx_pass++) {
      JXL_ASSERT(enc_state.coeffs[idx_pass]->Type() == ACType::k32);
      const int32_t* JXL_RESTRICT ac_rows[3] = {
          enc_state.coeffs[idx_pass]->PlaneRow(0, group_index, 0).ptr32,
          enc_state.coeffs[idx_pass]->PlaneRow(1, group_index, 0).ptr32,
          enc_state.coeffs[idx_pass]->PlaneRow(2, group_index, 0).ptr32,
      };
      // Ensure group cache is initialized.
      group_caches[thread].InitOnce();
      TokenizeCoefficients(
          &shared.coeff_orders[idx_pass * shared.coeff_order_size], rect,
          ac_rows, shared.ac_strategy, frame_header->chroma_subsampling,
          &group_caches[thread].num_nzeroes,
          &enc_state.passes[idx_pass].ac_tokens[group_index]);
    }
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, shared.frame_dim.num_groups,
                                tokenize_group_init, tokenize_group,
                                "TokenizeGroup"));

  *frame_header = shared.frame_header;
  // needs to happen *AFTER* VarDCT-ComputeEncodingData.
  JXL_RETURN_IF_ERROR(modular_frame_encoder->ComputeEncodingData(
      *frame_header, *ib.metadata(), pool, aux_out));

  JXL_RETURN_IF_ERROR(WriteFrameHeader(*frame_header, writer, aux_out));

  // DC global info + DC groups + AC global info + AC groups
  std::vector<BitWriter> group_codes(
      NumTocEntries(frame_dim.num_groups, frame_dim.num_dc_groups, 1, true));
  const size_t global_ac_index = frame_dim.num_dc_groups + 1;
  const bool is_small_image = frame_dim.num_groups == 1;
  const auto get_output = [&](const size_t index) {
    return &group_codes[is_small_image ? 0 : index];
  };

  {
    BitWriter* writer = get_output(0);
    BitWriter::Allotment allotment(writer, 2);
    writer->Write(1, 1);  // default quant dc
    ReclaimAndCharge(writer, &allotment, kLayerQuant, aux_out);
  }
  {
    BitWriter* writer = get_output(0);
    // Encode quantizer DC and global scale.
    JXL_RETURN_IF_ERROR(
        enc_state.shared.quantizer.Encode(writer, kLayerQuant, aux_out));
    writer->Write(1, 1);  // default BlockCtxMap
    ColorCorrelationMapEncodeDC(&enc_state.shared.cmap, writer, kLayerDC,
                                aux_out);
  }
  JXL_RETURN_IF_ERROR(
      modular_frame_encoder->EncodeGlobalInfo(get_output(0), aux_out));
  JXL_RETURN_IF_ERROR(modular_frame_encoder->EncodeStream(
      get_output(0), aux_out, kLayerModularGlobal, ModularStreamId::Global()));

  const auto process_dc_group = [&](const uint32_t group_index,
                                    const size_t thread) {
    AuxOut* my_aux_out = aux_out ? &aux_outs[thread] : nullptr;
    BitWriter* output = get_output(group_index + 1);
    {
      BitWriter::Allotment allotment(output, 2);
      output->Write(2, modular_frame_encoder->extra_dc_precision[group_index]);
      ReclaimAndCharge(output, &allotment, kLayerDC, my_aux_out);
      JXL_CHECK(modular_frame_encoder->EncodeStream(
          output, my_aux_out, kLayerDC,
          ModularStreamId::VarDCTDC(group_index)));
    }
    {
      const Rect& rect = enc_state.shared.DCGroupRect(group_index);
      size_t nb_bits = CeilLog2Nonzero(rect.xsize() * rect.ysize());
      if (nb_bits != 0) {
        BitWriter::Allotment allotment(output, nb_bits);
        output->Write(nb_bits,
                      modular_frame_encoder->ac_metadata_size[group_index] - 1);
        ReclaimAndCharge(output, &allotment, kLayerControlFields, my_aux_out);
      }
      JXL_CHECK(modular_frame_encoder->EncodeStream(
          output, my_aux_out, kLayerControlFields,
          ModularStreamId::ACMetadata(group_index)));
    }
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, frame_dim.num_dc_groups,
                                resize_aux_outs, process_dc_group,
                                "EncodeDCGroup"));
  {
    BitWriter* writer = get_output(global_ac_index);
    {
      BitWriter::Allotment allotment(writer, 1024);
      writer->Write(1, 1);  // all default quant matrices
      size_t num_histo_bits =
          CeilLog2Nonzero(enc_state.shared.frame_dim.num_groups);
      if (num_histo_bits != 0) {
        writer->Write(num_histo_bits, enc_state.shared.num_histograms - 1);
      }
      ReclaimAndCharge(writer, &allotment, kLayerAC, aux_out);
    }

    // Encode coefficient orders.
    size_t order_bits = 0;
    JXL_RETURN_IF_ERROR(
        U32Coder::CanEncode(kOrderEnc, enc_state.used_orders[0], &order_bits));
    BitWriter::Allotment allotment(writer, order_bits);
    JXL_CHECK(U32Coder::Write(kOrderEnc, enc_state.used_orders[0], writer));
    ReclaimAndCharge(writer, &allotment, kLayerOrder, aux_out);
    EncodeCoeffOrders(enc_state.used_orders[0],
                      &enc_state.shared.coeff_orders[0], writer, kLayerOrder,
                      aux_out);

    // Encode histograms.
    HistogramParams hist_params(enc_state.cparams.speed_tier,
                                enc_state.shared.block_ctx_map.NumACContexts());
    hist_params.lz77_method = HistogramParams::LZ77Method::kNone;
    if (enc_state.cparams.decoding_speed_tier >= 1) {
      hist_params.max_histograms = 6;
    }
    BuildAndEncodeHistograms(
        hist_params,
        enc_state.shared.num_histograms *
            enc_state.shared.block_ctx_map.NumACContexts(),
        enc_state.passes[0].ac_tokens, &enc_state.passes[0].codes,
        &enc_state.passes[0].context_map, writer, kLayerAC, aux_out);
  }

  std::atomic<int> num_errors{0};
  const auto process_group = [&](const uint32_t group_index,
                                 const size_t thread) {
    AuxOut* my_aux_out = aux_out ? &aux_outs[thread] : nullptr;
    BitWriter* writer = get_output(2 + frame_dim.num_dc_groups + group_index);
    if (!EncodeGroupTokenizedCoefficients(group_index, 0,
                                          enc_state.histogram_idx[group_index],
                                          enc_state, writer, my_aux_out)) {
      num_errors.fetch_add(1, std::memory_order_relaxed);
      return;
    }
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, num_groups, resize_aux_outs,
                                process_group, "EncodeGroupCoefficients"));

  // Resizing aux_outs to 0 also Assimilates the array.
  static_cast<void>(resize_aux_outs(0));
  JXL_RETURN_IF_ERROR(num_errors.load(std::memory_order_relaxed) == 0);

  for (BitWriter& bw : group_codes) {
    BitWriter::Allotment allotment(&bw, 8);
    bw.ZeroPadToByte();  // end of group.
    ReclaimAndCharge(&bw, &allotment, kLayerAC, aux_out);
  }

  std::vector<coeff_order_t>* permutation_ptr = nullptr;
  std::vector<coeff_order_t> permutation;
  if (cparams.centerfirst && num_groups > 1) {
    permutation_ptr = &permutation;
    // Don't permute global DC/AC or DC.
    permutation.resize(global_ac_index + 1);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::vector<coeff_order_t> ac_group_order(num_groups);
    std::iota(ac_group_order.begin(), ac_group_order.end(), 0);
    size_t group_dim = frame_dim.group_dim;

    // The center of the image is either given by parameters or chosen
    // to be the middle of the image by default if center_x, center_y resp.
    // are not provided.

    int64_t imag_cx;
    if (cparams.center_x != static_cast<size_t>(-1)) {
      JXL_RETURN_IF_ERROR(cparams.center_x < ib.xsize());
      imag_cx = cparams.center_x;
    } else {
      imag_cx = ib.xsize() / 2;
    }

    int64_t imag_cy;
    if (cparams.center_y != static_cast<size_t>(-1)) {
      JXL_RETURN_IF_ERROR(cparams.center_y < ib.ysize());
      imag_cy = cparams.center_y;
    } else {
      imag_cy = ib.ysize() / 2;
    }

    // The center of the group containing the center of the image.
    int64_t cx = (imag_cx / group_dim) * group_dim + group_dim / 2;
    int64_t cy = (imag_cy / group_dim) * group_dim + group_dim / 2;
    // This identifies in what area of the central group the center of the image
    // lies in.
    double direction = -std::atan2(imag_cy - cy, imag_cx - cx);
    // This identifies the side of the central group the center of the image
    // lies closest to. This can take values 0, 1, 2, 3 corresponding to left,
    // bottom, right, top.
    int64_t side = std::fmod((direction + 5 * kPi / 4), 2 * kPi) * 2 / kPi;
    auto get_distance_from_center = [&](size_t gid) {
      Rect r = enc_state.shared.GroupRect(gid);
      int64_t gcx = r.x0() + group_dim / 2;
      int64_t gcy = r.y0() + group_dim / 2;
      int64_t dx = gcx - cx;
      int64_t dy = gcy - cy;
      // The angle is determined by taking atan2 and adding an appropriate
      // starting point depending on the side we want to start on.
      double angle = std::remainder(
          std::atan2(dy, dx) + kPi / 4 + side * (kPi / 2), 2 * kPi);
      // Concentric squares in clockwise order.
      return std::make_pair(std::max(std::abs(dx), std::abs(dy)), angle);
    };
    std::sort(ac_group_order.begin(), ac_group_order.end(),
              [&](coeff_order_t a, coeff_order_t b) {
                return get_distance_from_center(a) <
                       get_distance_from_center(b);
              });
    std::vector<coeff_order_t> inv_ac_group_order(ac_group_order.size(), 0);
    for (size_t i = 0; i < ac_group_order.size(); i++) {
      inv_ac_group_order[ac_group_order[i]] = i;
    }
    size_t pass_start = permutation.size();
    for (coeff_order_t v : inv_ac_group_order) {
      permutation.push_back(pass_start + v);
    }
    std::vector<BitWriter> new_group_codes(group_codes.size());
    for (size_t i = 0; i < permutation.size(); i++) {
      new_group_codes[permutation[i]] = std::move(group_codes[i]);
    }
    group_codes = std::move(new_group_codes);
  }

  JXL_RETURN_IF_ERROR(
      WriteGroupOffsets(group_codes, permutation_ptr, writer, aux_out));
  writer->AppendByteAligned(group_codes);

  return true;
}

}  // namespace jxl
