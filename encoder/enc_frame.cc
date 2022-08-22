// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/enc_frame.h"

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

#include "encoder/ac_context.h"
#include "encoder/ac_strategy.h"
#include "encoder/ans_params.h"
#include "encoder/base/bits.h"
#include "encoder/base/compiler_specific.h"
#include "encoder/base/data_parallel.h"
#include "encoder/base/override.h"
#include "encoder/base/padded_bytes.h"
#include "encoder/base/profiler.h"
#include "encoder/base/status.h"
#include "encoder/chroma_from_luma.h"
#include "encoder/coeff_order.h"
#include "encoder/coeff_order_fwd.h"
#include "encoder/color_encoding_internal.h"
#include "encoder/common.h"
#include "encoder/dct_util.h"
#include "encoder/enc_ac_strategy.h"
#include "encoder/enc_adaptive_quantization.h"
#include "encoder/enc_ans.h"
#include "encoder/enc_bit_writer.h"
#include "encoder/enc_cache.h"
#include "encoder/enc_chroma_from_luma.h"
#include "encoder/enc_coeff_order.h"
#include "encoder/enc_entropy_coder.h"
#include "encoder/enc_group.h"
#include "encoder/enc_toc.h"
#include "encoder/enc_xyb.h"
#include "encoder/fields.h"
#include "encoder/frame_header.h"
#include "encoder/gaborish.h"
#include "encoder/image.h"
#include "encoder/image_ops.h"
#include "encoder/loop_filter.h"
#include "encoder/modular/encoding/context_predict.h"
#include "encoder/modular/encoding/enc_encoding.h"
#include "encoder/modular/encoding/encoding.h"
#include "encoder/modular/modular_image.h"
#include "encoder/modular/options.h"
#include "encoder/quant_weights.h"
#include "encoder/quantizer.h"
#include "encoder/toc.h"

namespace jxl {
namespace {

struct ModularStreamId {
  enum Kind {
    kGlobalData,
    kVarDCTDC,
    kModularDC,
    kACMetadata,
    kQuantTable,
    kModularAC
  };
  Kind kind;
  size_t quant_table_id;
  size_t group_id;  // DC or AC group id.
  size_t pass_id;   // Only for kModularAC.
  size_t ID(const FrameDimensions& frame_dim) const {
    size_t id = 0;
    switch (kind) {
      case kGlobalData:
        id = 0;
        break;
      case kVarDCTDC:
        id = 1 + group_id;
        break;
      case kModularDC:
        id = 1 + frame_dim.num_dc_groups + group_id;
        break;
      case kACMetadata:
        id = 1 + 2 * frame_dim.num_dc_groups + group_id;
        break;
      case kQuantTable:
        id = 1 + 3 * frame_dim.num_dc_groups + quant_table_id;
        break;
      case kModularAC:
        id = 1 + 3 * frame_dim.num_dc_groups + DequantMatrices::kNum +
             frame_dim.num_groups * pass_id + group_id;
        break;
    };
    return id;
  }
  static ModularStreamId Global() {
    return ModularStreamId{kGlobalData, 0, 0, 0};
  }
  static ModularStreamId VarDCTDC(size_t group_id) {
    return ModularStreamId{kVarDCTDC, 0, group_id, 0};
  }
  static ModularStreamId ModularDC(size_t group_id) {
    return ModularStreamId{kModularDC, 0, group_id, 0};
  }
  static ModularStreamId ACMetadata(size_t group_id) {
    return ModularStreamId{kACMetadata, 0, group_id, 0};
  }
  static ModularStreamId QuantTable(size_t quant_table_id) {
    JXL_ASSERT(quant_table_id < DequantMatrices::kNum);
    return ModularStreamId{kQuantTable, quant_table_id, 0, 0};
  }
  static ModularStreamId ModularAC(size_t group_id, size_t pass_id) {
    return ModularStreamId{kModularAC, 0, group_id, pass_id};
  }
  static size_t Num(const FrameDimensions& frame_dim, size_t passes) {
    return ModularAC(0, passes).ID(frame_dim);
  }
  std::string DebugString() const;
};

Status MakeFrameHeader(const float distance,
                       FrameHeader* JXL_RESTRICT frame_header) {
  frame_header->nonserialized_is_preview = false;
  frame_header->is_last = true;
  frame_header->save_before_color_transform = false;
  frame_header->frame_type = FrameType::kRegularFrame;
  frame_header->passes.num_passes = 1;
  frame_header->passes.num_downsample = 0;
  frame_header->passes.shift[0] = 0;
  frame_header->encoding = FrameEncoding::kVarDCT;
  frame_header->color_transform = ColorTransform::kXYB;
  frame_header->flags = 0;
  frame_header->dc_level = 0;
  frame_header->flags |= FrameHeader::kSkipAdaptiveDCSmoothing;
  frame_header->custom_size_or_origin = false;
  frame_header->upsampling = 1;
  frame_header->save_as_reference = 0;

  LoopFilter* loop_filter = &frame_header->loop_filter;
  loop_filter->gab = true;
  constexpr float kThresholds[3] = {0.7, 1.5, 4.0};
  loop_filter->epf_iters = 0;
  for (size_t i = 0; i < 3; i++) {
    if (distance >= kThresholds[i]) {
      loop_filter->epf_iters++;
    }
  }

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
  ModularFrameEncoder(const FrameDimensions& frame_dim)
      : frame_dim_(frame_dim) {
    size_t num_streams = ModularStreamId::Num(frame_dim_, 1);
    ModularOptions options;
    stream_images_.resize(num_streams);
    options.splitting_heuristics_node_threshold = 192;
    // Set properties.
    std::vector<uint32_t> prop_order;
    prop_order = {0, 1, 15, 9, 10, 11, 12, 13, 14, 2, 3, 4, 5, 6, 7, 8};
    options.splitting_heuristics_properties.assign(prop_order.begin(),
                                                   prop_order.begin() + 8);
    options.max_property_values = 32;
    // Gradient in previous channels.
    for (int i = 0; i < options.max_properties; i++) {
      options.splitting_heuristics_properties.push_back(kNumNonrefProperties +
                                                        i * 4 + 3);
    }

    if (options.predictor == static_cast<Predictor>(-1)) {
      // no explicit predictor(s) given, set a good default
      options.predictor = Predictor::Gradient;
    } else {
      delta_pred_ = options.predictor;
    }
    if (options.predictor == Predictor::Weighted ||
        options.predictor == Predictor::Variable ||
        options.predictor == Predictor::Best) {
      options.predictor = Predictor::Zero;
    }
    tree_splits_.push_back(0);
    options.fast_decode_multiplier = 1.0f;
    tree_splits_.push_back(ModularStreamId::VarDCTDC(0).ID(frame_dim_));
    tree_splits_.push_back(ModularStreamId::ModularDC(0).ID(frame_dim_));
    tree_splits_.push_back(ModularStreamId::ACMetadata(0).ID(frame_dim_));
    tree_splits_.push_back(ModularStreamId::QuantTable(0).ID(frame_dim_));
    tree_splits_.push_back(ModularStreamId::ModularAC(0, 0).ID(frame_dim_));
    ac_metadata_size.resize(frame_dim_.num_dc_groups);
    extra_dc_precision.resize(frame_dim_.num_dc_groups);
    tree_splits_.push_back(num_streams);
    options.max_chan_size = frame_dim_.group_dim;
    options.group_dim = frame_dim_.group_dim;

    // TODO(veluca): figure out how to use different predictor sets per channel.
    stream_options_.resize(num_streams, options);
  }
  Status ComputeEncodingData(ThreadPool* pool) {
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

    image_widths_.resize(num_streams);
    JXL_RETURN_IF_ERROR(RunOnPool(
        pool, 0, num_streams, ThreadPool::NoInit,
        [&](const uint32_t stream_id, size_t /* thread */) {
          tokens_[stream_id].clear();
          JXL_CHECK(ModularGenericCompress(
              stream_images_[stream_id], stream_options_[stream_id],
              /*writer=*/nullptr, stream_id,
              /*total_pixels=*/nullptr,
              /*tree=*/&tree_, /*header=*/&stream_headers_[stream_id],
              /*tokens=*/&tokens_[stream_id],
              /*widths=*/&image_widths_[stream_id]));
        },
        "ComputeTokens"));
    return true;
  }
  // Encodes global info (tree + histograms) in the `writer`.
  Status EncodeGlobalInfo(BitWriter* writer) {
    BitWriter::Allotment allotment(writer, 1);
    // If we are using brotli, or not using modular mode.
    if (tree_tokens_.empty() || tree_tokens_[0].empty()) {
      writer->Write(1, 0);
      allotment.Reclaim(writer);
      return true;
    }
    writer->Write(1, 1);
    allotment.Reclaim(writer);

    // Write tree
    HistogramParams params;
    params.clustering = HistogramParams::ClusteringType::kFast;
    params.lz77_method = HistogramParams::LZ77Method::kNone;
    BuildAndEncodeHistograms(params, kNumTreeContexts, tree_tokens_, &code_,
                             &context_map_, writer);
    WriteTokens(tree_tokens_[0], code_, context_map_, writer);
    params.image_widths = image_widths_;
    // Write histograms.
    BuildAndEncodeHistograms(params, (tree_.size() + 1) / 2, tokens_, &code_,
                             &context_map_, writer);
    return true;
  }
  // Encodes a specific modular image (identified by `stream`) in the `writer`,
  // assigning bits to the provided `layer`.
  Status EncodeStream(BitWriter* writer, const ModularStreamId& stream) {
    size_t stream_id = stream.ID(frame_dim_);
    if (stream_images_[stream_id].channel.empty()) {
      return true;  // Image with no channels, header never gets decoded.
    }
    JXL_RETURN_IF_ERROR(Bundle::Write(stream_headers_[stream_id], writer));
    WriteTokens(tokens_[stream_id], code_, context_map_, writer);
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
  }
  // Creates a modular image for the AC metadata of the given group
  // (`group_index`).
  void AddACMetadata(size_t group_index, PassesSharedState* shared) {
    const Rect r = shared->DCGroupRect(group_index);
    size_t stream_id = ModularStreamId::ACMetadata(group_index).ID(frame_dim_);
    stream_options_[stream_id].max_chan_size = 0xFFFFFF;
    stream_options_[stream_id].wp_tree_mode = ModularOptions::TreeMode::kNoWP;
    stream_options_[stream_id].tree_kind = ModularOptions::TreeKind::kACMeta;
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
  std::vector<size_t> tree_splits_;
  std::vector<ModularMultiplierInfo> multiplier_info_;
  std::vector<std::vector<uint32_t>> gi_channel_;
  std::vector<size_t> image_widths_;
  Predictor delta_pred_ = Predictor::Average4;
};

}  // namespace

Status EncodeFrame(const float distance, const CodecMetadata* metadata,
                   const Image3F& linear, ThreadPool* pool, BitWriter* writer) {
  PassesEncoderState enc_state;
  PassesSharedState& shared = enc_state.shared;
  shared.frame_header = FrameHeader(metadata);
  JXL_RETURN_IF_ERROR(MakeFrameHeader(distance, &shared.frame_header));
  shared.frame_dim = shared.frame_header.ToFrameDimensions();

  const FrameDimensions& frame_dim = shared.frame_dim;
  const size_t num_groups = frame_dim.num_groups;

  shared.ac_strategy =
      AcStrategyImage(frame_dim.xsize_blocks, frame_dim.ysize_blocks);
  shared.raw_quant_field =
      ImageI(frame_dim.xsize_blocks, frame_dim.ysize_blocks);
  shared.epf_sharpness = ImageB(frame_dim.xsize_blocks, frame_dim.ysize_blocks);
  shared.cmap = ColorCorrelationMap(frame_dim.xsize, frame_dim.ysize);

  // In the decoder, we allocate coeff orders afterwards, when we know how many
  // we will actually need.
  shared.coeff_order_size = kCoeffOrderMaxSize;
  shared.coeff_orders.resize(kCoeffOrderMaxSize);

  enc_state.passes.clear();

  std::unique_ptr<ModularFrameEncoder> modular_frame_encoder =
      jxl::make_unique<ModularFrameEncoder>(frame_dim);

  float x_qm_scale_steps[2] = {1.25f, 9.0f};
  shared.frame_header.x_qm_scale = 2;
  for (float x_qm_scale_step : x_qm_scale_steps) {
    if (distance > x_qm_scale_step) {
      shared.frame_header.x_qm_scale++;
    }
  }
  if (distance < 0.299f) {
    // Favor chromacity preservation for making images appear more
    // faithful to original even with extreme (5-10x) zooming.
    shared.frame_header.x_qm_scale++;
  }

  Image3F opsin(RoundUpToBlockDim(linear.xsize()),
                RoundUpToBlockDim(linear.ysize()));
  opsin.ShrinkTo(linear.xsize(), linear.ysize());
  ToXYB(linear, pool, &opsin);
  PadImageToBlockMultipleInPlace(&opsin);

  // Dependency graph:
  //
  // input: either XYB or input image
  //
  // XYB -> initial quant field
  // XYB -> Gaborished XYB
  // Gaborished XYB -> CfL1
  // initial quant field, Gaborished XYB, CfL1 -> ACS
  // initial quant field, ACS, Gaborished XYB -> EPF control field
  // initial quant field -> adjusted initial quant field
  // adjusted initial quant field, ACS -> raw quant field
  // raw quant field, ACS, Gaborished XYB -> CfL2
  //
  // output: Gaborished XYB, CfL, ACS, raw quant field, EPF control field.

  // Compute adaptive quantization field, relies on pre-gaborish values.
  float butteraugli_distance_for_iqf = distance;
  if (!shared.frame_header.loop_filter.gab) {
    butteraugli_distance_for_iqf *= 0.73f;
  }
  const float quant_dc = InitialQuantDC(distance);
  enc_state.initial_quant_field =
      InitialQuantField(butteraugli_distance_for_iqf, opsin, shared.frame_dim,
                        pool, 1.0f, &enc_state.initial_quant_masking);
  Quantizer& quantizer = shared.quantizer;
  quantizer.SetQuantField(quant_dc, enc_state.initial_quant_field, nullptr);

  // Apply inverse-gaborish.
  if (shared.frame_header.loop_filter.gab) {
    GaborishInverse(&opsin, 0.9908511000000001f, pool);
  }

  // Flat AR field.
  FillPlane(static_cast<uint8_t>(4), &shared.epf_sharpness);

  AcStrategyHeuristics acs_heuristics;
  CfLHeuristics cfl_heuristics;
  cfl_heuristics.Init(opsin);
  acs_heuristics.Init(opsin, distance, &enc_state);

  auto process_tile = [&](const uint32_t tid, const size_t thread) {
    size_t n_enc_tiles =
        DivCeil(shared.frame_dim.xsize_blocks, kColorTileDimInBlocks);
    size_t tx = tid % n_enc_tiles;
    size_t ty = tid / n_enc_tiles;
    size_t by0 = ty * kColorTileDimInBlocks;
    size_t by1 = std::min((ty + 1) * kColorTileDimInBlocks,
                          shared.frame_dim.ysize_blocks);
    size_t bx0 = tx * kColorTileDimInBlocks;
    size_t bx1 = std::min((tx + 1) * kColorTileDimInBlocks,
                          shared.frame_dim.xsize_blocks);
    Rect r(bx0, by0, bx1 - bx0, by1 - by0);

    // Choose block sizes.
    acs_heuristics.ProcessRect(r);

    // Always set the initial quant field, so we can compute the CfL map with
    // more accuracy. The initial quant field might change in slower modes, but
    // adjusting the quant field with butteraugli when all the other encoding
    // parameters are fixed is likely a more reliable choice anyway.
    AdjustQuantField(shared.ac_strategy, r, &enc_state.initial_quant_field);
    quantizer.SetQuantFieldRect(enc_state.initial_quant_field, r,
                                &shared.raw_quant_field);

    cfl_heuristics.ComputeTile(r, opsin, shared.matrices, &shared.ac_strategy,
                               &shared.quantizer,
                               /*fast=*/true, thread, &shared.cmap);
  };
  JXL_RETURN_IF_ERROR(RunOnPool(
      pool, 0,
      DivCeil(shared.frame_dim.xsize_blocks, kColorTileDimInBlocks) *
          DivCeil(shared.frame_dim.ysize_blocks, kColorTileDimInBlocks),
      [&](const size_t num_threads) {
        cfl_heuristics.PrepareForThreads(num_threads);
        return true;
      },
      process_tile, "Enc Heuristics"));

  cfl_heuristics.ComputeDC(/*fast=*/true, &shared.cmap);

  enc_state.histogram_idx.resize(shared.frame_dim.num_groups);

  enc_state.x_qm_multiplier =
      std::pow(1.25f, shared.frame_header.x_qm_scale - 2.0f);
  enc_state.b_qm_multiplier =
      std::pow(1.25f, shared.frame_header.b_qm_scale - 2.0f);

  // Allocate enough coefficients for each group on every row.
  enc_state.coeffs.push_back(make_unique<ACImageT<int32_t>>(
      kGroupDim * kGroupDim, shared.frame_dim.num_groups));

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

  enc_state.passes.resize(1);
  for (PassesEncoderState::PassData& pass : enc_state.passes) {
    pass.ac_tokens.resize(shared.frame_dim.num_groups);
  }

  auto used_orders_info = ComputeUsedOrders(
      enc_state.shared.ac_strategy, Rect(enc_state.shared.raw_quant_field));
  enc_state.used_orders.clear();
  enc_state.used_orders.resize(1, used_orders_info.second);
  ComputeCoeffOrder(*enc_state.coeffs[0], enc_state.shared.ac_strategy,
                    shared.frame_dim, enc_state.used_orders[0],
                    used_orders_info.first, &enc_state.shared.coeff_orders[0]);
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
          ac_rows, shared.ac_strategy, YCbCrChromaSubsampling(),
          &group_caches[thread].num_nzeroes,
          &enc_state.passes[idx_pass].ac_tokens[group_index]);
    }
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, shared.frame_dim.num_groups,
                                tokenize_group_init, tokenize_group,
                                "TokenizeGroup"));

  // needs to happen *AFTER* VarDCT-ComputeEncodingData.
  JXL_RETURN_IF_ERROR(modular_frame_encoder->ComputeEncodingData(pool));

  JXL_RETURN_IF_ERROR(Bundle::Write(shared.frame_header, writer));

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
    allotment.Reclaim(writer);
  }
  {
    BitWriter* writer = get_output(0);
    // Encode quantizer DC and global scale.
    JXL_RETURN_IF_ERROR(enc_state.shared.quantizer.Encode(writer));
    writer->Write(1, 1);  // default BlockCtxMap
    ColorCorrelationMapEncodeDC(&enc_state.shared.cmap, writer);
  }
  JXL_RETURN_IF_ERROR(modular_frame_encoder->EncodeGlobalInfo(get_output(0)));
  JXL_RETURN_IF_ERROR(modular_frame_encoder->EncodeStream(
      get_output(0), ModularStreamId::Global()));

  const auto process_dc_group = [&](const uint32_t group_index,
                                    const size_t thread) {
    BitWriter* output = get_output(group_index + 1);
    {
      BitWriter::Allotment allotment(output, 2);
      output->Write(2, modular_frame_encoder->extra_dc_precision[group_index]);
      allotment.Reclaim(output);
      JXL_CHECK(modular_frame_encoder->EncodeStream(
          output, ModularStreamId::VarDCTDC(group_index)));
    }
    {
      const Rect& rect = enc_state.shared.DCGroupRect(group_index);
      size_t nb_bits = CeilLog2Nonzero(rect.xsize() * rect.ysize());
      if (nb_bits != 0) {
        BitWriter::Allotment allotment(output, nb_bits);
        output->Write(nb_bits,
                      modular_frame_encoder->ac_metadata_size[group_index] - 1);
        allotment.Reclaim(output);
      }
      JXL_CHECK(modular_frame_encoder->EncodeStream(
          output, ModularStreamId::ACMetadata(group_index)));
    }
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, frame_dim.num_dc_groups,
                                ThreadPool::NoInit, process_dc_group,
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
      allotment.Reclaim(writer);
    }

    // Encode coefficient orders.
    size_t order_bits = 0;
    JXL_RETURN_IF_ERROR(
        U32Coder::CanEncode(kOrderEnc, enc_state.used_orders[0], &order_bits));
    BitWriter::Allotment allotment(writer, order_bits);
    JXL_CHECK(U32Coder::Write(kOrderEnc, enc_state.used_orders[0], writer));
    allotment.Reclaim(writer);
    EncodeCoeffOrders(enc_state.used_orders[0],
                      &enc_state.shared.coeff_orders[0], writer);

    // Encode histograms.
    HistogramParams hist_params;
    hist_params.lz77_method = HistogramParams::LZ77Method::kNone;
    BuildAndEncodeHistograms(hist_params,
                             enc_state.shared.num_histograms *
                                 enc_state.shared.block_ctx_map.NumACContexts(),
                             enc_state.passes[0].ac_tokens,
                             &enc_state.passes[0].codes,
                             &enc_state.passes[0].context_map, writer);
  }

  std::atomic<int> num_errors{0};
  const auto process_group = [&](const uint32_t group_index,
                                 const size_t thread) {
    BitWriter* writer = get_output(2 + frame_dim.num_dc_groups + group_index);
    if (!EncodeGroupTokenizedCoefficients(group_index, 0,
                                          enc_state.histogram_idx[group_index],
                                          enc_state, writer)) {
      num_errors.fetch_add(1, std::memory_order_relaxed);
      return;
    }
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, num_groups, ThreadPool::NoInit,
                                process_group, "EncodeGroupCoefficients"));

  JXL_RETURN_IF_ERROR(num_errors.load(std::memory_order_relaxed) == 0);

  for (BitWriter& bw : group_codes) {
    BitWriter::Allotment allotment(&bw, 8);
    bw.ZeroPadToByte();  // end of group.
    allotment.Reclaim(&bw);
  }

  JXL_RETURN_IF_ERROR(WriteGroupOffsets(group_codes, nullptr, writer));
  writer->AppendByteAligned(group_codes);

  return true;
}

}  // namespace jxl
