// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "lib/jxl/enc_modular.h"

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <atomic>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include "lib/jxl/aux_out.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/compressed_dc.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_cluster.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/gaborish.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/encoding/enc_debug_tree.h"
#include "lib/jxl/modular/encoding/enc_encoding.h"
#include "lib/jxl/modular/encoding/encoding.h"
#include "lib/jxl/modular/encoding/ma_common.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/toc.h"

namespace jxl {

namespace {
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

}  // namespace

ModularFrameEncoder::ModularFrameEncoder(const FrameHeader& frame_header,
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
        cparams_.options.wp_tree_mode = ModularOptions::TreeMode::kGradientOnly;
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

Status ModularFrameEncoder::ComputeEncodingData(
    const FrameHeader& frame_header, const ImageMetadata& metadata,
    Image3F* JXL_RESTRICT color, const std::vector<ImageF>& extra_channels,
    PassesEncoderState* JXL_RESTRICT enc_state, const JxlCmsInterface& cms,
    ThreadPool* pool, AuxOut* aux_out) {
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

Status ModularFrameEncoder::EncodeGlobalInfo(BitWriter* writer,
                                             AuxOut* aux_out) {
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
                           &context_map_, writer, kLayerModularGlobal, aux_out);
  return true;
}

Status ModularFrameEncoder::EncodeStream(BitWriter* writer, AuxOut* aux_out,
                                         size_t layer,
                                         const ModularStreamId& stream) {
  size_t stream_id = stream.ID(frame_dim_);
  if (stream_images_[stream_id].channel.empty()) {
    return true;  // Image with no channels, header never gets decoded.
  }
  JXL_RETURN_IF_ERROR(
      Bundle::Write(stream_headers_[stream_id], writer, layer, aux_out));
  WriteTokens(tokens_[stream_id], code_, context_map_, writer, layer, aux_out);
  return true;
}

void ModularFrameEncoder::AddVarDCTDC(const Image3F& dc, size_t group_index,
                                      PassesEncoderState* enc_state,
                                      bool jpeg_transcode) {
  const Rect r = enc_state->shared.DCGroupRect(group_index);
  extra_dc_precision[group_index] = 0;

  size_t stream_id = ModularStreamId::VarDCTDC(group_index).ID(frame_dim_);
  stream_options_[stream_id].max_chan_size = 0xFFFFFF;
  stream_options_[stream_id].predictor = Predictor::Gradient;
  stream_options_[stream_id].tree_kind =
      ModularOptions::TreeKind::kGradientFixedDC;

  stream_images_[stream_id] = Image(r.xsize(), r.ysize(), 8, 3);
  if (enc_state->shared.frame_header.chroma_subsampling.Is444()) {
    for (size_t c : {1, 0, 2}) {
      float inv_factor = enc_state->shared.quantizer.GetInvDcStep(c);
      float y_factor = enc_state->shared.quantizer.GetDcStep(1);
      float cfl_factor = enc_state->shared.cmap.DCFactors()[c];
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
          r.x0() >> enc_state->shared.frame_header.chroma_subsampling.HShift(c),
          r.y0() >> enc_state->shared.frame_header.chroma_subsampling.VShift(c),
          r.xsize() >>
              enc_state->shared.frame_header.chroma_subsampling.HShift(c),
          r.ysize() >>
              enc_state->shared.frame_header.chroma_subsampling.VShift(c));
      float inv_factor = enc_state->shared.quantizer.GetInvDcStep(c);
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

  DequantDC(r, &enc_state->shared.dc_storage, &enc_state->shared.quant_dc,
            stream_images_[stream_id], enc_state->shared.quantizer.MulDC(), 1.0,
            enc_state->shared.cmap.DCFactors(),
            enc_state->shared.frame_header.chroma_subsampling,
            enc_state->shared.block_ctx_map);
}

void ModularFrameEncoder::AddACMetadata(size_t group_index,
                                        PassesEncoderState* enc_state) {
  const Rect r = enc_state->shared.DCGroupRect(group_index);
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
  Rect cr(r.x0() >> 3, r.y0() >> 3, (r.xsize() + 7) >> 3, (r.ysize() + 7) >> 3);
  image.channel[0] = Channel(cr.xsize(), cr.ysize(), 3, 3);
  image.channel[1] = Channel(cr.xsize(), cr.ysize(), 3, 3);
  image.channel[2] = Channel(r.xsize() * r.ysize(), 2, 0, 0);
  ConvertPlaneAndClamp(cr, enc_state->shared.cmap.ytox_map,
                       Rect(image.channel[0].plane), &image.channel[0].plane);
  ConvertPlaneAndClamp(cr, enc_state->shared.cmap.ytob_map,
                       Rect(image.channel[1].plane), &image.channel[1].plane);
  size_t num = 0;
  for (size_t y = 0; y < r.ysize(); y++) {
    AcStrategyRow row_acs = enc_state->shared.ac_strategy.ConstRow(r, y);
    const int32_t* row_qf = r.ConstRow(enc_state->shared.raw_quant_field, y);
    const uint8_t* row_epf = r.ConstRow(enc_state->shared.epf_sharpness, y);
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

}  // namespace jxl
