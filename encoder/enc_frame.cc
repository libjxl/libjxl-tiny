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
#include "encoder/common.h"
#include "encoder/context_predict.h"
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
#include "encoder/enc_modular.h"
#include "encoder/enc_toc.h"
#include "encoder/enc_xyb.h"
#include "encoder/gaborish.h"
#include "encoder/image.h"
#include "encoder/image_ops.h"
#include "encoder/modular.h"
#include "encoder/quant_weights.h"
#include "encoder/quantizer.h"

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
        id = 1 + 3 * frame_dim.num_dc_groups + DequantMatrices::kNum + group_id;
        break;
    };
    return id;
  }
  static ModularStreamId Global() { return ModularStreamId{kGlobalData, 0, 0}; }
  static ModularStreamId VarDCTDC(size_t group_id) {
    return ModularStreamId{kVarDCTDC, 0, group_id};
  }
  static ModularStreamId ModularDC(size_t group_id) {
    return ModularStreamId{kModularDC, 0, group_id};
  }
  static ModularStreamId ACMetadata(size_t group_id) {
    return ModularStreamId{kACMetadata, 0, group_id};
  }
  static ModularStreamId QuantTable(size_t quant_table_id) {
    JXL_ASSERT(quant_table_id < DequantMatrices::kNum);
    return ModularStreamId{kQuantTable, quant_table_id, 0};
  }
  static ModularStreamId ModularAC(size_t group_id) {
    return ModularStreamId{kModularAC, 0, group_id};
  }
  static size_t Num(const FrameDimensions& frame_dim) {
    return ModularAC(0).ID(frame_dim) + frame_dim.num_groups;
  }
  std::string DebugString() const;
};

Tree MakeFixedTree(size_t num_dc_groups) {
  Tree tree(95);
  tree[0] = PropertyDecisionNode::Split(1, num_dc_groups + 1, 1, 28);
  // ACMetadata
  tree[1] = PropertyDecisionNode::Split(0, 1, 2);
  tree[2] = PropertyDecisionNode::Split(0, 2, 4);
  tree[3] = PropertyDecisionNode::Split(0, 0, 6);
  tree[4] = PropertyDecisionNode::Split(6, 0, 22);
  tree[5] = PropertyDecisionNode::Split(2, 0, 8);
  tree[6] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[7] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[8] = PropertyDecisionNode::Split(7, 5, 10);
  tree[9] = PropertyDecisionNode::Split(7, 5, 16);
  tree[10] = PropertyDecisionNode::Split(7, 11, 12);
  tree[11] = PropertyDecisionNode::Split(7, 3, 14);
  tree[12] = PropertyDecisionNode::Leaf(Predictor::Left);
  tree[13] = PropertyDecisionNode::Leaf(Predictor::Left);
  tree[14] = PropertyDecisionNode::Leaf(Predictor::Left);
  tree[15] = PropertyDecisionNode::Leaf(Predictor::Left);
  tree[16] = PropertyDecisionNode::Split(7, 11, 18);
  tree[17] = PropertyDecisionNode::Split(7, 3, 20);
  tree[18] = PropertyDecisionNode::Leaf(Predictor::Zero);
  tree[19] = PropertyDecisionNode::Leaf(Predictor::Zero);
  tree[20] = PropertyDecisionNode::Leaf(Predictor::Zero);
  tree[21] = PropertyDecisionNode::Leaf(Predictor::Zero);
  tree[22] = PropertyDecisionNode::Split(7, 0, 24);
  tree[23] = PropertyDecisionNode::Split(7, 0, 26);
  tree[24] = PropertyDecisionNode::Leaf(Predictor::Zero);
  tree[25] = PropertyDecisionNode::Leaf(Predictor::Zero);
  tree[26] = PropertyDecisionNode::Leaf(Predictor::Zero);
  tree[27] = PropertyDecisionNode::Leaf(Predictor::Zero);
  // VarDCTDC
  tree[28] = PropertyDecisionNode::Split(9, 0, 29);
  tree[29] = PropertyDecisionNode::Split(9, 47, 31);
  tree[30] = PropertyDecisionNode::Split(9, -31, 33);
  tree[31] = PropertyDecisionNode::Split(9, 191, 35);
  tree[32] = PropertyDecisionNode::Split(9, 11, 37);
  tree[33] = PropertyDecisionNode::Split(9, -7, 39);
  tree[34] = PropertyDecisionNode::Split(9, -127, 41);
  tree[35] = PropertyDecisionNode::Split(9, 392, 43);
  tree[36] = PropertyDecisionNode::Split(9, 95, 45);
  tree[37] = PropertyDecisionNode::Split(9, 23, 47);
  tree[38] = PropertyDecisionNode::Split(9, 5, 49);
  tree[39] = PropertyDecisionNode::Split(9, -3, 51);
  tree[40] = PropertyDecisionNode::Split(9, -15, 53);
  tree[41] = PropertyDecisionNode::Split(9, -63, 55);
  tree[42] = PropertyDecisionNode::Split(9, -255, 57);
  tree[43] = PropertyDecisionNode::Split(9, 500, 59);
  tree[44] = PropertyDecisionNode::Split(9, 255, 61);
  tree[45] = PropertyDecisionNode::Split(9, 127, 63);
  tree[46] = PropertyDecisionNode::Split(9, 63, 65);
  tree[47] = PropertyDecisionNode::Split(9, 31, 67);
  tree[48] = PropertyDecisionNode::Split(9, 15, 69);
  tree[49] = PropertyDecisionNode::Split(9, 7, 71);
  tree[50] = PropertyDecisionNode::Split(9, 3, 73);
  tree[51] = PropertyDecisionNode::Split(9, -1, 75);
  tree[52] = PropertyDecisionNode::Split(9, -4, 77);
  tree[53] = PropertyDecisionNode::Split(9, -11, 79);
  tree[54] = PropertyDecisionNode::Split(9, -23, 81);
  tree[55] = PropertyDecisionNode::Split(9, -47, 83);
  tree[56] = PropertyDecisionNode::Split(9, -95, 85);
  tree[57] = PropertyDecisionNode::Split(9, -191, 87);
  tree[58] = PropertyDecisionNode::Split(9, -392, 89);
  tree[59] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[60] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[61] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[62] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[63] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[64] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[65] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[66] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[67] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[68] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[69] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[70] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[71] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[72] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[73] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[74] = PropertyDecisionNode::Split(9, 1, 91);
  tree[75] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[76] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[77] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[78] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[79] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[80] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[81] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[82] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[83] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[84] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[85] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[86] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[87] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[88] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[89] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[90] = PropertyDecisionNode::Split(9, -500, 93);
  tree[91] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[92] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[93] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  tree[94] = PropertyDecisionNode::Leaf(Predictor::Gradient);
  return tree;
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
    size_t num_streams = ModularStreamId::Num(frame_dim_);
    stream_images_.resize(num_streams);
    ac_metadata_size.resize(frame_dim_.num_dc_groups);
  }
  Status ComputeEncodingData(ThreadPool* pool) {
    size_t num_streams = stream_images_.size();
    tokens_.resize(num_streams);
    tree_ = MakeFixedTree(frame_dim_.num_dc_groups);
    Tree decoded_tree;
    TokenizeTree(tree_, &tree_tokens_, &decoded_tree);
    tree_ = std::move(decoded_tree);
    image_widths_.resize(num_streams);
    JXL_RETURN_IF_ERROR(RunOnPool(
        pool, 0, num_streams, ThreadPool::NoInit,
        [&](const uint32_t stream_id, size_t /* thread */) {
          tokens_[stream_id].clear();
          JXL_CHECK(ModularGenericCompress(stream_images_[stream_id], stream_id,
                                           tree_, &tokens_[stream_id],
                                           &image_widths_[stream_id]));
        },
        "ComputeTokens"));
    return true;
  }
  // Encodes global info (tree + histograms) in the `writer`.
  Status EncodeGlobalInfo(BitWriter* writer) {
    BitWriter::Allotment allotment(writer, 1);
    writer->Write(1, 1);  // not an empty tree
    allotment.Reclaim(writer);

    // Write tree
    HistogramParams params;
    params.clustering = HistogramParams::ClusteringType::kFast;
    params.lz77_method = HistogramParams::LZ77Method::kNone;
    WriteHistogramsAndTokens(params, kNumTreeContexts, tree_tokens_, writer);
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
    BitWriter::Allotment allotment(writer, 1024);
    writer->Write(1, 1);  // use global tree
    writer->Write(1, 1);  // all default wp header
    writer->Write(2, 0);  // no transforms
    allotment.Reclaim(writer);
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

    size_t stream_id = ModularStreamId::VarDCTDC(group_index).ID(frame_dim_);

    stream_images_[stream_id] = Image(r.xsize(), r.ysize(), 3);
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
    // YToX, YToB, ACS + QF, EPF
    Image& image = stream_images_[stream_id];
    image = Image(r.xsize(), r.ysize(), 4);
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

 private:
  std::vector<Image> stream_images_;

  Tree tree_;
  std::vector<Token> tree_tokens_;
  std::vector<std::vector<Token>> tokens_;
  EntropyEncodingData code_;
  std::vector<uint8_t> context_map_;
  FrameDimensions frame_dim_;
  std::vector<size_t> image_widths_;
};

void WriteFrameHeader(uint32_t x_qm_scale, uint32_t epf_iters,
                      BitWriter* writer) {
  BitWriter::Allotment allotment(writer, 1024);
  writer->Write(1, 0);    // not all default
  writer->Write(2, 0);    // regular frame
  writer->Write(1, 0);    // vardct
  writer->Write(2, 2);    // flags selector bits (17 .. 272)
  writer->Write(8, 111);  // skip adaptive dc flag (128)
  writer->Write(2, 0);    // no upsampling
  writer->Write(3, x_qm_scale);
  writer->Write(3, 2);  // b_qm_scale
  writer->Write(2, 0);  // one pass
  writer->Write(1, 0);  // no custom frame size or origin
  writer->Write(2, 0);  // replace blend mode
  writer->Write(1, 1);  // last frame
  writer->Write(2, 0);  // no name
  if (epf_iters == 2) {
    writer->Write(1, 1);  // default loop filter
  } else {
    writer->Write(1, 0);  // not default loop filter
    writer->Write(1, 1);  // gaborish on
    writer->Write(1, 0);  // default gaborish
    writer->Write(2, epf_iters);
    if (epf_iters > 0) {
      writer->Write(1, 0);  // default epf sharpness
      writer->Write(1, 0);  // default epf weights
      writer->Write(1, 0);  // default epf sigma
    }
    writer->Write(2, 0);  // no loop filter extensions
  }
  writer->Write(2, 0);  // no frame header extensions
  allotment.Reclaim(writer);
}

}  // namespace

Status EncodeFrame(const float distance, const Image3F& linear,
                   ThreadPool* pool, BitWriter* writer) {
  PassesEncoderState enc_state;
  PassesSharedState& shared = enc_state.shared;
  FrameDimensions frame_dim;
  frame_dim.Set(linear.xsize(), linear.ysize());
  shared.frame_dim = frame_dim;

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

  std::unique_ptr<ModularFrameEncoder> modular_frame_encoder =
      jxl::make_unique<ModularFrameEncoder>(frame_dim);

  float x_qm_scale_steps[2] = {1.25f, 9.0f};
  uint32_t x_qm_scale = 2;
  for (float x_qm_scale_step : x_qm_scale_steps) {
    if (distance > x_qm_scale_step) {
      x_qm_scale++;
    }
  }
  if (distance < 0.299f) {
    // Favor chromacity preservation for making images appear more
    // faithful to original even with extreme (5-10x) zooming.
    x_qm_scale++;
  }
  constexpr float kEpfThresholds[3] = {0.7, 1.5, 4.0};
  uint32_t epf_iters = 0;
  for (size_t i = 0; i < 3; i++) {
    if (distance >= kEpfThresholds[i]) {
      epf_iters++;
    }
  }

  Image3F opsin(frame_dim.xsize_padded, frame_dim.ysize_padded);
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
  const float quant_dc = InitialQuantDC(distance);
  enc_state.initial_quant_field =
      InitialQuantField(distance, opsin, shared.frame_dim, pool, 1.0f,
                        &enc_state.initial_quant_masking);
  Quantizer& quantizer = shared.quantizer;
  quantizer.SetQuantField(quant_dc, enc_state.initial_quant_field, nullptr);

  // Apply inverse-gaborish.
  GaborishInverse(&opsin, 0.9908511000000001f, pool);

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

  enc_state.x_qm_multiplier = std::pow(1.25f, x_qm_scale - 2.0f);
  enc_state.b_qm_multiplier = 1.0;

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
  enc_state.passes[0].ac_tokens.resize(shared.frame_dim.num_groups);

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
    JXL_ASSERT(enc_state.coeffs[0]->Type() == ACType::k32);
    const int32_t* JXL_RESTRICT ac_rows[3] = {
        enc_state.coeffs[0]->PlaneRow(0, group_index, 0).ptr32,
        enc_state.coeffs[0]->PlaneRow(1, group_index, 0).ptr32,
        enc_state.coeffs[0]->PlaneRow(2, group_index, 0).ptr32,
    };
    // Ensure group cache is initialized.
    group_caches[thread].InitOnce();
    TokenizeCoefficients(&shared.coeff_orders[0], rect, ac_rows,
                         shared.ac_strategy, &group_caches[thread].num_nzeroes,
                         &enc_state.passes[0].ac_tokens[group_index]);
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, shared.frame_dim.num_groups,
                                tokenize_group_init, tokenize_group,
                                "TokenizeGroup"));

  // needs to happen *AFTER* VarDCT-ComputeEncodingData.
  JXL_RETURN_IF_ERROR(modular_frame_encoder->ComputeEncodingData(pool));

  WriteFrameHeader(x_qm_scale, epf_iters, writer);

  // DC global info + DC groups + AC global info + AC groups
  size_t num_toc_entries = 2 + frame_dim.num_dc_groups + frame_dim.num_groups;
  std::vector<BitWriter> group_codes(num_toc_entries);
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
      output->Write(2, 0);  // extra_dc_precision
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

    BitWriter::Allotment allotment(writer, 1024);
    writer->Write(2, 3);
    writer->Write(kNumOrders, enc_state.used_orders[0]);
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
