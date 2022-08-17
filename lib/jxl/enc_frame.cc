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
#include "lib/jxl/enc_adaptive_quantization.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_chroma_from_luma.h"
#include "lib/jxl/enc_coeff_order.h"
#include "lib/jxl/enc_entropy_coder.h"
#include "lib/jxl/enc_group.h"
#include "lib/jxl/enc_modular.h"
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
                            frame_header->encoding == FrameEncoding::kVarDCT &&
                            cparams.decoding_speed_tier < 4);

  if (cparams.epf != -1) {
    loop_filter->epf_iters = cparams.epf;
  } else {
    if (frame_header->encoding == FrameEncoding::kModular) {
      loop_filter->epf_iters = 0;
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
  }
  // Strength of EPF in modular mode.
  if (frame_header->encoding == FrameEncoding::kModular &&
      !cparams.IsLossless()) {
    // TODO(veluca): this formula is nonsense.
    loop_filter->epf_sigma_for_modular = cparams.butteraugli_distance;
  }
  if (frame_header->encoding == FrameEncoding::kModular &&
      cparams.lossy_palette) {
    loop_filter->epf_sigma_for_modular = 1.0f;
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

  if (cparams.modular_mode) {
    frame_header->encoding = FrameEncoding::kModular;
    frame_header->group_size_shift = cparams.modular_group_size_shift;
  }

  frame_header->chroma_subsampling = ib.chroma_subsampling;
  if (ib.IsJPEG()) {
    // we are transcoding a JPEG, so we don't get to choose
    frame_header->encoding = FrameEncoding::kVarDCT;
    frame_header->color_transform = ib.color_transform;
  } else {
    frame_header->color_transform = cparams.color_transform;
    if (!cparams.modular_mode &&
        (frame_header->chroma_subsampling.MaxHShift() != 0 ||
         frame_header->chroma_subsampling.MaxVShift() != 0)) {
      return JXL_FAILURE(
          "Chroma subsampling is not supported in VarDCT mode when not "
          "recompressing JPEGs");
    }
  }

  frame_header->flags = 0;

  JXL_RETURN_IF_ERROR(LoopFilterFromParams(cparams, frame_header));

  frame_header->dc_level = frame_info.dc_level;
  if (frame_header->dc_level > 0) {
    return JXL_FAILURE("progressive dc is not supported");
  }
  // Resized frames.
  if (frame_info.frame_type != FrameType::kDCFrame) {
    frame_header->frame_origin = ib.origin;
    frame_header->frame_size.xsize = ib.xsize();
    frame_header->frame_size.ysize = ib.ysize();
    if (ib.origin.x0 != 0 || ib.origin.y0 != 0 ||
        frame_header->frame_size.xsize != frame_header->default_xsize() ||
        frame_header->frame_size.ysize != frame_header->default_ysize()) {
      frame_header->custom_size_or_origin = true;
    }
  }
  // Upsampling.
  frame_header->upsampling = 1;
  const std::vector<ExtraChannelInfo>& extra_channels =
      frame_header->nonserialized_metadata->m.extra_channel_info;
  frame_header->extra_channel_upsampling.clear();
  frame_header->extra_channel_upsampling.resize(extra_channels.size(), 1);
  frame_header->save_as_reference = frame_info.save_as_reference;

  // Set blending-related information.
  if (ib.blend || frame_header->custom_size_or_origin) {
    // Set blend_channel to the first alpha channel. These values are only
    // encoded in case a blend mode involving alpha is used and there are more
    // than one extra channels.
    size_t index = 0;
    if (frame_info.alpha_channel == -1) {
      if (extra_channels.size() > 1) {
        for (size_t i = 0; i < extra_channels.size(); i++) {
          if (extra_channels[i].type == ExtraChannel::kAlpha) {
            index = i;
            break;
          }
        }
      }
    } else {
      index = static_cast<size_t>(frame_info.alpha_channel);
      JXL_ASSERT(index == 0 || index < extra_channels.size());
    }
    frame_header->blending_info.alpha_channel = index;
    frame_header->blending_info.mode =
        ib.blend ? ib.blendmode : BlendMode::kReplace;
    frame_header->blending_info.source = frame_info.source;
    frame_header->blending_info.clamp = frame_info.clamp;
    const auto& extra_channel_info = frame_info.extra_channel_blending_info;
    for (size_t i = 0; i < extra_channels.size(); i++) {
      if (i < extra_channel_info.size()) {
        frame_header->extra_channel_blending_info[i] = extra_channel_info[i];
      } else {
        frame_header->extra_channel_blending_info[i].alpha_channel = index;
        BlendMode default_blend = ib.blendmode;
        if (extra_channels[i].type != ExtraChannel::kBlack && i != index) {
          // K needs to be blended, spot colors and other stuff gets added
          default_blend = BlendMode::kAdd;
        }
        frame_header->extra_channel_blending_info[i].mode =
            ib.blend ? default_blend : BlendMode::kReplace;
        frame_header->extra_channel_blending_info[i].source = 1;
      }
    }
  }

  frame_header->animation_frame.duration = ib.duration;
  frame_header->animation_frame.timecode = ib.timecode;

  return true;
}

// Invisible (alpha = 0) pixels tend to be a mess in optimized PNGs.
// Since they have no visual impact whatsoever, we can replace them with
// something that compresses better and reduces artifacts near the edges. This
// does some kind of smooth stuff that seems to work.
// Replace invisible pixels with a weighted average of the pixel to the left,
// the pixel to the topright, and non-invisible neighbours.
// Produces downward-blurry smears, with in the upwards direction only a 1px
// edge duplication but not more. It would probably be better to smear in all
// directions. That requires an alpha-weighed convolution with a large enough
// kernel though, which might be overkill...
void SimplifyInvisible(Image3F* image, const ImageF& alpha, bool lossless) {
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < image->ysize(); ++y) {
      float* JXL_RESTRICT row = image->PlaneRow(c, y);
      const float* JXL_RESTRICT prow =
          (y > 0 ? image->PlaneRow(c, y - 1) : nullptr);
      const float* JXL_RESTRICT nrow =
          (y + 1 < image->ysize() ? image->PlaneRow(c, y + 1) : nullptr);
      const float* JXL_RESTRICT a = alpha.Row(y);
      const float* JXL_RESTRICT pa = (y > 0 ? alpha.Row(y - 1) : nullptr);
      const float* JXL_RESTRICT na =
          (y + 1 < image->ysize() ? alpha.Row(y + 1) : nullptr);
      for (size_t x = 0; x < image->xsize(); ++x) {
        if (a[x] == 0) {
          if (lossless) {
            row[x] = 0;
            continue;
          }
          float d = 0.f;
          row[x] = 0;
          if (x > 0) {
            row[x] += row[x - 1];
            d++;
            if (a[x - 1] > 0.f) {
              row[x] += row[x - 1];
              d++;
            }
          }
          if (x + 1 < image->xsize()) {
            if (y > 0) {
              row[x] += prow[x + 1];
              d++;
            }
            if (a[x + 1] > 0.f) {
              row[x] += 2.f * row[x + 1];
              d += 2.f;
            }
            if (y > 0 && pa[x + 1] > 0.f) {
              row[x] += 2.f * prow[x + 1];
              d += 2.f;
            }
            if (y + 1 < image->ysize() && na[x + 1] > 0.f) {
              row[x] += 2.f * nrow[x + 1];
              d += 2.f;
            }
          }
          if (y > 0 && pa[x] > 0.f) {
            row[x] += 2.f * prow[x];
            d += 2.f;
          }
          if (y + 1 < image->ysize() && na[x] > 0.f) {
            row[x] += 2.f * nrow[x];
            d += 2.f;
          }
          if (d > 1.f) row[x] /= d;
        }
      }
    }
  }
}

}  // namespace

class LossyFrameEncoder {
 public:
  LossyFrameEncoder(const CompressParams& cparams,
                    const FrameHeader& frame_header,
                    PassesEncoderState* JXL_RESTRICT enc_state,
                    const JxlCmsInterface& cms, ThreadPool* pool,
                    AuxOut* aux_out)
      : enc_state_(enc_state), cms_(cms), pool_(pool), aux_out_(aux_out) {
    JXL_CHECK(InitializePassesSharedState(frame_header, &enc_state_->shared,
                                          /*encoder=*/true));
    enc_state_->cparams = cparams;
    enc_state_->passes.clear();
  }

  Status ComputeEncodingData(const ImageBundle* linear,
                             Image3F* JXL_RESTRICT opsin,
                             const JxlCmsInterface& cms, ThreadPool* pool,
                             ModularFrameEncoder* modular_frame_encoder,
                             FrameHeader* frame_header) {
    PROFILER_ZONE("ComputeEncodingData uninstrumented");
    JXL_ASSERT((opsin->xsize() % kBlockDim) == 0 &&
               (opsin->ysize() % kBlockDim) == 0);
    PassesSharedState& shared = enc_state_->shared;

    float x_qm_scale_steps[2] = {1.25f, 9.0f};
    shared.frame_header.x_qm_scale = 2;
    for (float x_qm_scale_step : x_qm_scale_steps) {
      if (enc_state_->cparams.butteraugli_distance > x_qm_scale_step) {
        shared.frame_header.x_qm_scale++;
      }
    }
    if (enc_state_->cparams.butteraugli_distance < 0.299f) {
      // Favor chromacity preservation for making images appear more
      // faithful to original even with extreme (5-10x) zooming.
      shared.frame_header.x_qm_scale++;
    }

    JXL_RETURN_IF_ERROR(enc_state_->heuristics->LossyFrameHeuristics(
        enc_state_, modular_frame_encoder, linear, opsin, cms_, pool_,
        aux_out_));

    JXL_RETURN_IF_ERROR(InitializePassesEncoder(
        *opsin, cms, pool_, enc_state_, modular_frame_encoder, aux_out_));
    enc_state_->passes.resize(1);
    for (PassesEncoderState::PassData& pass : enc_state_->passes) {
      pass.ac_tokens.resize(shared.frame_dim.num_groups);
    }

    ComputeAllCoeffOrders(shared.frame_dim);
    shared.num_histograms = 1;

    const auto tokenize_group_init = [&](const size_t num_threads) {
      group_caches_.resize(num_threads);
      return true;
    };
    const auto tokenize_group = [&](const uint32_t group_index,
                                    const size_t thread) {
      // Tokenize coefficients.
      const Rect rect = shared.BlockGroupRect(group_index);
      for (size_t idx_pass = 0; idx_pass < enc_state_->passes.size();
           idx_pass++) {
        JXL_ASSERT(enc_state_->coeffs[idx_pass]->Type() == ACType::k32);
        const int32_t* JXL_RESTRICT ac_rows[3] = {
            enc_state_->coeffs[idx_pass]->PlaneRow(0, group_index, 0).ptr32,
            enc_state_->coeffs[idx_pass]->PlaneRow(1, group_index, 0).ptr32,
            enc_state_->coeffs[idx_pass]->PlaneRow(2, group_index, 0).ptr32,
        };
        // Ensure group cache is initialized.
        group_caches_[thread].InitOnce();
        TokenizeCoefficients(
            &shared.coeff_orders[idx_pass * shared.coeff_order_size], rect,
            ac_rows, shared.ac_strategy, frame_header->chroma_subsampling,
            &group_caches_[thread].num_nzeroes,
            &enc_state_->passes[idx_pass].ac_tokens[group_index]);
      }
    };
    JXL_RETURN_IF_ERROR(RunOnPool(pool_, 0, shared.frame_dim.num_groups,
                                  tokenize_group_init, tokenize_group,
                                  "TokenizeGroup"));

    *frame_header = shared.frame_header;
    return true;
  }

  Status EncodeGlobalDCInfo(const FrameHeader& frame_header,
                            BitWriter* writer) const {
    // Encode quantizer DC and global scale.
    JXL_RETURN_IF_ERROR(
        enc_state_->shared.quantizer.Encode(writer, kLayerQuant, aux_out_));
    writer->Write(1, 1);  // default BlockCtxMap
    ColorCorrelationMapEncodeDC(&enc_state_->shared.cmap, writer, kLayerDC,
                                aux_out_);
    return true;
  }

  Status EncodeGlobalACInfo(BitWriter* writer,
                            ModularFrameEncoder* modular_frame_encoder) {
    {
      BitWriter::Allotment allotment(writer, 1024);
      writer->Write(1, 1);  // all default quant matrices
      size_t num_histo_bits =
          CeilLog2Nonzero(enc_state_->shared.frame_dim.num_groups);
      if (num_histo_bits != 0) {
        writer->Write(num_histo_bits, enc_state_->shared.num_histograms - 1);
      }
      ReclaimAndCharge(writer, &allotment, kLayerAC, aux_out_);
    }

    // Encode coefficient orders.
    size_t order_bits = 0;
    JXL_RETURN_IF_ERROR(U32Coder::CanEncode(
        kOrderEnc, enc_state_->used_orders[0], &order_bits));
    BitWriter::Allotment allotment(writer, order_bits);
    JXL_CHECK(U32Coder::Write(kOrderEnc, enc_state_->used_orders[0], writer));
    ReclaimAndCharge(writer, &allotment, kLayerOrder, aux_out_);
    EncodeCoeffOrders(enc_state_->used_orders[0],
                      &enc_state_->shared.coeff_orders[0], writer, kLayerOrder,
                      aux_out_);

    // Encode histograms.
    HistogramParams hist_params(
        enc_state_->cparams.speed_tier,
        enc_state_->shared.block_ctx_map.NumACContexts());
    hist_params.lz77_method = HistogramParams::LZ77Method::kNone;
    if (enc_state_->cparams.decoding_speed_tier >= 1) {
      hist_params.max_histograms = 6;
    }
    BuildAndEncodeHistograms(
        hist_params,
        enc_state_->shared.num_histograms *
            enc_state_->shared.block_ctx_map.NumACContexts(),
        enc_state_->passes[0].ac_tokens, &enc_state_->passes[0].codes,
        &enc_state_->passes[0].context_map, writer, kLayerAC, aux_out_);

    return true;
  }

  Status EncodeACGroup(size_t pass, size_t group_index, BitWriter* group_code,
                       AuxOut* local_aux_out) {
    return EncodeGroupTokenizedCoefficients(
        group_index, pass, enc_state_->histogram_idx[group_index], *enc_state_,
        group_code, local_aux_out);
  }

  PassesEncoderState* State() { return enc_state_; }

 private:
  void ComputeAllCoeffOrders(const FrameDimensions& frame_dim) {
    PROFILER_FUNC;
    // No coefficient reordering in Falcon or faster.
    auto used_orders_info = ComputeUsedOrders(
        enc_state_->cparams.speed_tier, enc_state_->shared.ac_strategy,
        Rect(enc_state_->shared.raw_quant_field));
    enc_state_->used_orders.clear();
    enc_state_->used_orders.resize(1, used_orders_info.second);
    ComputeCoeffOrder(enc_state_->cparams.speed_tier, *enc_state_->coeffs[0],
                      enc_state_->shared.ac_strategy, frame_dim,
                      enc_state_->used_orders[0], used_orders_info.first,
                      &enc_state_->shared.coeff_orders[0]);
  }

  template <typename V, typename R>
  static inline void FindIndexOfSumMaximum(const V* array, const size_t len,
                                           R* idx, V* sum) {
    JXL_ASSERT(len > 0);
    V maxval = 0;
    V val = 0;
    R maxidx = 0;
    for (size_t i = 0; i < len; ++i) {
      val += array[i];
      if (val > maxval) {
        maxval = val;
        maxidx = i;
      }
    }
    *idx = maxidx;
    *sum = maxval;
  }

  PassesEncoderState* JXL_RESTRICT enc_state_;
  JxlCmsInterface cms_;
  ThreadPool* pool_;
  AuxOut* aux_out_;
  std::vector<EncCache> group_caches_;
};

Status EncodeFrame(const CompressParams& cparams, const FrameInfo& frame_info,
                   const CodecMetadata* metadata, const ImageBundle& ib,
                   PassesEncoderState* passes_enc_state,
                   const JxlCmsInterface& cms, ThreadPool* pool,
                   BitWriter* writer, AuxOut* aux_out) {
  ib.VerifyMetadata();

  passes_enc_state->special_frames.clear();

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

  // Assert that this metadata is correctly set up for the compression params,
  // this should have been done by enc_file.cc
  JXL_ASSERT(metadata->m.xyb_encoded ==
             (cparams.color_transform == ColorTransform::kXYB));
  std::unique_ptr<FrameHeader> frame_header =
      jxl::make_unique<FrameHeader>(metadata);
  JXL_RETURN_IF_ERROR(MakeFrameHeader(cparams,
                                      frame_info, ib, frame_header.get()));
  // Check that if the codestream header says xyb_encoded, the color_transform
  // matches the requirement. This is checked from the cparams here, even though
  // optimally we'd be able to check this against what has actually been written
  // in the main codestream header, but since ib is a const object and the data
  // written to the main codestream header is (in modified form) in ib, the
  // encoder cannot indicate this fact in the ib's metadata.
  if (cparams.color_transform == ColorTransform::kXYB) {
    if (frame_header->color_transform != ColorTransform::kXYB) {
      return JXL_FAILURE(
          "The color transform of frames must be xyb if the codestream is xyb "
          "encoded");
    }
  } else {
    if (frame_header->color_transform == ColorTransform::kXYB) {
      return JXL_FAILURE(
          "The color transform of frames cannot be xyb if the codestream is "
          "not xyb encoded");
    }
  }

  FrameDimensions frame_dim = frame_header->ToFrameDimensions();

  const size_t num_groups = frame_dim.num_groups;

  Image3F opsin;
  const ColorEncoding& c_linear = ColorEncoding::LinearSRGB(ib.IsGray());
  std::unique_ptr<ImageMetadata> metadata_linear =
      jxl::make_unique<ImageMetadata>();
  metadata_linear->xyb_encoded =
      (cparams.color_transform == ColorTransform::kXYB);
  metadata_linear->color_encoding = c_linear;
  ImageBundle linear_storage(metadata_linear.get());

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

  LossyFrameEncoder lossy_frame_encoder(cparams, *frame_header,
                                        passes_enc_state, cms, pool, aux_out);
  std::unique_ptr<ModularFrameEncoder> modular_frame_encoder =
      jxl::make_unique<ModularFrameEncoder>(*frame_header, cparams);

  const std::vector<ImageF>* extra_channels = &ib.extra_channels();
  std::vector<ImageF> extra_channels_storage;
  // Clear patches
  passes_enc_state->shared.image_features.patches = PatchDictionary();
  passes_enc_state->shared.image_features.patches.SetPassesSharedState(
      &passes_enc_state->shared);

  if (ib.IsJPEG()) {
    return JXL_FAILURE("JPEG transcoding not implemented");
  } else if (!lossy_frame_encoder.State()->heuristics->HandlesColorConversion(
                 cparams, ib) ||
             frame_header->encoding != FrameEncoding::kVarDCT) {
    // Allocating a large enough image avoids a copy when padding.
    opsin =
        Image3F(RoundUpToBlockDim(ib.xsize()), RoundUpToBlockDim(ib.ysize()));
    opsin.ShrinkTo(ib.xsize(), ib.ysize());

    const ImageBundle* JXL_RESTRICT ib_or_linear = &ib;

    if (frame_header->color_transform == ColorTransform::kXYB &&
        frame_info.ib_needs_color_transform) {
      // linear_storage would only be used by the Butteraugli loop (passing
      // linear sRGB avoids a color conversion there). Otherwise, don't
      // fill it to reduce memory usage.
      ib_or_linear = ToXYB(ib, pool, &opsin, cms, nullptr);
    } else {  // RGB or YCbCr: don't do anything (forward YCbCr is not
              // implemented, this is only used when the input is already in
              // YCbCr)
              // If encoding a special DC or reference frame, don't do anything:
              // input is already in XYB.
      CopyImageTo(ib.color(), &opsin);
    }
    bool lossless = cparams.IsLossless();
    if (ib.HasAlpha() && !ib.AlphaIsPremultiplied() &&
        frame_header->frame_type == FrameType::kRegularFrame &&
        !ApplyOverride(cparams.keep_invisible, lossless)) {
      // simplify invisible pixels
      SimplifyInvisible(&opsin, ib.alpha(), lossless);
    }
    if (aux_out != nullptr) {
      JXL_RETURN_IF_ERROR(
          aux_out->InspectImage3F("enc_frame:OpsinDynamicsImage", opsin));
    }
    if (frame_header->encoding == FrameEncoding::kVarDCT) {
      PadImageToBlockMultipleInPlace(&opsin);
      JXL_RETURN_IF_ERROR(lossy_frame_encoder.ComputeEncodingData(
          ib_or_linear, &opsin, cms, pool, modular_frame_encoder.get(),
          frame_header.get()));
    }
  } else {
    JXL_RETURN_IF_ERROR(lossy_frame_encoder.ComputeEncodingData(
        &ib, &opsin, cms, pool, modular_frame_encoder.get(),
        frame_header.get()));
  }
  // needs to happen *AFTER* VarDCT-ComputeEncodingData.
  JXL_RETURN_IF_ERROR(modular_frame_encoder->ComputeEncodingData(
      *frame_header, *ib.metadata(), &opsin, *extra_channels,
      lossy_frame_encoder.State(), cms, pool, aux_out));

  writer->AppendByteAligned(lossy_frame_encoder.State()->special_frames);
  frame_header->UpdateFlag(
      lossy_frame_encoder.State()->shared.image_features.splines.HasAny(),
      FrameHeader::kSplines);
  JXL_RETURN_IF_ERROR(WriteFrameHeader(*frame_header, writer, aux_out));

  const size_t num_passes = 1;

  // DC global info + DC groups + AC global info + AC groups *
  // num_passes.
  const bool has_ac_global = true;
  std::vector<BitWriter> group_codes(NumTocEntries(frame_dim.num_groups,
                                                   frame_dim.num_dc_groups,
                                                   num_passes, has_ac_global));
  const size_t global_ac_index = frame_dim.num_dc_groups + 1;
  const bool is_small_image = frame_dim.num_groups == 1 && num_passes == 1;
  const auto get_output = [&](const size_t index) {
    return &group_codes[is_small_image ? 0 : index];
  };
  auto ac_group_code = [&](size_t pass, size_t group) {
    return get_output(AcGroupIndex(pass, group, frame_dim.num_groups,
                                   frame_dim.num_dc_groups, has_ac_global));
  };

  {
    BitWriter* writer = get_output(0);
    BitWriter::Allotment allotment(writer, 2);
    writer->Write(1, 1);  // default quant dc
    ReclaimAndCharge(writer, &allotment, kLayerQuant, aux_out);
  }
  if (frame_header->encoding == FrameEncoding::kVarDCT) {
    JXL_RETURN_IF_ERROR(
        lossy_frame_encoder.EncodeGlobalDCInfo(*frame_header, get_output(0)));
  }
  JXL_RETURN_IF_ERROR(
      modular_frame_encoder->EncodeGlobalInfo(get_output(0), aux_out));
  JXL_RETURN_IF_ERROR(modular_frame_encoder->EncodeStream(
      get_output(0), aux_out, kLayerModularGlobal, ModularStreamId::Global()));

  const auto process_dc_group = [&](const uint32_t group_index,
                                    const size_t thread) {
    AuxOut* my_aux_out = aux_out ? &aux_outs[thread] : nullptr;
    BitWriter* output = get_output(group_index + 1);
    if (frame_header->encoding == FrameEncoding::kVarDCT &&
        !(frame_header->flags & FrameHeader::kUseDcFrame)) {
      BitWriter::Allotment allotment(output, 2);
      output->Write(2, modular_frame_encoder->extra_dc_precision[group_index]);
      ReclaimAndCharge(output, &allotment, kLayerDC, my_aux_out);
      JXL_CHECK(modular_frame_encoder->EncodeStream(
          output, my_aux_out, kLayerDC,
          ModularStreamId::VarDCTDC(group_index)));
    }
    JXL_CHECK(modular_frame_encoder->EncodeStream(
        output, my_aux_out, kLayerModularDcGroup,
        ModularStreamId::ModularDC(group_index)));
    if (frame_header->encoding == FrameEncoding::kVarDCT) {
      const Rect& rect =
          lossy_frame_encoder.State()->shared.DCGroupRect(group_index);
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

  if (frame_header->encoding == FrameEncoding::kVarDCT) {
    JXL_RETURN_IF_ERROR(lossy_frame_encoder.EncodeGlobalACInfo(
        get_output(global_ac_index), modular_frame_encoder.get()));
  }

  std::atomic<int> num_errors{0};
  const auto process_group = [&](const uint32_t group_index,
                                 const size_t thread) {
    AuxOut* my_aux_out = aux_out ? &aux_outs[thread] : nullptr;

    for (size_t i = 0; i < num_passes; i++) {
      if (frame_header->encoding == FrameEncoding::kVarDCT) {
        if (!lossy_frame_encoder.EncodeACGroup(
                i, group_index, ac_group_code(i, group_index), my_aux_out)) {
          num_errors.fetch_add(1, std::memory_order_relaxed);
          return;
        }
      }
      // Write all modular encoded data (color?, alpha, depth, extra channels)
      if (!modular_frame_encoder->EncodeStream(
              ac_group_code(i, group_index), my_aux_out, kLayerModularAcGroup,
              ModularStreamId::ModularAC(group_index, i))) {
        num_errors.fetch_add(1, std::memory_order_relaxed);
        return;
      }
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
  if (cparams.centerfirst && !(num_passes == 1 && num_groups == 1)) {
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
      Rect r = passes_enc_state->shared.GroupRect(gid);
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
    for (size_t i = 0; i < num_passes; i++) {
      size_t pass_start = permutation.size();
      for (coeff_order_t v : inv_ac_group_order) {
        permutation.push_back(pass_start + v);
      }
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
