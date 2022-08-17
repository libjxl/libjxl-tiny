// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "lib/jxl/enc_heuristics.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <numeric>
#include <string>

#include "lib/jxl/enc_ac_strategy.h"
#include "lib/jxl/enc_adaptive_quantization.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_chroma_from_luma.h"
#include "lib/jxl/enc_modular.h"
#include "lib/jxl/enc_xyb.h"
#include "lib/jxl/gaborish.h"

namespace jxl {

Status DefaultEncoderHeuristics::LossyFrameHeuristics(
    PassesEncoderState* enc_state, ModularFrameEncoder* modular_frame_encoder,
    const ImageBundle* original_pixels, Image3F* opsin,
    const JxlCmsInterface& cms, ThreadPool* pool, AuxOut* aux_out) {
  PROFILER_ZONE("JxlLossyFrameHeuristics uninstrumented");

  CompressParams& cparams = enc_state->cparams;
  PassesSharedState& shared = enc_state->shared;

  if (cparams.butteraugli_distance < 0) {
    return JXL_FAILURE("Expected non-negative distance");
  }

  static const float kAcQuant = 0.79f;
  const float quant_dc = InitialQuantDC(cparams.butteraugli_distance);
  Quantizer& quantizer = enc_state->shared.quantizer;
  // We don't know the quant field yet, but for computing the global scale
  // assuming that it will be the same as for Falcon mode is good enough.
  quantizer.ComputeGlobalScaleAndQuant(
      quant_dc, kAcQuant / cparams.butteraugli_distance, 0);

  // Dependency graph:
  //
  // input: either XYB or input image
  //
  // input image -> XYB [optional]
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

  AcStrategyHeuristics acs_heuristics;
  CfLHeuristics cfl_heuristics;

  *opsin = Image3F(RoundUpToBlockDim(original_pixels->xsize()),
                   RoundUpToBlockDim(original_pixels->ysize()));
  opsin->ShrinkTo(original_pixels->xsize(), original_pixels->ysize());
  ToXYB(*original_pixels, pool, opsin, cms, /*linear=*/nullptr);
  PadImageToBlockMultipleInPlace(opsin);

  // Compute an initial estimate of the quantization field.
  // Call InitialQuantField only in Hare mode or slower. Otherwise, rely
  // on simple heuristics in FindBestAcStrategy, or set a constant for Falcon
  // mode.
  if (cparams.speed_tier > SpeedTier::kHare) {
    enc_state->initial_quant_field =
        ImageF(shared.frame_dim.xsize_blocks, shared.frame_dim.ysize_blocks);
    float q = kAcQuant / cparams.butteraugli_distance;
    FillImage(q, &enc_state->initial_quant_field);
  } else {
    // Call this here, as it relies on pre-gaborish values.
    float butteraugli_distance_for_iqf = cparams.butteraugli_distance;
    if (!shared.frame_header.loop_filter.gab) {
      butteraugli_distance_for_iqf *= 0.73f;
    }
    enc_state->initial_quant_field = InitialQuantField(
        butteraugli_distance_for_iqf, *opsin, shared.frame_dim, pool, 1.0f,
        &enc_state->initial_quant_masking);
    quantizer.SetQuantField(quant_dc, enc_state->initial_quant_field, nullptr);
  }

  // TODO(veluca): do something about animations.

  // Apply inverse-gaborish.
  if (shared.frame_header.loop_filter.gab) {
    GaborishInverse(opsin, 0.9908511000000001f, pool);
  }

  FillPlane(static_cast<uint8_t>(4), &enc_state->shared.epf_sharpness);

  cfl_heuristics.Init(*opsin);
  acs_heuristics.Init(*opsin, enc_state);

  auto process_tile = [&](const uint32_t tid, const size_t thread) {
    size_t n_enc_tiles =
        DivCeil(enc_state->shared.frame_dim.xsize_blocks, kEncTileDimInBlocks);
    size_t tx = tid % n_enc_tiles;
    size_t ty = tid / n_enc_tiles;
    size_t by0 = ty * kEncTileDimInBlocks;
    size_t by1 = std::min((ty + 1) * kEncTileDimInBlocks,
                          enc_state->shared.frame_dim.ysize_blocks);
    size_t bx0 = tx * kEncTileDimInBlocks;
    size_t bx1 = std::min((tx + 1) * kEncTileDimInBlocks,
                          enc_state->shared.frame_dim.xsize_blocks);
    Rect r(bx0, by0, bx1 - bx0, by1 - by0);

    // For speeds up to Wombat, we only compute the color correlation map
    // once we know the transform type and the quantization map.
    if (cparams.speed_tier <= SpeedTier::kSquirrel) {
      cfl_heuristics.ComputeTile(r, *opsin, enc_state->shared.matrices,
                                 /*ac_strategy=*/nullptr,
                                 /*quantizer=*/nullptr, /*fast=*/false, thread,
                                 &enc_state->shared.cmap);
    }

    // Choose block sizes.
    acs_heuristics.ProcessRect(r);

    // Always set the initial quant field, so we can compute the CfL map with
    // more accuracy. The initial quant field might change in slower modes, but
    // adjusting the quant field with butteraugli when all the other encoding
    // parameters are fixed is likely a more reliable choice anyway.
    AdjustQuantField(enc_state->shared.ac_strategy, r,
                     &enc_state->initial_quant_field);
    quantizer.SetQuantFieldRect(enc_state->initial_quant_field, r,
                                &enc_state->shared.raw_quant_field);

    // Compute a non-default CfL map if we are at Hare speed, or slower.
    if (cparams.speed_tier <= SpeedTier::kHare) {
      cfl_heuristics.ComputeTile(
          r, *opsin, enc_state->shared.matrices, &enc_state->shared.ac_strategy,
          &enc_state->shared.quantizer,
          /*fast=*/cparams.speed_tier >= SpeedTier::kWombat, thread,
          &enc_state->shared.cmap);
    }
  };
  JXL_RETURN_IF_ERROR(RunOnPool(
      pool, 0,
      DivCeil(enc_state->shared.frame_dim.xsize_blocks, kEncTileDimInBlocks) *
          DivCeil(enc_state->shared.frame_dim.ysize_blocks,
                  kEncTileDimInBlocks),
      [&](const size_t num_threads) {
        cfl_heuristics.PrepareForThreads(num_threads);
        return true;
      },
      process_tile, "Enc Heuristics"));

  acs_heuristics.Finalize(aux_out);
  if (cparams.speed_tier <= SpeedTier::kHare) {
    cfl_heuristics.ComputeDC(/*fast=*/cparams.speed_tier >= SpeedTier::kWombat,
                             &enc_state->shared.cmap);
  }

  return true;
}

}  // namespace jxl
