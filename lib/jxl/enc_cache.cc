// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "lib/jxl/enc_cache.h"

#include <stddef.h>
#include <stdint.h>

#include <type_traits>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/aux_out.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/profiler.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/common.h"
#include "lib/jxl/compressed_dc.h"
#include "lib/jxl/dct_scales.h"
#include "lib/jxl/dct_util.h"
#include "lib/jxl/dec_frame.h"
#include "lib/jxl/enc_frame.h"
#include "lib/jxl/enc_group.h"
#include "lib/jxl/enc_modular.h"
#include "lib/jxl/enc_quant_weights.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/passes_state.h"
#include "lib/jxl/quantizer.h"

namespace jxl {

Status InitializePassesEncoder(const Image3F& opsin, const JxlCmsInterface& cms,
                               ThreadPool* pool, PassesEncoderState* enc_state,
                               ModularFrameEncoder* modular_frame_encoder,
                               AuxOut* aux_out) {
  PROFILER_FUNC;

  PassesSharedState& JXL_RESTRICT shared = enc_state->shared;

  enc_state->histogram_idx.resize(shared.frame_dim.num_groups);

  enc_state->x_qm_multiplier =
      std::pow(1.25f, shared.frame_header.x_qm_scale - 2.0f);
  enc_state->b_qm_multiplier =
      std::pow(1.25f, shared.frame_header.b_qm_scale - 2.0f);

  if (enc_state->coeffs.size() < shared.frame_header.passes.num_passes) {
    enc_state->coeffs.reserve(shared.frame_header.passes.num_passes);
    for (size_t i = enc_state->coeffs.size();
         i < shared.frame_header.passes.num_passes; i++) {
      // Allocate enough coefficients for each group on every row.
      enc_state->coeffs.emplace_back(make_unique<ACImageT<int32_t>>(
          kGroupDim * kGroupDim, shared.frame_dim.num_groups));
    }
  }
  while (enc_state->coeffs.size() > shared.frame_header.passes.num_passes) {
    enc_state->coeffs.pop_back();
  }

  Image3F dc(shared.frame_dim.xsize_blocks, shared.frame_dim.ysize_blocks);
  JXL_RETURN_IF_ERROR(RunOnPool(
      pool, 0, shared.frame_dim.num_groups, ThreadPool::NoInit,
      [&](size_t group_idx, size_t _) {
        ComputeCoefficients(group_idx, enc_state, opsin, &dc);
      },
      "Compute coeffs"));

  auto compute_dc_coeffs = [&](int group_index, int /* thread */) {
    modular_frame_encoder->AddVarDCTDC(
        dc, group_index,
        enc_state, /*jpeg_transcode=*/false);
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, shared.frame_dim.num_dc_groups,
                                ThreadPool::NoInit, compute_dc_coeffs,
                                "Compute DC coeffs"));
  // TODO(veluca): this is only useful in tests and if inspection is enabled.
  if (!(shared.frame_header.flags & FrameHeader::kSkipAdaptiveDCSmoothing)) {
    AdaptiveDCSmoothing(shared.quantizer.MulDC(), &shared.dc_storage, pool);
  }
  auto compute_ac_meta = [&](int group_index, int /* thread */) {
    modular_frame_encoder->AddACMetadata(group_index, /*jpeg_transcode=*/false,
                                         enc_state);
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, shared.frame_dim.num_dc_groups,
                                ThreadPool::NoInit, compute_ac_meta,
                                "Compute AC Metadata"));

  if (aux_out != nullptr) {
    aux_out->InspectImage3F("compressed_image:InitializeFrameEncCache:dc_dec",
                            shared.dc_storage);
  }
  return true;
}

void EncCache::InitOnce() {
  PROFILER_FUNC;

  if (num_nzeroes.xsize() == 0) {
    num_nzeroes = Image3I(kGroupDimInBlocks, kGroupDimInBlocks);
  }
}

}  // namespace jxl
