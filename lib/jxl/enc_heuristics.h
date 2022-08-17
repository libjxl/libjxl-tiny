// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef LIB_JXL_ENC_HEURISTICS_H_
#define LIB_JXL_ENC_HEURISTICS_H_

// Hook for custom encoder heuristics (VarDCT only for now).

#include <stddef.h>
#include <stdint.h>

#include <string>

#include "lib/jxl/aux_out_fwd.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/image.h"
#include "lib/jxl/modular/encoding/enc_ma.h"

namespace jxl {

struct PassesEncoderState;
class ImageBundle;
class ModularFrameEncoder;

class EncoderHeuristics {
 public:
  virtual ~EncoderHeuristics() = default;
  // Initializes encoder structures in `enc_state` using the original image data
  // in `original_pixels`, and the XYB image data in `opsin`. Also modifies the
  // `opsin` image by applying Gaborish, and doing other modifications if
  // necessary. `pool` is used for running the computations on multiple threads.
  // `aux_out` collects statistics and can be used to print debug images.
  virtual Status LossyFrameHeuristics(
      PassesEncoderState* enc_state, ModularFrameEncoder* modular_frame_encoder,
      const ImageBundle* original_pixels, Image3F* opsin,
      const JxlCmsInterface& cms, ThreadPool* pool, AuxOut* aux_out) = 0;
};

class DefaultEncoderHeuristics : public EncoderHeuristics {
 public:
  Status LossyFrameHeuristics(PassesEncoderState* enc_state,
                              ModularFrameEncoder* modular_frame_encoder,
                              const ImageBundle* original_pixels,
                              Image3F* opsin, const JxlCmsInterface& cms,
                              ThreadPool* pool, AuxOut* aux_out) override;
};

}  // namespace jxl

#endif  // LIB_JXL_ENC_HEURISTICS_H_
