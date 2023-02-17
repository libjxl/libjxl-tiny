// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_ENC_GROUP_H_
#define ENCODER_ENC_GROUP_H_

#include <stddef.h>

#include "encoder/ac_strategy.h"
#include "encoder/dc_group_data.h"
#include "encoder/enc_bit_writer.h"
#include "encoder/entropy_code.h"
#include "encoder/image.h"
#include "encoder/quant_weights.h"

namespace jxl {

struct GroupProcessorMemory {
  GroupProcessorMemory() {
    mem_dct = hwy::AllocateAligned<float>(kMaxCoeffArea * 4);
    mem_coeff = hwy::AllocateAligned<int32_t>(kMaxCoeffArea * 3);
  }
  float* block_storage() { return mem_dct.get(); }
  float* scratch_space() { return mem_dct.get() + 3 * kMaxCoeffArea; }
  int32_t* coeff_storage() { return mem_coeff.get(); }
  hwy::AlignedFreeUniquePtr<float[]> mem_dct;
  hwy::AlignedFreeUniquePtr<int32_t[]> mem_coeff;
};

void WriteACGroup(const Image3F& opsin, const Rect& group_brect,
                  const DequantMatrices& matrices, const float scale,
                  const float scale_dc, const uint32_t x_qm_scale,
                  DCGroupData* dc_data, const EntropyCode& ac_code,
                  Image3B* num_nzeros, GroupProcessorMemory* mem,
                  BitWriter* writer);

}  // namespace jxl

#endif  // ENCODER_ENC_GROUP_H_
