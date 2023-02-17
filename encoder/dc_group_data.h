// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_DC_GROUP_DATA_H_
#define ENCODER_DC_GROUP_DATA_H_

#include <stddef.h>
#include <stdint.h>

#include "encoder/ac_strategy.h"
#include "encoder/common.h"
#include "encoder/image.h"

namespace jxl {

struct DCGroupData {
  DCGroupData(size_t xsize_blocks, size_t ysize_blocks)
      : quant_dc(xsize_blocks, ysize_blocks),
        raw_quant_field(xsize_blocks, ysize_blocks),
        ac_strategy(xsize_blocks, ysize_blocks),
        ytox_map(DivCeil(xsize_blocks * kBlockDim, kColorTileDim),
                 DivCeil(ysize_blocks * kBlockDim, kColorTileDim)),
        ytob_map(DivCeil(xsize_blocks * kBlockDim, kColorTileDim),
                 DivCeil(ysize_blocks * kBlockDim, kColorTileDim)) {
    ac_strategy.FillDCT8();
    ZeroFillImage(&ytox_map);
    ZeroFillImage(&ytob_map);
  }
  Image3S quant_dc;
  ImageB raw_quant_field;
  AcStrategyImage ac_strategy;
  ImageSB ytox_map;
  ImageSB ytob_map;
};

}  // namespace jxl

#endif  // ENCODER_DC_GROUP_DATA_H_
