// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_COLOR_MANAGEMENT_H_
#define ENCODER_COLOR_MANAGEMENT_H_

// ICC profiles and color space conversions.

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "encoder/base/padded_bytes.h"
#include "encoder/base/status.h"
#include "encoder/color_encoding_internal.h"
#include "encoder/common.h"
#include "encoder/image.h"

namespace jxl {

enum class ExtraTF {
  kNone,
  kPQ,
  kHLG,
  kSRGB,
};

Status MaybeCreateProfile(const ColorEncoding& c,
                          PaddedBytes* JXL_RESTRICT icc);

Status CIEXYZFromWhiteCIExy(const CIExy& xy, float XYZ[3]);

}  // namespace jxl

#endif  // ENCODER_COLOR_MANAGEMENT_H_
