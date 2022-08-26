// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/dct_scales.h"

namespace jxl {

// Definition of constexpr arrays.
constexpr float DCTResampleScales<1, 8>::kScales[];
constexpr float DCTResampleScales<2, 16>::kScales[];
constexpr float DCTResampleScales<8, 1>::kScales[];
constexpr float DCTResampleScales<16, 2>::kScales[];
constexpr float WcMultipliers<4>::kMultipliers[];
constexpr float WcMultipliers<8>::kMultipliers[];
constexpr float WcMultipliers<16>::kMultipliers[];

}  // namespace jxl
