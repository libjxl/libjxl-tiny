// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/ac_strategy.h"

#include <stddef.h>

namespace jxl {

// These definitions are needed before C++17.
constexpr size_t AcStrategy::kMaxCoeffBlocks;
constexpr size_t AcStrategy::kMaxBlockDim;
constexpr size_t AcStrategy::kMaxCoeffArea;

}  // namespace jxl
