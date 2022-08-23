// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_MODULAR_OPTIONS_H_
#define ENCODER_MODULAR_OPTIONS_H_

#include <stdint.h>

#include <array>
#include <vector>

namespace jxl {

using PropertyVal = int32_t;
using Properties = std::vector<PropertyVal>;

enum class Predictor : uint32_t {
  Zero = 0,
  Left = 1,
  Top = 2,
  Average0 = 3,
  Select = 4,
  Gradient = 5,
  TopRight = 7,
  TopLeft = 8,
  LeftLeft = 9,
  Average1 = 10,
  Average2 = 11,
  Average3 = 12,
  Average4 = 13,
};

static constexpr ssize_t kNumStaticProperties = 2;  // channel, group_id.

}  // namespace jxl

#endif  // ENCODER_MODULAR_OPTIONS_H_
