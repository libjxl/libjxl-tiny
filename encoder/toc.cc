// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/toc.h"

#include <stdint.h>

#include "encoder/coeff_order.h"
#include "encoder/coeff_order_fwd.h"
#include "encoder/common.h"
#include "encoder/fields.h"

namespace jxl {
size_t MaxBits(const size_t num_sizes) {
  const size_t entry_bits = U32Coder::MaxEncodedBits(kTocDist) * num_sizes;
  // permutation bit (not its tokens!), padding, entries, padding.
  return 1 + kBitsPerByte + entry_bits + kBitsPerByte;
}
}  // namespace jxl
