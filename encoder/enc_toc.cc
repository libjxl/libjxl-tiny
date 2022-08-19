// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/enc_toc.h"

#include <stdint.h>

#include "encoder/coeff_order.h"
#include "encoder/coeff_order_fwd.h"
#include "encoder/common.h"
#include "encoder/enc_coeff_order.h"
#include "encoder/field_encodings.h"
#include "encoder/fields.h"
#include "encoder/toc.h"

namespace jxl {
Status WriteGroupOffsets(const std::vector<BitWriter>& group_codes,
                         const std::vector<coeff_order_t>* permutation,
                         BitWriter* JXL_RESTRICT writer) {
  BitWriter::Allotment allotment(writer, MaxBits(group_codes.size()));
  if (permutation && !group_codes.empty()) {
    // Don't write a permutation at all for an empty group_codes.
    writer->Write(1, 1);  // permutation
    JXL_DASSERT(permutation->size() == group_codes.size());
    EncodePermutation(permutation->data(), /*skip=*/0, permutation->size(),
                      writer);

  } else {
    writer->Write(1, 0);  // no permutation
  }
  writer->ZeroPadToByte();  // before TOC entries

  for (size_t i = 0; i < group_codes.size(); i++) {
    JXL_ASSERT(group_codes[i].BitsWritten() % kBitsPerByte == 0);
    const size_t group_size = group_codes[i].BitsWritten() / kBitsPerByte;
    JXL_RETURN_IF_ERROR(U32Coder::Write(kTocDist, group_size, writer));
  }
  writer->ZeroPadToByte();  // before first group
  allotment.Reclaim(writer);
  return true;
}

}  // namespace jxl
