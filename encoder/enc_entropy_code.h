// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_ENC_ENTROPY_CODE_H_
#define ENCODER_ENC_ENTROPY_CODE_H_

#include <stddef.h>
#include <stdint.h>

#include "encoder/enc_bit_writer.h"
#include "encoder/entropy_code.h"
#include "encoder/token.h"

namespace jxl {

void OptimizePrefixCodes(const std::vector<Token>& tokens, EntropyCode* code);

void OptimizeEntropyCode(const std::vector<Token>& tokens, EntropyCode* code);

void WriteEntropyCode(const EntropyCode& code, BitWriter* writer);

static inline void WriteToken(const Token& token, const EntropyCode& code,
                              BitWriter* writer) {
  // TODO(szabadka) Move this out of here.
  BitWriter::Allotment allotment(writer, 32);
  uint32_t tok, nbits, bits;
  UintCoder().Encode(token.value, &tok, &nbits, &bits);
  const PrefixCode& pc = code.prefix_codes[code.context_map[token.context]];
  uint64_t data = pc.bits[tok];
  data |= bits << pc.depths[tok];
  writer->Write(pc.depths[tok] + nbits, data);
  allotment.Reclaim(writer);
}

}  // namespace jxl
#endif  // ENCODER_ENC_ENTROPY_CODE_H_
