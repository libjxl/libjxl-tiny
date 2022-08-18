// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef LIB_JXL_ENC_FILE_H_
#define LIB_JXL_ENC_FILE_H_

// Facade for JXL encoding.

#include "lib/jxl/aux_out.h"
#include "lib/jxl/aux_out_fwd.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_params.h"

namespace jxl {

// Write preview from `io`.
Status EncodePreview(const CompressParams& cparams, const ImageBundle& ib,
                     const CodecMetadata* metadata, const JxlCmsInterface& cms,
                     ThreadPool* pool, BitWriter* JXL_RESTRICT writer);

// Write headers from the CodecMetadata. Also may modify nonserialized_...
// fields of the metadata.
Status WriteHeaders(CodecMetadata* metadata, BitWriter* writer,
                    AuxOut* aux_out);

// Compresses pixels from `io` (given in any ColorEncoding).
// `io->metadata.m.original` must be set.
Status EncodeFile(const CompressParams& params, const CodecInOut* io,
                  PaddedBytes* compressed, const JxlCmsInterface& cms,
                  AuxOut* aux_out = nullptr, ThreadPool* pool = nullptr);

Status EncodeFile(const Image3F& input, float distance,
                  std::vector<uint8_t>* output);

}  // namespace jxl

#endif  // LIB_JXL_ENC_FILE_H_
