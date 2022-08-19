// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_ENC_FRAME_H_
#define ENCODER_ENC_FRAME_H_

#include "encoder/base/data_parallel.h"
#include "encoder/base/status.h"
#include "encoder/enc_bit_writer.h"
#include "encoder/image.h"
#include "encoder/image_metadata.h"

namespace jxl {

// Encodes a single frame (including its header) into a byte stream.  Groups may
// be processed in parallel by `pool`. metadata is the ImageMetadata encoded in
// the codestream, and must be used for the FrameHeaders, do not use
// ib.metadata.
Status EncodeFrame(const float distance, const CodecMetadata* metadata,
                   const Image3F& linear, ThreadPool* pool, BitWriter* writer);

}  // namespace jxl

#endif  // ENCODER_ENC_FRAME_H_