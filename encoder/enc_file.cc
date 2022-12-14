// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/enc_file.h"

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "encoder/base/data_parallel.h"
#include "encoder/enc_bit_writer.h"
#include "encoder/enc_frame.h"
#include "encoder/image.h"

namespace jxl {

namespace {

// Reserved by ISO/IEC 10918-1. LF causes files opened in text mode to be
// rejected because the marker changes to 0x0D instead. The 0xFF prefix also
// ensures there were no 7-bit transmission limitations.
static constexpr uint8_t kCodestreamMarker = 0x0A;

void WriteSize(uint32_t size, BitWriter* writer) {
  size -= 1;
  uint32_t kBits[4] = {9, 13, 18, 30};
  for (size_t i = 0; i < 4; ++i) {
    if (size < (1u << kBits[i])) {
      writer->Write(2, i);
      writer->Write(kBits[i], size);
      return;
    }
  }
}

Status WriteSizeHeader(size_t xsize64, size_t ysize64, BitWriter* writer) {
  if (xsize64 > 0x3FFFFFFFull || ysize64 > 0x3FFFFFFFull) {
    return JXL_FAILURE("Image too large");
  }
  const uint32_t xsize32 = static_cast<uint32_t>(xsize64);
  const uint32_t ysize32 = static_cast<uint32_t>(ysize64);
  writer->Write(1, 0);  // small
  WriteSize(ysize32, writer);
  writer->Write(3, 0);  // ratio
  WriteSize(xsize32, writer);
  return true;
}

}  // namespace

bool EncodeFile(const Image3F& input, float distance,
                std::vector<uint8_t>* output) {
  if (distance < 0.0) {
    return JXL_FAILURE("Invalid butteraugli distance (%f)", distance);
  } else if (distance == 0.0) {
    return JXL_FAILURE("Lossless compression is not supported.");
  } else if (distance <= 0.03) {
    // Distance where the average BPP is still slightly smaller on photographs
    // than for lossless JPEG XL.
    distance = 0.03;
  }
  if (input.xsize() == 0 || input.ysize() == 0) {
    return JXL_FAILURE("Empty image");
  }

  BitWriter writer;
  BitWriter::Allotment allotment(&writer, 1024);
  writer.Write(8, 0xFF);
  writer.Write(8, kCodestreamMarker);
  JXL_RETURN_IF_ERROR(WriteSizeHeader(input.xsize(), input.ysize(), &writer));
  writer.Write(1, 0);  // not all default image metadata
  writer.Write(1, 0);  // no extra fields in image metadata
  writer.Write(1, 1);  // floating point samples
  writer.Write(2, 0);  // 32 bits per sample
  writer.Write(4, 7);  // 8 exponent bits per sample
  writer.Write(1, 0);  // modular 16 bit sufficient
  writer.Write(2, 0);  // no extra channels
  writer.Write(1, 1);  // xyb encoded
  writer.Write(1, 0);  // not all default color encoding
  writer.Write(1, 0);  // no icc
  writer.Write(2, 0);  // RGB color space
  writer.Write(2, 1);  // D65 white point
  writer.Write(2, 1);  // SRGB primaries
  writer.Write(1, 0);  // no gamma
  writer.Write(2, 2);  // transfer function selector bits (2 .. 17)
  writer.Write(4, 6);  // linear transfer function (enum value 8)
  writer.Write(2, 1);  // relative rendering intent
  writer.Write(2, 0);  // no extensions
  writer.Write(1, 1);  // all default transform data
  writer.ZeroPadToByte();
  allotment.Reclaim(&writer);

  ThreadPool pool;
  JXL_RETURN_IF_ERROR(EncodeFrame(distance, input, &pool, &writer));

  PaddedBytes compressed;
  compressed = std::move(writer).TakeBytes();
  output->assign(compressed.data(), compressed.data() + compressed.size());

  return true;
}

}  // namespace jxl
