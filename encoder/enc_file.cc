// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/enc_file.h"

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_color_management.h"
#include "lib/jxl/enc_frame.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"

namespace jxl {

bool EncodeFile(const Image3F& input, float distance,
                std::vector<uint8_t>* output) {
  BitWriter writer;
  CodecMetadata metadata;
  {
    BitWriter::Allotment allotment(&writer, 16);
    writer.Write(8, 0xFF);
    writer.Write(8, kCodestreamMarker);
    ReclaimAndCharge(&writer, &allotment, 0, nullptr);
  }
  JXL_RETURN_IF_ERROR(metadata.size.Set(input.xsize(), input.ysize()));
  JXL_RETURN_IF_ERROR(Bundle::Write(metadata.size, &writer, 0, nullptr));
  metadata.m.SetFloat32Samples();
  metadata.m.xyb_encoded = true;
  metadata.m.color_encoding = ColorEncoding::LinearSRGB();
  JXL_RETURN_IF_ERROR(Bundle::Write(metadata.m, &writer, 0, nullptr));
  {
    BitWriter::Allotment allotment(&writer, 16);
    writer.Write(1, 1);  // all default transform data
    writer.ZeroPadToByte();
    ReclaimAndCharge(&writer, &allotment, 0, nullptr);
  }

  CompressParams cparams;
  cparams.butteraugli_distance = distance;
  FrameInfo info;
  info.is_last = true;
  ImageBundle ib(&metadata.m);
  ib.SetFromImage(CopyImage(input), metadata.m.color_encoding);
  JXL_RETURN_IF_ERROR(EncodeFrame(cparams, info, &metadata, ib, GetJxlCms(),
                                  nullptr, &writer, nullptr));

  PaddedBytes compressed;
  compressed = std::move(writer).TakeBytes();
  output->assign(compressed.data(), compressed.data() + compressed.size());

  return true;
}

}  // namespace jxl
