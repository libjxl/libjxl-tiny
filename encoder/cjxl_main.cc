// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include <stdio.h>

#include "encoder/read_pfm.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/enc_file.h"
#include "lib/jxl/image.h"
#include "tools/cmdline.h"
#include "tools/file_io.h"

namespace {
struct CompressArgs {
  void AddCommandLineOptions(jpegxl::tools::CommandLineParser* cmdline) {
    cmdline->AddPositionalOption("INPUT", /* required = */ true,
                                 "the input is PFM in linear sRGB colorspace",
                                 &file_in);
    cmdline->AddPositionalOption(
        "OUTPUT", /* required = */ false,
        "the compressed JXL output file (can be omitted for benchmarking)",
        &file_out);
    cmdline->AddOptionValue('d', "distance", "maxError",
                            "Target butteraugli distance.", &distance,
                            jpegxl::tools::ParseFloat);
  }
  const char* file_in = nullptr;
  const char* file_out = nullptr;
  float distance = 1.0;
};
}  // namespace

int main(int argc, char** argv) {
  jpegxl::tools::CommandLineParser cmdline;
  CompressArgs args;
  args.AddCommandLineOptions(&cmdline);

  if (!cmdline.Parse(argc, const_cast<const char**>(argv)) || !args.file_in) {
    fprintf(stderr, "Use '%s -h' for more information\n", argv[0]);
    return EXIT_FAILURE;
  }

  jxl::Image3F image;
  if (!jxl::ReadPFM(args.file_in, &image)) {
    fprintf(stderr, "Error reading PFM input file.\n");
    return EXIT_FAILURE;
  }

  fprintf(stderr, "Read %" PRIuS "x%" PRIuS " pixels input image.\n",
          image.xsize(), image.ysize());

  std::vector<uint8_t> output;
  if (!jxl::EncodeFile(image, args.distance, &output)) {
    fprintf(stderr, "Encoding failed.\n");
    return EXIT_FAILURE;
  }

  fprintf(stderr, "Compressed to %" PRIuS " bytes.\n", output.size());

  if (args.file_out && !jpegxl::tools::WriteFile(args.file_out, output)) {
    fprintf(stderr, "Failed to write to output file %s\n", args.file_out);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
