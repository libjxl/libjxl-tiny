// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include <errno.h>
#include <stdio.h>

#include "encoder/base/printf_macros.h"
#include "encoder/enc_file.h"
#include "encoder/image.h"
#include "encoder/read_pfm.h"

namespace {
struct CompressArgs {
  const char* file_in = nullptr;
  const char* file_out = nullptr;
  float distance = 1.0;
};

bool WriteFile(const char* filename, const std::vector<uint8_t>& bytes) {
  FILE* file = fopen(filename, "wb");
  if (!file) {
    fprintf(stderr, "Could not open %s for writing\nError: %s", filename,
            strerror(errno));
    return false;
  }
  if (fwrite(bytes.data(), 1, bytes.size(), file) != bytes.size()) {
    fprintf(stderr, "Could not write to file\nError: %s", strerror(errno));
    return false;
  }
  if (fclose(file) != 0) {
    fprintf(stderr, "Could not close file\nError: %s", strerror(errno));
    return false;
  }
  return true;
}

void PrintHelp(char* arg0) {
  fprintf(stderr,
          "Usage: %s <file in> [<file out>] [-d distance]\n\n"
          "  NOTE: <file in> is a .pfm file in linear SRGB colorspace\n",
          arg0);
}

}  // namespace

int main(int argc, char** argv) {
  CompressArgs args;
  for (int i = 1; i < argc; i++) {
    if (!strcmp("-h", argv[i]) || !strcmp("--help", argv[i])) {
      PrintHelp(argv[0]);
      return EXIT_SUCCESS;
    }
    if (argv[i][0] == '-' && argv[i][1] == 'd') {
      char* arg = argv[i][2] != '\0' ? &argv[i][2] : argv[++i];
      if (i == argc) {
        fprintf(stderr, "-d requires an argument\n");
        return EXIT_FAILURE;
      }
      char* end;
      args.distance = static_cast<float>(strtod(arg, &end));
      if (*end != '\0') {
        fprintf(stderr, "Unable to interpret as float: %s\n", arg);
        return EXIT_FAILURE;
      }
      continue;
    }
    if (!args.file_in) {
      args.file_in = argv[i];
    } else if (!args.file_out) {
      args.file_out = argv[i];
    }
  }
  if (!args.file_in) {
    fprintf(stderr, "Missing input file.\n");
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

  if (args.file_out && !WriteFile(args.file_out, output)) {
    fprintf(stderr, "Failed to write to output file %s\n", args.file_out);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
