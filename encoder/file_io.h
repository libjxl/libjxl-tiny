// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_FILE_IO_H_
#define ENCODER_FILE_IO_H_

#include <stdint.h>

#include <vector>

namespace jpegxl {
namespace tools {

bool ReadFile(const char* filename, std::vector<uint8_t>* out);

bool WriteFile(const char* filename, const std::vector<uint8_t>& bytes);

}  // namespace tools
}  // namespace jpegxl

#endif  // ENCODER_FILE_IO_H_
