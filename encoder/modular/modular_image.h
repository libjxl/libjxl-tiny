// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_MODULAR_MODULAR_IMAGE_H_
#define ENCODER_MODULAR_MODULAR_IMAGE_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <utility>
#include <vector>

#include "encoder/base/compiler_specific.h"
#include "encoder/base/data_parallel.h"
#include "encoder/base/status.h"
#include "encoder/image.h"
#include "encoder/image_ops.h"

namespace jxl {

typedef int32_t pixel_type;  // can use int16_t if it's only for 8-bit images.
                             // Need some wiggle room for YCoCg / Squeeze etc

typedef int64_t pixel_type_w;

class Channel {
 public:
  jxl::Plane<pixel_type> plane;
  size_t w, h;
  int hshift, vshift;  // w ~= image.w >> hshift;  h ~= image.h >> vshift
  Channel(size_t iw, size_t ih, int hsh = 0, int vsh = 0)
      : plane(iw, ih), w(iw), h(ih), hshift(hsh), vshift(vsh) {}

  Channel(const Channel& other) = delete;
  Channel& operator=(const Channel& other) = delete;

  // Move assignment
  Channel& operator=(Channel&& other) noexcept {
    w = other.w;
    h = other.h;
    hshift = other.hshift;
    vshift = other.vshift;
    plane = std::move(other.plane);
    return *this;
  }

  // Move constructor
  Channel(Channel&& other) noexcept = default;

  void shrink() {
    if (plane.xsize() == w && plane.ysize() == h) return;
    jxl::Plane<pixel_type> resizedplane(w, h);
    plane = std::move(resizedplane);
  }
  void shrink(int nw, int nh) {
    w = nw;
    h = nh;
    shrink();
  }

  JXL_INLINE pixel_type* Row(const size_t y) { return plane.Row(y); }
  JXL_INLINE const pixel_type* Row(const size_t y) const {
    return plane.Row(y);
  }
};

class Image {
 public:
  // image data
  std::vector<Channel> channel;

  // image dimensions
  size_t w, h;
  int bitdepth;
  size_t nb_meta_channels;  // first few channels might contain palette(s)
  bool error;               // true if a fatal error occurred, false otherwise

  Image(size_t iw, size_t ih, int bitdepth, int nb_chans);
  Image();

  Image(const Image& other) = delete;
  Image& operator=(const Image& other) = delete;

  Image& operator=(Image&& other) noexcept;
  Image(Image&& other) noexcept = default;

  bool empty() const {
    for (const auto& ch : channel) {
      if (ch.w && ch.h) return false;
    }
    return true;
  }

  Image clone();

  std::string DebugString() const;
};

}  // namespace jxl

#endif  // ENCODER_MODULAR_MODULAR_IMAGE_H_
