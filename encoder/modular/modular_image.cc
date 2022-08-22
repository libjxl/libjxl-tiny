// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/modular/modular_image.h"

#include <sstream>

#include "encoder/base/status.h"
#include "encoder/common.h"

namespace jxl {

Image::Image(size_t iw, size_t ih, int bitdepth, int nb_chans)
    : w(iw), h(ih), bitdepth(bitdepth), nb_meta_channels(0), error(false) {
  for (int i = 0; i < nb_chans; i++) channel.emplace_back(Channel(iw, ih));
}

Image::Image() : w(0), h(0), bitdepth(8), nb_meta_channels(0), error(true) {}

Image &Image::operator=(Image &&other) noexcept {
  w = other.w;
  h = other.h;
  bitdepth = other.bitdepth;
  nb_meta_channels = other.nb_meta_channels;
  error = other.error;
  channel = std::move(other.channel);
  return *this;
}

Image Image::clone() {
  Image c(w, h, bitdepth, 0);
  c.nb_meta_channels = nb_meta_channels;
  c.error = error;
  for (Channel &ch : channel) {
    Channel a(ch.w, ch.h, ch.hshift, ch.vshift);
    CopyImageTo(ch.plane, &a.plane);
    c.channel.push_back(std::move(a));
  }
  return c;
}

std::string Image::DebugString() const {
  std::ostringstream os;
  os << w << "x" << h << ", depth: " << bitdepth;
  if (!channel.empty()) {
    os << ", channels:";
    for (size_t i = 0; i < channel.size(); ++i) {
      os << " " << channel[i].w << "x" << channel[i].h
         << "(shift: " << channel[i].hshift << "," << channel[i].vshift << ")";
      if (i < nb_meta_channels) os << "*";
    }
  }
  return os.str();
}

}  // namespace jxl
