// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "plugins/gimp/common.h"

namespace jxl {

JpegXlGimpProgress::JpegXlGimpProgress(const char *message) {
  cur_progress = 0;
  max_progress = 100;

  gimp_progress_init_printf("%s\n", message);
}

void JpegXlGimpProgress::update() {
  gimp_progress_update((float)++cur_progress / (float)max_progress);
  return;
}

void JpegXlGimpProgress::finished() {
  gimp_progress_update(1.0);
  return;
}

}  // namespace jxl
