// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/modular/transform/transform.h"

#include "encoder/base/printf_macros.h"
#include "encoder/fields.h"
#include "encoder/modular/modular_image.h"

namespace jxl {

SqueezeParams::SqueezeParams() { Bundle::Init(this); }
Transform::Transform(TransformId id) {
  Bundle::Init(this);
  this->id = id;
}

}  // namespace jxl
