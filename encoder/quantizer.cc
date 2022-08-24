// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/quantizer.h"

#include <stdio.h>
#include <string.h>

#include <algorithm>

#include "encoder/base/compiler_specific.h"
#include "encoder/image.h"
#include "encoder/image_ops.h"
#include "encoder/quant_weights.h"

namespace jxl {

static const int32_t kDefaultQuant = 64;

constexpr int32_t Quantizer::kQuantMax;

Quantizer::Quantizer(const DequantMatrices* dequant)
    : Quantizer(dequant, kDefaultQuant, kGlobalScaleDenom / kDefaultQuant) {}

Quantizer::Quantizer(const DequantMatrices* dequant, int quant_dc,
                     int global_scale)
    : global_scale_(global_scale), quant_dc_(quant_dc), dequant_(dequant) {
  JXL_ASSERT(dequant_ != nullptr);
  RecomputeFromGlobalScale();
  inv_quant_dc_ = inv_global_scale_ / quant_dc_;

  memcpy(zero_bias_, kZeroBiasDefault, sizeof(kZeroBiasDefault));
}

void Quantizer::ComputeGlobalScaleAndQuant(float quant_dc, float quant_median,
                                           float quant_median_absd) {
  // Target value for the median value in the quant field.
  const float kQuantFieldTarget = 5;
  // We reduce the median of the quant field by the median absolute deviation:
  // higher resolution on highly varying quant fields.
  float scale = kGlobalScaleDenom * (quant_median - quant_median_absd) /
                kQuantFieldTarget;
  // Ensure that new_global_scale is positive and no more than 1<<15.
  if (scale < 1) scale = 1;
  if (scale > (1 << 15)) scale = 1 << 15;
  int new_global_scale = static_cast<int>(scale);
  // Ensure that quant_dc_ will always be at least
  // 0.625 * kGlobalScaleDenom/kGlobalScaleNumerator = 10.
  const int scaled_quant_dc =
      static_cast<int>(quant_dc * kGlobalScaleNumerator * 1.6);
  if (new_global_scale > scaled_quant_dc) {
    new_global_scale = scaled_quant_dc;
    if (new_global_scale <= 0) new_global_scale = 1;
  }
  global_scale_ = new_global_scale;
  // Code below uses inv_global_scale_.
  RecomputeFromGlobalScale();

  float fval = quant_dc * inv_global_scale_ + 0.5f;
  fval = std::min<float>(1 << 16, fval);
  const int new_quant_dc = static_cast<int>(fval);
  quant_dc_ = new_quant_dc;

  // quant_dc_ was updated, recompute values.
  RecomputeFromGlobalScale();
}

void Quantizer::SetQuantFieldRect(const ImageF& qf, const Rect& rect,
                                  ImageI* JXL_RESTRICT raw_quant_field) const {
  for (size_t y = 0; y < rect.ysize(); ++y) {
    const float* JXL_RESTRICT row_qf = rect.ConstRow(qf, y);
    int32_t* JXL_RESTRICT row_qi = rect.Row(raw_quant_field, y);
    for (size_t x = 0; x < rect.xsize(); ++x) {
      int val = ClampVal(row_qf[x] * inv_global_scale_ + 0.5f);
      row_qi[x] = val;
    }
  }
}

void Quantizer::SetQuantField(const float quant_dc, const ImageF& qf,
                              ImageI* JXL_RESTRICT raw_quant_field) {
  std::vector<float> data(qf.xsize() * qf.ysize());
  for (size_t y = 0; y < qf.ysize(); ++y) {
    const float* JXL_RESTRICT row_qf = qf.Row(y);
    for (size_t x = 0; x < qf.xsize(); ++x) {
      float quant = row_qf[x];
      data[qf.xsize() * y + x] = quant;
    }
  }
  std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
  const float quant_median = data[data.size() / 2];
  std::vector<float> deviations(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    deviations[i] = fabsf(data[i] - quant_median);
  }
  std::nth_element(deviations.begin(),
                   deviations.begin() + deviations.size() / 2,
                   deviations.end());
  const float quant_median_absd = deviations[deviations.size() / 2];
  ComputeGlobalScaleAndQuant(quant_dc, quant_median, quant_median_absd);
  if (raw_quant_field) {
    JXL_CHECK(SameSize(*raw_quant_field, qf));
    SetQuantFieldRect(qf, Rect(qf), raw_quant_field);
  }
}

void Quantizer::SetQuant(float quant_dc, float quant_ac,
                         ImageI* JXL_RESTRICT raw_quant_field) {
  ComputeGlobalScaleAndQuant(quant_dc, quant_ac, 0);
  int32_t val = ClampVal(quant_ac * inv_global_scale_ + 0.5f);
  FillImage(val, raw_quant_field);
}

void Quantizer::Encode(BitWriter* writer) const {
  if (global_scale_ < 2049) {
    writer->Write(2, 0);
    writer->Write(11, global_scale_ - 1);
  } else if (global_scale_ < 4097) {
    writer->Write(2, 1);
    writer->Write(11, global_scale_ - 2049);
  } else if (global_scale_ < 8193) {
    writer->Write(2, 2);
    writer->Write(12, global_scale_ - 4097);
  } else {
    writer->Write(2, 3);
    writer->Write(16, global_scale_ - 8193);
  }
  if (quant_dc_ == 16) {
    writer->Write(2, 0);
  } else if (quant_dc_ < 33) {
    writer->Write(2, 1);
    writer->Write(5, quant_dc_ - 1);
  } else if (quant_dc_ < 257) {
    writer->Write(2, 2);
    writer->Write(8, quant_dc_ - 1);
  } else {
    writer->Write(2, 3);
    writer->Write(16, quant_dc_ - 1);
  }
}

void Quantizer::DumpQuantizationMap(const ImageI& raw_quant_field) const {
  printf("Global scale: %d (%.7f)\nDC quant: %d\n", global_scale_,
         global_scale_ * 1.0 / kGlobalScaleDenom, quant_dc_);
  printf("AC quantization Map:\n");
  for (size_t y = 0; y < raw_quant_field.ysize(); ++y) {
    for (size_t x = 0; x < raw_quant_field.xsize(); ++x) {
      printf(" %3d", raw_quant_field.Row(y)[x]);
    }
    printf("\n");
  }
}

}  // namespace jxl
