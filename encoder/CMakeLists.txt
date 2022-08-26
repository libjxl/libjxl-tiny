# Copyright (c) the JPEG XL Project Authors.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

add_library(jxl_tiny STATIC EXCLUDE_FROM_ALL
  ac_strategy.cc
  base/cache_aligned.cc
  base/padded_bytes.cc
  enc_ans.cc
  image.cc
  enc_adaptive_quantization.cc
  enc_ac_strategy.cc
  enc_ans.cc
  enc_bit_writer.cc
  enc_chroma_from_luma.cc
  enc_cluster.cc
  enc_coeff_order.cc
  enc_entropy_coder.cc
  enc_file.cc
  enc_frame.cc
  enc_group.cc
  enc_toc.cc
  enc_xyb.cc
  read_pfm.cc
  quant_weights.cc
  base/data_parallel.cc
  ans_common.cc
  entropy_coder.cc
  quantizer.cc
  gaborish.cc
  convolve_symmetric5.cc
  cmdline.cc
  file_io.cc
  dct_scales.cc
)
target_compile_options(jxl_tiny PUBLIC "${JPEGXL_INTERNAL_FLAGS}")
target_include_directories(jxl_tiny PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(jxl_tiny hwy)


add_executable(cjxl_tiny cjxl_main.cc)
target_link_libraries(cjxl_tiny jxl_tiny)
