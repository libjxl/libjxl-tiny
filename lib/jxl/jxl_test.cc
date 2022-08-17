// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "lib/extras/dec/jxl.h"

#include <stdint.h>
#include <stdio.h>

#include <array>
#include <future>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "lib/extras/codec.h"
#include "lib/jxl/aux_out.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/codec_y4m_testonly.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/color_management.h"
#include "lib/jxl/enc_butteraugli_comparator.h"
#include "lib/jxl/enc_butteraugli_pnorm.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_file.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/fake_parallel_runner_testonly.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/image_test_utils.h"
#include "lib/jxl/jpeg/dec_jpeg_data.h"
#include "lib/jxl/jpeg/dec_jpeg_data_writer.h"
#include "lib/jxl/jpeg/enc_jpeg_data.h"
#include "lib/jxl/jpeg/jpeg_data.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testdata.h"
#include "tools/box/box.h"

namespace jxl {
namespace {
using test::Roundtrip;

#define JXL_TEST_NL 0  // Disabled in code

void CreateImage1x1(CodecInOut* io) {
  Image3F image(1, 1);
  ZeroFillImage(&image);
  io->metadata.m.SetUintSamples(8);
  io->metadata.m.color_encoding = ColorEncoding::SRGB();
  io->SetFromImage(std::move(image), io->metadata.m.color_encoding);
}

TEST(JxlTest, RoundtripSinglePixel) {
  CodecInOut io;
  CreateImage1x1(&io);

  CompressParams cparams;
  cparams.butteraugli_distance = 1.0;
  ThreadPool* pool = nullptr;
  CodecInOut io2;
  Roundtrip(&io, cparams, {}, pool, &io2);
}

// Changing serialized signature causes Decode to fail.
#ifndef JXL_CRASH_ON_ERROR
TEST(JxlTest, RoundtripMarker) {
  CodecInOut io;
  CreateImage1x1(&io);

  CompressParams cparams;
  cparams.butteraugli_distance = 1.0;
  AuxOut* aux_out = nullptr;
  ThreadPool* pool = nullptr;

  for (size_t i = 0; i < 2; ++i) {
    PaddedBytes compressed;
    EXPECT_TRUE(
        EncodeFile(cparams, &io, &compressed, GetJxlCms(), aux_out, pool));
    compressed[i] ^= 0xFF;
    CodecInOut io2;
    EXPECT_FALSE(test::DecodeFile({}, compressed, &io2, pool));
  }
}
#endif

TEST(JxlTest, RoundtripTinyFast) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, pool));
  io.ShrinkTo(32, 32);

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;
  cparams.butteraugli_distance = 4.0f;

  CodecInOut io2;
  const size_t enc_bytes = Roundtrip(&io, cparams, {}, pool, &io2);
  printf("32x32 image size %" PRIuS " bytes\n", enc_bytes);
}

TEST(JxlTest, RoundtripSmallD1) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CompressParams cparams;
  cparams.butteraugli_distance = 1.0;

  CodecInOut io_out;
  size_t compressed_size;

  {
    CodecInOut io;
    ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, pool));
    io.ShrinkTo(io.xsize() / 8, io.ysize() / 8);

    compressed_size = Roundtrip(&io, cparams, {}, pool, &io_out);
    EXPECT_LE(compressed_size, 1000u);
    EXPECT_THAT(ButteraugliDistance(io, io_out, cparams.ba_params, GetJxlCms(),
                                    /*distmap=*/nullptr, pool),
                IsSlightlyBelow(1.0));
  }

  {
    // And then, with a lower intensity target than the default, the bitrate
    // should be smaller.
    CodecInOut io_dim;
    ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io_dim, pool));
    io_dim.metadata.m.SetIntensityTarget(100);
    io_dim.ShrinkTo(io_dim.xsize() / 8, io_dim.ysize() / 8);
    EXPECT_LT(Roundtrip(&io_dim, cparams, {}, pool, &io_out), compressed_size);
    EXPECT_THAT(
        ButteraugliDistance(io_dim, io_out, cparams.ba_params, GetJxlCms(),
                            /*distmap=*/nullptr, pool),
        IsSlightlyBelow(1.1));
    EXPECT_EQ(io_dim.metadata.m.IntensityTarget(),
              io_out.metadata.m.IntensityTarget());
  }
}

// Roundtrip the image using a parallel runner that executes single-threaded but
// in random order.
TEST(JxlTest, RoundtripOutOfOrderProcessing) {
  FakeParallelRunner fake_pool(/*order_seed=*/123, /*num_threads=*/8);
  ThreadPool pool(&JxlFakeParallelRunner, &fake_pool);
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, &pool));
  // Image size is selected so that the block border needed is larger than the
  // amount of pixels available on the next block.
  io.ShrinkTo(513, 515);

  CompressParams cparams;
  // Force epf so we end up needing a lot of border.
  cparams.epf = 3;

  CodecInOut io2;
  Roundtrip(&io, cparams, {}, &pool, &io2);

  EXPECT_GE(1.5, ButteraugliDistance(io, io2, cparams.ba_params, GetJxlCms(),
                                     /*distmap=*/nullptr, &pool));
}

TEST(JxlTest, RoundtripUnalignedD2) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, pool));
  io.ShrinkTo(io.xsize() / 12, io.ysize() / 7);

  CompressParams cparams;
  cparams.butteraugli_distance = 2.0;

  CodecInOut io2;
  EXPECT_LE(Roundtrip(&io, cparams, {}, pool, &io2), 700u);
  EXPECT_THAT(ButteraugliDistance(io, io2, cparams.ba_params, GetJxlCms(),
                                  /*distmap=*/nullptr, pool),
              IsSlightlyBelow(1.7));
}

#if JXL_TEST_NL

TEST(JxlTest, RoundtripMultiGroupNL) {
  ThreadPoolInternal pool(4);
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, &pool));
  io.ShrinkTo(600, 1024);  // partial X, full Y group

  CompressParams cparams;

  cparams.fast_mode = true;
  cparams.butteraugli_distance = 1.0f;
  CodecInOut io2;
  Roundtrip(&io, cparams, {}, &pool, &io2);
  EXPECT_THAT(ButteraugliDistance(io, io2, cparams.ba_params, GetJxlCms(),
                                  /*distmap=*/nullptr, &pool),
              IsSlightlyBelow(0.9f));

  cparams.butteraugli_distance = 2.0f;
  CodecInOut io3;
  EXPECT_LE(Roundtrip(&io, cparams, {}, &pool, &io3), 80000u);
  EXPECT_THAT(ButteraugliDistance(io, io3, cparams.ba_params, GetJxlCms(),
                                  /*distmap=*/nullptr, &pool),
              IsSlightlyBelow(1.5f));
}

#endif

TEST(JxlTest, RoundtripMultiGroup) {
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png");
  CodecInOut io;
  {
    ThreadPoolInternal pool(4);
    ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, &pool));
  }
  io.ShrinkTo(600, 1024);

  auto test = [&](jxl::SpeedTier speed_tier, float target_distance,
                  size_t expected_size, float expected_distance) {
    ThreadPoolInternal pool(4);
    CompressParams cparams;
    cparams.butteraugli_distance = target_distance;
    cparams.speed_tier = speed_tier;
    CodecInOut io2;
    EXPECT_LE(Roundtrip(&io, cparams, {}, &pool, &io2), expected_size);
    EXPECT_THAT(ComputeDistance2(io.Main(), io2.Main(), GetJxlCms()),
                IsSlightlyBelow(expected_distance));
  };

  auto run_wombat = std::async(std::launch::async, test, SpeedTier::kWombat,
                               2.0f, 34500u, 18.5);
}

TEST(JxlTest, RoundtripRGBToGrayscale) {
  ThreadPoolInternal pool(4);
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, &pool));
  io.ShrinkTo(600, 1024);

  CompressParams cparams;
  cparams.butteraugli_distance = 1.0f;
  cparams.speed_tier = SpeedTier::kFalcon;

  extras::JXLDecompressParams dparams;
  dparams.color_space = "Gra_D65_Rel_SRG";

  CodecInOut io2;
  EXPECT_FALSE(io.Main().IsGray());
  EXPECT_LE(Roundtrip(&io, cparams, dparams, &pool, &io2), 56000u);
  EXPECT_TRUE(io2.Main().IsGray());

  // Convert original to grayscale here, because TransformTo refuses to
  // convert between grayscale and RGB.
  ColorEncoding srgb_lin = ColorEncoding::LinearSRGB(/*is_gray=*/false);
  ASSERT_TRUE(io.TransformTo(srgb_lin, GetJxlCms(), &pool));
  Image3F* color = io.Main().color();
  for (size_t y = 0; y < color->ysize(); ++y) {
    float* row_r = color->PlaneRow(0, y);
    float* row_g = color->PlaneRow(1, y);
    float* row_b = color->PlaneRow(2, y);
    for (size_t x = 0; x < color->xsize(); ++x) {
      float luma = 0.2126 * row_r[x] + 0.7152 * row_g[x] + 0.0722 * row_b[x];
      row_r[x] = row_g[x] = row_b[x] = luma;
    }
  }
  ColorEncoding srgb_gamma = ColorEncoding::SRGB(/*is_gray=*/false);
  ASSERT_TRUE(io.TransformTo(srgb_gamma, GetJxlCms(), &pool));
  io.metadata.m.color_encoding = io2.Main().c_current();
  io.Main().OverrideProfile(io2.Main().c_current());
  EXPECT_THAT(ButteraugliDistance(io, io2, cparams.ba_params, GetJxlCms(),
                                  /*distmap=*/nullptr, &pool),
              IsSlightlyBelow(1.7));
}

TEST(JxlTest, RoundtripLargeFast) {
  ThreadPoolInternal pool(8);
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, &pool));

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;

  CodecInOut io2;
  EXPECT_LE(Roundtrip(&io, cparams, {}, &pool, &io2), 465000u);
}

// Checks for differing size/distance in two consecutive runs of distance 2,
// which involves additional processing including adaptive reconstruction.
// Failing this may be a sign of race conditions or invalid memory accesses.
TEST(JxlTest, RoundtripD2Consistent) {
  ThreadPoolInternal pool(8);
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, &pool));

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;
  cparams.butteraugli_distance = 2.0;

  // Try each xsize mod kBlockDim to verify right border handling.
  for (size_t xsize = 48; xsize > 40; --xsize) {
    io.ShrinkTo(xsize, 15);

    CodecInOut io2;
    const size_t size2 = Roundtrip(&io, cparams, {}, &pool, &io2);

    CodecInOut io3;
    const size_t size3 = Roundtrip(&io, cparams, {}, &pool, &io3);

    // Exact same compressed size.
    EXPECT_EQ(size2, size3);

    // Exact same distance.
    const float dist2 = ComputeDistance2(io.Main(), io2.Main(), GetJxlCms());
    const float dist3 = ComputeDistance2(io.Main(), io3.Main(), GetJxlCms());
    EXPECT_EQ(dist2, dist3);
  }
}

// Same as above, but for full image, testing multiple groups.
TEST(JxlTest, RoundtripLargeConsistent) {
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png");
  CodecInOut io;
  {
    ThreadPoolInternal pool(8);
    ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, &pool));
  }

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;
  cparams.butteraugli_distance = 2.0;

  auto roundtrip_and_compare = [&]() {
    ThreadPoolInternal pool(8);
    CodecInOut io2;
    size_t size = Roundtrip(&io, cparams, {}, &pool, &io2);
    double dist = ComputeDistance2(io.Main(), io2.Main(), GetJxlCms());
    return std::tuple<size_t, double>(size, dist);
  };

  // Try each xsize mod kBlockDim to verify right border handling.
  auto future2 = std::async(std::launch::async, roundtrip_and_compare);
  auto future3 = std::async(std::launch::async, roundtrip_and_compare);

  const auto result2 = future2.get();
  const auto result3 = future3.get();

  // Exact same compressed size.
  EXPECT_EQ(std::get<0>(result2), std::get<0>(result3));

  // Exact same distance.
  EXPECT_EQ(std::get<1>(result2), std::get<1>(result3));
}

#if JXL_TEST_NL

TEST(JxlTest, RoundtripSmallNL) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, pool));
  io.ShrinkTo(io.xsize() / 8, io.ysize() / 8);

  CompressParams cparams;
  cparams.butteraugli_distance = 1.0;

  CodecInOut io2;
  EXPECT_LE(Roundtrip(&io, cparams, {}, pool, &io2), 1500u);
  EXPECT_THAT(ButteraugliDistance(io, io2, cparams.ba_params, GetJxlCms(),
                                  /*distmap=*/nullptr, pool),
              IsSlightlyBelow(1.7));
}

#endif

TEST(JxlTest, RoundtripNoGaborishNoAR) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, pool));

  CompressParams cparams;
  cparams.gaborish = Override::kOff;
  cparams.epf = 0;
  cparams.butteraugli_distance = 1.0;

  CodecInOut io2;
  EXPECT_LE(Roundtrip(&io, cparams, {}, pool, &io2), 40000u);
  EXPECT_THAT(ButteraugliDistance(io, io2, cparams.ba_params, GetJxlCms(),
                                  /*distmap=*/nullptr, pool),
              IsSlightlyBelow(2.0));
}

TEST(JxlTest, RoundtripSmallNoGaborish) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, pool));
  io.ShrinkTo(io.xsize() / 8, io.ysize() / 8);

  CompressParams cparams;
  cparams.gaborish = Override::kOff;
  cparams.butteraugli_distance = 1.0;

  CodecInOut io2;
  EXPECT_LE(Roundtrip(&io, cparams, {}, pool, &io2), 920u);
  EXPECT_THAT(ButteraugliDistance(io, io2, cparams.ba_params, GetJxlCms(),
                                  /*distmap=*/nullptr, pool),
              IsSlightlyBelow(1.2));
}

// Test header encoding of original bits per sample
TEST(JxlTest, RoundtripImageBundleOriginalBits) {
  ThreadPool* pool = nullptr;

  // Image does not matter, only io.metadata.m and io2.metadata.m are tested.
  Image3F image(1, 1);
  ZeroFillImage(&image);
  CodecInOut io;
  io.metadata.m.color_encoding = ColorEncoding::LinearSRGB();
  io.SetFromImage(std::move(image), ColorEncoding::LinearSRGB());

  CompressParams cparams;

  // Test unsigned integers from 1 to 32 bits
  for (uint32_t bit_depth = 1; bit_depth <= 32; bit_depth++) {
    if (bit_depth == 32) {
      // TODO(lode): allow testing 32, however the code below ends up in
      // enc_modular which does not support 32. We only want to test the header
      // encoding though, so try without modular.
      break;
    }

    io.metadata.m.SetUintSamples(bit_depth);
    CodecInOut io2;
    Roundtrip(&io, cparams, {}, pool, &io2);

    EXPECT_EQ(bit_depth, io2.metadata.m.bit_depth.bits_per_sample);
    EXPECT_FALSE(io2.metadata.m.bit_depth.floating_point_sample);
    EXPECT_EQ(0u, io2.metadata.m.bit_depth.exponent_bits_per_sample);
    EXPECT_EQ(0u, io2.metadata.m.GetAlphaBits());
  }

  // Test various existing and non-existing floating point formats
  for (uint32_t bit_depth = 8; bit_depth <= 32; bit_depth++) {
    if (bit_depth != 32) {
      // TODO: test other float types once they work
      break;
    }

    uint32_t exponent_bit_depth;
    if (bit_depth < 10) {
      exponent_bit_depth = 2;
    } else if (bit_depth < 12) {
      exponent_bit_depth = 3;
    } else if (bit_depth < 16) {
      exponent_bit_depth = 4;
    } else if (bit_depth < 20) {
      exponent_bit_depth = 5;
    } else if (bit_depth < 24) {
      exponent_bit_depth = 6;
    } else if (bit_depth < 28) {
      exponent_bit_depth = 7;
    } else {
      exponent_bit_depth = 8;
    }

    io.metadata.m.bit_depth.bits_per_sample = bit_depth;
    io.metadata.m.bit_depth.floating_point_sample = true;
    io.metadata.m.bit_depth.exponent_bits_per_sample = exponent_bit_depth;

    CodecInOut io2;
    Roundtrip(&io, cparams, {}, pool, &io2);

    EXPECT_EQ(bit_depth, io2.metadata.m.bit_depth.bits_per_sample);
    EXPECT_TRUE(io2.metadata.m.bit_depth.floating_point_sample);
    EXPECT_EQ(exponent_bit_depth,
              io2.metadata.m.bit_depth.exponent_bits_per_sample);
    EXPECT_EQ(0u, io2.metadata.m.GetAlphaBits());
  }
}

TEST(JxlTest, RoundtripGrayscale) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig = ReadTestData(
      "external/wesaturate/500px/cvo9xd_keong_macan_grayscale.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, pool));
  ASSERT_NE(io.xsize(), 0u);
  io.ShrinkTo(128, 128);
  EXPECT_TRUE(io.Main().IsGray());
  EXPECT_EQ(8u, io.metadata.m.bit_depth.bits_per_sample);
  EXPECT_FALSE(io.metadata.m.bit_depth.floating_point_sample);
  EXPECT_EQ(0u, io.metadata.m.bit_depth.exponent_bits_per_sample);
  EXPECT_TRUE(io.metadata.m.color_encoding.tf.IsSRGB());

  AuxOut* aux_out = nullptr;

  {
    CompressParams cparams;
    cparams.butteraugli_distance = 1.0;

    PaddedBytes compressed;
    EXPECT_TRUE(
        EncodeFile(cparams, &io, &compressed, GetJxlCms(), aux_out, pool));
    CodecInOut io2;
    EXPECT_TRUE(test::DecodeFile({}, compressed, &io2, pool));
    EXPECT_TRUE(io2.Main().IsGray());

    EXPECT_LE(compressed.size(), 7000u);
    EXPECT_THAT(ButteraugliDistance(io, io2, cparams.ba_params, GetJxlCms(),
                                    /*distmap=*/nullptr, pool),
                IsSlightlyBelow(1.6));
  }

  // Test with larger butteraugli distance and other settings enabled so
  // different jxl codepaths trigger.
  {
    CompressParams cparams;
    cparams.butteraugli_distance = 8.0;

    PaddedBytes compressed;
    EXPECT_TRUE(
        EncodeFile(cparams, &io, &compressed, GetJxlCms(), aux_out, pool));
    CodecInOut io2;
    EXPECT_TRUE(test::DecodeFile({}, compressed, &io2, pool));
    EXPECT_TRUE(io2.Main().IsGray());

    EXPECT_LE(compressed.size(), 1300u);
    EXPECT_THAT(ButteraugliDistance(io, io2, cparams.ba_params, GetJxlCms(),
                                    /*distmap=*/nullptr, pool),
                IsSlightlyBelow(8.0));
  }

  {
    CompressParams cparams;
    cparams.butteraugli_distance = 1.0;

    PaddedBytes compressed;
    EXPECT_TRUE(
        EncodeFile(cparams, &io, &compressed, GetJxlCms(), aux_out, pool));

    CodecInOut io2;
    extras::JXLDecompressParams dparams;
    dparams.color_space = "RGB_D65_SRG_Rel_SRG";
    EXPECT_TRUE(test::DecodeFile(dparams, compressed, &io2, pool));
    EXPECT_FALSE(io2.Main().IsGray());

    EXPECT_LE(compressed.size(), 7000u);
    EXPECT_THAT(ButteraugliDistance(io, io2, cparams.ba_params, GetJxlCms(),
                                    /*distmap=*/nullptr, pool),
                IsSlightlyBelow(1.6));
  }
}

}  // namespace
}  // namespace jxl
