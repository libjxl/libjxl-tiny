// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "lib/jxl/render_pipeline/render_pipeline.h"

#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "lib/extras/codec.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/dec_frame.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/fake_parallel_runner_testonly.h"
#include "lib/jxl/icc_codec.h"
#include "lib/jxl/image_test_utils.h"
#include "lib/jxl/jpeg/enc_jpeg_data.h"
#include "lib/jxl/render_pipeline/test_render_pipeline_stages.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testdata.h"

namespace jxl {
namespace {

Status DecodeFile(const Span<const uint8_t> file, bool use_slow_pipeline,
                  CodecInOut* io, ThreadPool* pool) {
  Status ret = true;
  {
    BitReader reader(file);
    BitReaderScopedCloser reader_closer(&reader, &ret);
    JXL_RETURN_IF_ERROR(reader.ReadFixedBits<16>() == 0x0AFF);
    JXL_RETURN_IF_ERROR(ReadSizeHeader(&reader, &io->metadata.size));
    JXL_RETURN_IF_ERROR(ReadImageMetadata(&reader, &io->metadata.m));
    io->metadata.transform_data.nonserialized_xyb_encoded =
        io->metadata.m.xyb_encoded;
    JXL_RETURN_IF_ERROR(Bundle::Read(&reader, &io->metadata.transform_data));
    size_t xsize = io->metadata.xsize();
    size_t ysize = io->metadata.ysize();
    JXL_RETURN_IF_ERROR(VerifyDimensions(&io->constraints, xsize, ysize));
    if (io->metadata.m.color_encoding.WantICC()) {
      PaddedBytes icc;
      JXL_RETURN_IF_ERROR(ReadICC(&reader, &icc));
      JXL_RETURN_IF_ERROR(io->metadata.m.color_encoding.SetICC(std::move(icc)));
    }
    PassesDecoderState dec_state;
    JXL_RETURN_IF_ERROR(
        dec_state.output_encoding_info.SetFromMetadata(io->metadata));
    JXL_RETURN_IF_ERROR(reader.JumpToByteBoundary());
    io->frames.clear();
    do {
      io->frames.emplace_back(&io->metadata.m);
      // Skip frames that are not displayed.
      do {
        size_t frame_start = reader.TotalBitsConsumed() / kBitsPerByte;
        size_t size_left = file.size() - frame_start;
        JXL_RETURN_IF_ERROR(
            DecodeFrame(&dec_state, pool, file.data() + frame_start, size_left,
                        &io->frames.back(), io->metadata, use_slow_pipeline));
        reader.SkipBits(io->frames.back().decoded_bytes() * kBitsPerByte);
      } while (dec_state.shared->frame_header.frame_type !=
                   FrameType::kRegularFrame &&
               dec_state.shared->frame_header.frame_type !=
                   FrameType::kSkipProgressive);
    } while (!dec_state.shared->frame_header.is_last);

    if (io->frames.empty()) return JXL_FAILURE("Not enough data.");

    if (reader.TotalBitsConsumed() != file.size() * kBitsPerByte) {
      return JXL_FAILURE("Reader position not at EOF.");
    }
    if (!reader.AllReadsWithinBounds()) {
      return JXL_FAILURE("Reader out of bounds read.");
    }
    io->CheckMetadata();
    // reader is closed here.
  }
  return ret;
}

TEST(RenderPipelineTest, Build) {
  RenderPipeline::Builder builder(/*num_c=*/1);
  builder.AddStage(jxl::make_unique<UpsampleXSlowStage>());
  builder.AddStage(jxl::make_unique<UpsampleYSlowStage>());
  builder.AddStage(jxl::make_unique<Check0FinalStage>());
  builder.UseSimpleImplementation();
  FrameDimensions frame_dimensions;
  frame_dimensions.Set(/*xsize=*/1024, /*ysize=*/1024, /*group_size_shift=*/0,
                       /*max_hshift=*/0, /*max_vshift=*/0,
                       /*modular_mode=*/false, /*upsampling=*/1);
  std::move(builder).Finalize(frame_dimensions);
}

TEST(RenderPipelineTest, CallAllGroups) {
  RenderPipeline::Builder builder(/*num_c=*/1);
  builder.AddStage(jxl::make_unique<UpsampleXSlowStage>());
  builder.AddStage(jxl::make_unique<UpsampleYSlowStage>());
  builder.AddStage(jxl::make_unique<Check0FinalStage>());
  builder.UseSimpleImplementation();
  FrameDimensions frame_dimensions;
  frame_dimensions.Set(/*xsize=*/1024, /*ysize=*/1024, /*group_size_shift=*/0,
                       /*max_hshift=*/0, /*max_vshift=*/0,
                       /*modular_mode=*/false, /*upsampling=*/1);
  auto pipeline = std::move(builder).Finalize(frame_dimensions);
  ASSERT_TRUE(pipeline->PrepareForThreads(1, /*use_group_ids=*/false));

  for (size_t i = 0; i < frame_dimensions.num_groups; i++) {
    auto input_buffers = pipeline->GetInputBuffers(i, 0);
    FillPlane(0.0f, input_buffers.GetBuffer(0).first,
              input_buffers.GetBuffer(0).second);
    input_buffers.Done();
  }

  EXPECT_EQ(pipeline->PassesWithAllInput(), 1);
}

TEST(RenderPipelineTest, BuildFast) {
  RenderPipeline::Builder builder(/*num_c=*/1);
  builder.AddStage(jxl::make_unique<UpsampleXSlowStage>());
  builder.AddStage(jxl::make_unique<UpsampleYSlowStage>());
  builder.AddStage(jxl::make_unique<Check0FinalStage>());
  FrameDimensions frame_dimensions;
  frame_dimensions.Set(/*xsize=*/1024, /*ysize=*/1024, /*group_size_shift=*/0,
                       /*max_hshift=*/0, /*max_vshift=*/0,
                       /*modular_mode=*/false, /*upsampling=*/1);
  std::move(builder).Finalize(frame_dimensions);
}

TEST(RenderPipelineTest, CallAllGroupsFast) {
  RenderPipeline::Builder builder(/*num_c=*/1);
  builder.AddStage(jxl::make_unique<UpsampleXSlowStage>());
  builder.AddStage(jxl::make_unique<UpsampleYSlowStage>());
  builder.AddStage(jxl::make_unique<Check0FinalStage>());
  builder.UseSimpleImplementation();
  FrameDimensions frame_dimensions;
  frame_dimensions.Set(/*xsize=*/1024, /*ysize=*/1024, /*group_size_shift=*/0,
                       /*max_hshift=*/0, /*max_vshift=*/0,
                       /*modular_mode=*/false, /*upsampling=*/1);
  auto pipeline = std::move(builder).Finalize(frame_dimensions);
  ASSERT_TRUE(pipeline->PrepareForThreads(1, /*use_group_ids=*/false));

  for (size_t i = 0; i < frame_dimensions.num_groups; i++) {
    auto input_buffers = pipeline->GetInputBuffers(i, 0);
    FillPlane(0.0f, input_buffers.GetBuffer(0).first,
              input_buffers.GetBuffer(0).second);
    input_buffers.Done();
  }

  EXPECT_EQ(pipeline->PassesWithAllInput(), 1);
}

struct RenderPipelineTestInputSettings {
  // Input image.
  std::string input_path;
  size_t xsize, ysize;
  // Encoding settings.
  CompressParams cparams;
  // Short name for the encoder settings.
  std::string cparams_descr;

  bool add_spot_color = false;
};

class RenderPipelineTestParam
    : public ::testing::TestWithParam<RenderPipelineTestInputSettings> {};

TEST_P(RenderPipelineTestParam, PipelineTest) {
  RenderPipelineTestInputSettings config = GetParam();

  // Use a parallel runner that randomly shuffles tasks to detect possible
  // border handling bugs.
  FakeParallelRunner fake_pool(/*order_seed=*/123, /*num_threads=*/8);
  ThreadPool pool(&JxlFakeParallelRunner, &fake_pool);
  const PaddedBytes orig = ReadTestData(config.input_path);

  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, &pool));
  io.ShrinkTo(config.xsize, config.ysize);

  if (config.add_spot_color) {
    jxl::ImageF spot(config.xsize, config.ysize);
    jxl::ZeroFillImage(&spot);

    for (size_t y = 0; y < config.ysize; y++) {
      float* JXL_RESTRICT row = spot.Row(y);
      for (size_t x = 0; x < config.xsize; x++) {
        row[x] = ((x ^ y) & 255) * (1.f / 255.f);
      }
    }
    ExtraChannelInfo info;
    info.bit_depth.bits_per_sample = 8;
    info.dim_shift = 0;
    info.type = jxl::ExtraChannel::kSpotColor;
    info.spot_color[0] = 0.5f;
    info.spot_color[1] = 0.2f;
    info.spot_color[2] = 1.f;
    info.spot_color[3] = 0.5f;

    io.metadata.m.extra_channel_info.push_back(info);
    std::vector<jxl::ImageF> ec;
    ec.push_back(std::move(spot));
    io.frames[0].SetExtraChannels(std::move(ec));
  }

  PaddedBytes compressed;

  PassesEncoderState enc_state;
  ASSERT_TRUE(EncodeFile(config.cparams, &io, &enc_state, &compressed,
                         GetJxlCms(), /*aux_out=*/nullptr, &pool));


  CodecInOut io_default;
  ASSERT_TRUE(DecodeFile(Span<const uint8_t>(compressed),
                         /*use_slow_pipeline=*/false, &io_default, &pool));
  CodecInOut io_slow_pipeline;
  ASSERT_TRUE(DecodeFile(Span<const uint8_t>(compressed),
                         /*use_slow_pipeline=*/true, &io_slow_pipeline, &pool));

  ASSERT_EQ(io_default.frames.size(), io_slow_pipeline.frames.size());
  for (size_t i = 0; i < io_default.frames.size(); i++) {
#if JXL_HIGH_PRECISION
    constexpr float kMaxError = 1e-5;
#else
    constexpr float kMaxError = 1e-4;
#endif
    Image3F def = std::move(*io_default.frames[i].color());
    Image3F pip = std::move(*io_slow_pipeline.frames[i].color());
    VerifyRelativeError(pip, def, kMaxError, kMaxError);
    for (size_t ec = 0; ec < io_default.frames[i].extra_channels().size();
         ec++) {
      VerifyRelativeError(io_slow_pipeline.frames[i].extra_channels()[ec],
                          io_default.frames[i].extra_channels()[ec], kMaxError,
                          kMaxError);
    }
  }
}

std::vector<RenderPipelineTestInputSettings> GeneratePipelineTests() {
  std::vector<RenderPipelineTestInputSettings> all_tests;

  std::pair<size_t, size_t> sizes[] = {
      {3, 8}, {128, 128}, {256, 256}, {258, 258}, {533, 401}, {777, 777},
  };

  for (auto size : sizes) {
    RenderPipelineTestInputSettings settings;
    settings.input_path = "jxl/flower/flower.png";
    settings.xsize = size.first;
    settings.ysize = size.second;

    // Base settings.
    settings.cparams.butteraugli_distance = 1.0;
    settings.cparams.gaborish = Override::kOff;
    settings.cparams.epf = 0;
    settings.cparams.color_transform = ColorTransform::kXYB;

    {
      auto s = settings;
      s.cparams_descr = "NoGabNoEpfNoPatches";
      all_tests.push_back(s);
    }

    {
      auto s = settings;
      s.cparams.color_transform = ColorTransform::kNone;
      s.cparams_descr = "NoGabNoEpfNoPatchesNoXYB";
      all_tests.push_back(s);
    }

    {
      auto s = settings;
      s.cparams.gaborish = Override::kOn;
      s.cparams_descr = "GabNoEpfNoPatches";
      all_tests.push_back(s);
    }

    {
      auto s = settings;
      s.cparams.epf = 1;
      s.cparams_descr = "NoGabEpf1NoPatches";
      all_tests.push_back(s);
    }

    {
      auto s = settings;
      s.cparams.epf = 2;
      s.cparams_descr = "NoGabEpf2NoPatches";
      all_tests.push_back(s);
    }

    {
      auto s = settings;
      s.cparams_descr = "ModularLossless";
      s.cparams.modular_mode = true;
      s.cparams.butteraugli_distance = 0;
      all_tests.push_back(s);
    }

    {
      auto s = settings;
      s.cparams_descr = "ModularLossy";
      s.cparams.modular_mode = true;
      s.cparams.butteraugli_distance = 1.f;
      all_tests.push_back(s);
    }

    {
      auto s = settings;
      s.input_path = "jxl/flower/flower_alpha.png";
      s.cparams_descr = "AlphaVarDCT";
      all_tests.push_back(s);
    }

    {
      auto s = settings;
      s.cparams.modular_mode = true;
      s.cparams.butteraugli_distance = 0;
      s.input_path = "jxl/flower/flower_alpha.png";
      s.cparams_descr = "AlphaLossless";
      all_tests.push_back(s);
    }

    {
      auto s = settings;
      s.cparams_descr = "SpotColor";
      s.add_spot_color = true;
      all_tests.push_back(s);
    }
  }

  {
    RenderPipelineTestInputSettings settings;
    settings.input_path = "jxl/grayscale_patches.png";
    settings.xsize = 1011;
    settings.ysize = 277;
    settings.cparams_descr = "Patches";
    all_tests.push_back(settings);
  }

  return all_tests;
}

std::ostream& operator<<(std::ostream& os,
                         const RenderPipelineTestInputSettings& c) {
  std::string filename;
  size_t pos = c.input_path.find_last_of('/');
  if (pos == std::string::npos) {
    filename = c.input_path;
  } else {
    filename = c.input_path.substr(pos + 1);
  }
  std::replace_if(
      filename.begin(), filename.end(), [](char c) { return !isalnum(c); },
      '_');
  os << filename << "_" << c.xsize << "x" << c.ysize << "_" << c.cparams_descr;
  return os;
}

std::string PipelineTestDescription(
    const testing::TestParamInfo<RenderPipelineTestParam::ParamType>& info) {
  std::stringstream name;
  name << info.param;
  return name.str();
}

JXL_GTEST_INSTANTIATE_TEST_SUITE_P(RenderPipelineTest, RenderPipelineTestParam,
                                   testing::ValuesIn(GeneratePipelineTests()),
                                   PipelineTestDescription);

TEST(RenderPipelineDecodingTest, Animation) {
  FakeParallelRunner fake_pool(/*order_seed=*/123, /*num_threads=*/8);
  ThreadPool pool(&JxlFakeParallelRunner, &fake_pool);

  PaddedBytes compressed =
      ReadTestData("jxl/blending/cropped_traffic_light.jxl");

  CodecInOut io_default;
  ASSERT_TRUE(DecodeFile(Span<const uint8_t>(compressed),
                         /*use_slow_pipeline=*/false, &io_default, &pool));
  CodecInOut io_slow_pipeline;
  ASSERT_TRUE(DecodeFile(Span<const uint8_t>(compressed),
                         /*use_slow_pipeline=*/true, &io_slow_pipeline, &pool));

  ASSERT_EQ(io_default.frames.size(), io_slow_pipeline.frames.size());
  for (size_t i = 0; i < io_default.frames.size(); i++) {
#if JXL_HIGH_PRECISION
    constexpr float kMaxError = 1e-5;
#else
    constexpr float kMaxError = 1e-4;
#endif

    Image3F fast_pipeline = std::move(*io_default.frames[i].color());
    Image3F slow_pipeline = std::move(*io_slow_pipeline.frames[i].color());
    VerifyRelativeError(slow_pipeline, fast_pipeline, kMaxError, kMaxError);
    for (size_t ec = 0; ec < io_default.frames[i].extra_channels().size();
         ec++) {
      VerifyRelativeError(io_slow_pipeline.frames[i].extra_channels()[ec],
                          io_default.frames[i].extra_channels()[ec], kMaxError,
                          kMaxError);
    }
  }
}

}  // namespace
}  // namespace jxl
