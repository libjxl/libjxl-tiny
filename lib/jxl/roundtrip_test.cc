// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include <cmath>  // std::abs
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "gtest/gtest.h"
#include "jxl/codestream_header.h"
#include "jxl/decode.h"
#include "jxl/decode_cxx.h"
#include "jxl/encode.h"
#include "jxl/encode_cxx.h"
#include "jxl/types.h"
#include "lib/extras/codec.h"
#include "lib/jxl/dec_external_image.h"
#include "lib/jxl/enc_butteraugli_comparator.h"
#include "lib/jxl/enc_comparator.h"
#include "lib/jxl/enc_external_image.h"
#include "lib/jxl/encode_internal.h"
#include "lib/jxl/icc_codec.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testdata.h"

namespace {

// Converts a test image to a CodecInOut.
// icc_profile can be empty to automatically deduce profile from the pixel
// format, or filled in to force this ICC profile
jxl::CodecInOut ConvertTestImage(const std::vector<uint8_t>& buf,
                                 const size_t xsize, const size_t ysize,
                                 const JxlPixelFormat& pixel_format,
                                 const jxl::PaddedBytes& icc_profile) {
  jxl::CodecInOut io;
  io.SetSize(xsize, ysize);

  bool is_gray = pixel_format.num_channels < 3;
  bool has_alpha =
      pixel_format.num_channels == 2 || pixel_format.num_channels == 4;

  io.metadata.m.color_encoding.SetColorSpace(is_gray ? jxl::ColorSpace::kGray
                                                     : jxl::ColorSpace::kRGB);
  if (has_alpha) {
    // Note: alpha > 16 not yet supported by the C++ codec
    switch (pixel_format.data_type) {
      case JXL_TYPE_UINT8:
        io.metadata.m.SetAlphaBits(8);
        break;
      case JXL_TYPE_UINT16:
      case JXL_TYPE_FLOAT:
      case JXL_TYPE_FLOAT16:
        io.metadata.m.SetAlphaBits(16);
        break;
      default:
        EXPECT_TRUE(false) << "Roundtrip tests for data type "
                           << pixel_format.data_type << " not yet implemented.";
    }
  }
  size_t bitdepth = 0;
  bool float_in = false;
  switch (pixel_format.data_type) {
    case JXL_TYPE_FLOAT:
      bitdepth = 32;
      float_in = true;
      io.metadata.m.SetFloat32Samples();
      break;
    case JXL_TYPE_FLOAT16:
      bitdepth = 16;
      float_in = true;
      io.metadata.m.SetFloat16Samples();
      break;
    case JXL_TYPE_UINT8:
      bitdepth = 8;
      float_in = false;
      io.metadata.m.SetUintSamples(8);
      break;
    case JXL_TYPE_UINT16:
      bitdepth = 16;
      float_in = false;
      io.metadata.m.SetUintSamples(16);
      break;
    default:
      EXPECT_TRUE(false) << "Roundtrip tests for data type "
                         << pixel_format.data_type << " not yet implemented.";
  }
  jxl::ColorEncoding color_encoding;
  if (!icc_profile.empty()) {
    jxl::PaddedBytes icc_profile_copy(icc_profile);
    EXPECT_TRUE(color_encoding.SetICC(std::move(icc_profile_copy)));
  } else if (pixel_format.data_type == JXL_TYPE_FLOAT) {
    color_encoding = jxl::ColorEncoding::LinearSRGB(is_gray);
  } else {
    color_encoding = jxl::ColorEncoding::SRGB(is_gray);
  }
  EXPECT_TRUE(ConvertFromExternal(
      jxl::Span<const uint8_t>(buf.data(), buf.size()), xsize, ysize,
      color_encoding, pixel_format.num_channels,
      /*alpha_is_premultiplied=*/false,
      /*bits_per_sample=*/bitdepth, pixel_format.endianness,
      /*pool=*/nullptr, &io.Main(), float_in,
      /*align=*/0));
  return io;
}

template <typename T>
T ConvertTestPixel(const float val);

template <>
float ConvertTestPixel<float>(const float val) {
  return val;
}

template <>
uint16_t ConvertTestPixel<uint16_t>(const float val) {
  return (uint16_t)(val * UINT16_MAX);
}

template <>
uint8_t ConvertTestPixel<uint8_t>(const float val) {
  return (uint8_t)(val * UINT8_MAX);
}

// Returns a test image.
template <typename T>
std::vector<uint8_t> GetTestImage(const size_t xsize, const size_t ysize,
                                  const JxlPixelFormat& pixel_format) {
  std::vector<T> pixels(xsize * ysize * pixel_format.num_channels);
  for (size_t y = 0; y < ysize; y++) {
    for (size_t x = 0; x < xsize; x++) {
      for (size_t chan = 0; chan < pixel_format.num_channels; chan++) {
        float val;
        switch (chan % 4) {
          case 0:
            val = static_cast<float>(y) / static_cast<float>(ysize);
            break;
          case 1:
            val = static_cast<float>(x) / static_cast<float>(xsize);
            break;
          case 2:
            val = static_cast<float>(x + y) / static_cast<float>(xsize + ysize);
            break;
          case 3:
            val = static_cast<float>(x * y) / static_cast<float>(xsize * ysize);
            break;
        }
        pixels[(y * xsize + x) * pixel_format.num_channels + chan] =
            ConvertTestPixel<T>(val);
      }
    }
  }
  std::vector<uint8_t> bytes(pixels.size() * sizeof(T));
  memcpy(bytes.data(), pixels.data(), sizeof(T) * pixels.size());
  return bytes;
}

void EncodeWithEncoder(JxlEncoder* enc, std::vector<uint8_t>* compressed) {
  compressed->resize(64);
  uint8_t* next_out = compressed->data();
  size_t avail_out = compressed->size() - (next_out - compressed->data());
  JxlEncoderStatus process_result = JXL_ENC_NEED_MORE_OUTPUT;
  while (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
    process_result = JxlEncoderProcessOutput(enc, &next_out, &avail_out);
    if (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
      size_t offset = next_out - compressed->data();
      compressed->resize(compressed->size() * 2);
      next_out = compressed->data() + offset;
      avail_out = compressed->size() - offset;
    }
  }
  compressed->resize(next_out - compressed->data());
  EXPECT_EQ(JXL_ENC_SUCCESS, process_result);
}

// Generates some pixels using using some dimensions and pixel_format,
// compresses them, and verifies that the decoded version is similar to the
// original pixels.
// TODO(firsching): change this to be a parameterized test, like in
// decode_test.cc
template <typename T>
void VerifyRoundtripCompression(
    const size_t xsize, const size_t ysize,
    const JxlPixelFormat& input_pixel_format,
    const JxlPixelFormat& output_pixel_format, const bool use_container,
    const std::vector<std::pair<JxlExtraChannelType, std::string>>&
        extra_channels = {}) {
  size_t orig_xsize = xsize;
  size_t orig_ysize = ysize;

  JxlPixelFormat extra_channel_pixel_format = input_pixel_format;
  extra_channel_pixel_format.num_channels = 1;
  const std::vector<uint8_t> extra_channel_bytes =
      GetTestImage<T>(xsize, ysize, extra_channel_pixel_format);
  const std::vector<uint8_t> original_bytes =
      GetTestImage<T>(orig_xsize, orig_ysize, input_pixel_format);
  jxl::CodecInOut original_io = ConvertTestImage(
      original_bytes, orig_xsize, orig_ysize, input_pixel_format, {});

  JxlEncoder* enc = JxlEncoderCreate(nullptr);
  EXPECT_NE(nullptr, enc);
  EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetCodestreamLevel(enc, 10));
  EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderUseContainer(enc, use_container));
  JxlBasicInfo basic_info;
  jxl::test::JxlBasicInfoSetFromPixelFormat(&basic_info, &input_pixel_format);
  basic_info.xsize = xsize;
  basic_info.ysize = ysize;
  basic_info.uses_original_profile = false;
  uint32_t num_channels = input_pixel_format.num_channels;
  size_t has_interleaved_alpha = num_channels == 2 || num_channels == 4;
  JxlPixelFormat output_pixel_format_with_extra_channel_alpha =
      output_pixel_format;

  // In the case where we have an alpha channel, but it is provided as an extra
  // channel and not interleaved, we do two things here:
  // 1. modify the original_io to have the correct alpha channel
  // 2. change the output_format_with_extra_alpha to have an alpha channel
  bool alpha_in_extra_channels_vector = false;
  for (const auto& extra_channel : extra_channels) {
    if (extra_channel.first == JXL_CHANNEL_ALPHA) {
      alpha_in_extra_channels_vector = true;
    }
  }
  if (alpha_in_extra_channels_vector && !has_interleaved_alpha) {
    jxl::ImageF alpha_channel(xsize, ysize);

    EXPECT_EQ(
        jxl::ConvertFromExternal(
            jxl::Span<const uint8_t>(extra_channel_bytes.data(),
                                     extra_channel_bytes.size()),
            xsize, ysize, basic_info.bits_per_sample,
            input_pixel_format.endianness, /*pool=*/nullptr, &alpha_channel,
            /*float_in=*/input_pixel_format.data_type == JXL_TYPE_FLOAT,
            /*align=*/0),
        true);

    original_io.metadata.m.SetAlphaBits(basic_info.bits_per_sample);
    original_io.Main().SetAlpha(std::move(alpha_channel), false);
    output_pixel_format_with_extra_channel_alpha.num_channels++;
  }
  // Those are the num_extra_channels including a potential alpha channel.
  basic_info.num_extra_channels = extra_channels.size() + has_interleaved_alpha;
  EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetBasicInfo(enc, &basic_info));
  EXPECT_EQ(enc->metadata.m.num_extra_channels,
            extra_channels.size() + has_interleaved_alpha);
  JxlColorEncoding color_encoding;
  if (input_pixel_format.data_type == JXL_TYPE_FLOAT) {
    JxlColorEncodingSetToLinearSRGB(
        &color_encoding,
        /*is_gray=*/input_pixel_format.num_channels < 3);
  } else {
    JxlColorEncodingSetToSRGB(&color_encoding,
                              /*is_gray=*/input_pixel_format.num_channels < 3);
  }

  std::vector<JxlExtraChannelInfo> channel_infos;
  for (const auto& extra_channel : extra_channels) {
    auto channel_type = extra_channel.first;
    JxlExtraChannelInfo channel_info;
    JxlEncoderInitExtraChannelInfo(channel_type, &channel_info);
    channel_info.bits_per_sample = 8;
    channel_info.exponent_bits_per_sample = 0;
    channel_infos.push_back(channel_info);
  }
  for (size_t index = 0; index < channel_infos.size(); index++) {
    EXPECT_EQ(JXL_ENC_SUCCESS,
              JxlEncoderSetExtraChannelInfo(enc, index + has_interleaved_alpha,
                                            &channel_infos[index]));
    std::string name = extra_channels[index].second;
    EXPECT_EQ(JXL_ENC_SUCCESS,
              JxlEncoderSetExtraChannelName(enc, index + has_interleaved_alpha,
                                            name.c_str(), name.length()));
  }
  EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetColorEncoding(enc, &color_encoding));
  JxlEncoderFrameSettings* frame_settings =
      JxlEncoderFrameSettingsCreate(enc, nullptr);
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderAddImageFrame(frame_settings, &input_pixel_format,
                                    (void*)original_bytes.data(),
                                    original_bytes.size()));
  EXPECT_EQ(frame_settings->enc->input_queue.back()
                .frame->frame.extra_channels()
                .size(),
            has_interleaved_alpha + extra_channels.size());
  EXPECT_EQ(frame_settings->enc->input_queue.empty(), false);
  for (size_t index = 0; index < channel_infos.size(); index++) {
    EXPECT_EQ(JXL_ENC_SUCCESS,
              JxlEncoderSetExtraChannelBuffer(
                  frame_settings, &input_pixel_format,
                  (void*)extra_channel_bytes.data(), extra_channel_bytes.size(),
                  index + has_interleaved_alpha));
  }
  JxlEncoderCloseInput(enc);
  EXPECT_EQ(frame_settings->enc->input_queue.back()
                .frame->frame.extra_channels()
                .size(),
            has_interleaved_alpha + extra_channels.size());
  std::vector<uint8_t> compressed;
  EncodeWithEncoder(enc, &compressed);
  JxlEncoderDestroy(enc);

  JxlDecoder* dec = JxlDecoderCreate(nullptr);
  EXPECT_NE(nullptr, dec);

  const uint8_t* next_in = compressed.data();
  size_t avail_in = compressed.size();

  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO |
                                               JXL_DEC_COLOR_ENCODING |
                                               JXL_DEC_FULL_IMAGE));

  JxlDecoderSetInput(dec, next_in, avail_in);
  EXPECT_EQ(JXL_DEC_BASIC_INFO, JxlDecoderProcessInput(dec));
  size_t buffer_size;
  EXPECT_EQ(
      JXL_DEC_SUCCESS,
      JxlDecoderImageOutBufferSize(
          dec, &output_pixel_format_with_extra_channel_alpha, &buffer_size));
  if (&input_pixel_format == &output_pixel_format_with_extra_channel_alpha) {
    EXPECT_EQ(buffer_size, original_bytes.size());
  }

  JxlBasicInfo info;
  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderGetBasicInfo(dec, &info));
  EXPECT_EQ(xsize, info.xsize);
  EXPECT_EQ(ysize, info.ysize);
  EXPECT_EQ(extra_channels.size() + has_interleaved_alpha,
            info.num_extra_channels);

  EXPECT_EQ(JXL_DEC_COLOR_ENCODING, JxlDecoderProcessInput(dec));

  size_t icc_profile_size;
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderGetICCProfileSize(
                dec, &output_pixel_format_with_extra_channel_alpha,
                JXL_COLOR_PROFILE_TARGET_DATA, &icc_profile_size));
  jxl::PaddedBytes icc_profile(icc_profile_size);
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderGetColorAsICCProfile(
                dec, &output_pixel_format, JXL_COLOR_PROFILE_TARGET_DATA,
                icc_profile.data(), icc_profile.size()));

  std::vector<uint8_t> decoded_bytes(buffer_size);

  EXPECT_EQ(JXL_DEC_NEED_IMAGE_OUT_BUFFER, JxlDecoderProcessInput(dec));

  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSetImageOutBuffer(
                dec, &output_pixel_format_with_extra_channel_alpha,
                decoded_bytes.data(), decoded_bytes.size()));
  std::vector<std::vector<uint8_t>> extra_channel_decoded_bytes(
      info.num_extra_channels - has_interleaved_alpha);

  for (size_t index = has_interleaved_alpha; index < info.num_extra_channels;
       index++) {
    JxlExtraChannelInfo channel_info;
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderGetExtraChannelInfo(dec, index, &channel_info));
    EXPECT_EQ(channel_info.type,
              extra_channels[index - has_interleaved_alpha].first);
    std::string input_name =
        extra_channels[index - has_interleaved_alpha].second;
    const size_t name_length = channel_info.name_length;
    EXPECT_EQ(input_name.size(), name_length);
    std::vector<char> output_name(name_length + 1);
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderGetExtraChannelName(dec, index, output_name.data(),
                                            output_name.size()));
    EXPECT_EQ(0,
              memcmp(input_name.data(), output_name.data(), input_name.size()));
    size_t extra_buffer_size;
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderExtraChannelBufferSize(dec, &output_pixel_format,
                                               &extra_buffer_size, index));
    std::vector<uint8_t> extra_decoded_bytes(extra_buffer_size);
    extra_channel_decoded_bytes[index - has_interleaved_alpha] =
        std::move(extra_decoded_bytes);
    EXPECT_EQ(
        JXL_DEC_SUCCESS,
        JxlDecoderSetExtraChannelBuffer(
            dec, &output_pixel_format,
            extra_channel_decoded_bytes[index - has_interleaved_alpha].data(),
            extra_channel_decoded_bytes[index - has_interleaved_alpha].size(),
            index));
  }
  EXPECT_EQ(JXL_DEC_FULL_IMAGE, JxlDecoderProcessInput(dec));
  // Check if there are no further errors after getting the full image, e.g.
  // check that the final codestream box is actually marked as last.
  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderProcessInput(dec));

  JxlDecoderDestroy(dec);

  jxl::CodecInOut decoded_io = ConvertTestImage(
      decoded_bytes, xsize, ysize, output_pixel_format_with_extra_channel_alpha,
      icc_profile);

  jxl::ButteraugliParams ba;
  float butteraugli_score =
      ButteraugliDistance(original_io, decoded_io, ba, jxl::GetJxlCms(),
                          /*distmap=*/nullptr, nullptr);
  EXPECT_LE(butteraugli_score, 2.0f);
  for (auto& extra_channel : extra_channel_decoded_bytes) {
    EXPECT_EQ(extra_channel.size(), extra_channel_bytes.size());
  }
}

}  // namespace

TEST(RoundtripTest, FloatFrameRoundtripTest) {
  std::vector<std::vector<std::pair<JxlExtraChannelType, std::string>>>
      extra_channels_cases = {{}};
  for (int use_container = 0; use_container < 2; use_container++) {
    for (uint32_t num_channels = 1; num_channels < 4; num_channels++) {
      for (auto& extra_channels : extra_channels_cases) {
        uint32_t has_alpha = static_cast<uint32_t>(num_channels % 2 == 0);
        uint32_t total_extra_channels = has_alpha + extra_channels.size();
        // There's no support (yet) for lossless extra float
        // channels, so we don't test it.
        if (total_extra_channels == 0) {
          JxlPixelFormat pixel_format = JxlPixelFormat{
              num_channels, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
          VerifyRoundtripCompression<float>(63, 129, pixel_format, pixel_format,
                                            (bool)use_container,
                                            extra_channels);
        }
      }
    }
  }
}

TEST(RoundtripTest, Uint16FrameRoundtripTest) {
  for (int use_container = 0; use_container < 2; use_container++) {
    JxlPixelFormat pixel_format =
        JxlPixelFormat{3, JXL_TYPE_UINT16, JXL_NATIVE_ENDIAN, 0};
    VerifyRoundtripCompression<uint16_t>(63, 129, pixel_format, pixel_format,
                                         (bool)use_container);
  }
}

TEST(RoundtripTest, Uint8FrameRoundtripTest) {
  for (int use_container = 0; use_container < 2; use_container++) {
    JxlPixelFormat pixel_format =
        JxlPixelFormat{3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
    VerifyRoundtripCompression<uint8_t>(63, 129, pixel_format, pixel_format,
                                        (bool)use_container);
  }
}

TEST(RoundtripTest, TestNonlinearSrgbAsXybEncoded) {
  for (int use_container = 0; use_container < 2; use_container++) {
    JxlPixelFormat pixel_format_in =
        JxlPixelFormat{3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
    JxlPixelFormat pixel_format_out =
        JxlPixelFormat{3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
    VerifyRoundtripCompression<uint8_t>(
        63, 129, pixel_format_in, pixel_format_out, (bool)use_container, {});
  }
}
