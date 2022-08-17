// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef LIB_JXL_ENC_PARAMS_H_
#define LIB_JXL_ENC_PARAMS_H_

// Parameters and flags that govern JXL compression.

#include <stddef.h>
#include <stdint.h>

#include <string>

#include "lib/jxl/base/override.h"
#include "lib/jxl/butteraugli/butteraugli.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/modular/transform/transform.h"

namespace jxl {

enum class SpeedTier {
  // Turns on FindBestQuantizationHQ loop. Equivalent to "guetzli" mode.
  kTortoise = 1,
  // Turns on FindBestQuantization butteraugli loop.
  kKitten = 2,
  // Turns on dots, patches, and spline detection by default, as well as full
  // context clustering. Default.
  kSquirrel = 3,
  // Turns on error diffusion and full AC strategy heuristics. Equivalent to
  // "fast" mode.
  kWombat = 4,
  // Turns on gaborish by default, non-default cmap, initial quant field.
  kHare = 5,
  // Turns on simple heuristics for AC strategy, quant field, and clustering;
  // also enables coefficient reordering.
  kCheetah = 6,
  // Turns off most encoder features. Does context clustering.
  // Modular: uses fixed tree with Weighted predictor.
  kFalcon = 7,
  // Currently fastest possible setting for VarDCT.
  // Modular: uses fixed tree with Gradient predictor.
  kThunder = 8,
  // VarDCT: same as kThunder.
  // Modular: no tree, Gradient predictor, fast histograms
  kLightning = 9
};

inline bool ParseSpeedTier(const std::string& s, SpeedTier* out) {
  if (s == "lightning") {
    *out = SpeedTier::kLightning;
    return true;
  } else if (s == "thunder") {
    *out = SpeedTier::kThunder;
    return true;
  } else if (s == "falcon") {
    *out = SpeedTier::kFalcon;
    return true;
  } else if (s == "cheetah") {
    *out = SpeedTier::kCheetah;
    return true;
  } else if (s == "hare") {
    *out = SpeedTier::kHare;
    return true;
  } else if (s == "fast" || s == "wombat") {
    *out = SpeedTier::kWombat;
    return true;
  } else if (s == "squirrel") {
    *out = SpeedTier::kSquirrel;
    return true;
  } else if (s == "kitten") {
    *out = SpeedTier::kKitten;
    return true;
  } else if (s == "guetzli" || s == "tortoise") {
    *out = SpeedTier::kTortoise;
    return true;
  }
  size_t st = 10 - static_cast<size_t>(strtoull(s.c_str(), nullptr, 0));
  if (st <= static_cast<size_t>(SpeedTier::kLightning) &&
      st >= static_cast<size_t>(SpeedTier::kTortoise)) {
    *out = SpeedTier(st);
    return true;
  }
  return false;
}

inline const char* SpeedTierName(SpeedTier speed_tier) {
  switch (speed_tier) {
    case SpeedTier::kLightning:
      return "lightning";
    case SpeedTier::kThunder:
      return "thunder";
    case SpeedTier::kFalcon:
      return "falcon";
    case SpeedTier::kCheetah:
      return "cheetah";
    case SpeedTier::kHare:
      return "hare";
    case SpeedTier::kWombat:
      return "wombat";
    case SpeedTier::kSquirrel:
      return "squirrel";
    case SpeedTier::kKitten:
      return "kitten";
    case SpeedTier::kTortoise:
      return "tortoise";
  }
  return "INVALID";
}

// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
struct CompressParams {
  float butteraugli_distance = 1.0f;

  SpeedTier speed_tier = SpeedTier::kSquirrel;
  int brotli_effort = -1;

  // 0 = default.
  // 1 = slightly worse quality.
  // 4 = fastest speed, lowest quality
  // TODO(veluca): hook this up to the C API.
  size_t decoding_speed_tier = 0;

  ColorTransform color_transform = ColorTransform::kXYB;
  YCbCrChromaSubsampling chroma_subsampling;

  // If true, the "modular mode options" members below are used.
  bool modular_mode = false;

  // Change group size in modular mode (0=128, 1=256, 2=512, 3=1024).
  size_t modular_group_size_shift = 1;

  Override preview = Override::kDefault;
  Override gaborish = Override::kDefault;
  int epf = -1;

  // Put center groups first in the bitstream.
  bool centerfirst = false;

  // Pixel coordinates of the center. First group will contain that center.
  size_t center_x = static_cast<size_t>(-1);
  size_t center_y = static_cast<size_t>(-1);

  // If on: preserve color of invisible pixels (if off: don't care)
  // Default: on for lossless, off for lossy
  Override keep_invisible = Override::kDefault;

  // Currently unused as of 2020-01.
  bool clear_metadata = false;

  // Prints extra information during/after encoding.
  bool verbose = false;

  ButteraugliParams ba_params;

  // modular mode options below
  ModularOptions options;
  int responsive = -1;
  // empty for default squeeze
  std::vector<SqueezeParams> squeezes;
  int colorspace = -1;
  // Use Global channel palette if #colors < this percentage of range
  float channel_colors_pre_transform_percent = 95.f;
  // Use Local channel palette if #colors < this percentage of range
  float channel_colors_percent = 80.f;
  int palette_colors = 1 << 10;  // up to 10-bit palette is probably worthwhile
  bool lossy_palette = false;

  // Returns whether these params are lossless as defined by SetLossless();
  bool IsLossless() const {
    // YCbCr is also considered lossless here since it's intended for
    // source material that is already YCbCr (we don't do the fwd transform)
    return modular_mode && butteraugli_distance == 0.0f &&
           color_transform != jxl::ColorTransform::kXYB;
  }

  // Sets the parameters required to make the codec lossless.
  void SetLossless() {
    modular_mode = true;
    butteraugli_distance = 0.0f;
    color_transform = jxl::ColorTransform::kNone;
  }

  // Codestream level to conform to.
  // -1: don't care
  int level = -1;
};

static constexpr float kMinButteraugliForDynamicAR = 0.5f;
static constexpr float kMinButteraugliForDots = 3.0f;
static constexpr float kMinButteraugliToSubtractOriginalPatches = 3.0f;
static constexpr float kMinButteraugliDistanceForProgressiveDc = 4.5f;

// Minimum butteraugli distance the encoder accepts.
static constexpr float kMinButteraugliDistance = 0.01f;

// Tile size for encoder-side processing. Must be equal to color tile dim in the
// current implementation.
static constexpr size_t kEncTileDim = 64;
static constexpr size_t kEncTileDimInBlocks = kEncTileDim / kBlockDim;

}  // namespace jxl

#endif  // LIB_JXL_ENC_PARAMS_H_
