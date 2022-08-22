// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_MODULAR_ENCODING_CONTEXT_PREDICT_H_
#define ENCODER_MODULAR_ENCODING_CONTEXT_PREDICT_H_

#include <utility>
#include <vector>

#include "encoder/modular/modular_image.h"
#include "encoder/modular/options.h"

namespace jxl {

// Stores a node and its two children at the same time. This significantly
// reduces the number of branches needed during decoding.
struct FlatDecisionNode {
  // Property + splitval of the top node.
  int32_t property0;  // -1 if leaf.
  union {
    PropertyVal splitval0;
    Predictor predictor;
  };
  uint32_t childID;  // childID is ctx id if leaf.
  // Property+splitval of the two child nodes.
  union {
    PropertyVal splitvals[2];
    int32_t multiplier;
  };
  union {
    int32_t properties[2];
    int64_t predictor_offset;
  };
};
using FlatTree = std::vector<FlatDecisionNode>;

class MATreeLookup {
 public:
  explicit MATreeLookup(const FlatTree &tree) : nodes_(tree) {}
  struct LookupResult {
    uint32_t context;
    Predictor predictor;
    int64_t offset;
    int32_t multiplier;
  };
  JXL_INLINE LookupResult Lookup(const Properties &properties) const {
    uint32_t pos = 0;
    while (true) {
      const FlatDecisionNode &node = nodes_[pos];
      if (node.property0 < 0) {
        return {node.childID, node.predictor, node.predictor_offset,
                node.multiplier};
      }
      bool p0 = properties[node.property0] <= node.splitval0;
      uint32_t off0 = properties[node.properties[0]] <= node.splitvals[0];
      uint32_t off1 =
          2 | (properties[node.properties[1]] <= node.splitvals[1] ? 1 : 0);
      pos = node.childID + (p0 ? off1 : off0);
    }
  }

 private:
  const FlatTree &nodes_;
};

static constexpr size_t kExtraPropsPerChannel = 4;
static constexpr size_t kNumNonrefProperties = kNumStaticProperties + 14;

constexpr size_t kGradientProp = 9;

// Clamps gradient to the min/max of n, w (and l, implicitly).
static JXL_INLINE int32_t ClampedGradient(const int32_t n, const int32_t w,
                                          const int32_t l) {
  const int32_t m = std::min(n, w);
  const int32_t M = std::max(n, w);
  // The end result of this operation doesn't overflow or underflow if the
  // result is between m and M, but the intermediate value may overflow, so we
  // do the intermediate operations in uint32_t and check later if we had an
  // overflow or underflow condition comparing m, M and l directly.
  // grad = M + m - l = n + w - l
  const int32_t grad =
      static_cast<int32_t>(static_cast<uint32_t>(n) + static_cast<uint32_t>(w) -
                           static_cast<uint32_t>(l));
  // We use two sets of ternary operators to force the evaluation of them in
  // any case, allowing the compiler to avoid branches and use cmovl/cmovg in
  // x86.
  const int32_t grad_clamp_M = (l < m) ? M : grad;
  return (l > M) ? m : grad_clamp_M;
}

inline pixel_type_w Select(pixel_type_w a, pixel_type_w b, pixel_type_w c) {
  pixel_type_w p = a + b - c;
  pixel_type_w pa = std::abs(p - a);
  pixel_type_w pb = std::abs(p - b);
  return pa < pb ? a : b;
}

inline void PrecomputeReferences(const Channel &ch, size_t y,
                                 const Image &image, uint32_t i,
                                 Channel *references) {
  ZeroFillImage(&references->plane);
  uint32_t offset = 0;
  size_t num_extra_props = references->w;
  intptr_t onerow = references->plane.PixelsPerRow();
  for (int32_t j = static_cast<int32_t>(i) - 1;
       j >= 0 && offset < num_extra_props; j--) {
    if (image.channel[j].w != image.channel[i].w ||
        image.channel[j].h != image.channel[i].h) {
      continue;
    }
    if (image.channel[j].hshift != image.channel[i].hshift) continue;
    if (image.channel[j].vshift != image.channel[i].vshift) continue;
    pixel_type *JXL_RESTRICT rp = references->Row(0) + offset;
    const pixel_type *JXL_RESTRICT rpp = image.channel[j].Row(y);
    const pixel_type *JXL_RESTRICT rpprev = image.channel[j].Row(y ? y - 1 : 0);
    for (size_t x = 0; x < ch.w; x++, rp += onerow) {
      pixel_type_w v = rpp[x];
      rp[0] = std::abs(v);
      rp[1] = v;
      pixel_type_w vleft = (x ? rpp[x - 1] : 0);
      pixel_type_w vtop = (y ? rpprev[x] : vleft);
      pixel_type_w vtopleft = (x && y ? rpprev[x - 1] : vleft);
      pixel_type_w vpredicted = ClampedGradient(vleft, vtop, vtopleft);
      rp[2] = std::abs(v - vpredicted);
      rp[3] = v - vpredicted;
    }

    offset += kExtraPropsPerChannel;
  }
}

struct PredictionResult {
  int context = 0;
  pixel_type_w guess = 0;
  Predictor predictor;
  int32_t multiplier;
};

inline void InitPropsRow(
    Properties *p,
    const std::array<pixel_type, kNumStaticProperties> &static_props,
    const int y) {
  for (size_t i = 0; i < kNumStaticProperties; i++) {
    (*p)[i] = static_props[i];
  }
  (*p)[2] = y;
  (*p)[9] = 0;  // local gradient.
}

namespace detail {
enum PredictorMode {
  kUseTree = 1,
};

JXL_INLINE pixel_type_w PredictOne(Predictor p, pixel_type_w left,
                                   pixel_type_w top, pixel_type_w toptop,
                                   pixel_type_w topleft, pixel_type_w topright,
                                   pixel_type_w leftleft,
                                   pixel_type_w toprightright) {
  switch (p) {
    case Predictor::Zero:
      return pixel_type_w{0};
    case Predictor::Left:
      return left;
    case Predictor::Top:
      return top;
    case Predictor::Select:
      return Select(left, top, topleft);
    case Predictor::Gradient:
      return pixel_type_w{ClampedGradient(left, top, topleft)};
    case Predictor::TopLeft:
      return topleft;
    case Predictor::TopRight:
      return topright;
    case Predictor::LeftLeft:
      return leftleft;
    case Predictor::Average0:
      return (left + top) / 2;
    case Predictor::Average1:
      return (left + topleft) / 2;
    case Predictor::Average2:
      return (topleft + top) / 2;
    case Predictor::Average3:
      return (top + topright) / 2;
    case Predictor::Average4:
      return (6 * top - 2 * toptop + 7 * left + 1 * leftleft +
              1 * toprightright + 3 * topright + 8) /
             16;
    default:
      return pixel_type_w{0};
  }
}

template <int mode>
JXL_INLINE PredictionResult Predict(Properties *p, size_t w,
                                    const pixel_type *JXL_RESTRICT pp,
                                    const intptr_t onerow, const size_t x,
                                    const size_t y, Predictor predictor,
                                    const MATreeLookup *lookup,
                                    const Channel *references,
                                    pixel_type_w *predictions) {
  // We start in position 3 because of 2 static properties + y.
  size_t offset = 3;
  constexpr bool compute_properties = mode & kUseTree;
  pixel_type_w left = (x ? pp[-1] : (y ? pp[-onerow] : 0));
  pixel_type_w top = (y ? pp[-onerow] : left);
  pixel_type_w topleft = ((x && y) ? pp[-1 - onerow] : left);
  pixel_type_w topright = ((x + 1 < w && y) ? pp[1 - onerow] : top);
  pixel_type_w leftleft = (x > 1 ? pp[-2] : left);
  pixel_type_w toptop = (y > 1 ? pp[-onerow - onerow] : top);
  pixel_type_w toprightright = ((x + 2 < w && y) ? pp[2 - onerow] : topright);

  if (compute_properties) {
    // location
    (*p)[offset++] = x;
    // neighbors
    (*p)[offset++] = std::abs(top);
    (*p)[offset++] = std::abs(left);
    (*p)[offset++] = top;
    (*p)[offset++] = left;

    // local gradient
    (*p)[offset] = left - (*p)[offset + 1];
    offset++;
    // local gradient
    (*p)[offset++] = left + top - topleft;

    // FFV1 context properties
    (*p)[offset++] = left - topleft;
    (*p)[offset++] = topleft - top;
    (*p)[offset++] = top - topright;
    (*p)[offset++] = top - toptop;
    (*p)[offset++] = left - leftleft;
  }

  if (compute_properties) {
    offset += 1;
    // Extra properties.
    const pixel_type *JXL_RESTRICT rp = references->Row(x);
    for (size_t i = 0; i < references->w; i++) {
      (*p)[offset++] = rp[i];
    }
  }
  PredictionResult result;
  if (mode & kUseTree) {
    MATreeLookup::LookupResult lr = lookup->Lookup(*p);
    result.context = lr.context;
    result.guess = lr.offset;
    result.multiplier = lr.multiplier;
    predictor = lr.predictor;
  }
  result.guess += PredictOne(predictor, left, top, toptop, topleft, topright,
                             leftleft, toprightright);
  result.predictor = predictor;

  return result;
}
}  // namespace detail

inline PredictionResult PredictNoTreeNoWP(size_t w,
                                          const pixel_type *JXL_RESTRICT pp,
                                          const intptr_t onerow, const int x,
                                          const int y, Predictor predictor) {
  return detail::Predict</*mode=*/0>(
      /*p=*/nullptr, w, pp, onerow, x, y, predictor, /*lookup=*/nullptr,
      /*references=*/nullptr, /*predictions=*/nullptr);
}

inline PredictionResult PredictTreeNoWP(Properties *p, size_t w,
                                        const pixel_type *JXL_RESTRICT pp,
                                        const intptr_t onerow, const int x,
                                        const int y,
                                        const MATreeLookup &tree_lookup,
                                        const Channel &references) {
  return detail::Predict<detail::kUseTree>(
      p, w, pp, onerow, x, y, Predictor::Zero, &tree_lookup, &references,
      /*predictions=*/nullptr);
}

}  // namespace jxl

#endif  // ENCODER_MODULAR_ENCODING_CONTEXT_PREDICT_H_
