// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/image.h"

#include <algorithm>  // swap

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "encoder/image.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "encoder/base/sanitizer_definitions.h"
#include "encoder/common.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {

namespace HWY_NAMESPACE {
size_t GetVectorSize() { return HWY_LANES(uint8_t); }
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE

}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {
namespace {

HWY_EXPORT(GetVectorSize);  // Local function.

size_t VectorSize() {
  static size_t bytes = HWY_DYNAMIC_DISPATCH(GetVectorSize)();
  return bytes;
}

// Returns distance [bytes] between the start of two consecutive rows, a
// multiple of vector/cache line size but NOT CacheAligned::kAlias - see below.
size_t BytesPerRow(const size_t xsize, const size_t sizeof_t) {
  const size_t vec_size = VectorSize();
  size_t valid_bytes = xsize * sizeof_t;

  // Allow unaligned accesses starting at the last valid value - this may raise
  // msan errors.
  // Skip for the scalar case because no extra lanes will be loaded.
  if (vec_size != 0) {
    valid_bytes += vec_size - sizeof_t;
  }

  // Round up to vector and cache line size.
  const size_t align = std::max(vec_size, CacheAligned::kAlignment);
  size_t bytes_per_row = RoundUpTo(valid_bytes, align);

  // During the lengthy window before writes are committed to memory, CPUs
  // guard against read after write hazards by checking the address, but
  // only the lower 11 bits. We avoid a false dependency between writes to
  // consecutive rows by ensuring their sizes are not multiples of 2 KiB.
  // Avoid2K prevents the same problem for the planes of an Image3.
  if (bytes_per_row % CacheAligned::kAlias == 0) {
    bytes_per_row += align;
  }

  JXL_ASSERT(bytes_per_row % align == 0);
  return bytes_per_row;
}

}  // namespace

PlaneBase::PlaneBase(const size_t xsize, const size_t ysize,
                     const size_t sizeof_t)
    : xsize_(static_cast<uint32_t>(xsize)),
      ysize_(static_cast<uint32_t>(ysize)),
      orig_xsize_(static_cast<uint32_t>(xsize)),
      orig_ysize_(static_cast<uint32_t>(ysize)) {
  JXL_CHECK(xsize == xsize_);
  JXL_CHECK(ysize == ysize_);

  JXL_ASSERT(sizeof_t == 1 || sizeof_t == 2 || sizeof_t == 4 || sizeof_t == 8);

  bytes_per_row_ = 0;
  // Dimensions can be zero, e.g. for lazily-allocated images. Only allocate
  // if nonzero, because "zero" bytes still have padding/bookkeeping overhead.
  if (xsize != 0 && ysize != 0) {
    bytes_per_row_ = BytesPerRow(xsize, sizeof_t);
    bytes_ = AllocateArray(bytes_per_row_ * ysize);
    JXL_CHECK(bytes_.get());
    InitializePadding(sizeof_t, Padding::kRoundUp);
  }
}

void PlaneBase::InitializePadding(const size_t sizeof_t, Padding padding) {
#if defined(MEMORY_SANITIZER) || HWY_IDE
  if (xsize_ == 0 || ysize_ == 0) return;

  const size_t vec_size = VectorSize();
  if (vec_size == 0) return;  // Scalar mode: no padding needed

  const size_t valid_size = xsize_ * sizeof_t;
  const size_t initialize_size = padding == Padding::kRoundUp
                                     ? RoundUpTo(valid_size, vec_size)
                                     : valid_size + vec_size - sizeof_t;
  if (valid_size == initialize_size) return;

  for (size_t y = 0; y < ysize_; ++y) {
    uint8_t* JXL_RESTRICT row = static_cast<uint8_t*>(VoidRow(y));
#if defined(__clang__) && (__clang_major__ <= 6)
    // There's a bug in msan in clang-6 when handling AVX2 operations. This
    // workaround allows tests to pass on msan, although it is slower and
    // prevents msan warnings from uninitialized images.
    std::fill(row, msan::kSanitizerSentinelByte, initialize_size);
#else
    memset(row + valid_size, msan::kSanitizerSentinelByte,
           initialize_size - valid_size);
#endif  // clang6
  }
#endif  // MEMORY_SANITIZER
}

void PlaneBase::Swap(PlaneBase& other) {
  std::swap(xsize_, other.xsize_);
  std::swap(ysize_, other.ysize_);
  std::swap(orig_xsize_, other.orig_xsize_);
  std::swap(orig_ysize_, other.orig_ysize_);
  std::swap(bytes_per_row_, other.bytes_per_row_);
  std::swap(bytes_, other.bytes_);
}

constexpr inline size_t RoundUpToBlockDim(size_t dim) {
  return (dim + 7) & ~size_t(7);
}

}  // namespace jxl
#endif  // HWY_ONCE
