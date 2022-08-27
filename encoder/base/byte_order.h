// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_BASE_BYTE_ORDER_H_
#define ENCODER_BASE_BYTE_ORDER_H_

#include <stdint.h>
#include <string.h>  // memcpy

#include "encoder/base/compiler_specific.h"

#if JXL_COMPILER_MSVC
#include <intrin.h>  // _byteswap_*
#endif

#if (defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__))
#define JXL_BYTE_ORDER_LITTLE 1
#else
// This means that we don't know that the byte order is little endian, in
// this case we use endian-neutral code that works for both little- and
// big-endian.
#define JXL_BYTE_ORDER_LITTLE 0
#endif

// Returns whether the system is little-endian (least-significant byte first).
#if JXL_BYTE_ORDER_LITTLE
static constexpr bool IsLittleEndian() { return true; }
#else
static inline bool IsLittleEndian() {
  const uint32_t multibyte = 1;
  uint8_t byte;
  memcpy(&byte, &multibyte, 1);
  return byte == 1;
}
#endif

#if JXL_COMPILER_MSVC
#define JXL_BSWAP32(x) _byteswap_ulong(x)
#else
#define JXL_BSWAP32(x) __builtin_bswap32(x)
#endif

static JXL_INLINE float BSwapFloat(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  uint32_t uswap = JXL_BSWAP32(u);
  float xswap;
  memcpy(&xswap, &uswap, 4);
  return xswap;
}

#endif  // ENCODER_BASE_BYTE_ORDER_H_
