// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_BASE_SANITIZER_DEFINITIONS_H_
#define ENCODER_BASE_SANITIZER_DEFINITIONS_H_

#ifdef MEMORY_SANITIZER
#define JXL_MEMORY_SANITIZER 1
#elif defined(__has_feature)
#if __has_feature(memory_sanitizer)
#define JXL_MEMORY_SANITIZER 1
#else
#define JXL_MEMORY_SANITIZER 0
#endif
#else
#define JXL_MEMORY_SANITIZER 0
#endif

#ifdef ADDRESS_SANITIZER
#define JXL_ADDRESS_SANITIZER 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define JXL_ADDRESS_SANITIZER 1
#else
#define JXL_ADDRESS_SANITIZER 0
#endif
#else
#define JXL_ADDRESS_SANITIZER 0
#endif

#ifdef THREAD_SANITIZER
#define JXL_THREAD_SANITIZER 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define JXL_THREAD_SANITIZER 1
#else
#define JXL_THREAD_SANITIZER 0
#endif
#else
#define JXL_THREAD_SANITIZER 0
#endif

#if JXL_MEMORY_SANITIZER
#include <stddef.h>
#include <stdint.h>

#include "sanitizer/msan_interface.h"
namespace jxl {
namespace msan {
// Chosen so that kSanitizerSentinel is four copies of kSanitizerSentinelByte.
constexpr uint8_t kSanitizerSentinelByte = 0x48;
constexpr float kSanitizerSentinel = 205089.125f;
}  // namespace msan
}  // namespace jxl
#endif  // JXL_MEMORY_SANITIZER

#endif  // ENCODER_BASE_SANITIZER_DEFINITIONS_H
