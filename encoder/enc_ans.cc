// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/enc_ans.h"

#include <stdint.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <hwy/cache_control.h>  // Prefetch
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "encoder/base/bits.h"
#include "encoder/fast_math-inl.h"

namespace jxl {

namespace {

#define ANS_LOG_TAB_SIZE 12u
#define ANS_TAB_SIZE (1 << ANS_LOG_TAB_SIZE)
#define ANS_TAB_MASK (ANS_TAB_SIZE - 1)

// Largest possible symbol to be encoded by ANS coding.
#define ANS_MAX_ALPHABET_SIZE 256

#define ANS_SIGNATURE 0x13  // Initial state, used as CRC.

// precision must be equal to:  #bits(state_) + #bits(freq)
#define RECIPROCAL_PRECISION (32 + ANS_LOG_TAB_SIZE)

static const int kMaxNumSymbolsForSmallCode = 4;

// Returns the precision (number of bits) that should be used to store
// a histogram count such that Log2Floor(count) == logcount.
static JXL_INLINE uint32_t GetPopulationCountPrecision(uint32_t logcount,
                                                       uint32_t shift) {
  int32_t r = std::min<int>(
      logcount, int(shift) - int((ANS_LOG_TAB_SIZE - logcount) >> 1));
  if (r < 0) return 0;
  return r;
}

// An alias table implements a mapping from the [0, ANS_TAB_SIZE) range into
// the [0, ANS_MAX_ALPHABET_SIZE) range, satisfying the following conditions:
// - each symbol occurs as many times as specified by any valid distribution
//   of frequencies of the symbols. A valid distribution here is an array of
//   ANS_MAX_ALPHABET_SIZE that contains numbers in the range [0, ANS_TAB_SIZE],
//   and whose sum is ANS_TAB_SIZE.
// - lookups can be done in constant time, and also return how many smaller
//   input values map into the same symbol, according to some well-defined order
//   of input values.
// - the space used by the alias table is given by a small constant times the
//   index of the largest symbol with nonzero probability in the distribution.
// Each of the entries in the table covers a range of `entry_size` values in the
// [0, ANS_TAB_SIZE) range; consecutive entries represent consecutive
// sub-ranges. In the range covered by entry `i`, the first `cutoff` values map
// to symbol `i`, while the others map to symbol `right_value`.
//
// TODO(veluca): consider making the order used for computing offsets easier to
// define - it is currently defined by the algorithm to compute the alias table.
// Beware of breaking the implicit assumption that symbols that come after the
// cutoff value should have an offset at least as big as the cutoff.

struct AliasTable {
  struct Symbol {
    size_t value;
    size_t offset;
    size_t freq;
  };

// Working set size matters here (~64 tables x 256 entries).
// offsets0 is always zero (beginning of [0] side among the same symbol).
// offsets1 is an offset of (pos >= cutoff) side decremented by cutoff.
#pragma pack(push, 1)
  struct Entry {
    uint8_t cutoff;       // < kEntrySizeMinus1 when used by ANS.
    uint8_t right_value;  // < alphabet size.
    uint16_t freq0;

    // Only used if `greater` (see Lookup)
    uint16_t offsets1;         // <= ANS_TAB_SIZE
    uint16_t freq1_xor_freq0;  // for branchless ternary in Lookup
  };
#pragma pack(pop)

  // Dividing `value` by `entry_size` determines `i`, the entry which is
  // responsible for the input. If the remainder is below `cutoff`, then the
  // mapped symbol is `i`; since `offsets[0]` stores the number of occurrences
  // of `i` "before" the start of this entry, the offset of the input will be
  // `offsets[0] + remainder`. If the remainder is above cutoff, the mapped
  // symbol is `right_value`; since `offsets[1]` stores the number of
  // occurrences of `right_value` "before" this entry, minus the `cutoff` value,
  // the input offset is then `remainder + offsets[1]`.
  static JXL_INLINE Symbol Lookup(const Entry* JXL_RESTRICT table, size_t value,
                                  size_t log_entry_size,
                                  size_t entry_size_minus_1) {
    const size_t i = value >> log_entry_size;
    const size_t pos = value & entry_size_minus_1;

#if JXL_BYTE_ORDER_LITTLE
    uint64_t entry;
    memcpy(&entry, &table[i].cutoff, sizeof(entry));
    const size_t cutoff = entry & 0xFF;              // = MOVZX
    const size_t right_value = (entry >> 8) & 0xFF;  // = MOVZX
    const size_t freq0 = (entry >> 16) & 0xFFFF;
#else
    // Generates multiple loads with complex addressing.
    const size_t cutoff = table[i].cutoff;
    const size_t right_value = table[i].right_value;
    const size_t freq0 = table[i].freq0;
#endif

    const bool greater = pos >= cutoff;

#if JXL_BYTE_ORDER_LITTLE
    const uint64_t conditional = greater ? entry : 0;  // = CMOV
    const size_t offsets1_or_0 = (conditional >> 32) & 0xFFFF;
    const size_t freq1_xor_freq0_or_0 = conditional >> 48;
#else
    const size_t offsets1_or_0 = greater ? table[i].offsets1 : 0;
    const size_t freq1_xor_freq0_or_0 = greater ? table[i].freq1_xor_freq0 : 0;
#endif

    // WARNING: moving this code may interfere with CMOV heuristics.
    Symbol s;
    s.value = greater ? right_value : i;
    s.offset = offsets1_or_0 + pos;
    s.freq = freq0 ^ freq1_xor_freq0_or_0;  // = greater ? freq1 : freq0
    // XOR avoids implementation-defined conversion from unsigned to signed.
    // Alternatives considered: BEXTR is 2 cycles on HSW, SET+shift causes
    // spills, simple ternary has a long dependency chain.

    return s;
  }

  static HWY_INLINE void Prefetch(const Entry* JXL_RESTRICT table, size_t value,
                                  size_t log_entry_size) {
    const size_t i = value >> log_entry_size;
    hwy::Prefetch(table + i);
  }
};

// First, all trailing non-occuring symbols are removed from the distribution;
// if this leaves the distribution empty, a dummy symbol with max weight is
// added. This ensures that the resulting distribution sums to total table size.
// Then, `entry_size` is chosen to be the largest power of two so that
// `table_size` = ANS_TAB_SIZE/`entry_size` is at least as big as the
// distribution size.
// Note that each entry will only ever contain two different symbols, and
// consecutive ranges of offsets, which allows us to use a compact
// representation.
// Each entry is initialized with only the (symbol=i, offset) pairs; then
// positions for which the entry overflows (i.e. distribution[i] > entry_size)
// or is not full are computed, and put into a stack in increasing order.
// Missing symbols in the distribution are padded with 0 (because `table_size`
// >= number of symbols). The `cutoff` value for each entry is initialized to
// the number of occupied slots in that entry (i.e. `distributions[i]`). While
// the overflowing-symbol stack is not empty (which implies that the
// underflowing-symbol stack also is not), the top overfull and underfull
// positions are popped from the stack; the empty slots in the underfull entry
// are then filled with as many slots as needed from the overfull entry; such
// slots are placed after the slots in the overfull entry, and `offsets[1]` is
// computed accordingly. The formerly underfull entry is thus now neither
// underfull nor overfull, and represents exactly two symbols. The overfull
// entry might be either overfull or underfull, and is pushed into the
// corresponding stack.
void InitAliasTable(std::vector<int32_t> distribution, uint32_t range,
                    size_t log_alpha_size, AliasTable::Entry* JXL_RESTRICT a) {
  while (!distribution.empty() && distribution.back() == 0) {
    distribution.pop_back();
  }
  // Ensure that a valid table is always returned, even for an empty
  // alphabet. Otherwise, a specially-crafted stream might crash the
  // decoder.
  if (distribution.empty()) {
    distribution.emplace_back(range);
  }
  const size_t table_size = 1 << log_alpha_size;
#if JXL_ENABLE_ASSERT
  int sum = std::accumulate(distribution.begin(), distribution.end(), 0);
#endif  // JXL_ENABLE_ASSERT
  JXL_ASSERT(static_cast<uint32_t>(sum) == range);
  // range must be a power of two
  JXL_ASSERT((range & (range - 1)) == 0);
  JXL_ASSERT(distribution.size() <= table_size);
  JXL_ASSERT(table_size <= range);
  const uint32_t entry_size = range >> log_alpha_size;  // this is exact
  // Special case for single-symbol distributions, that ensures that the state
  // does not change when decoding from such a distribution. Note that, since we
  // hardcode offset0 == 0, it is not straightforward (if at all possible) to
  // fix the general case to produce this result.
  for (size_t sym = 0; sym < distribution.size(); sym++) {
    if (distribution[sym] == ANS_TAB_SIZE) {
      for (size_t i = 0; i < table_size; i++) {
        a[i].right_value = sym;
        a[i].cutoff = 0;
        a[i].offsets1 = entry_size * i;
        a[i].freq0 = 0;
        a[i].freq1_xor_freq0 = ANS_TAB_SIZE;
      }
      return;
    }
  }
  std::vector<uint32_t> underfull_posn;
  std::vector<uint32_t> overfull_posn;
  std::vector<uint32_t> cutoffs(1 << log_alpha_size);
  // Initialize entries.
  for (size_t i = 0; i < distribution.size(); i++) {
    cutoffs[i] = distribution[i];
    if (cutoffs[i] > entry_size) {
      overfull_posn.push_back(i);
    } else if (cutoffs[i] < entry_size) {
      underfull_posn.push_back(i);
    }
  }
  for (uint32_t i = distribution.size(); i < table_size; i++) {
    cutoffs[i] = 0;
    underfull_posn.push_back(i);
  }
  // Reassign overflow/underflow values.
  while (!overfull_posn.empty()) {
    uint32_t overfull_i = overfull_posn.back();
    overfull_posn.pop_back();
    JXL_ASSERT(!underfull_posn.empty());
    uint32_t underfull_i = underfull_posn.back();
    underfull_posn.pop_back();
    uint32_t underfull_by = entry_size - cutoffs[underfull_i];
    cutoffs[overfull_i] -= underfull_by;
    // overfull positions have their original symbols
    a[underfull_i].right_value = overfull_i;
    a[underfull_i].offsets1 = cutoffs[overfull_i];
    // Slots in the right part of entry underfull_i were taken from the end
    // of the symbols in entry overfull_i.
    if (cutoffs[overfull_i] < entry_size) {
      underfull_posn.push_back(overfull_i);
    } else if (cutoffs[overfull_i] > entry_size) {
      overfull_posn.push_back(overfull_i);
    }
  }
  for (uint32_t i = 0; i < table_size; i++) {
    // cutoffs[i] is properly initialized but the clang-analyzer doesn't infer
    // it since it is partially initialized across two for-loops.
    // NOLINTNEXTLINE(clang-analyzer-core.UndefinedBinaryOperatorResult)
    if (cutoffs[i] == entry_size) {
      a[i].right_value = i;
      a[i].offsets1 = 0;
      a[i].cutoff = 0;
    } else {
      // Note that, if cutoff is not equal to entry_size,
      // a[i].offsets1 was initialized with (overfull cutoff) -
      // (entry_size - a[i].cutoff). Thus, subtracting
      // a[i].cutoff cannot make it negative.
      a[i].offsets1 -= cutoffs[i];
      a[i].cutoff = cutoffs[i];
    }
    const size_t freq0 = i < distribution.size() ? distribution[i] : 0;
    const size_t i1 = a[i].right_value;
    const size_t freq1 = i1 < distribution.size() ? distribution[i1] : 0;
    a[i].freq0 = static_cast<uint16_t>(freq0);
    a[i].freq1_xor_freq0 = static_cast<uint16_t>(freq1 ^ freq0);
  }
}

void ANSBuildInfoTable(const int32_t* counts, const AliasTable::Entry* table,
                       size_t alphabet_size, size_t log_alpha_size,
                       ANSEncSymbolInfo* info) {
  size_t log_entry_size = ANS_LOG_TAB_SIZE - log_alpha_size;
  size_t entry_size_minus_1 = (1 << log_entry_size) - 1;
  // create valid alias table for empty streams.
  for (size_t s = 0; s < std::max<size_t>(1, alphabet_size); ++s) {
    const int32_t freq = s == alphabet_size ? ANS_TAB_SIZE : counts[s];
    info[s].freq_ = static_cast<uint16_t>(freq);
    if (freq != 0) {
      info[s].ifreq_ =
          ((1ull << RECIPROCAL_PRECISION) + info[s].freq_ - 1) / info[s].freq_;
    } else {
      info[s].ifreq_ = 1;  // shouldn't matter (symbol shouldn't occur), but...
    }
    info[s].reverse_map_.resize(freq);
  }
  for (int i = 0; i < ANS_TAB_SIZE; i++) {
    AliasTable::Symbol s =
        AliasTable::Lookup(table, i, log_entry_size, entry_size_minus_1);
    info[s.value].reverse_map_[s.offset] = i;
  }
}

// Static Huffman code for encoding logcounts. The last symbol is used as RLE
// sequence.
static const uint8_t kLogCountBitLengths[ANS_LOG_TAB_SIZE + 2] = {
    5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 6, 7, 7,
};
static const uint8_t kLogCountSymbols[ANS_LOG_TAB_SIZE + 2] = {
    17, 11, 15, 3, 9, 7, 4, 2, 5, 6, 0, 33, 1, 65,
};

// Returns the difference between largest count that can be represented and is
// smaller than "count" and smallest representable count larger than "count".
static int SmallestIncrement(uint32_t count, uint32_t shift) {
  int bits = count == 0 ? -1 : FloorLog2Nonzero(count);
  int drop_bits = bits - GetPopulationCountPrecision(bits, shift);
  return drop_bits < 0 ? 1 : (1 << drop_bits);
}

template <bool minimize_error_of_sum>
bool RebalanceHistogram(const float* targets, int max_symbol, int table_size,
                        uint32_t shift, int* omit_pos, int32_t* counts) {
  int sum = 0;
  float sum_nonrounded = 0.0;
  int remainder_pos = 0;  // if all of them are handled in first loop
  int remainder_log = -1;
  for (int n = 0; n < max_symbol; ++n) {
    if (targets[n] > 0 && targets[n] < 1.0f) {
      counts[n] = 1;
      sum_nonrounded += targets[n];
      sum += counts[n];
    }
  }
  const float discount_ratio =
      (table_size - sum) / (table_size - sum_nonrounded);
  JXL_ASSERT(discount_ratio > 0);
  JXL_ASSERT(discount_ratio <= 1.0f);
  // Invariant for minimize_error_of_sum == true:
  // abs(sum - sum_nonrounded)
  //   <= SmallestIncrement(max(targets[])) + max_symbol
  for (int n = 0; n < max_symbol; ++n) {
    if (targets[n] >= 1.0f) {
      sum_nonrounded += targets[n];
      counts[n] =
          static_cast<int32_t>(targets[n] * discount_ratio);  // truncate
      if (counts[n] == 0) counts[n] = 1;
      if (counts[n] == table_size) counts[n] = table_size - 1;
      // Round the count to the closest nonzero multiple of SmallestIncrement
      // (when minimize_error_of_sum is false) or one of two closest so as to
      // keep the sum as close as possible to sum_nonrounded.
      int inc = SmallestIncrement(counts[n], shift);
      counts[n] -= counts[n] & (inc - 1);
      // TODO(robryk): Should we rescale targets[n]?
      const float target =
          minimize_error_of_sum ? (sum_nonrounded - sum) : targets[n];
      if (counts[n] == 0 ||
          (target > counts[n] + inc / 2 && counts[n] + inc < table_size)) {
        counts[n] += inc;
      }
      sum += counts[n];
      const int count_log = FloorLog2Nonzero(static_cast<uint32_t>(counts[n]));
      if (count_log > remainder_log) {
        remainder_pos = n;
        remainder_log = count_log;
      }
    }
  }
  JXL_ASSERT(remainder_pos != -1);
  // NOTE: This is the only place where counts could go negative. We could
  // detect that, return false and make int32_t uint32_t.
  counts[remainder_pos] -= sum - table_size;
  *omit_pos = remainder_pos;
  return counts[remainder_pos] > 0;
}

Status NormalizeCounts(int32_t* counts, int* omit_pos, const int length,
                       const int precision_bits, uint32_t shift,
                       int* num_symbols, int* symbols) {
  const int32_t table_size = 1 << precision_bits;  // target sum / table size
  uint64_t total = 0;
  int max_symbol = 0;
  int symbol_count = 0;
  for (int n = 0; n < length; ++n) {
    total += counts[n];
    if (counts[n] > 0) {
      if (symbol_count < kMaxNumSymbolsForSmallCode) {
        symbols[symbol_count] = n;
      }
      ++symbol_count;
      max_symbol = n + 1;
    }
  }
  *num_symbols = symbol_count;
  if (symbol_count == 0) {
    return true;
  }
  if (symbol_count == 1) {
    counts[symbols[0]] = table_size;
    return true;
  }
  if (symbol_count > table_size)
    return JXL_FAILURE("Too many entries in an ANS histogram");

  const float norm = 1.f * table_size / total;
  std::vector<float> targets(max_symbol);
  for (size_t n = 0; n < targets.size(); ++n) {
    targets[n] = norm * counts[n];
  }
  if (!RebalanceHistogram<false>(&targets[0], max_symbol, table_size, shift,
                                 omit_pos, counts)) {
    // Use an alternative rebalancing mechanism if the one above failed
    // to create a histogram that is positive wherever the original one was.
    if (!RebalanceHistogram<true>(&targets[0], max_symbol, table_size, shift,
                                  omit_pos, counts)) {
      return JXL_FAILURE("Logic error: couldn't rebalance a histogram");
    }
  }
  return true;
}

template <typename Writer>
void StoreVarLenUint8(size_t n, Writer* writer) {
  JXL_DASSERT(n <= 255);
  if (n == 0) {
    writer->Write(1, 0);
  } else {
    writer->Write(1, 1);
    size_t nbits = FloorLog2Nonzero(n);
    writer->Write(3, nbits);
    writer->Write(nbits, n - (1ULL << nbits));
  }
}

template <typename Writer>
bool EncodeCounts(const int32_t* counts, const int alphabet_size,
                  const int omit_pos, const int num_symbols, uint32_t shift,
                  const int* symbols, Writer* writer) {
  bool ok = true;
  if (num_symbols <= 2) {
    // Small tree marker to encode 1-2 symbols.
    writer->Write(1, 1);
    if (num_symbols == 0) {
      writer->Write(1, 0);
      StoreVarLenUint8(0, writer);
    } else {
      writer->Write(1, num_symbols - 1);
      for (int i = 0; i < num_symbols; ++i) {
        StoreVarLenUint8(symbols[i], writer);
      }
    }
    if (num_symbols == 2) {
      writer->Write(ANS_LOG_TAB_SIZE, counts[symbols[0]]);
    }
  } else {
    // Mark non-small tree.
    writer->Write(1, 0);
    // Mark non-flat histogram.
    writer->Write(1, 0);

    // Precompute sequences for RLE encoding. Contains the number of identical
    // values starting at a given index. Only contains the value at the first
    // element of the series.
    std::vector<uint32_t> same(alphabet_size, 0);
    int last = 0;
    for (int i = 1; i < alphabet_size; i++) {
      // Store the sequence length once different symbol reached, or we're at
      // the end, or the length is longer than we can encode, or we are at
      // the omit_pos. We don't support including the omit_pos in an RLE
      // sequence because this value may use a different amount of log2 bits
      // than standard, it is too complex to handle in the decoder.
      if (counts[i] != counts[last] || i + 1 == alphabet_size ||
          (i - last) >= 255 || i == omit_pos || i == omit_pos + 1) {
        same[last] = (i - last);
        last = i + 1;
      }
    }

    int length = 0;
    std::vector<int> logcounts(alphabet_size);
    int omit_log = 0;
    for (int i = 0; i < alphabet_size; ++i) {
      JXL_ASSERT(counts[i] <= ANS_TAB_SIZE);
      JXL_ASSERT(counts[i] >= 0);
      if (i == omit_pos) {
        length = i + 1;
      } else if (counts[i] > 0) {
        logcounts[i] = FloorLog2Nonzero(static_cast<uint32_t>(counts[i])) + 1;
        length = i + 1;
        if (i < omit_pos) {
          omit_log = std::max(omit_log, logcounts[i] + 1);
        } else {
          omit_log = std::max(omit_log, logcounts[i]);
        }
      }
    }
    logcounts[omit_pos] = omit_log;

    // Elias gamma-like code for shift. Only difference is that if the number
    // of bits to be encoded is equal to FloorLog2(ANS_LOG_TAB_SIZE+1), we skip
    // the terminating 0 in unary coding.
    int upper_bound_log = FloorLog2Nonzero(ANS_LOG_TAB_SIZE + 1);
    int log = FloorLog2Nonzero(shift + 1);
    writer->Write(log, (1 << log) - 1);
    if (log != upper_bound_log) writer->Write(1, 0);
    writer->Write(log, ((1 << log) - 1) & (shift + 1));

    // Since num_symbols >= 3, we know that length >= 3, therefore we encode
    // length - 3.
    if (length - 3 > 255) {
      // Pretend that everything is OK, but complain about correctness later.
      StoreVarLenUint8(255, writer);
      ok = false;
    } else {
      StoreVarLenUint8(length - 3, writer);
    }

    // The logcount values are encoded with a static Huffman code.
    static const size_t kMinReps = 4;
    size_t rep = ANS_LOG_TAB_SIZE + 1;
    for (int i = 0; i < length; ++i) {
      if (i > 0 && same[i - 1] > kMinReps) {
        // Encode the RLE symbol and skip the repeated ones.
        writer->Write(kLogCountBitLengths[rep], kLogCountSymbols[rep]);
        StoreVarLenUint8(same[i - 1] - kMinReps - 1, writer);
        i += same[i - 1] - 2;
        continue;
      }
      writer->Write(kLogCountBitLengths[logcounts[i]],
                    kLogCountSymbols[logcounts[i]]);
    }
    for (int i = 0; i < length; ++i) {
      if (i > 0 && same[i - 1] > kMinReps) {
        // Skip symbols encoded by RLE.
        i += same[i - 1] - 2;
        continue;
      }
      if (logcounts[i] > 1 && i != omit_pos) {
        int bitcount = GetPopulationCountPrecision(logcounts[i] - 1, shift);
        int drop_bits = logcounts[i] - 1 - bitcount;
        JXL_CHECK((counts[i] & ((1 << drop_bits) - 1)) == 0);
        writer->Write(bitcount, (counts[i] >> drop_bits) - (1 << bitcount));
      }
    }
  }
  return ok;
}

// Returns an estimate of the cost of encoding this histogram and the
// corresponding data.
void BuildAndStoreANSEncodingData(const int32_t* histogram,
                                  size_t alphabet_size, size_t log_alpha_size,
                                  ANSEncSymbolInfo* info, BitWriter* writer) {
  JXL_ASSERT(alphabet_size <= ANS_TAB_SIZE);
  int num_symbols;
  int symbols[kMaxNumSymbolsForSmallCode] = {};
  std::vector<int32_t> counts(histogram, histogram + alphabet_size);
  if (!counts.empty()) {
    size_t sum = 0;
    for (size_t i = 0; i < counts.size(); i++) {
      sum += counts[i];
    }
    if (sum == 0) {
      counts[0] = ANS_TAB_SIZE;
    }
  }
  int omit_pos = 0;
  JXL_CHECK(NormalizeCounts(counts.data(), &omit_pos, alphabet_size,
                            ANS_LOG_TAB_SIZE, ANS_LOG_TAB_SIZE, &num_symbols,
                            symbols));
  AliasTable::Entry a[ANS_MAX_ALPHABET_SIZE];
  InitAliasTable(counts, ANS_TAB_SIZE, log_alpha_size, a);
  ANSBuildInfoTable(counts.data(), a, alphabet_size, log_alpha_size, info);
  EncodeCounts(counts.data(), alphabet_size, omit_pos, num_symbols,
               ANS_LOG_TAB_SIZE, symbols, writer);
  return;
}

class ANSCoder {
 public:
  ANSCoder() : state_(ANS_SIGNATURE << 16) {}

  uint32_t PutSymbol(const ANSEncSymbolInfo& t, uint8_t* nbits) {
    uint32_t bits = 0;
    *nbits = 0;
    if ((state_ >> (32 - ANS_LOG_TAB_SIZE)) >= t.freq_) {
      bits = state_ & 0xffff;
      state_ >>= 16;
      *nbits = 16;
    }
    // We use mult-by-reciprocal trick, but that requires 64b calc.
    const uint32_t v = (state_ * t.ifreq_) >> RECIPROCAL_PRECISION;
    const uint32_t offset = t.reverse_map_[state_ - v * t.freq_];
    state_ = (v << ANS_LOG_TAB_SIZE) + offset;
    return bits;
  }

  uint32_t GetState() const { return state_; }

 private:
  uint32_t state_;
};

}  // namespace

void WriteHistograms(const std::vector<Histogram>& histograms,
                     EntropyEncodingData* codes, BitWriter* writer) {
  BitWriter::Allotment allotment(writer, 1024 + histograms.size() * 16);
  writer->Write(1, 0);  // use_prefix_code
  writer->Write(2, 3);  // log_alpha_size = 8
  for (size_t i = 0; i < histograms.size(); ++i) {
    writer->Write(4, 4);  // split_exponent
    writer->Write(3, 2);  // msb_in_token
    writer->Write(2, 0);  // lsb_in_token
  }
  allotment.Reclaim(writer);
  for (size_t c = 0; c < histograms.size(); ++c) {
    size_t num_symbol = 0;
    for (size_t i = 0; i < histograms[c].data_.size(); i++) {
      if (histograms[c].data_[i]) num_symbol = i + 1;
    }
    codes->encoding_info.emplace_back();
    codes->encoding_info.back().resize(std::max<size_t>(1, num_symbol));

    BitWriter::Allotment allotment(writer, 256 + num_symbol * 24);
    BuildAndStoreANSEncodingData(histograms[c].data_.data(), num_symbol,
                                 /*log_alpha_size=*/8,
                                 codes->encoding_info.back().data(), writer);
    allotment.Reclaim(writer);
  }
}

void WriteTokens(const std::vector<Token>& tokens,
                 const EntropyEncodingData& codes,
                 const std::vector<uint8_t>& context_map, BitWriter* writer) {
  BitWriter::Allotment allotment(writer, 32 * tokens.size() + 32 * 1024 * 4);
  std::vector<uint64_t> out;
  std::vector<uint8_t> out_nbits;
  out.reserve(tokens.size());
  out_nbits.reserve(tokens.size());
  uint64_t allbits = 0;
  size_t numallbits = 0;
  // Writes in *reversed* order.
  auto addbits = [&](size_t bits, size_t nbits) {
    if (JXL_UNLIKELY(nbits)) {
      JXL_DASSERT(bits >> nbits == 0);
      if (JXL_UNLIKELY(numallbits + nbits > BitWriter::kMaxBitsPerCall)) {
        out.push_back(allbits);
        out_nbits.push_back(numallbits);
        numallbits = allbits = 0;
      }
      allbits <<= nbits;
      allbits |= bits;
      numallbits += nbits;
    }
  };
  const int end = tokens.size();
  ANSCoder ans;
  UintCoder uint_conder;
  if (context_map.size() > 1) {
    for (int i = end - 1; i >= 0; --i) {
      const Token token = tokens[i];
      const uint8_t histo = context_map[token.context];
      uint32_t tok, nbits, bits;
      uint_conder.Encode(tokens[i].value, &tok, &nbits, &bits);
      const ANSEncSymbolInfo& info = codes.encoding_info[histo][tok];
      // Extra bits first as this is reversed.
      addbits(bits, nbits);
      uint8_t ans_nbits = 0;
      uint32_t ans_bits = ans.PutSymbol(info, &ans_nbits);
      addbits(ans_bits, ans_nbits);
    }
  } else {
    for (int i = end - 1; i >= 0; --i) {
      uint32_t tok, nbits, bits;
      uint_conder.Encode(tokens[i].value, &tok, &nbits, &bits);
      const ANSEncSymbolInfo& info = codes.encoding_info[0][tok];
      // Extra bits first as this is reversed.
      addbits(bits, nbits);
      uint8_t ans_nbits = 0;
      uint32_t ans_bits = ans.PutSymbol(info, &ans_nbits);
      addbits(ans_bits, ans_nbits);
    }
  }
  const uint32_t state = ans.GetState();
  writer->Write(32, state);
  writer->Write(numallbits, allbits);
  for (int i = out.size(); i > 0; --i) {
    writer->Write(out_nbits[i - 1], out[i - 1]);
  }
  allotment.Reclaim(writer);
}

}  // namespace jxl
