// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "lib/jxl/enc_ans.h"

#include <stdint.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "lib/jxl/ans_common.h"
#include "lib/jxl/aux_out.h"
#include "lib/jxl/aux_out_fwd.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/enc_cluster.h"
#include "lib/jxl/enc_context_map.h"
#include "lib/jxl/enc_huffman.h"
#include "lib/jxl/fast_math-inl.h"
#include "lib/jxl/fields.h"

namespace jxl {

namespace {

bool ans_fuzzer_friendly_ = false;

static const int kMaxNumSymbolsForSmallCode = 4;

void ANSBuildInfoTable(const ANSHistBin* counts, const AliasTable::Entry* table,
                       size_t alphabet_size, size_t log_alpha_size,
                       ANSEncSymbolInfo* info) {
  size_t log_entry_size = ANS_LOG_TAB_SIZE - log_alpha_size;
  size_t entry_size_minus_1 = (1 << log_entry_size) - 1;
  // create valid alias table for empty streams.
  for (size_t s = 0; s < std::max<size_t>(1, alphabet_size); ++s) {
    const ANSHistBin freq = s == alphabet_size ? ANS_TAB_SIZE : counts[s];
    info[s].freq_ = static_cast<uint16_t>(freq);
#ifdef USE_MULT_BY_RECIPROCAL
    if (freq != 0) {
      info[s].ifreq_ =
          ((1ull << RECIPROCAL_PRECISION) + info[s].freq_ - 1) / info[s].freq_;
    } else {
      info[s].ifreq_ = 1;  // shouldn't matter (symbol shouldn't occur), but...
    }
#endif
    info[s].reverse_map_.resize(freq);
  }
  for (int i = 0; i < ANS_TAB_SIZE; i++) {
    AliasTable::Symbol s =
        AliasTable::Lookup(table, i, log_entry_size, entry_size_minus_1);
    info[s.value].reverse_map_[s.offset] = i;
  }
}

float EstimateDataBits(const ANSHistBin* histogram, const ANSHistBin* counts,
                       size_t len) {
  float sum = 0.0f;
  int total_histogram = 0;
  int total_counts = 0;
  for (size_t i = 0; i < len; ++i) {
    total_histogram += histogram[i];
    total_counts += counts[i];
    if (histogram[i] > 0) {
      JXL_ASSERT(counts[i] > 0);
      // += histogram[i] * -log(counts[i]/total_counts)
      sum += histogram[i] *
             std::max(0.0f, ANS_LOG_TAB_SIZE - FastLog2f(counts[i]));
    }
  }
  if (total_histogram > 0) {
    JXL_ASSERT(total_counts == ANS_TAB_SIZE);
  }
  return sum;
}

float EstimateDataBitsFlat(const ANSHistBin* histogram, size_t len) {
  const float flat_bits = std::max(FastLog2f(len), 0.0f);
  float total_histogram = 0;
  for (size_t i = 0; i < len; ++i) {
    total_histogram += histogram[i];
  }
  return total_histogram * flat_bits;
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
                        uint32_t shift, int* omit_pos, ANSHistBin* counts) {
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
          static_cast<ANSHistBin>(targets[n] * discount_ratio);  // truncate
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
  // detect that, return false and make ANSHistBin uint32_t.
  counts[remainder_pos] -= sum - table_size;
  *omit_pos = remainder_pos;
  return counts[remainder_pos] > 0;
}

Status NormalizeCounts(ANSHistBin* counts, int* omit_pos, const int length,
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

struct SizeWriter {
  size_t size = 0;
  void Write(size_t num, size_t bits) { size += num; }
};

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
void StoreVarLenUint16(size_t n, Writer* writer) {
  JXL_DASSERT(n <= 65535);
  if (n == 0) {
    writer->Write(1, 0);
  } else {
    writer->Write(1, 1);
    size_t nbits = FloorLog2Nonzero(n);
    writer->Write(4, nbits);
    writer->Write(nbits, n - (1ULL << nbits));
  }
}

template <typename Writer>
bool EncodeCounts(const ANSHistBin* counts, const int alphabet_size,
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

void EncodeFlatHistogram(const int alphabet_size, BitWriter* writer) {
  // Mark non-small tree.
  writer->Write(1, 0);
  // Mark uniform histogram.
  writer->Write(1, 1);
  JXL_ASSERT(alphabet_size > 0);
  // Encode alphabet size.
  StoreVarLenUint8(alphabet_size - 1, writer);
}

float ComputeHistoAndDataCost(const ANSHistBin* histogram, size_t alphabet_size,
                              uint32_t method) {
  if (method == 0) {  // Flat code
    return ANS_LOG_TAB_SIZE + 2 +
           EstimateDataBitsFlat(histogram, alphabet_size);
  }
  // Non-flat: shift = method-1.
  uint32_t shift = method - 1;
  std::vector<ANSHistBin> counts(histogram, histogram + alphabet_size);
  int omit_pos = 0;
  int num_symbols;
  int symbols[kMaxNumSymbolsForSmallCode] = {};
  JXL_CHECK(NormalizeCounts(counts.data(), &omit_pos, alphabet_size,
                            ANS_LOG_TAB_SIZE, shift, &num_symbols, symbols));
  SizeWriter writer;
  // Ignore the correctness, no real encoding happens at this stage.
  (void)EncodeCounts(counts.data(), alphabet_size, omit_pos, num_symbols, shift,
                     symbols, &writer);
  return writer.size +
         EstimateDataBits(histogram, counts.data(), alphabet_size);
}

uint32_t ComputeBestMethod(
    const ANSHistBin* histogram, size_t alphabet_size, float* cost,
    HistogramParams::ANSHistogramStrategy ans_histogram_strategy) {
  size_t method = 0;
  float fcost = ComputeHistoAndDataCost(histogram, alphabet_size, 0);
  auto try_shift = [&](size_t shift) {
    float c = ComputeHistoAndDataCost(histogram, alphabet_size, shift + 1);
    if (c < fcost) {
      method = shift + 1;
      fcost = c;
    }
  };
  switch (ans_histogram_strategy) {
    case HistogramParams::ANSHistogramStrategy::kPrecise: {
      for (uint32_t shift = 0; shift <= ANS_LOG_TAB_SIZE; shift++) {
        try_shift(shift);
      }
      break;
    }
    case HistogramParams::ANSHistogramStrategy::kApproximate: {
      for (uint32_t shift = 0; shift <= ANS_LOG_TAB_SIZE; shift += 2) {
        try_shift(shift);
      }
      break;
    }
    case HistogramParams::ANSHistogramStrategy::kFast: {
      try_shift(0);
      try_shift(ANS_LOG_TAB_SIZE / 2);
      try_shift(ANS_LOG_TAB_SIZE);
      break;
    }
  };
  *cost = fcost;
  return method;
}

}  // namespace

// Returns an estimate of the cost of encoding this histogram and the
// corresponding data.
void BuildAndStoreANSEncodingData(
    HistogramParams::ANSHistogramStrategy ans_histogram_strategy,
    const ANSHistBin* histogram, size_t alphabet_size, size_t log_alpha_size,
    bool use_prefix_code, ANSEncSymbolInfo* info, BitWriter* writer) {
  if (use_prefix_code) {
    if (alphabet_size <= 1) return;
    std::vector<uint32_t> histo(alphabet_size);
    for (size_t i = 0; i < alphabet_size; i++) {
      histo[i] = histogram[i];
      JXL_CHECK(histogram[i] >= 0);
    }
    {
      std::vector<uint8_t> depths(alphabet_size);
      std::vector<uint16_t> bits(alphabet_size);
      BuildAndStoreHuffmanTree(histo.data(), alphabet_size, depths.data(),
                               bits.data(), writer);
      for (size_t i = 0; i < alphabet_size; i++) {
        info[i].bits = depths[i] == 0 ? 0 : bits[i];
        info[i].depth = depths[i];
      }
    }
    return;
  }
  JXL_ASSERT(alphabet_size <= ANS_TAB_SIZE);
  // Ensure we ignore trailing zeros in the histogram.
  if (alphabet_size != 0) {
    size_t largest_symbol = 0;
    for (size_t i = 0; i < alphabet_size; i++) {
      if (histogram[i] != 0) largest_symbol = i;
    }
    alphabet_size = largest_symbol + 1;
  }
  float cost;
  uint32_t method = ComputeBestMethod(histogram, alphabet_size, &cost,
                                      ans_histogram_strategy);
  JXL_ASSERT(cost >= 0);
  int num_symbols;
  int symbols[kMaxNumSymbolsForSmallCode] = {};
  std::vector<ANSHistBin> counts(histogram, histogram + alphabet_size);
  if (!counts.empty()) {
    size_t sum = 0;
    for (size_t i = 0; i < counts.size(); i++) {
      sum += counts[i];
    }
    if (sum == 0) {
      counts[0] = ANS_TAB_SIZE;
    }
  }
  if (method == 0) {
    counts = CreateFlatHistogram(alphabet_size, ANS_TAB_SIZE);
    AliasTable::Entry a[ANS_MAX_ALPHABET_SIZE];
    InitAliasTable(counts, ANS_TAB_SIZE, log_alpha_size, a);
    ANSBuildInfoTable(counts.data(), a, alphabet_size, log_alpha_size, info);
    if (writer != nullptr) {
      EncodeFlatHistogram(alphabet_size, writer);
    }
    return;
  }
  int omit_pos = 0;
  uint32_t shift = method - 1;
  JXL_CHECK(NormalizeCounts(counts.data(), &omit_pos, alphabet_size,
                            ANS_LOG_TAB_SIZE, shift, &num_symbols, symbols));
  AliasTable::Entry a[ANS_MAX_ALPHABET_SIZE];
  InitAliasTable(counts, ANS_TAB_SIZE, log_alpha_size, a);
  ANSBuildInfoTable(counts.data(), a, alphabet_size, log_alpha_size, info);
  if (writer != nullptr) {
    bool ok = EncodeCounts(counts.data(), alphabet_size, omit_pos, num_symbols,
                           shift, symbols, writer);
    (void)ok;
    JXL_DASSERT(ok);
  }
  return;
}

template <typename Writer>
void EncodeUintConfig(const HybridUintConfig uint_config, Writer* writer,
                      size_t log_alpha_size) {
  writer->Write(CeilLog2Nonzero(log_alpha_size + 1),
                uint_config.split_exponent);
  if (uint_config.split_exponent == log_alpha_size) {
    return;  // msb/lsb don't matter.
  }
  size_t nbits = CeilLog2Nonzero(uint_config.split_exponent + 1);
  writer->Write(nbits, uint_config.msb_in_token);
  nbits = CeilLog2Nonzero(uint_config.split_exponent -
                          uint_config.msb_in_token + 1);
  writer->Write(nbits, uint_config.lsb_in_token);
}
template <typename Writer>
void EncodeUintConfigs(const std::vector<HybridUintConfig>& uint_config,
                       Writer* writer, size_t log_alpha_size) {
  // TODO(veluca): RLE?
  for (size_t i = 0; i < uint_config.size(); i++) {
    EncodeUintConfig(uint_config[i], writer, log_alpha_size);
  }
}
template void EncodeUintConfigs(const std::vector<HybridUintConfig>&,
                                BitWriter*, size_t);

namespace {

void ChooseUintConfigs(const HistogramParams& params,
                       const std::vector<std::vector<Token>>& tokens,
                       const std::vector<uint8_t>& context_map,
                       std::vector<Histogram>* clustered_histograms,
                       EntropyEncodingData* codes, size_t* log_alpha_size) {
  codes->uint_config.resize(clustered_histograms->size());

  if (params.uint_method == HistogramParams::HybridUintMethod::kNone) return;
  if (params.uint_method == HistogramParams::HybridUintMethod::k000) {
    codes->uint_config.clear();
    codes->uint_config.resize(clustered_histograms->size(),
                              HybridUintConfig(0, 0, 0));
    return;
  }
  if (params.uint_method == HistogramParams::HybridUintMethod::kContextMap) {
    codes->uint_config.clear();
    codes->uint_config.resize(clustered_histograms->size(),
                              HybridUintConfig(2, 0, 1));
    return;
  }
}

class HistogramBuilder {
 public:
  explicit HistogramBuilder(const size_t num_contexts)
      : histograms_(num_contexts) {}

  void VisitSymbol(int symbol, size_t histo_idx) {
    JXL_DASSERT(histo_idx < histograms_.size());
    histograms_[histo_idx].Add(symbol);
  }

  // NOTE: `layer` is only for clustered_entropy; caller does ReclaimAndCharge.
  void BuildAndStoreEntropyCodes(const HistogramParams& params,
                                 const std::vector<std::vector<Token>>& tokens,
                                 EntropyEncodingData* codes,
                                 std::vector<uint8_t>* context_map,
                                 bool use_prefix_code, BitWriter* writer,
                                 size_t layer, AuxOut* aux_out) const {
    codes->encoding_info.clear();
    std::vector<Histogram> clustered_histograms(histograms_);
    context_map->resize(histograms_.size());
    if (histograms_.size() > 1) {
      if (!ans_fuzzer_friendly_) {
        std::vector<uint32_t> histogram_symbols;
        ClusterHistograms(params, histograms_, kClustersLimit,
                          &clustered_histograms, &histogram_symbols);
        for (size_t c = 0; c < histograms_.size(); ++c) {
          (*context_map)[c] = static_cast<uint8_t>(histogram_symbols[c]);
        }
      } else {
        fill(context_map->begin(), context_map->end(), 0);
        size_t max_symbol = 0;
        for (const Histogram& h : histograms_) {
          max_symbol = std::max(h.data_.size(), max_symbol);
        }
        size_t num_symbols = 1 << CeilLog2Nonzero(max_symbol + 1);
        clustered_histograms.resize(1);
        clustered_histograms[0].Clear();
        for (size_t i = 0; i < num_symbols; i++) {
          clustered_histograms[0].Add(i);
        }
      }
      if (writer != nullptr) {
        EncodeContextMap(*context_map, clustered_histograms.size(), writer,
                         layer, aux_out);
      }
    }
    if (aux_out != nullptr) {
      for (size_t i = 0; i < clustered_histograms.size(); ++i) {
        aux_out->layers[layer].clustered_entropy +=
            clustered_histograms[i].ShannonEntropy();
      }
    }
    codes->use_prefix_code = use_prefix_code;
    size_t log_alpha_size = codes->lz77.enabled ? 8 : 7;  // Sane default.
    if (ans_fuzzer_friendly_) {
      codes->uint_config.clear();
      codes->uint_config.resize(1, HybridUintConfig(7, 0, 0));
    } else {
      ChooseUintConfigs(params, tokens, *context_map, &clustered_histograms,
                        codes, &log_alpha_size);
    }
    if (log_alpha_size < 5) log_alpha_size = 5;
    if (writer) writer->Write(1, use_prefix_code);

    if (use_prefix_code) {
      log_alpha_size = PREFIX_MAX_BITS;
    }
    if (writer) {
      if (!use_prefix_code) writer->Write(2, log_alpha_size - 5);
      EncodeUintConfigs(codes->uint_config, writer, log_alpha_size);
    }
    if (use_prefix_code) {
      for (size_t c = 0; c < clustered_histograms.size(); ++c) {
        size_t num_symbol = 1;
        for (size_t i = 0; i < clustered_histograms[c].data_.size(); i++) {
          if (clustered_histograms[c].data_[i]) num_symbol = i + 1;
        }
        if (writer) {
          StoreVarLenUint16(num_symbol - 1, writer);
        }
      }
    }
    for (size_t c = 0; c < clustered_histograms.size(); ++c) {
      size_t num_symbol = 1;
      for (size_t i = 0; i < clustered_histograms[c].data_.size(); i++) {
        if (clustered_histograms[c].data_[i]) num_symbol = i + 1;
      }
      codes->encoding_info.emplace_back();
      codes->encoding_info.back().resize(std::max<size_t>(1, num_symbol));

      BitWriter::Allotment allotment(writer, 256 + num_symbol * 24);
      BuildAndStoreANSEncodingData(params.ans_histogram_strategy,
                                   clustered_histograms[c].data_.data(),
                                   num_symbol, log_alpha_size, use_prefix_code,
                                   codes->encoding_info.back().data(), writer);
      allotment.FinishedHistogram(writer);
      ReclaimAndCharge(writer, &allotment, layer, aux_out);
    }
    return;
  }

  const Histogram& Histo(size_t i) const { return histograms_[i]; }

 private:
  std::vector<Histogram> histograms_;
};

class SymbolCostEstimator {
 public:
  SymbolCostEstimator(size_t num_contexts, bool force_huffman,
                      const std::vector<std::vector<Token>>& tokens,
                      const LZ77Params& lz77) {
    HistogramBuilder builder(num_contexts);
    // Build histograms for estimating lz77 savings.
    HybridUintConfig uint_config;
    for (size_t i = 0; i < tokens.size(); ++i) {
      for (size_t j = 0; j < tokens[i].size(); ++j) {
        const Token token = tokens[i][j];
        uint32_t tok, nbits, bits;
        (token.is_lz77_length ? lz77.length_uint_config : uint_config)
            .Encode(token.value, &tok, &nbits, &bits);
        tok += token.is_lz77_length ? lz77.min_symbol : 0;
        builder.VisitSymbol(tok, token.context);
      }
    }
    max_alphabet_size_ = 0;
    for (size_t i = 0; i < num_contexts; i++) {
      max_alphabet_size_ =
          std::max(max_alphabet_size_, builder.Histo(i).data_.size());
    }
    bits_.resize(num_contexts * max_alphabet_size_);
    // TODO(veluca): SIMD?
    add_symbol_cost_.resize(num_contexts);
    for (size_t i = 0; i < num_contexts; i++) {
      float inv_total = 1.0f / (builder.Histo(i).total_count_ + 1e-8f);
      float total_cost = 0;
      for (size_t j = 0; j < builder.Histo(i).data_.size(); j++) {
        size_t cnt = builder.Histo(i).data_[j];
        float cost = 0;
        if (cnt != 0 && cnt != builder.Histo(i).total_count_) {
          cost = -FastLog2f(cnt * inv_total);
          if (force_huffman) cost = std::ceil(cost);
        } else if (cnt == 0) {
          cost = ANS_LOG_TAB_SIZE;  // Highest possible cost.
        }
        bits_[i * max_alphabet_size_ + j] = cost;
        total_cost += cost * builder.Histo(i).data_[j];
      }
      // Penalty for adding a lz77 symbol to this contest (only used for static
      // cost model). Higher penalty for contexts that have a very low
      // per-symbol entropy.
      add_symbol_cost_[i] = std::max(0.0f, 6.0f - total_cost * inv_total);
    }
  }
  float Bits(size_t ctx, size_t sym) const {
    return bits_[ctx * max_alphabet_size_ + sym];
  }
  float LenCost(size_t ctx, size_t len, const LZ77Params& lz77) const {
    uint32_t nbits, bits, tok;
    lz77.length_uint_config.Encode(len, &tok, &nbits, &bits);
    tok += lz77.min_symbol;
    return nbits + Bits(ctx, tok);
  }
  float DistCost(size_t len, const LZ77Params& lz77) const {
    uint32_t nbits, bits, tok;
    HybridUintConfig().Encode(len, &tok, &nbits, &bits);
    return nbits + Bits(lz77.nonserialized_distance_context, tok);
  }
  float AddSymbolCost(size_t idx) const { return add_symbol_cost_[idx]; }

 private:
  size_t max_alphabet_size_;
  std::vector<float> bits_;
  std::vector<float> add_symbol_cost_;
};

void ApplyLZ77_RLE(const HistogramParams& params, size_t num_contexts,
                   const std::vector<std::vector<Token>>& tokens,
                   LZ77Params& lz77,
                   std::vector<std::vector<Token>>& tokens_lz77) {
  // TODO(veluca): tune heuristics here.
  SymbolCostEstimator sce(num_contexts, params.force_huffman, tokens, lz77);
  float bit_decrease = 0;
  size_t total_symbols = 0;
  tokens_lz77.resize(tokens.size());
  std::vector<float> sym_cost;
  HybridUintConfig uint_config;
  for (size_t stream = 0; stream < tokens.size(); stream++) {
    size_t distance_multiplier =
        params.image_widths.size() > stream ? params.image_widths[stream] : 0;
    const auto& in = tokens[stream];
    auto& out = tokens_lz77[stream];
    total_symbols += in.size();
    // Cumulative sum of bit costs.
    sym_cost.resize(in.size() + 1);
    for (size_t i = 0; i < in.size(); i++) {
      uint32_t tok, nbits, unused_bits;
      uint_config.Encode(in[i].value, &tok, &nbits, &unused_bits);
      sym_cost[i + 1] = sce.Bits(in[i].context, tok) + nbits + sym_cost[i];
    }
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); i++) {
      size_t num_to_copy = 0;
      size_t distance_symbol = 0;  // 1 for RLE.
      if (distance_multiplier != 0) {
        distance_symbol = 1;  // Special distance 1 if enabled.
        JXL_DASSERT(kSpecialDistances[1][0] == 1);
        JXL_DASSERT(kSpecialDistances[1][1] == 0);
      }
      if (i > 0) {
        for (; i + num_to_copy < in.size(); num_to_copy++) {
          if (in[i + num_to_copy].value != in[i - 1].value) {
            break;
          }
        }
      }
      if (num_to_copy == 0) {
        out.push_back(in[i]);
        continue;
      }
      float cost = sym_cost[i + num_to_copy] - sym_cost[i];
      // This subtraction might overflow, but that's OK.
      size_t lz77_len = num_to_copy - lz77.min_length;
      float lz77_cost = num_to_copy >= lz77.min_length
                            ? CeilLog2Nonzero(lz77_len + 1) + 1
                            : 0;
      if (num_to_copy < lz77.min_length || cost <= lz77_cost) {
        for (size_t j = 0; j < num_to_copy; j++) {
          out.push_back(in[i + j]);
        }
        i += num_to_copy - 1;
        continue;
      }
      // Output the LZ77 length
      out.emplace_back(in[i].context, lz77_len);
      out.back().is_lz77_length = true;
      i += num_to_copy - 1;
      bit_decrease += cost - lz77_cost;
      // Output the LZ77 copy distance.
      out.emplace_back(lz77.nonserialized_distance_context, distance_symbol);
    }
  }

  if (bit_decrease > total_symbols * 0.2 + 16) {
    lz77.enabled = true;
  }
}

void ApplyLZ77(const HistogramParams& params, size_t num_contexts,
               const std::vector<std::vector<Token>>& tokens, LZ77Params& lz77,
               std::vector<std::vector<Token>>& tokens_lz77) {
  lz77.enabled = false;
  if (params.force_huffman) {
    lz77.min_symbol = std::min(PREFIX_MAX_ALPHABET_SIZE - 32, 512);
  } else {
    lz77.min_symbol = 224;
  }
  if (params.lz77_method == HistogramParams::LZ77Method::kNone) {
    return;
  }
  ApplyLZ77_RLE(params, num_contexts, tokens, lz77, tokens_lz77);
}
}  // namespace

void BuildAndEncodeHistograms(const HistogramParams& params,
                              size_t num_contexts,
                              std::vector<std::vector<Token>>& tokens,
                              EntropyEncodingData* codes,
                              std::vector<uint8_t>* context_map,
                              BitWriter* writer, size_t layer,
                              AuxOut* aux_out) {
  codes->lz77.nonserialized_distance_context = num_contexts;
  std::vector<std::vector<Token>> tokens_lz77;
  ApplyLZ77(params, num_contexts, tokens, codes->lz77, tokens_lz77);
  if (ans_fuzzer_friendly_) {
    codes->lz77.length_uint_config = HybridUintConfig(10, 0, 0);
    codes->lz77.min_symbol = 2048;
  }

  const size_t max_contexts = std::min(num_contexts, kClustersLimit);
  BitWriter::Allotment allotment(writer,
                                 128 + num_contexts * 40 + max_contexts * 96);
  if (writer) {
    JXL_CHECK(Bundle::Write(codes->lz77, writer, layer, aux_out));
  }
  if (codes->lz77.enabled) {
    if (writer) {
      EncodeUintConfig(codes->lz77.length_uint_config, writer,
                       /*log_alpha_size=*/8);
    }
    num_contexts += 1;
    tokens = std::move(tokens_lz77);
  }
  size_t total_tokens = 0;
  // Build histograms.
  HistogramBuilder builder(num_contexts);
  HybridUintConfig uint_config;  //  Default config for clustering.
  // Unless we are using the kContextMap histogram option.
  if (params.uint_method == HistogramParams::HybridUintMethod::kContextMap) {
    uint_config = HybridUintConfig(2, 0, 1);
  }
  if (params.uint_method == HistogramParams::HybridUintMethod::k000) {
    uint_config = HybridUintConfig(0, 0, 0);
  }
  if (ans_fuzzer_friendly_) {
    uint_config = HybridUintConfig(10, 0, 0);
  }
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (codes->lz77.enabled) {
      for (size_t j = 0; j < tokens[i].size(); ++j) {
        const Token& token = tokens[i][j];
        total_tokens++;
        uint32_t tok, nbits, bits;
        (token.is_lz77_length ? codes->lz77.length_uint_config : uint_config)
            .Encode(token.value, &tok, &nbits, &bits);
        tok += token.is_lz77_length ? codes->lz77.min_symbol : 0;
        builder.VisitSymbol(tok, token.context);
      }
    } else if (num_contexts == 1) {
      for (size_t j = 0; j < tokens[i].size(); ++j) {
        const Token& token = tokens[i][j];
        total_tokens++;
        uint32_t tok, nbits, bits;
        uint_config.Encode(token.value, &tok, &nbits, &bits);
        builder.VisitSymbol(tok, /*token.context=*/0);
      }
    } else {
      for (size_t j = 0; j < tokens[i].size(); ++j) {
        const Token& token = tokens[i][j];
        total_tokens++;
        uint32_t tok, nbits, bits;
        uint_config.Encode(token.value, &tok, &nbits, &bits);
        builder.VisitSymbol(tok, token.context);
      }
    }
  }

  bool use_prefix_code =
      params.force_huffman || total_tokens < 100 ||
      params.clustering == HistogramParams::ClusteringType::kFastest ||
      ans_fuzzer_friendly_;
  if (!use_prefix_code) {
    bool all_singleton = true;
    for (size_t i = 0; i < num_contexts; i++) {
      if (builder.Histo(i).ShannonEntropy() >= 1e-5) {
        all_singleton = false;
      }
    }
    if (all_singleton) {
      use_prefix_code = true;
    }
  }

  // Encode histograms.
  builder.BuildAndStoreEntropyCodes(params, tokens, codes, context_map,
                                    use_prefix_code, writer, layer, aux_out);
  allotment.FinishedHistogram(writer);
  ReclaimAndCharge(writer, &allotment, layer, aux_out);

  if (aux_out != nullptr) {
    aux_out->layers[layer].num_clustered_histograms +=
        codes->encoding_info.size();
  }
}

size_t WriteTokens(const std::vector<Token>& tokens,
                   const EntropyEncodingData& codes,
                   const std::vector<uint8_t>& context_map, BitWriter* writer) {
  size_t num_extra_bits = 0;
  if (codes.use_prefix_code) {
    for (size_t i = 0; i < tokens.size(); i++) {
      uint32_t tok, nbits, bits;
      const Token& token = tokens[i];
      size_t histo = context_map[token.context];
      (token.is_lz77_length ? codes.lz77.length_uint_config
                            : codes.uint_config[histo])
          .Encode(token.value, &tok, &nbits, &bits);
      tok += token.is_lz77_length ? codes.lz77.min_symbol : 0;
      // Combine two calls to the BitWriter. Equivalent to:
      // writer->Write(codes.encoding_info[histo][tok].depth,
      //               codes.encoding_info[histo][tok].bits);
      // writer->Write(nbits, bits);
      uint64_t data = codes.encoding_info[histo][tok].bits;
      data |= bits << codes.encoding_info[histo][tok].depth;
      writer->Write(codes.encoding_info[histo][tok].depth + nbits, data);
      num_extra_bits += nbits;
    }
    return num_extra_bits;
  }
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
  if (codes.lz77.enabled || context_map.size() > 1) {
    for (int i = end - 1; i >= 0; --i) {
      const Token token = tokens[i];
      const uint8_t histo = context_map[token.context];
      uint32_t tok, nbits, bits;
      (token.is_lz77_length ? codes.lz77.length_uint_config
                            : codes.uint_config[histo])
          .Encode(tokens[i].value, &tok, &nbits, &bits);
      tok += token.is_lz77_length ? codes.lz77.min_symbol : 0;
      const ANSEncSymbolInfo& info = codes.encoding_info[histo][tok];
      // Extra bits first as this is reversed.
      addbits(bits, nbits);
      num_extra_bits += nbits;
      uint8_t ans_nbits = 0;
      uint32_t ans_bits = ans.PutSymbol(info, &ans_nbits);
      addbits(ans_bits, ans_nbits);
    }
  } else {
    for (int i = end - 1; i >= 0; --i) {
      uint32_t tok, nbits, bits;
      codes.uint_config[0].Encode(tokens[i].value, &tok, &nbits, &bits);
      const ANSEncSymbolInfo& info = codes.encoding_info[0][tok];
      // Extra bits first as this is reversed.
      addbits(bits, nbits);
      num_extra_bits += nbits;
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
  return num_extra_bits;
}

void WriteTokens(const std::vector<Token>& tokens,
                 const EntropyEncodingData& codes,
                 const std::vector<uint8_t>& context_map, BitWriter* writer,
                 size_t layer, AuxOut* aux_out) {
  BitWriter::Allotment allotment(writer, 32 * tokens.size() + 32 * 1024 * 4);
  size_t num_extra_bits = WriteTokens(tokens, codes, context_map, writer);
  ReclaimAndCharge(writer, &allotment, layer, aux_out);
  if (aux_out != nullptr) {
    aux_out->layers[layer].extra_bits += num_extra_bits;
  }
}

void SetANSFuzzerFriendly(bool ans_fuzzer_friendly) {
#if JXL_IS_DEBUG_BUILD  // Guard against accidental / malicious changes.
  ans_fuzzer_friendly_ = ans_fuzzer_friendly;
#endif
}
}  // namespace jxl
