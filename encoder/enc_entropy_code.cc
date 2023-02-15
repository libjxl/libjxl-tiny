// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/enc_entropy_code.h"

#include "encoder/enc_cluster.h"
#include "encoder/enc_huffman_tree.h"
#include "encoder/histogram.h"

namespace jxl {

namespace {

constexpr int kCodeLengthCodes = 18;

void StoreHuffmanTreeOfHuffmanTreeToBitMask(const int num_codes,
                                            const uint8_t* code_length_bitdepth,
                                            BitWriter* writer) {
  static const uint8_t kStorageOrder[kCodeLengthCodes] = {
      1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  // The bit lengths of the Huffman code over the code length alphabet
  // are compressed with the following static Huffman code:
  //   Symbol   Code
  //   ------   ----
  //   0          00
  //   1        1110
  //   2         110
  //   3          01
  //   4          10
  //   5        1111
  static const uint8_t kHuffmanBitLengthHuffmanCodeSymbols[6] = {0, 7, 3,
                                                                 2, 1, 15};
  static const uint8_t kHuffmanBitLengthHuffmanCodeBitLengths[6] = {2, 4, 3,
                                                                    2, 2, 4};

  // Throw away trailing zeros:
  size_t codes_to_store = kCodeLengthCodes;
  if (num_codes > 1) {
    for (; codes_to_store > 0; --codes_to_store) {
      if (code_length_bitdepth[kStorageOrder[codes_to_store - 1]] != 0) {
        break;
      }
    }
  }
  size_t skip_some = 0;  // skips none.
  if (code_length_bitdepth[kStorageOrder[0]] == 0 &&
      code_length_bitdepth[kStorageOrder[1]] == 0) {
    skip_some = 2;  // skips two.
    if (code_length_bitdepth[kStorageOrder[2]] == 0) {
      skip_some = 3;  // skips three.
    }
  }
  writer->Write(2, skip_some);
  for (size_t i = skip_some; i < codes_to_store; ++i) {
    size_t l = code_length_bitdepth[kStorageOrder[i]];
    writer->Write(kHuffmanBitLengthHuffmanCodeBitLengths[l],
                  kHuffmanBitLengthHuffmanCodeSymbols[l]);
  }
}

void StoreHuffmanTreeToBitMask(const size_t huffman_tree_size,
                               const uint8_t* huffman_tree,
                               const uint8_t* huffman_tree_extra_bits,
                               const uint8_t* code_length_bitdepth,
                               const uint16_t* code_length_bitdepth_symbols,
                               BitWriter* writer) {
  for (size_t i = 0; i < huffman_tree_size; ++i) {
    size_t ix = huffman_tree[i];
    writer->Write(code_length_bitdepth[ix], code_length_bitdepth_symbols[ix]);
    // Extra bits
    switch (ix) {
      case 16:
        writer->Write(2, huffman_tree_extra_bits[i]);
        break;
      case 17:
        writer->Write(3, huffman_tree_extra_bits[i]);
        break;
    }
  }
}

void StoreSimpleHuffmanTree(const uint8_t* depths, size_t symbols[4],
                            size_t num_symbols, size_t max_bits,
                            BitWriter* writer) {
  // value of 1 indicates a simple Huffman code
  writer->Write(2, 1);
  writer->Write(2, num_symbols - 1);  // NSYM - 1

  // Sort
  for (size_t i = 0; i < num_symbols; i++) {
    for (size_t j = i + 1; j < num_symbols; j++) {
      if (depths[symbols[j]] < depths[symbols[i]]) {
        std::swap(symbols[j], symbols[i]);
      }
    }
  }

  if (num_symbols == 2) {
    writer->Write(max_bits, symbols[0]);
    writer->Write(max_bits, symbols[1]);
  } else if (num_symbols == 3) {
    writer->Write(max_bits, symbols[0]);
    writer->Write(max_bits, symbols[1]);
    writer->Write(max_bits, symbols[2]);
  } else {
    writer->Write(max_bits, symbols[0]);
    writer->Write(max_bits, symbols[1]);
    writer->Write(max_bits, symbols[2]);
    writer->Write(max_bits, symbols[3]);
    // tree-select
    writer->Write(1, depths[symbols[0]] == 1 ? 1 : 0);
  }
}

void Reverse(uint8_t* v, size_t start, size_t end) {
  --end;
  while (start < end) {
    uint8_t tmp = v[start];
    v[start] = v[end];
    v[end] = tmp;
    ++start;
    --end;
  }
}

void WriteHuffmanTreeRepetitions(const uint8_t previous_value,
                                 const uint8_t value, size_t repetitions,
                                 size_t* tree_size, uint8_t* tree,
                                 uint8_t* extra_bits_data) {
  JXL_DASSERT(repetitions > 0);
  if (previous_value != value) {
    tree[*tree_size] = value;
    extra_bits_data[*tree_size] = 0;
    ++(*tree_size);
    --repetitions;
  }
  if (repetitions == 7) {
    tree[*tree_size] = value;
    extra_bits_data[*tree_size] = 0;
    ++(*tree_size);
    --repetitions;
  }
  if (repetitions < 3) {
    for (size_t i = 0; i < repetitions; ++i) {
      tree[*tree_size] = value;
      extra_bits_data[*tree_size] = 0;
      ++(*tree_size);
    }
  } else {
    repetitions -= 3;
    size_t start = *tree_size;
    while (true) {
      tree[*tree_size] = 16;
      extra_bits_data[*tree_size] = repetitions & 0x3;
      ++(*tree_size);
      repetitions >>= 2;
      if (repetitions == 0) {
        break;
      }
      --repetitions;
    }
    Reverse(tree, start, *tree_size);
    Reverse(extra_bits_data, start, *tree_size);
  }
}

void WriteHuffmanTreeRepetitionsZeros(size_t repetitions, size_t* tree_size,
                                      uint8_t* tree, uint8_t* extra_bits_data) {
  if (repetitions == 11) {
    tree[*tree_size] = 0;
    extra_bits_data[*tree_size] = 0;
    ++(*tree_size);
    --repetitions;
  }
  if (repetitions < 3) {
    for (size_t i = 0; i < repetitions; ++i) {
      tree[*tree_size] = 0;
      extra_bits_data[*tree_size] = 0;
      ++(*tree_size);
    }
  } else {
    repetitions -= 3;
    size_t start = *tree_size;
    while (true) {
      tree[*tree_size] = 17;
      extra_bits_data[*tree_size] = repetitions & 0x7;
      ++(*tree_size);
      repetitions >>= 3;
      if (repetitions == 0) {
        break;
      }
      --repetitions;
    }
    Reverse(tree, start, *tree_size);
    Reverse(extra_bits_data, start, *tree_size);
  }
}

static void DecideOverRleUse(const uint8_t* depth, const size_t length,
                             bool* use_rle_for_non_zero,
                             bool* use_rle_for_zero) {
  size_t total_reps_zero = 0;
  size_t total_reps_non_zero = 0;
  size_t count_reps_zero = 1;
  size_t count_reps_non_zero = 1;
  for (size_t i = 0; i < length;) {
    const uint8_t value = depth[i];
    size_t reps = 1;
    for (size_t k = i + 1; k < length && depth[k] == value; ++k) {
      ++reps;
    }
    if (reps >= 3 && value == 0) {
      total_reps_zero += reps;
      ++count_reps_zero;
    }
    if (reps >= 4 && value != 0) {
      total_reps_non_zero += reps;
      ++count_reps_non_zero;
    }
    i += reps;
  }
  *use_rle_for_non_zero = total_reps_non_zero > count_reps_non_zero * 2;
  *use_rle_for_zero = total_reps_zero > count_reps_zero * 2;
}

// Write a Huffman tree from bit depths into the bitstream representation
// of a Huffman tree. The generated Huffman tree is to be compressed once
// more using a Huffman tree
void WriteHuffmanTree(const uint8_t* depth, size_t length, size_t* tree_size,
                      uint8_t* tree, uint8_t* extra_bits_data) {
  uint8_t previous_value = 8;

  // Throw away trailing zeros.
  size_t new_length = length;
  for (size_t i = 0; i < length; ++i) {
    if (depth[length - i - 1] == 0) {
      --new_length;
    } else {
      break;
    }
  }

  // First gather statistics on if it is a good idea to do rle.
  bool use_rle_for_non_zero = false;
  bool use_rle_for_zero = false;
  if (length > 50) {
    // Find rle coding for longer codes.
    // Shorter codes seem not to benefit from rle.
    DecideOverRleUse(depth, new_length, &use_rle_for_non_zero,
                     &use_rle_for_zero);
  }

  // Actual rle coding.
  for (size_t i = 0; i < new_length;) {
    const uint8_t value = depth[i];
    size_t reps = 1;
    if ((value != 0 && use_rle_for_non_zero) ||
        (value == 0 && use_rle_for_zero)) {
      for (size_t k = i + 1; k < new_length && depth[k] == value; ++k) {
        ++reps;
      }
    }
    if (value == 0) {
      WriteHuffmanTreeRepetitionsZeros(reps, tree_size, tree, extra_bits_data);
    } else {
      WriteHuffmanTreeRepetitions(previous_value, value, reps, tree_size, tree,
                                  extra_bits_data);
      previous_value = value;
    }
    i += reps;
  }
}

namespace {

uint16_t ReverseBits(int num_bits, uint16_t bits) {
  static const size_t kLut[16] = {// Pre-reversed 4-bit values.
                                  0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe,
                                  0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf};
  size_t retval = kLut[bits & 0xf];
  for (int i = 4; i < num_bits; i += 4) {
    retval <<= 4;
    bits = static_cast<uint16_t>(bits >> 4);
    retval |= kLut[bits & 0xf];
  }
  retval >>= (-num_bits & 0x3);
  return static_cast<uint16_t>(retval);
}

}  // namespace

// Get the actual bit values for a tree of bit depths.
void ConvertBitDepthsToSymbols(const uint8_t* depth, size_t len,
                               uint16_t* bits) {
  // In Brotli, all bit depths are [1..15]
  // 0 bit depth means that the symbol does not exist.
  const int kMaxBits = 16;  // 0..15 are values for bits
  uint16_t bl_count[kMaxBits] = {0};
  {
    for (size_t i = 0; i < len; ++i) {
      ++bl_count[depth[i]];
    }
    bl_count[0] = 0;
  }
  uint16_t next_code[kMaxBits];
  next_code[0] = 0;
  {
    int code = 0;
    for (size_t i = 1; i < kMaxBits; ++i) {
      code = (code + bl_count[i - 1]) << 1;
      next_code[i] = static_cast<uint16_t>(code);
    }
  }
  for (size_t i = 0; i < len; ++i) {
    if (depth[i]) {
      bits[i] = ReverseBits(depth[i], next_code[depth[i]]++);
    }
  }
}

// num = alphabet size
// depths = symbol depths
void StoreHuffmanTree(const uint8_t* depths, size_t num, BitWriter* writer) {
  // Write the Huffman tree into the compact representation.
  std::unique_ptr<uint8_t[]> arena(new uint8_t[2 * num]);
  uint8_t* huffman_tree = arena.get();
  uint8_t* huffman_tree_extra_bits = arena.get() + num;
  size_t huffman_tree_size = 0;
  WriteHuffmanTree(depths, num, &huffman_tree_size, huffman_tree,
                   huffman_tree_extra_bits);

  // Calculate the statistics of the Huffman tree in the compact representation.
  uint32_t huffman_tree_histogram[kCodeLengthCodes] = {0};
  for (size_t i = 0; i < huffman_tree_size; ++i) {
    ++huffman_tree_histogram[huffman_tree[i]];
  }

  int num_codes = 0;
  int code = 0;
  for (int i = 0; i < kCodeLengthCodes; ++i) {
    if (huffman_tree_histogram[i]) {
      if (num_codes == 0) {
        code = i;
        num_codes = 1;
      } else if (num_codes == 1) {
        num_codes = 2;
        break;
      }
    }
  }

  // Calculate another Huffman tree to use for compressing both the
  // earlier Huffman tree with.
  uint8_t code_length_bitdepth[kCodeLengthCodes] = {0};
  uint16_t code_length_bitdepth_symbols[kCodeLengthCodes] = {0};
  CreateHuffmanTree(&huffman_tree_histogram[0], kCodeLengthCodes, 5,
                    &code_length_bitdepth[0]);
  ConvertBitDepthsToSymbols(code_length_bitdepth, kCodeLengthCodes,
                            &code_length_bitdepth_symbols[0]);

  // Now, we have all the data, let's start storing it
  StoreHuffmanTreeOfHuffmanTreeToBitMask(num_codes, code_length_bitdepth,
                                         writer);

  if (num_codes == 1) {
    code_length_bitdepth[code] = 0;
  }

  // Store the real huffman tree now.
  StoreHuffmanTreeToBitMask(huffman_tree_size, huffman_tree,
                            huffman_tree_extra_bits, &code_length_bitdepth[0],
                            code_length_bitdepth_symbols, writer);
}

void StoreVarLenUint16(size_t n, BitWriter* writer) {
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

void WritePrefixCode(const PrefixCode& code, BitWriter* writer) {
  size_t count = 0;
  size_t s4[4] = {0};
  size_t length = 0;
  for (size_t i = 0; i < kAlphabetSize; i++) {
    if (code.depths[i]) {
      if (count < 4) {
        s4[count] = i;
      }
      count++;
      length = i + 1;
    }
  }

  size_t max_bits_counter = length - 1;
  size_t max_bits = 0;
  while (max_bits_counter) {
    max_bits_counter >>= 1;
    ++max_bits;
  }

  if (count <= 1) {
    // Output symbol bits and depths are initialized with 0, nothing to do.
    writer->Write(4, 1);
    writer->Write(max_bits, s4[0]);
    return;
  }

  if (count <= 4) {
    StoreSimpleHuffmanTree(code.depths, s4, count, max_bits, writer);
  } else {
    StoreHuffmanTree(code.depths, length, writer);
  }
}

void WritePrefixCodes(const PrefixCode* prefix_codes, size_t num,
                      BitWriter* writer) {
  BitWriter::Allotment allotment(writer, 1024 + num * 16);
  writer->Write(1, 1);  // use_prefix_code
  for (size_t i = 0; i < num; ++i) {
    writer->Write(4, 4);  // split_exponent
    writer->Write(3, 2);  // msb_in_token
    writer->Write(2, 0);  // lsb_in_token
  }
  for (size_t c = 0; c < num; ++c) {
    size_t num_symbol = 1;
    for (size_t i = 0; i < kAlphabetSize; i++) {
      if (prefix_codes[c].depths[i]) num_symbol = i + 1;
    }
    StoreVarLenUint16(num_symbol - 1, writer);
  }
  allotment.Reclaim(writer);
  for (size_t c = 0; c < num; ++c) {
    size_t num_symbol = 1;
    for (size_t i = 0; i < kAlphabetSize; i++) {
      if (prefix_codes[c].depths[i]) num_symbol = i + 1;
    }
    BitWriter::Allotment allotment(writer, 256 + num_symbol * 24);
    if (num_symbol > 1) {
      WritePrefixCode(prefix_codes[c], writer);
    }
    allotment.Reclaim(writer);
  }
}

void WriteContextMap(const uint8_t* context_map, size_t num_contexts,
                     BitWriter* writer) {
  if (num_contexts == 0) {
    return;
  }
  if (*std::max_element(context_map, context_map + num_contexts) == 0) {
    writer->AllocateAndWrite(3, 1);  // simple code, 0 bits per entry
    return;
  }
  writer->AllocateAndWrite(3, 0);  // no simple code, no MTF, no LZ77
  std::vector<Token> tokens;
  for (size_t i = 0; i < num_contexts; i++) {
    tokens.emplace_back(0, context_map[i]);
  }
  uint8_t dummy_ctx_map = 0;
  EntropyCode code(&dummy_ctx_map, 1, nullptr, 1);
  OptimizePrefixCodes(tokens, &code);
  WritePrefixCodes(code.prefix_codes, code.num_prefix_codes, writer);
  for (const Token& t : tokens) {
    WriteToken(t, code, writer);
  }
}

void BuildHistograms(const std::vector<Token>& tokens,
                     const uint8_t* context_map, size_t num_contexts,
                     std::vector<Histogram>* histograms) {
  for (const Token& t : tokens) {
    uint32_t tok, nbits, bits;
    UintCoder().Encode(t.value, &tok, &nbits, &bits);
    uint32_t context = t.context;
    if (context_map) {
      JXL_ASSERT(context < num_contexts);
      context = context_map[context];
    }
    JXL_ASSERT(context < histograms->size());
    JXL_ASSERT(tok < kAlphabetSize);
    (*histograms)[context].Add(tok);
  }
}

void BuildHuffmanCodes(const std::vector<Histogram>& histograms,
                       EntropyCode* code) {
  code->num_prefix_codes = histograms.size();
  code->prefix_code_storage.resize(histograms.size());
  for (size_t i = 0; i < code->num_prefix_codes; ++i) {
    PrefixCode& prefix_code = code->prefix_code_storage[i];
    const uint32_t* counts = histograms[i].counts;
    size_t length = kAlphabetSize;
    while (length > 0 && counts[length - 1] == 0) --length;
    CreateHuffmanTree(counts, length, 15, prefix_code.depths);
    ConvertBitDepthsToSymbols(prefix_code.depths, length, prefix_code.bits);
  }
  code->prefix_codes = code->prefix_code_storage.data();
}

}  // namespace

void OptimizePrefixCodes(const std::vector<Token>& tokens, EntropyCode* code) {
  std::vector<Histogram> histograms(code->num_prefix_codes);
  BuildHistograms(tokens, code->context_map, code->num_contexts, &histograms);
  BuildHuffmanCodes(histograms, code);
}

void OptimizeEntropyCode(const std::vector<Token>& tokens, EntropyCode* code) {
  std::vector<Histogram> histograms(code->num_contexts);
  BuildHistograms(tokens, nullptr, code->num_contexts, &histograms);
  ClusterHistograms(&histograms, &code->context_map_storage);
  code->context_map = code->context_map_storage.data();
  JXL_ASSERT(code->context_map_storage.size() == code->num_contexts);
  BuildHuffmanCodes(histograms, code);
}

void WriteEntropyCode(const EntropyCode& code, BitWriter* writer) {
  WriteContextMap(code.context_map, code.num_contexts, writer);
  WritePrefixCodes(code.prefix_codes, code.num_prefix_codes, writer);
}

}  // namespace jxl
