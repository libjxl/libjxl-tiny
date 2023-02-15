// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Library for creating Huffman codes from population counts.

#ifndef ENCODER_ENC_HUFFMAN_TREE_H_
#define ENCODER_ENC_HUFFMAN_TREE_H_

#include <stdint.h>
#include <stdlib.h>

namespace jxl {

// This function will create a Huffman tree.
//
// The (data,length) contains the population counts.
// The tree_limit is the maximum bit depth of the Huffman codes.
//
// The depth contains the tree, i.e., how many bits are used for
// the symbol.
//
// See http://en.wikipedia.org/wiki/Huffman_coding
void CreateHuffmanTree(const uint32_t* data, const size_t length,
                       const int tree_limit, uint8_t* depth);

}  // namespace jxl
#endif  // ENCODER_ENC_HUFFMAN_TREE_H_
