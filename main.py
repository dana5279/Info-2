"""
Modified Huffman - COMPRESSION

Lecture idea:
  * Pick a Minimum Limit of Symbol Probabilities (e.g. 0.05).
  * Symbols with probability <= limit are grouped into one group: "Others".
  * Build a normal Huffman tree over: frequent symbols + "Others".
  * Frequent symbol  -> its Huffman code.
  * Symbol in Others -> Huffman code of "Others" + its ORIGINAL (fixed-length) code.

Usage (from your main file):
    from huffman_compress import compress
    compress("input.txt", "out.bin", 0.05)

Compressed file layout (for the decompression code):
    [4 bytes big-endian: header length N]
    [N bytes: JSON header]
    [packed code bits, last byte padded with zeros]

JSON header keys:
    symbols     original number of bytes
    padding     number of unused bits at the end of the last byte
    codes       {"<byte value>": "<huffman code>"} for frequent symbols
    others_code Huffman code of the Others group (null if no Others)
    others      sorted list of byte values inside Others
    width       bits of the original code; index of the symbol in `others`
                written in binary with `width` bits (0 bits if only one)
"""

import heapq
import itertools
import json
import math
import struct
from collections import Counter

OTHERS = "OTHERS"  # name of the grouped symbol inside the Huffman tree


# ---------------------------------------------------------------- Huffman
def build_huffman_codes(freqs):
    """freqs: {symbol: count}  ->  {symbol: '0101'} (standard Huffman)."""
    if len(freqs) == 1:                      # only one symbol -> give it "0"
        return {next(iter(freqs)): "0"}

    tie = itertools.count()                  # keeps heap order deterministic
    heap = [(c, next(tie), s) for s, c in freqs.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        c1, _, left = heapq.heappop(heap)    # two lowest probabilities
        c2, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (c1 + c2, next(tie), (left, right)))

    codes = {}

    def walk(node, path):
        if isinstance(node, tuple):          # internal node
            walk(node[0], path + "0")
            walk(node[1], path + "1")
        else:                                # leaf
            codes[node] = path

    walk(heap[0][2], "")
    return codes


# ------------------------------------------------------------ compression
def compress(in_path, out_path, limit):
    with open(in_path, "rb") as f:
        data = f.read()

    total = len(data)
    counts = Counter(data)

    # 1) split symbols: frequent vs "Others" (probability <= limit)
    frequent = {s: c for s, c in counts.items() if c / total > limit}
    others = sorted(s for s, c in counts.items() if c / total <= limit)

    tree_freqs = dict(frequent)
    if others:
        tree_freqs[OTHERS] = sum(counts[s] for s in others)

    # 2) Huffman codes (frequent symbols + Others)
    codes = build_huffman_codes(tree_freqs) if tree_freqs else {}

    # 3) original (fixed-length) code for symbols inside Others
    width = (len(others) - 1).bit_length() if others else 0
    orig_code = {s: format(i, "0{}b".format(width)) if width else ""
                 for i, s in enumerate(others)}

    # 4) encode the whole file into a bit string, then pack into bytes
    full_code = {}
    for s in frequent:
        full_code[s] = codes[s]
    for s in others:
        full_code[s] = codes[OTHERS] + orig_code[s]

    out = bytearray()
    acc = 0          # bit accumulator
    nbits = 0        # bits currently in accumulator
    total_bits = 0
    for b in data:
        code = full_code[b]
        acc = (acc << len(code)) | int(code, 2)
        nbits += len(code)
        total_bits += len(code)
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)
        acc &= (1 << nbits) - 1
    padding = (8 - nbits) % 8
    if nbits:
        out.append((acc << padding) & 0xFF)

    # 5) header (overhead, any format is allowed) -> JSON, then the bits
    header = {
        "symbols": total,                                    # original byte count
        "padding": padding,                                  # unused bits in last byte
        "codes": {str(s): codes[s] for s in frequent},       # symbol -> huffman code
        "others_code": codes.get(OTHERS),                    # huffman code of Others
        "others": others,                                    # list of Others symbols
        "width": width,                                      # bits of original code
    }
    header_bytes = json.dumps(header).encode("utf-8")

    with open(out_path, "wb") as f:
        f.write(struct.pack(">I", len(header_bytes)))        # header length
        f.write(header_bytes)
        f.write(out)

    # 6) report
    print("Limit:", limit)
    print("Symbol  Prob     Code")
    for s in sorted(frequent, key=lambda x: -frequent[x]):
        print("{:<7} {:<8.4f} {}".format(repr(chr(s)), frequent[s] / total, codes[s]))
    if others:
        p = tree_freqs[OTHERS] / total
        print("{:<7} {:<8.4f} {}  (+ {} bit original code)".format(
            "Others", p, codes[OTHERS], width))
    print("Original size  :", total * 8, "bits")
    print("Compressed bits:", total_bits, "(+ header", len(header_bytes) + 4, "bytes)")
    print("Written to     :", out_path)


import json
import struct


# ------------------------------------------------------------ decompression
def decompress(in_path, out_path):
    with open(in_path, "rb") as f:
        raw = f.read()

    
    header_len = struct.unpack(">I", raw[:4])[0]
    header = json.loads(raw[4:4 + header_len].decode("utf-8"))
    stream = raw[4 + header_len:]

    total = header["symbols"]        
    padding = header["padding"]      
    others = header["others"]        
    others_code = header["others_code"]
    width = header["width"]          

    
    decode = {code: int(s) for s, code in header["codes"].items()}

    
    bits = "".join(format(b, "08b") for b in stream)
    if padding:
        bits = bits[:-padding]

    
    out = bytearray()
    cur = ""
    i = 0
    while len(out) < total:
        cur += bits[i]
        i += 1
        if cur in decode:                         
            out.append(decode[cur])
            cur = ""
        elif others_code is not None and cur == others_code:
            
            idx = int(bits[i:i + width], 2) if width else 0
            i += width
            out.append(others[idx])
            cur = ""

    with open(out_path, "wb") as f:
        f.write(out)

    print("Decoded symbols:", len(out))
    print("Written to     :", out_path)




