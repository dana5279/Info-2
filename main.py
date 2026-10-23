

import heapq
import itertools
import json
import os
import struct
from collections import Counter

OTHERS = "OTHERS"  

INPUT_FILE = "input.txt"             
COMPRESSED_FILE = "compressed.bin"   
DECOMPRESSED_FILE = "decompressed.txt"  

def build_huffman_codes(freqs):
    """freqs: {symbol: count}  ->  {symbol: '0101'} (standard Huffman)."""
    if len(freqs) == 1:                      
        return {next(iter(freqs)): "0"}

    tie = itertools.count()                  
    heap = [(c, next(tie), s) for s, c in freqs.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        c1, _, left = heapq.heappop(heap)    
        c2, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (c1 + c2, next(tie), (left, right)))

    codes = {}

    def walk(node, path):
        if isinstance(node, tuple):          
            walk(node[0], path + "0")
            walk(node[1], path + "1")
        else:                                
            codes[node] = path

    walk(heap[0][2], "")
    return codes


def compress(in_path, out_path, limit):
    with open(in_path, "rb") as f:
        data = f.read()

    total = len(data)
    counts = Counter(data)

    frequent = {s: c for s, c in counts.items() if c / total > limit}
    others = sorted(s for s, c in counts.items() if c / total <= limit)

    tree_freqs = dict(frequent)
    if others:
        tree_freqs[OTHERS] = sum(counts[s] for s in others)

   
    codes = build_huffman_codes(tree_freqs) if tree_freqs else {}

   
    width = (len(others) - 1).bit_length() if others else 0
    orig_code = {s: format(i, "0{}b".format(width)) if width else ""
                 for i, s in enumerate(others)}

   
    full_code = {}
    for s in frequent:
        full_code[s] = codes[s]
    for s in others:
        full_code[s] = codes[OTHERS] + orig_code[s]

    out = bytearray()
    acc = 0          
    nbits = 0        
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

    
    header = {
        "symbols": total,
        "padding": padding,
        "codes": {str(s): codes[s] for s in frequent},
        "others_code": codes.get(OTHERS),
        "others": others,
        "width": width,
    }
    header_bytes = json.dumps(header).encode("utf-8")

    with open(out_path, "wb") as f:
        f.write(struct.pack(">I", len(header_bytes)))       
        f.write(header_bytes)
        f.write(out)


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



def read_text():
    """Read several lines from the user until an empty line is entered."""
    print("Type your text. Press Enter on an empty line to finish:")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)


def read_limit():
    raw = input("Minimum limit of symbol probability (default 0.05): ").strip()
    if not raw:
        return 0.05
    try:
        value = float(raw)
        if 0 <= value < 1:
            return value
    except ValueError:
        pass
    print("Invalid value, using 0.05")
    return 0.05


def main():
    while True:
        print("\n===== Modified Huffman =====")
        print("1) Compress")
        print("2) Decompress")
        print("3) Exit")
        choice = input("Choose: ").strip()

        if choice == "1":
            text = read_text()
            if not text:
                print("Nothing to compress.")
                continue
            with open(INPUT_FILE, "w", encoding="utf-8") as f:
                f.write(text)
            print("Saved your text to", INPUT_FILE)
            limit = read_limit()
            compress(INPUT_FILE, COMPRESSED_FILE, limit)

        elif choice == "2":
            if not os.path.exists(COMPRESSED_FILE):
                print("No", COMPRESSED_FILE, "found. Compress something first.")
                continue
            decompress(COMPRESSED_FILE, DECOMPRESSED_FILE)

        elif choice == "3":
            print("Bye!")
            break

        else:
            print("Invalid choice, enter 1, 2 or 3.")


if __name__ == "__main__":
    main()