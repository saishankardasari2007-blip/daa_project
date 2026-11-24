# Lossless Image Compression with Huffman Coding Before IPFS Storage

C program that compresses image pixel data using Huffman coding (greedy
algorithm, min-heap, prefix codes), stores it by SHA-256 hash (simulated
IPFS) and anchors the hash in a ledger (simulated blockchain).

## Build
gcc -O2 -o c_code c_code.c -lm

## Run
./c_code demo photo.bmp