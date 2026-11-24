/*
 * Lossless Image Compression with Huffman Coding Before IPFS Storage
 * ------------------------------------------------------------------
 * DAA concepts: Greedy algorithm, priority queue (min-heap), prefix codes,
 *               optimality of Huffman coding.
 *
 * Build : gcc -O2 -o huffman_ipfs huffman_ipfs.c -lm
 * Usage :
 *   ./huffman_ipfs compress   <input>  <output.huf>
 *   ./huffman_ipfs decompress <input.huf> <output>
 *   ./huffman_ipfs demo       <input>        (full pipeline: compress ->
 *                                             hash -> "IPFS" store -> ledger
 *                                             -> retrieve -> verify)
 *   ./huffman_ipfs retrieve   <hash> <output>
 *
 * Works on ANY file's bytes. For best results on images use raw-pixel
 * formats (BMP / PPM / PGM); PNG/JPG are already compressed so gains are small.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(p) _mkdir(p)
#else
#include <sys/stat.h>
#define MKDIR(p) mkdir(p, 0755)
#endif

#define ALPHABET 256
#define MAGIC "HUF1"
#define STORE_DIR "ipfs_store"
#define LEDGER_FILE "ledger.txt"

/* =====================================================================
 *  1. HUFFMAN TREE NODE
 * ===================================================================== */
typedef struct Node {
    uint64_t freq;
    int symbol;          /* 0..255 for leaves, -1 for internal nodes */
    int order;           /* deterministic tie-breaker (same in encoder/decoder) */
    struct Node *left, *right;
} Node;

static Node *new_node(uint64_t freq, int symbol, int order, Node *l, Node *r) {
    Node *n = (Node *)malloc(sizeof(Node));
    if (!n) { fprintf(stderr, "Out of memory\n"); exit(1); }
    n->freq = freq; n->symbol = symbol; n->order = order;
    n->left = l; n->right = r;
    return n;
}

static void free_tree(Node *n) {
    if (!n) return;
    free_tree(n->left);
    free_tree(n->right);
    free(n);
}

static int is_leaf(const Node *n) { return n->left == NULL && n->right == NULL; }

/* =====================================================================
 *  2. MIN-HEAP PRIORITY QUEUE (hand-written, O(log n) push/pop)
 * ===================================================================== */
typedef struct {
    Node *a[ALPHABET + 1];
    int size;
} MinHeap;

static int node_less(const Node *x, const Node *y) {
    if (x->freq != y->freq) return x->freq < y->freq;
    return x->order < y->order;
}

static void heap_push(MinHeap *h, Node *n) {
    int i = h->size++;
    h->a[i] = n;
    while (i > 0) {                       /* sift up */
        int p = (i - 1) / 2;
        if (!node_less(h->a[i], h->a[p])) break;
        Node *t = h->a[i]; h->a[i] = h->a[p]; h->a[p] = t;
        i = p;
    }
}

static Node *heap_pop(MinHeap *h) {
    Node *top = h->a[0];
    h->a[0] = h->a[--h->size];
    int i = 0;
    for (;;) {                            /* sift down */
        int l = 2 * i + 1, r = 2 * i + 2, m = i;
        if (l < h->size && node_less(h->a[l], h->a[m])) m = l;
        if (r < h->size && node_less(h->a[r], h->a[m])) m = r;
        if (m == i) break;
        Node *t = h->a[i]; h->a[i] = h->a[m]; h->a[m] = t;
        i = m;
    }
    return top;
}

/* =====================================================================
 *  3. FREQUENCY TABLE + GREEDY HUFFMAN TREE
 * ===================================================================== */
static void build_frequency_table(const uint8_t *data, uint64_t n, uint64_t freq[ALPHABET]) {
    memset(freq, 0, sizeof(uint64_t) * ALPHABET);
    for (uint64_t i = 0; i < n; i++) freq[data[i]]++;
}

/* GREEDY CHOICE: always merge the two least-frequent nodes. */
static Node *build_huffman_tree(const uint64_t freq[ALPHABET]) {
    MinHeap h; h.size = 0;
    for (int s = 0; s < ALPHABET; s++)
        if (freq[s] > 0) heap_push(&h, new_node(freq[s], s, s, NULL, NULL));
    if (h.size == 0) return NULL;

    int order = ALPHABET;
    while (h.size > 1) {
        Node *a = heap_pop(&h);
        Node *b = heap_pop(&h);
        heap_push(&h, new_node(a->freq + b->freq, -1, order++, a, b));
    }
    return heap_pop(&h);
}

/* =====================================================================
 *  4. PREFIX CODE GENERATION
 * ===================================================================== */
typedef struct {
    uint8_t bits[ALPHABET];   /* one 0/1 value per bit, MSB first */
    int len;
} Code;

static void gen_codes_rec(const Node *n, uint8_t *path, int depth, Code codes[ALPHABET]) {
    if (is_leaf(n)) {
        if (depth == 0) { path[0] = 0; depth = 1; }   /* single-symbol edge case */
        codes[n->symbol].len = depth;
        memcpy(codes[n->symbol].bits, path, depth);
        return;
    }
    path[depth] = 0; gen_codes_rec(n->left,  path, depth + 1, codes);
    path[depth] = 1; gen_codes_rec(n->right, path, depth + 1, codes);
}

static void generate_codes(const Node *root, Code codes[ALPHABET]) {
    uint8_t path[ALPHABET + 1];
    memset(codes, 0, sizeof(Code) * ALPHABET);
    if (root) gen_codes_rec(root, path, 0, codes);
}

/* =====================================================================
 *  5. STATISTICS (entropy H, average code length L, H <= L < H+1)
 * ===================================================================== */
typedef struct {
    uint64_t orig_size, comp_size;
    double entropy, avg_len;
    int distinct;
} Stats;

static void compute_stats(const uint64_t freq[ALPHABET], const Code codes[ALPHABET],
                          uint64_t n, Stats *st) {
    double H = 0.0, L = 0.0;
    int distinct = 0;
    for (int s = 0; s < ALPHABET; s++) {
        if (!freq[s]) continue;
        double p = (double)freq[s] / (double)n;
        H -= p * log2(p);
        L += p * codes[s].len;
        distinct++;
    }
    st->entropy = H; st->avg_len = L; st->distinct = distinct;
}

static void print_code_table(const uint64_t freq[ALPHABET], const Code codes[ALPHABET]) {
    int used[ALPHABET] = {0};
    printf("\n  Top code table entries (byte | freq | len | code)\n");
    for (int k = 0; k < 10; k++) {
        int best = -1;
        for (int s = 0; s < ALPHABET; s++)
            if (freq[s] && !used[s] && (best < 0 || freq[s] > freq[best])) best = s;
        if (best < 0) break;
        used[best] = 1;
        printf("   0x%02X | %10llu | %3d | ", best, (unsigned long long)freq[best], codes[best].len);
        for (int i = 0; i < codes[best].len && i < 40; i++) putchar('0' + codes[best].bits[i]);
        if (codes[best].len > 40) printf("...");
        putchar('\n');
    }
}

static void print_stats(const Stats *st) {
    double ratio = st->comp_size ? (double)st->orig_size / (double)st->comp_size : 0.0;
    double saved = st->orig_size ? 100.0 * (1.0 - (double)st->comp_size / (double)st->orig_size) : 0.0;
    printf("\n========== COMPRESSION STATS ==========\n");
    printf("  Original size   : %llu bytes\n", (unsigned long long)st->orig_size);
    printf("  Compressed size : %llu bytes (incl. header)\n", (unsigned long long)st->comp_size);
    printf("  Ratio           : %.3f : 1\n", ratio);
    printf("  Space saved     : %.2f %%\n", saved);
    printf("  Distinct symbols: %d\n", st->distinct);
    printf("  Entropy H       : %.4f bits/symbol\n", st->entropy);
    printf("  Avg code len L  : %.4f bits/symbol\n", st->avg_len);
    printf("  Check H <= L < H+1 : %s\n",
           (st->avg_len + 1e-9 >= st->entropy && st->avg_len < st->entropy + 1.0) ? "PASS" : "FAIL");
}

/* =====================================================================
 *  6. ENCODE / DECODE
 *  File format: "HUF1" | u64 N | u16 nsym | nsym * (u8 symbol, u64 freq) | bitstream
 * ===================================================================== */
static void put_u64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static uint64_t get_u64(const uint8_t *p) { uint64_t v = 0; for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i); return v; }

static uint8_t *huffman_compress(const uint8_t *data, uint64_t n, uint64_t *out_len, Stats *st) {
    uint64_t freq[ALPHABET];
    Code *codes = (Code *)malloc(sizeof(Code) * ALPHABET);
    build_frequency_table(data, n, freq);
    Node *root = build_huffman_tree(freq);
    generate_codes(root, codes);

    int nsym = 0;
    uint64_t total_bits = 0;
    for (int s = 0; s < ALPHABET; s++)
        if (freq[s]) { nsym++; total_bits += freq[s] * (uint64_t)codes[s].len; }

    uint64_t header = 4 + 8 + 2 + (uint64_t)nsym * 9;
    uint64_t payload = (total_bits + 7) / 8;
    uint8_t *out = (uint8_t *)calloc(header + payload + 1, 1);
    if (!out) { fprintf(stderr, "Out of memory\n"); exit(1); }

    /* header */
    memcpy(out, MAGIC, 4);
    put_u64(out + 4, n);
    out[12] = (uint8_t)(nsym & 0xFF);
    out[13] = (uint8_t)(nsym >> 8);
    uint8_t *p = out + 14;
    for (int s = 0; s < ALPHABET; s++)
        if (freq[s]) { *p++ = (uint8_t)s; put_u64(p, freq[s]); p += 8; }

    /* bit-packed payload */
    uint8_t *bs = out + header;
    uint64_t pos = 0;
    for (uint64_t i = 0; i < n; i++) {
        const Code *c = &codes[data[i]];
        for (int b = 0; b < c->len; b++, pos++)
            if (c->bits[b]) bs[pos >> 3] |= (uint8_t)(0x80 >> (pos & 7));
    }

    *out_len = header + payload;
    st->orig_size = n; st->comp_size = *out_len;
    compute_stats(freq, codes, n ? n : 1, st);
    if (n) print_code_table(freq, codes);

    free_tree(root);
    free(codes);
    return out;
}

static uint8_t *huffman_decompress(const uint8_t *in, uint64_t in_len, uint64_t *out_n) {
    if (in_len < 14 || memcmp(in, MAGIC, 4) != 0) { fprintf(stderr, "Invalid .huf file\n"); return NULL; }
    uint64_t n = get_u64(in + 4);
    int nsym = in[12] | (in[13] << 8);
    uint64_t header = 14 + (uint64_t)nsym * 9;
    if (in_len < header) { fprintf(stderr, "Truncated header\n"); return NULL; }

    uint64_t freq[ALPHABET] = {0};
    const uint8_t *p = in + 14;
    for (int i = 0; i < nsym; i++) { freq[p[0]] = get_u64(p + 1); p += 9; }

    *out_n = n;
    uint8_t *out = (uint8_t *)malloc(n ? n : 1);
    if (!out) { fprintf(stderr, "Out of memory\n"); return NULL; }
    if (n == 0) return out;

    Node *root = build_huffman_tree(freq);       /* identical tree as encoder */
    if (!root) { free(out); return NULL; }

    if (is_leaf(root)) {                         /* single-symbol image */
        memset(out, root->symbol, n);
        free_tree(root);
        return out;
    }

    const uint8_t *bs = in + header;
    uint64_t total_bits = (in_len - header) * 8, pos = 0, produced = 0;
    const Node *cur = root;
    while (produced < n) {
        if (pos >= total_bits) { fprintf(stderr, "Corrupt bitstream\n"); free(out); free_tree(root); return NULL; }
        int bit = (bs[pos >> 3] >> (7 - (pos & 7))) & 1;
        pos++;
        cur = bit ? cur->right : cur->left;
        if (is_leaf(cur)) { out[produced++] = (uint8_t)cur->symbol; cur = root; }
    }
    free_tree(root);
    return out;
}

/* =====================================================================
 *  7. SHA-256  (used as a mock IPFS CID / on-chain hash)
 * ===================================================================== */
typedef struct { uint32_t h[8]; uint64_t len; uint8_t buf[64]; size_t buflen; } Sha256;

static const uint32_t K256[64] = {
 0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
 0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
 0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
 0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
 0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
 0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
 0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
 0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

#define ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

static void sha256_block(Sha256 *s, const uint8_t *b) {
    uint32_t w[64], a, bb, c, d, e, f, g, h;
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)b[4*i] << 24) | ((uint32_t)b[4*i+1] << 16) | ((uint32_t)b[4*i+2] << 8) | b[4*i+3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ROR(w[i-15], 7) ^ ROR(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = ROR(w[i-2], 17) ^ ROR(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    a = s->h[0]; bb = s->h[1]; c = s->h[2]; d = s->h[3];
    e = s->h[4]; f = s->h[5]; g = s->h[6]; h = s->h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = ROR(e, 6) ^ ROR(e, 11) ^ ROR(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + K256[i] + w[i];
        uint32_t S0 = ROR(a, 2) ^ ROR(a, 13) ^ ROR(a, 22);
        uint32_t mj = (a & bb) ^ (a & c) ^ (bb & c);
        uint32_t t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1; d = c; c = bb; bb = a; a = t1 + t2;
    }
    s->h[0] += a; s->h[1] += bb; s->h[2] += c; s->h[3] += d;
    s->h[4] += e; s->h[5] += f; s->h[6] += g; s->h[7] += h;
}

static void sha256_init(Sha256 *s) {
    static const uint32_t iv[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                                   0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    memcpy(s->h, iv, sizeof(iv)); s->len = 0; s->buflen = 0;
}

static void sha256_update(Sha256 *s, const uint8_t *d, size_t n) {
    s->len += n;
    while (n > 0) {
        size_t take = 64 - s->buflen; if (take > n) take = n;
        memcpy(s->buf + s->buflen, d, take);
        s->buflen += take; d += take; n -= take;
        if (s->buflen == 64) { sha256_block(s, s->buf); s->buflen = 0; }
    }
}

static void sha256_final(Sha256 *s, char hex[65]) {
    uint64_t bits = s->len * 8;
    uint8_t pad = 0x80;
    sha256_update(s, &pad, 1);
    pad = 0;
    while (s->buflen != 56) sha256_update(s, &pad, 1);
    uint8_t lenb[8];
    for (int i = 0; i < 8; i++) lenb[i] = (uint8_t)(bits >> (56 - 8 * i));
    sha256_update(s, lenb, 8);
    for (int i = 0; i < 8; i++) sprintf(hex + 8 * i, "%08x", s->h[i]);
    hex[64] = 0;
}

static void sha256_hex(const uint8_t *d, size_t n, char hex[65]) {
    Sha256 s; sha256_init(&s); sha256_update(&s, d, n); sha256_final(&s, hex);
}

/* =====================================================================
 *  8. FILE HELPERS
 * ===================================================================== */
static uint8_t *read_file(const char *path, uint64_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open '%s'\n", path); return NULL; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t *)malloc(sz > 0 ? (size_t)sz : 1);
    if (!buf) { fclose(f); return NULL; }
    if (sz > 0 && fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); return NULL; }
    fclose(f);
    *n = (uint64_t)sz;
    return buf;
}

static int write_file(const char *path, const uint8_t *d, uint64_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write '%s'\n", path); return 0; }
    if (n) fwrite(d, 1, (size_t)n, f);
    fclose(f);
    return 1;
}

/* =====================================================================
 *  9. SIMULATED IPFS (content-addressed store) + BLOCKCHAIN LEDGER
 * ===================================================================== */
static void ipfs_store(const uint8_t *comp, uint64_t clen, uint64_t orig, char hash[65]) {
    sha256_hex(comp, (size_t)clen, hash);                 /* mock CID */
    MKDIR(STORE_DIR);
    char path[256];
    snprintf(path, sizeof(path), "%s/%s.huf", STORE_DIR, hash);
    write_file(path, comp, clen);

    /* "on-chain" anchor: append-only ledger record */
    FILE *l = fopen(LEDGER_FILE, "a");
    if (l) {
        fprintf(l, "ts=%ld | hash=%s | original=%llu | compressed=%llu\n",
                (long)time(NULL), hash, (unsigned long long)orig, (unsigned long long)clen);
        fclose(l);
    }
    printf("\n[IPFS ] Stored off-chain : %s\n", path);
    printf("[CHAIN] Anchored hash    : %s  (see %s)\n", hash, LEDGER_FILE);
}

static uint8_t *ipfs_fetch(const char *hash, uint64_t *clen) {
    char path[256];
    snprintf(path, sizeof(path), "%s/%s.huf", STORE_DIR, hash);
    uint8_t *d = read_file(path, clen);
    if (!d) return NULL;
    char check[65];
    sha256_hex(d, (size_t)*clen, check);                  /* integrity check */
    if (strcmp(check, hash) != 0) { fprintf(stderr, "Hash mismatch: data tampered!\n"); free(d); return NULL; }
    return d;
}

/* =====================================================================
 *  10. MAIN
 * ===================================================================== */
static int cmd_compress(const char *in, const char *out) {
    uint64_t n; uint8_t *data = read_file(in, &n);
    if (!data) return 1;
    uint64_t clen; Stats st;
    uint8_t *comp = huffman_compress(data, n, &clen, &st);
    print_stats(&st);
    int ok = write_file(out, comp, clen);
    free(data); free(comp);
    return ok ? 0 : 1;
}

static int cmd_decompress(const char *in, const char *out) {
    uint64_t clen; uint8_t *comp = read_file(in, &clen);
    if (!comp) return 1;
    uint64_t n; uint8_t *data = huffman_decompress(comp, clen, &n);
    int ok = data && write_file(out, data, n);
    if (ok) printf("Decompressed %llu bytes -> %s\n", (unsigned long long)n, out);
    free(comp); free(data);
    return ok ? 0 : 1;
}

static int cmd_demo(const char *in) {
    uint64_t n; uint8_t *data = read_file(in, &n);
    if (!data) return 1;

    printf("[1] Compressing '%s' ...\n", in);
    uint64_t clen; Stats st;
    uint8_t *comp = huffman_compress(data, n, &clen, &st);
    print_stats(&st);

    printf("\n[2] Storing on simulated IPFS + anchoring hash ...");
    char hash[65];
    ipfs_store(comp, clen, n, hash);

    printf("\n[3] Retrieving by hash and decompressing ...\n");
    uint64_t rlen; uint8_t *fetched = ipfs_fetch(hash, &rlen);
    if (!fetched) return 1;
    uint64_t dn; uint8_t *restored = huffman_decompress(fetched, rlen, &dn);
    if (!restored) return 1;

    char h1[65], h2[65];
    sha256_hex(data, (size_t)n, h1);
    sha256_hex(restored, (size_t)dn, h2);
    int lossless = (n == dn) && (n == 0 || memcmp(data, restored, (size_t)n) == 0) && strcmp(h1, h2) == 0;
    printf("\n[4] Verification\n  SHA-256 original : %s\n  SHA-256 restored : %s\n  Lossless? %s\n",
           h1, h2, lossless ? "YES (byte-for-byte identical)" : "NO");

    free(data); free(comp); free(fetched); free(restored);
    return lossless ? 0 : 1;
}

static int cmd_retrieve(const char *hash, const char *out) {
    uint64_t clen; uint8_t *comp = ipfs_fetch(hash, &clen);
    if (!comp) return 1;
    uint64_t n; uint8_t *data = huffman_decompress(comp, clen, &n);
    int ok = data && write_file(out, data, n);
    if (ok) printf("Retrieved & decompressed %llu bytes -> %s\n", (unsigned long long)n, out);
    free(comp); free(data);
    return ok ? 0 : 1;
}

static void usage(const char *p) {
    fprintf(stderr,
        "Usage:\n"
        "  %s compress   <input> <output.huf>\n"
        "  %s decompress <input.huf> <output>\n"
        "  %s demo       <input>\n"
        "  %s retrieve   <hash> <output>\n", p, p, p, p);
}

int main(int argc, char **argv) {
    if (argc >= 4 && !strcmp(argv[1], "compress"))   return cmd_compress(argv[2], argv[3]);
    if (argc >= 4 && !strcmp(argv[1], "decompress")) return cmd_decompress(argv[2], argv[3]);
    if (argc >= 3 && !strcmp(argv[1], "demo"))       return cmd_demo(argv[2]);
    if (argc >= 4 && !strcmp(argv[1], "retrieve"))   return cmd_retrieve(argv[2], argv[3]);
    usage(argv[0]);
    return 1;
}