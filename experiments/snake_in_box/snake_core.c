/* snake_core.c — bitboard inner loop for snake-in-the-box.
 *
 * The Python searchers (priority_skeleton.solve, complete_certificate.complete)
 * are correct but interpreter-bound: extending a path and scoring successors is
 * a handful of array ops run millions of times per attempt. This file is the
 * same two hot loops in C, called via ctypes. Semantics match the Python
 * reference exactly in the deterministic case (sib_complete with rng_seed < 0
 * uses bit order 0..n-1, identical to Python's unshuffled DFS), which is the
 * correctness anchor the tests check.
 *
 * Vertices are ints in [0, 2^n). on_snake[] and neighbor_count[] are dense
 * arrays of size 2^n (n <= 13 → <= 8192 bytes each, cache-resident). Legality:
 * a vertex may extend the snake iff it is unvisited and has exactly one snake
 * neighbour (the current head). "length" everywhere means edges = vertices - 1.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline double next_unit(uint64_t *s) {
    /* uniform in [0,1) with 53 bits of mantissa */
    return (splitmix64(s) >> 11) * (1.0 / 9007199254740992.0);
}

/* Count legal continuations from v assuming v has just been chosen but not yet
 * committed — matches Python _legal_with_counts inner loop: neighbours w of v
 * that are unvisited, not the current head, and currently have zero snake
 * neighbours (so after v joins they would have exactly one). */
static inline int succ_after(int n, int v, int cur,
                             const uint8_t *on_snake, const uint8_t *nc) {
    int succ = 0;
    for (int bb = 0; bb < n; ++bb) {
        int w = v ^ (1 << bb);
        if (!on_snake[w] && w != cur && nc[w] == 0) ++succ;
    }
    return succ;
}

/* Greedy Kinny rollout with noise tie-break, restarting until budget spent.
 * Mirrors priority_skeleton.solve: score = (succ==0 ? -100+noise : -succ+0.1*noise),
 * pick argmax; budget charged 1 + (#options) per step. Writes best snake to
 * out[] (capacity >= 2^n + 1) and returns its edge length. */
int sib_rollout(int n, uint64_t seed, long long budget, int *out) {
    int size = 1 << n;
    uint8_t *on_snake = calloc(size, 1);
    uint8_t *nc = calloc(size, 1);
    int *path = malloc(sizeof(int) * (size + 1));
    int best_len = 0;
    out[0] = 0;
    uint64_t rng = seed ? seed : 0x1234567ULL;
    long long evals = 0;

    while (evals < budget) {
        memset(on_snake, 0, size);
        memset(nc, 0, size);
        int plen = 1;
        path[0] = 0;
        on_snake[0] = 1;
        for (int b = 0; b < n; ++b) nc[1 << b] += 1;

        while (1) {
            int cur = path[plen - 1];
            int best_v = -1;
            double best_score = -1e18;
            int noptions = 0;
            for (int b = 0; b < n; ++b) {
                int v = cur ^ (1 << b);
                if (on_snake[v] || nc[v] != 1) continue;
                ++noptions;
                int s = succ_after(n, v, cur, on_snake, nc);
                double noise = next_unit(&rng);
                double score = (s == 0) ? (-100.0 + noise) : (-(double)s + 0.1 * noise);
                if (score > best_score) { best_score = score; best_v = v; }
            }
            evals += 1 + noptions;
            if (noptions == 0 || evals >= budget) break;
            int v = best_v;
            on_snake[v] = 1;
            path[plen++] = v;
            for (int b = 0; b < n; ++b) nc[v ^ (1 << b)] += 1;
        }
        if (plen - 1 > best_len) {
            best_len = plen - 1;
            memcpy(out, path, sizeof(int) * plen);
        }
    }
    free(on_snake); free(nc); free(path);
    return best_len;
}

/* Exhaustive DFS over chord-free extensions of prefix[0..plen-1]. rng_seed < 0
 * → deterministic bit order 0..n-1 (matches Python complete() exactly);
 * rng_seed >= 0 → per-node shuffled order. forbid_first >= 0 bans that vertex
 * as the very first extension. Writes best snake to out[], best edge length is
 * the return; *nodes_out and *exhausted_out report the search. */
typedef struct {
    int n, size;
    uint8_t *on_snake, *nc;
    int *path, plen0, *best, *bestlen;
    long long nodes, node_cap;
    int exhausted, use_rng, forbid_first;
    uint64_t rng;
} dfs_ctx;

static void dfs(dfs_ctx *c, int cur, int plen) {
    if (c->nodes >= c->node_cap) { c->exhausted = 0; return; }
    c->nodes += 1;
    int order[16];
    for (int i = 0; i < c->n; ++i) order[i] = i;
    if (c->use_rng) {
        for (int i = c->n - 1; i > 0; --i) {
            int j = (int)(splitmix64(&c->rng) % (uint64_t)(i + 1));
            int t = order[i]; order[i] = order[j]; order[j] = t;
        }
    }
    for (int i = 0; i < c->n; ++i) {
        int b = order[i];
        int v = cur ^ (1 << b);
        if (c->on_snake[v] || c->nc[v] != 1) continue;
        if (c->forbid_first >= 0 && plen == c->plen0 && v == c->forbid_first) continue;
        c->on_snake[v] = 1;
        c->path[plen] = v;
        for (int bb = 0; bb < c->n; ++bb) c->nc[v ^ (1 << bb)] += 1;
        if (plen + 1 > *c->bestlen) {
            *c->bestlen = plen + 1;
            memcpy(c->best, c->path, sizeof(int) * (plen + 1));
        }
        dfs(c, v, plen + 1);
        for (int bb = 0; bb < c->n; ++bb) c->nc[v ^ (1 << bb)] -= 1;
        c->on_snake[v] = 0;
        if (c->nodes >= c->node_cap) return;
    }
}

/* As sib_complete, but with a blocked vertex list: blocked vertices are
 * excluded from the search as if visited, WITHOUT contributing neighbor
 * counts (they are not on the snake — the path may run adjacent to them).
 * This is the complement-constrained search used by Wynn-style joint
 * construction: longest induced path avoiding an existing fragment. */
int sib_complete_blocked(int n, const int *prefix, int plen,
                         const int *blocked, int nblocked,
                         long long node_cap, long long rng_seed,
                         int *out, long long *nodes_out, int *exhausted_out);

int sib_complete(int n, const int *prefix, int plen, long long node_cap,
                 long long rng_seed, int forbid_first,
                 int *out, long long *nodes_out, int *exhausted_out) {
    int size = 1 << n;
    dfs_ctx c;
    c.n = n; c.size = size;
    c.on_snake = calloc(size, 1);
    c.nc = calloc(size, 1);
    c.path = malloc(sizeof(int) * (size + 1));
    int bestlen = plen;
    c.best = out; c.bestlen = &bestlen;
    c.plen0 = plen; c.nodes = 0; c.node_cap = node_cap;
    c.exhausted = 1;
    c.use_rng = (rng_seed >= 0);
    c.rng = (uint64_t)(rng_seed >= 0 ? rng_seed : 0) + 0x9E3779B9ULL;
    c.forbid_first = forbid_first;

    for (int i = 0; i < plen; ++i) {
        int v = prefix[i];
        c.on_snake[v] = 1;
        c.path[i] = v;
        for (int b = 0; b < n; ++b) c.nc[v ^ (1 << b)] += 1;
    }
    memcpy(out, prefix, sizeof(int) * plen);
    dfs(&c, prefix[plen - 1], plen);

    *nodes_out = c.nodes;
    *exhausted_out = c.exhausted;
    free(c.on_snake); free(c.nc); free(c.path);
    return bestlen - 1;  /* edge length */
}

int sib_complete_blocked(int n, const int *prefix, int plen,
                         const int *blocked, int nblocked,
                         long long node_cap, long long rng_seed,
                         int *out, long long *nodes_out, int *exhausted_out) {
    int size = 1 << n;
    dfs_ctx c;
    c.n = n; c.size = size;
    c.on_snake = calloc(size, 1);
    c.nc = calloc(size, 1);
    c.path = malloc(sizeof(int) * (size + 1));
    int bestlen = plen;
    c.best = out; c.bestlen = &bestlen;
    c.plen0 = plen; c.nodes = 0; c.node_cap = node_cap;
    c.exhausted = 1;
    c.use_rng = (rng_seed >= 0);
    c.rng = (uint64_t)(rng_seed >= 0 ? rng_seed : 0) + 0x9E3779B9ULL;
    c.forbid_first = -1;

    /* blocked vertices: excluded like visited, but contribute no neighbor
     * counts — the snake may run adjacent to them */
    for (int i = 0; i < nblocked; ++i) c.on_snake[blocked[i]] = 1;

    for (int i = 0; i < plen; ++i) {
        int v = prefix[i];
        c.on_snake[v] = 1;
        c.path[i] = v;
        for (int b = 0; b < n; ++b) c.nc[v ^ (1 << b)] += 1;
    }
    memcpy(out, prefix, sizeof(int) * plen);
    dfs(&c, prefix[plen - 1], plen);

    *nodes_out = c.nodes;
    *exhausted_out = c.exhausted;
    free(c.on_snake); free(c.nc); free(c.path);
    return bestlen - 1;
}

/* Combined-context completion for the fragment router: `frozen` vertices
 * are committed SAME-LAYER snake vertices (excluded AND contributing
 * neighbor counts — running adjacent to them is a chord); `blocked` are
 * label-only exclusions (the other layer's shadow). The prefix start's
 * legality against frozen is the caller's concern via nc at entry. */
int sib_complete_frozen(int n, const int *prefix, int plen,
                        const int *frozen, int nfrozen,
                        const int *blocked, int nblocked,
                        long long node_cap, long long rng_seed,
                        int *out, long long *nodes_out, int *exhausted_out) {
    int size = 1 << n;
    dfs_ctx c;
    c.n = n; c.size = size;
    c.on_snake = calloc(size, 1);
    c.nc = calloc(size, 1);
    c.path = malloc(sizeof(int) * (size + 1));
    int bestlen = plen;
    c.best = out; c.bestlen = &bestlen;
    c.plen0 = plen; c.nodes = 0; c.node_cap = node_cap;
    c.exhausted = 1;
    c.use_rng = (rng_seed >= 0);
    c.rng = (uint64_t)(rng_seed >= 0 ? rng_seed : 0) + 0x9E3779B9ULL;
    c.forbid_first = -1;

    for (int i = 0; i < nblocked; ++i) c.on_snake[blocked[i]] = 1;
    for (int i = 0; i < nfrozen; ++i) {
        int v = frozen[i];
        c.on_snake[v] = 1;
        for (int b = 0; b < n; ++b) c.nc[v ^ (1 << b)] += 1;
    }
    for (int i = 0; i < plen; ++i) {
        int v = prefix[i];
        c.on_snake[v] = 1;
        c.path[i] = v;
        for (int b = 0; b < n; ++b) c.nc[v ^ (1 << b)] += 1;
    }
    memcpy(out, prefix, sizeof(int) * plen);
    dfs(&c, prefix[plen - 1], plen);

    *nodes_out = c.nodes;
    *exhausted_out = c.exhausted;
    free(c.on_snake); free(c.nc); free(c.path);
    return bestlen - 1;
}
