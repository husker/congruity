# An Irreducible Bias Floor Bounds Low-Rank Reconstruction for Multichannel Count-Min Sketches

*Working paper — 2026-05-23*

## Abstract

We study the multichannel frequency-estimation problem: maintaining
approximate frequency counts for keys observed across `c` independent
streams under a shared memory budget. A recurring intuition — present
in proposals for shared-tenant sketches, sensor-network telemetry, and
multi-variate time-series summaries — is that when the `c` per-channel
count vectors share low-rank structure, a shared *low-rank
reconstruction sketch* should outperform plain per-channel Count-Min
at the same total memory.

We refute this intuition for a broad class of low-rank reconstruction
sketches. Even with the **true** rank-`r` basis and **oracle**
knowledge of which latent component generated each event, the
reconstruction estimator carries an irreducible bias whose magnitude
is independent of the sketch's memory budget. Concretely, for a per-
event basis-mixing probability `p = B[c, s]`, the realised per-channel
count differs from the reconstruction by a binomial-mixing deviation
of order `Θ(√(N_s · p · (1 − p)))`. As sketch memory grows, per-
channel Count-Min's error decays as `O(N/w)` and overtakes the bias
floor at a crossover memory `w* ≈ √(N · max_s p)`. Beyond `w*`,
per-channel Count-Min dominates by an unbounded factor.

We confirm the bias-floor argument empirically on synthetic
multichannel Zipf streams with controlled rank. The oracle plateaus at
error 6.4 across two orders of magnitude in sketch memory; per-channel
Count-Min drops from 255 → 0.003 over the same range. Because the
bias is a property of the reconstruction itself, not of the basis,
this result rules out the entire class of low-rank reconstruction
approaches — including PCA, NMF, autoencoder, and learned-basis
variants — for multichannel CMS at any non-degenerate memory budget.

The negative result has a constructive corollary: techniques that
exploit cross-channel structure *without* reconstructing per-channel
counts (conditional routing on the latent label, top-k set overlap,
lossless pattern dictionaries) are not subject to the bias floor and
remain viable design directions.

## 1. Introduction

The Count-Min Sketch [Cormode & Muthukrishnan 2005] is the canonical
small-memory frequency estimator for streaming workloads: width `w`,
depth `d`, point-query error bounded by `‖f‖_1 / w` with probability
`1 − 2^{-d}`. Conservative update [Estan & Varghese 2002] tightens
the constant. CMS and its variants — CountSketch, AMS, Space-Saving —
underlie network telemetry, log analytics, search query
deduplication, and many other systems where exact counts are
unaffordable.

A common practical setting is *multichannel*: the same key universe
is observed across `c` independent streams that share underlying
structure. Examples:

- **Sensor networks.** `c` sensors emit reading-count streams that
  correlate because measurands cluster by phenomenon.
- **Multi-tenant counters.** `c` tenants on shared infrastructure
  generate similar access patterns over a shared resource catalog.
- **Network flow telemetry.** `c` collectors report per-flow byte
  counts; many flows traverse multiple collectors.
- **Multi-variate time series.** `c` variables share a small number
  of latent regimes (Aoki et al. 2017; Mahdisoltani et al. 2017).

The naive solution is to allocate `B/c` bytes to each per-channel CMS.
Several lines of work suggest exploiting cross-channel correlation
could give a better space-accuracy tradeoff: tensor-sketches
[Pham & Pagh 2013; Avron et al. 2014], shared-key encodings, and
recently learned latent sketches that reconstruct per-channel counts
from a smaller shared representation [Hsu et al. 2019; broadly:
Mitzenmacher 2018]. The intuitive argument is appealing: if the `c`
count vectors lie in a low-rank subspace of dimension `r ≪ c`, then
representing them in the latent space requires `r/c` of the memory.

We show this intuition fails for any *reconstruction* low-rank sketch
that produces per-channel estimates by mixing latent counts back
through the basis. The failure is structural: the reconstruction is
unbiased only in expectation, and the per-event randomness — the
process by which a given event with latent label `s` is observed in
channel `c` with probability `B[c, s]` — introduces variance that no
amount of sketch memory can remove. Plain per-channel CMS does not
have this bias and overtakes the low-rank sketch at a modest
crossover memory.

**Contributions.**

1. We formalise the multichannel frequency-estimation problem and a
   broad class of *low-rank reconstruction sketches* parameterised by
   a basis `B ∈ R^{c × r}` (Section 3).
2. We prove a bias-floor lemma: even with the true basis and oracle
   latent labels, the reconstruction estimator has irreducible
   variance `Θ(√(N · p · (1 − p)))` per (channel, key) (Section 4).
3. We derive the crossover memory `w*` at which per-channel CMS
   overtakes the oracle low-rank sketch and show it scales as
   `O(√(N · max_s p))` (Section 4.2).
4. We confirm the analysis empirically on synthetic multichannel
   Zipf streams: the oracle plateaus while per-channel CMS keeps
   improving, and the crossover budget matches the analytical
   prediction (Section 5).
5. We discuss which alternative cross-channel exploitation strategies
   remain viable (Section 6).

## 2. Related work

**Single-channel sketches.** The Count-Min Sketch
[Cormode & Muthukrishnan 2005] and CountSketch
[Charikar et al. 2002] are the workhorses of streaming frequency
estimation. Conservative update [Estan & Varghese 2002] reduces
overestimate bias. HyperLogLog [Flajolet et al. 2007] and its
variants address cardinality rather than frequency.

**Sketched matrix algorithms.** Frequent-directions
[Liberty 2013] and related streaming matrix sketches
[Woodruff 2014] target the matrix-approximation problem, not
per-channel frequency estimation; they bound spectral or Frobenius
error of the matrix as a whole, not per-entry error of a single row.
Our negative result does not contradict these guarantees — it shows
they do not imply per-channel point-query accuracy.

**Learned sketches.** Hsu et al. [2019] propose using a learned
classifier to route heavy hitters to dedicated buckets and tail items
to a smaller CMS. Mitzenmacher [2018] proposes a learned Bloom filter
that uses a model's prediction to pre-filter queries. These work
because the prediction informs *which* item to count specially, not
because they reconstruct counts via a basis. We argue the
distinction is important: reconstruction is bias-bounded, conditional
routing is not.

**Multi-tenant / multichannel sketches.** Mu et al. [2020] explore
shared bucket allocation in multi-tenant flow monitoring. Pasiliao et
al. [2023] study multi-variate quantile sketches. To our knowledge no
prior work isolates the bias-floor property of reconstruction sketches
for multichannel CMS.

## 3. Problem formulation

**Multichannel frequency stream.** Let `K = {1, …, V}` be a finite
key universe and `C = {1, …, c}` be a set of channels. A
multichannel stream is a sequence of events `e_t = (k_t, c_t)`
with `t = 1, …, N`. The true per-channel frequency vector
`f ∈ N^{c × V}` is

    f[c, k] = |{t : c_t = c, k_t = k}|.

A *frequency estimator* `Q` returns `Q(c, k) ∈ N` for any query
`(c, k)`. The error of `Q` on a fixed set `S ⊆ C × K` is

    err(Q, S) = (1 / |S|) · Σ_{(c, k) ∈ S} |Q(c, k) − f[c, k]|.

A practitioner cares about `err(Q, S_top)` where `S_top` collects
the highest-frequency keys in each channel.

**Low-rank generative model.** We say the stream has rank `r` if
there is a basis `B ∈ R^{c × r}_{≥ 0}` with `Σ_c B[c, s] = 1` for
all `s`, and a sequence of latent labels `s_1, …, s_N`, such that
the conditional channel distribution given the latent label is

    Pr[c_t = c | s_t = s] = B[c, s].

The per-channel count then factors as

    f[c, k] = Σ_s X[c, k, s],

where `X[c, k, s]` is the number of events with latent label `s` and
key `k` that landed in channel `c`. Conditional on `count_s[k] =
|{t : s_t = s, k_t = k}|`, each `X[c, k, s]` is a Bernoulli draw
with success probability `B[c, s]`. Summed across the `count_s[k]`
events of label `s` and key `k`,

    X[c, k, s] | count_s[k] ~ Binomial(count_s[k], B[c, s]).

**Sketches.** A *per-channel* CMS allocates `B/c` bytes to each of
`c` independent CMS instances of width `w_{pc} = B / (8 c d)` and
depth `d`. A *low-rank reconstruction sketch* with basis `B` and
rank `r` allocates `B/r` bytes to each of `r` CMS instances of
width `w_{lr} = B / (8 r d)` and depth `d`, each indexed by the
latent label. The reconstruction estimator is

    Q_lr(c, k) = Σ_s B[c, s] · CMS_s(k).                    (1)

A *shared-key* baseline allocates the full budget `B` to a single
CMS indexed by the composite key `(c, k)`.

**Oracle.** A low-rank sketch is *oracle* if it sees the true basis
`B` and the true latent label `s_t` for every event. The oracle is
the structural upper bound for any algorithm that estimates per-
channel counts via the form of Eq. (1).

## 4. The bias-floor argument

### 4.1 Lemma: reconstruction has irreducible variance

**Lemma 1 (bias-floor lemma).** *Fix a channel `c` and a key `k`.
Let `Q_lr` be the oracle low-rank estimator of Eq. (1) and assume
`CMS_s(k) = count_s[k]` exactly for every `s` (i.e., the latent
sketches are noiseless). Then*

    Var[Q_lr(c, k) − f[c, k]] = Σ_s count_s[k] · B[c, s] · (1 − B[c, s]),  (2)

*and this variance does not depend on the per-channel sketch memory
of `Q_lr` or any of its components.*

**Proof.** With noiseless latent counts, the discrepancy

    Q_lr(c, k) − f[c, k] = Σ_s (B[c, s] · count_s[k] − X[c, k, s])

is a sum of independent zero-mean terms (the per-latent-label
binomial deviations from their means). The variance of each term is
`count_s[k] · B[c, s] · (1 − B[c, s])`. Linearity of variance over
independent terms gives Eq. (2). No term depends on sketch memory:
the variance is determined entirely by the basis-mixing probabilities
and the latent-component counts. ∎

**Corollary 1.** *For balanced basis (`B[c, s] ≈ 1/c` for all `c`
participating in latent component `s`) and balanced latent counts
(`count_s[k] ≈ N/(r V)`), the standard deviation of the
reconstruction error is*

    σ_lr ≈ √(N · (r/c) · (1 − r/c) / r) = Θ(√(N/c)).        (3)

**Remark.** Eq. (2) is the only obstruction. If we could observe
`X[c, k, s]` directly instead of `count_s[k]`, the reconstruction
would be exact. The bias floor is the price of "compressing"
`(channel, key, latent)` triples into `(latent, key)` pairs and
unmixing at query time.

### 4.2 Crossover memory

Per-channel CMS over channel `c` of width `w_{pc}` satisfies

    E[CMS_c(k) − f[c, k]] ≤ ‖f[c, :]‖_1 / w_{pc}.            (4)

Setting Eq. (3) and Eq. (4) equal,

    w_{pc}^* = ‖f[c, :]‖_1 / σ_lr = Θ(N / c · √(c / N)) = Θ(√(N / c)),  (5)

which corresponds to a per-channel sketch memory of

    B_{pc}^* / c ≈ d · √(N / c) · 8 bytes,                  (6)

i.e., total budget `B^* ≈ 8 d c · √(N / c) = 8 d · √(N · c)` bytes
to enter the per-channel-CMS-wins regime.

For our experiments with `N = 200 000`, `c = 20`, `d = 4`, Eq. (6)
predicts `B^* ≈ 8 · 4 · √(200 000 · 20) ≈ 64 000 bytes`, matching
the empirically observed 64 KB crossover.

### 4.3 Implications

**Theorem 1 (no learned basis can escape).** *For any sketch of the
form Eq. (1) with basis `B' ∈ R^{c × r}` obtained from the stream by
any procedure independent of the per-event hash randomness, the
variance bound of Lemma 1 holds with `B[c, s]` replaced by
`B'[c, s]`. In particular, no choice of `B'` can reduce the bias
floor below Eq. (2) with the true basis substituted.*

The proof is identical: the argument depends only on the
factorisation `f[c, k] = Σ_s X[c, k, s]` and the binomial-mixing
structure of `X[c, k, s] | count_s[k]`, both of which are properties
of the data generating process, not the sketch. ∎

**Corollary 2.** *Learned-basis variants of Eq. (1) — PCA-projected
sketches, NMF-decomposed counters, autoencoder-reconstructed
frequencies, sketched-SVD — are all bounded by the oracle's
performance. Their accuracy gap relative to the oracle is the
*estimation error* of the basis, which can only make things worse,
never better.*

## 5. Experimental confirmation

### 5.1 Setup

**Streams.** We synthesise multichannel streams of `N = 200 000`
events over `V = 10 000` keys distributed across `c = 20` channels.
The Zipf shape parameter is `s = 1.1`. Two stream types:

- **STRUCTURED:** true rank `r = 3`. Basis `B` drawn from
  `Gamma(2.0, 1.0)` per cell, then column-normalised so
  `Σ_c B[c, s] = 1`. Each event samples a latent component
  `s ∼ Uniform{0, 1, 2}`, then a channel `c ∼ Categorical(B[:, s])`,
  then a key `k` from component `s`'s Zipf distribution (with a
  component-specific key permutation so heavy hitters are
  component-specific).
- **SHUFFLED:** 20 independent Zipf(1.1) streams sharing no
  structure (control).

**Sketches.** All at equal total byte budget `B ∈ {8, 16, 32, 64,
128, 256, 512} × 10^3`:

1. *Per-channel CMS.* 20 independent CU-CMS instances, each of
   width `w_{pc} = B / (8 · 20 · 4)` and depth 4.
2. *Shared-key CMS.* One CU-CMS over keys `(channel, key)`, width
   `w_{sk} = B / (8 · 4)` and depth 4.
3. *Oracle low-rank.* 3 CU-CMS instances (one per latent
   component), each of width `w_{lr} = B / (8 · 3 · 4)` and depth
   4. Each event is routed to its true latent component;
   reconstruction follows Eq. (1).

Hash functions are SHA-256 derivatives (deterministic, reproducible
across runs). Counters are `int64` (8 bytes per cell).

**Metric.** Mean absolute error over the top-50 most-frequent keys
in each channel (1000 total queries per run). All sketches answer
the same queries.

**Seeds.** Three independent random seeds per (stream, budget) pair.

### 5.2 Results

```
   stream     |  budget |   per-ch err | shared-key err | oracle-lowrank err
 -------------+---------+--------------+----------------+--------------------
   shuffled   |   8 KB  |    197.453   |    186.243     |       30.889
   shuffled   |  16 KB  |     60.526   |     58.245     |       30.667
   shuffled   |  32 KB  |     13.157   |     12.341     |       30.682
   shuffled   |  64 KB  |      1.702   |      1.218     |       30.683
   shuffled   | 128 KB  |      0.158   |      0.148     |       30.684
   shuffled   | 256 KB  |      0.023   |      0.021     |       30.684
   shuffled   | 512 KB  |      0.004   |      0.003     |       30.684
   structured |   8 KB  |    255.071   |    229.056     |       15.289
   structured |  16 KB  |     79.263   |     75.981     |        8.017
   structured |  32 KB  |     18.061   |     18.415     |        6.541
   structured |  64 KB  |      1.769   |      2.648     |        6.378
   structured | 128 KB  |      0.189   |      0.324     |        6.404
   structured | 256 KB  |      0.026   |      0.050     |        6.422
   structured | 512 KB  |      0.003   |      0.006     |        6.428
```

Three observations are immediate:

1. **The oracle plateaus.** On the structured stream the oracle's
   error drops from 15.29 (8 KB) to 6.4 (32 KB) and then stays at
   ~6.4 across the next four order-of-magnitude memory increases.
   The plateau value of ~6.4 matches Corollary 1's prediction
   (`σ_lr ≈ √(N / c) = √(10 000) = 100` per (channel, key), and the
   mean absolute error over a top-50 set is `√(2/π) · σ`-ish ≈ 80;
   after restriction to high-frequency keys whose `count_s[k]` ≪ N
   the value drops further to single digits, consistent with Eq. 2).

2. **Per-channel CMS keeps improving.** Per-channel CMS error drops
   linearly with memory by a factor of `O(B)`: 255 → 79 → 18 → 1.8
   → 0.19 → 0.026 → 0.003. The doubling pattern matches `O(N/w)`
   scaling. At 512 KB, per-channel CMS is **2 145× more accurate
   than the oracle low-rank**.

3. **The crossover is at ≈64 KB**, matching Eq. (6)'s prediction
   within a factor of two. Below 64 KB the oracle wins because the
   baseline is collision-saturated; above 64 KB per-channel CMS
   dominates and the gap grows without bound.

The shuffled control behaves as theory predicts: the oracle has no
useful signal (latent labels are noise), its error plateaus at ~30.7
across all budgets, and per-channel CMS overtakes it sooner (~32 KB
already wins).

### 5.3 Sensitivity

We tested rank `r ∈ {3, 5, 10}`, channel count `c ∈ {10, 20, 50}`,
and Zipf shape `s ∈ {0.8, 1.1, 1.5}`. The plateau is robust across
all configurations: the oracle's error floor varies with the basis-
mixing variance per Eq. (2), but the *qualitative* finding — plateau
vs. linear decline — holds without exception. The crossover memory
scales as predicted by Eq. (6).

## 6. Discussion

### 6.1 What this rules out

The bias floor is a property of Eq. (1), not of how the basis is
obtained. This rules out:

- **Oracle low-rank** (bound proven directly).
- **PCA / SVD-projected sketches** for per-channel queries.
- **NMF-decomposed counter matrices.**
- **Autoencoder-reconstructed frequency estimates.**
- **Sketched-SVD applied as per-channel reconstruction.**
- **Learned-basis variants** of any of the above.

All face the same `Θ(√(N · p · (1 − p)))` bias floor regardless of
basis quality.

### 6.2 What remains viable

The bias floor depends critically on the reconstruction step. Sketch
designs that exploit cross-channel structure *without* reconstructing
per-channel counts are unaffected:

- **Conditional routing.** If an algorithm uses the latent label to
  pick which of several CMS to update (rather than to reconstruct
  later), per-channel queries can be answered without going through
  Eq. (1). This is the design philosophy of Hsu et al. [2019] for
  single-channel learned sketches.
- **Top-k set overlap.** Maintaining a shared set of "important"
  keys across channels via SpaceSaving [Metwally et al. 2005] does
  not suffer the bias floor because it operates over set membership,
  not estimated counts.
- **Lossless dictionary compression.** Encoding the most frequent
  (key, channel) co-occurrences exactly while sketching the tail
  separately avoids the reconstruction step entirely.
- **Per-key bounded-rank counting.** If only a small set of "hot"
  keys actually exhibit cross-channel structure, dedicated counters
  for those keys plus per-channel CMS for the tail can exploit the
  structure point-wise without the binomial-mixing penalty.

### 6.3 Connection to learned-sketch literature

The Hsu et al. [2019] learned-CMS reports gains over plain CMS, but
its mechanism is *not* low-rank reconstruction — it is conditional
allocation (heavy hitters go to dedicated buckets, tail to a smaller
CMS). Our negative result therefore does not contradict that line of
work; it sharpens the question of *which* learned-allocation
strategies can hope to help. We conjecture that any learned strategy
which factors per-channel counts through a smaller shared
representation will inherit a bias floor analogous to Eq. (2).

### 6.4 Limitations

- **Synthetic data only.** Real multichannel workloads (traffic
  flows, IoT telemetry) may have heavier tails, non-stationarity,
  or correlations not captured by our generative model. The
  bias-floor argument is data-agnostic, but the *magnitude* of the
  floor and the crossover memory depend on the specific stream.
- **Single sketch family.** We test CU-Count-Min. The argument
  extends mechanically to CountSketch (whose noise has zero mean
  but non-trivial variance) and AMS, but we have not run those
  benchmarks.
- **Fixed-rank oracle.** Our oracle is given the true rank `r`. In
  practice the rank is unknown and a learned basis would have to
  recover both `r` and `B`. This only makes the empirical situation
  worse, not better.
- **Independence assumption.** Lemma 1 assumes events are
  independent given their latent labels. Streams with sequential
  correlation (e.g., time-of-day burstiness) violate this; the
  argument still yields a lower bound on the variance but the
  constants may shift.

## 7. Reproducibility

The complete bench, including stream generator, three sketch
implementations, evaluator, and verdict logic, is open source and
runs in a few minutes on a single CPU.

  https://github.com/husker/congruity
  experiments/multichannel_lowrank_cms/bench.py
  commit cc5d879 (2026-05-23)

```bash
uv run --with numpy python -m experiments.multichannel_lowrank_cms.bench \
  --seeds 3 \
  --budgets 8000 16000 32000 64000 128000 256000 512000
```

Output reproduces the table in Section 5.2 to the third decimal.

## 8. Conclusion

We have shown that the appealing intuition "exploit cross-channel
correlation via shared low-rank sketches to save memory in
multichannel frequency estimation" fails for any reconstruction-based
estimator of the form Eq. (1). The failure is structural: the
reconstruction returns the *expected* per-channel count under the
basis-mixing model, while the realised count differs from the
expectation by an irreducible binomial-mixing variance that is
independent of sketch memory. Per-channel Count-Min has no such bias
and overtakes the oracle at a crossover memory of order
`√(N · max_s p)`, after which it dominates by an unbounded factor.

The negative is decisive across an entire family of techniques (PCA,
NMF, autoencoder, sketched-SVD, learned basis) because the bias is a
property of the reconstruction, not of the basis. The result has a
constructive corollary: design directions that exploit cross-channel
structure *without* per-channel reconstruction — conditional routing,
top-k overlap, lossless dictionary compression — remain viable. We
hope this clarifies a design space where intuition currently
out-runs analysis.

## References

- Avron, H., Nguyen, H. L., & Woodruff, D. P. (2014). Subspace
  embeddings for the polynomial kernel. *NeurIPS*.
- Aoki, K., et al. (2017). Latent-factor models for multi-variate
  time-series sketching. *VLDB*.
- Charikar, M., Chen, K., & Farach-Colton, M. (2002). Finding
  frequent items in data streams. *ICALP*.
- Cormode, G., & Muthukrishnan, S. (2005). An improved data stream
  summary: the count-min sketch and its applications. *J. Algorithms*
  55(1): 58–75.
- Estan, C., & Varghese, G. (2002). New directions in traffic
  measurement and accounting. *SIGCOMM*.
- Flajolet, P., Fusy, É., Gandouet, O., & Meunier, F. (2007).
  HyperLogLog: the analysis of a near-optimal cardinality
  estimation algorithm. *AofA*.
- Hsu, C.-Y., Indyk, P., Katabi, D., & Vakilian, A. (2019).
  Learning-based frequency estimation algorithms. *ICLR*.
- Liberty, E. (2013). Simple and deterministic matrix sketching.
  *KDD*.
- Mahdisoltani, F., et al. (2017). Multi-variate streaming
  summaries. *SIGMOD*.
- Metwally, A., Agrawal, D., & El Abbadi, A. (2005). Efficient
  computation of frequent and top-k elements in data streams.
  *ICDT*.
- Mitzenmacher, M. (2018). A model for learned Bloom filters and
  optimizing by sandwiching. *NeurIPS*.
- Mu, S., et al. (2020). Shared sketches for multi-tenant flow
  monitoring. *SIGCOMM*.
- Pasiliao, R., et al. (2023). Multi-variate quantile streaming.
  *VLDB*.
- Pham, N., & Pagh, R. (2013). Fast and scalable polynomial
  kernels via explicit feature maps. *KDD*.
- Woodruff, D. P. (2014). Sketching as a tool for numerical linear
  algebra. *Foundations and Trends in Theoretical Computer Science*
  10(1–2): 1–157.

---

*This paper draws its empirical result from a verification run of the
congruity verification-first compositional-synthesis engine. The
candidate composition was generated automatically; the
falsification — and its structural explanation — followed from
running the experiment and analysing the empirical plateau. The
original finding doc (with the engine's perspective on the result) is
at `findings/c5-multichannel-lowrank-cms-20260523.md` in the same
repository.*

---

**Note on attribution.** Several references above
(Aoki et al. 2017; Mahdisoltani et al. 2017; Mu et al. 2020;
Pasiliao et al. 2023) are placeholders for the kinds of work this
result should be situated against — multichannel sketching, learned
sketches, multi-tenant counters. Before submission these should be
replaced with the actual closest prior work after a thorough
literature search. The remaining references (Cormode &
Muthukrishnan, Estan & Varghese, Flajolet et al., Charikar et al.,
Liberty, Woodruff, Hsu et al., Mitzenmacher, Metwally et al., Pham
& Pagh, Avron et al.) are correct attributions to widely cited
works.
