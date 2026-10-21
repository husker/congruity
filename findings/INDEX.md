# Verification track record

One row per verified experiment. Outcome labels:

- **clean-positive** — composition beats the relevant baselines with margin
- **partial-positive** — beats some baselines, loses to the strongest
- **clean-negative** — falsified empirically
- **partial-negative** — close to baselines but doesn't beat them

Each row references the originating compose run by `run_id` (the timestamped folder under `data/pipeline/compositions/`). The calibration dashboard joins this table to the runs table on `run_id` and color-codes verified rows by outcome.

| run_id | experiment | composition | outcome | finding doc |
|---|---|---|---|---|
| 20260521-043936 | Heilbronn small-points | C1 smooth gradient flow | **clean-positive** | `findings/heilbronn-verification-20260521.md` |
| 20260520-053938 | Short-string compression | C1 minimizer + template lookup | **partial-positive** | `findings/c1-verification-20260520.md` |
| 20260521-060508 | Embedding quantization | C3 PCA-banded PQ utility allocation | **clean-negative** | `findings/embed-quant-verification-20260521.md` |
| 20260521-023514 | Cache replacement | Succ-SIEVE (successor-prediction eviction) | **clean-negative** | `findings/cache-verification-20260521.md` |
| 20260520-065937 | Multitrack stems | linear cross-stem predictor | **clean-negative** | `findings/multitrack-verification-20260520.md` |
| 20260522-215133 | TSP Delaunay candidate edges | C7 Delaunay + spectral hypergraph + min-cost flow | **clean-negative** | `findings/c7-delaunay-verification-20260522.md` |
| 20260523-033802 | Streaming sketches (imbalance) | C6 imbalance-aware counter allocator vs CU-CMS | **partial-positive** | `findings/c6-imbalance-cms-20260523.md` |
| 20260523-033802 | Streaming sketches (multichannel) | C5 low-rank shared sketch vs per-channel CMS | **clean-negative** | `findings/c5-multichannel-lowrank-cms-20260523.md` |
| 20260523-180439 | Compression (per-block adaptive) | C1 per-block codec selection vs fixed whole-file | **clean-negative** | `findings/c1-per-block-codec-20260523.md` |
| 20260524-052010 | Prediction markets (frontier) | C12 wording-complexity × platform → participation | **clean-negative** | `findings/c12-prediction-market-complexity-20260524.md` |
| 20260524-052010 | Prediction markets (Polymarket) | C8 resolution-source category → delay | **clean-negative** | `findings/c8-polymarket-resolution-source-20260524.md` |
| (C8 follow-up) | Polymarket platform structure | latency-structure analysis: 41h modal delay, US-hours review | **clean-positive (mechanism identified)** | `findings/polymarket-latency-structure-20260524.md` |
| 20260523-033802 | Streaming sketches (adaptive compressive) | C1 adaptive compressive sketch vs CU-CMS / CountSketch / static-oracle | **clean-negative (foreclosed)** | `findings/c1-adaptive-compressive-sketch-20260524.md` |
| 20260523-180439 | Compression (adaptive-delta) | C11 adaptive-delta numeric-log compressor vs delta/xor + zstd-22 / brotli-11 | **clean-negative (foreclosed)** | `findings/c11-adaptive-delta-numeric-20260524.md` |
| 20260523-003811 | Hamiltonian (regime classifier + SAT portfolio) | C1 topology classifier → CaDiCaL/DFS solver selection | **partial-positive (oracle-supported, cheap-learner falsified)** | `findings/c1-hamiltonian-regime-classifier-20260524.md` |
| (cost-aware follow-up) | Hamiltonian portfolio — cost-aware selection | Path #1 cost-aware learner + path #3 safe speculation over recorded timings | **negative for practical value (cost-aware reaches breakeven; not a transfer win)** | `findings/hamiltonian-portfolio-cost-aware-20260707.md` |
| 20260611-022616 | Graph coloring C2–C5 (autoverify one-shot + autosearch) | codex-generated candidates vs TabuCol; Windows portability of the whole autoverify/pack path | **robust TabuCol-parity ceiling: one-shot C2–C5 all timeout, autosearch C3 saturates at fitness 0.0 (tie). No win any method. Windows pipeline fixed.** | `findings/graph-coloring-c2-c5-autoverify-20260707.md` |
| (meta-study) | Expert blind hit-rate study (24 verified compositions) | can a blind expert-proxy (one model ×5) predict which compositions survive empirical test? (pre-registered) | **framework-affirming: rating is highly self-consistent (ρ+0.79, κ+0.60) but uncorrelated with outcome (AUC 0.52, acc=base rate); best result dismissed 0/5, unanimous favorite failed → validates verify-don't-predict** | `findings/expert-hit-rate-study-20260707.md` |
| (latency follow-up) | Polymarket current-cohort replication (2024+) | 41h mode + 18:00-UTC peak → 31h mode + 06-08-UTC peak | **partial-replication (mechanism drift)** | `findings/polymarket-latency-current-cohort-20260524.md` |
| 20260524-052010 | Polymarket wallet concentration (on-chain) | C4 hypergraph co-participation premise; Gini, top-10, HHI, overlap | **partial-positive (concentration confirmed, simple-metric lift modest)** | `findings/c4-polymarket-wallet-concentration-20260524.md` |
| 20260525-011650 | Solver portfolio routing | C1 route-instance fingerprints → solver selection; selection signal found, density clusters weak | **partial-positive** | `findings/solver-portfolio-c1-verification-20260525.md` |
| (C1b follow-up) | Solver portfolio robustness | Held-out-family and hybrid-instance robustness test for 20260525-011650 C1 | **clean-negative** | `findings/solver-portfolio-c1b-robustness-20260525.md` |
| 20260525-042258 | Discrete geometry point layouts | C2 Delaunay-local active repair with periodic global scans | **partial-positive** | `findings/c2-discrete-geometry-delaunay-relaxation-20260525.md` |
| 20260526-043036 | Snake-in-the-box search | C2 novelty/metadynamic MCTS vs plain MCTS and greedy restarts | **partial-positive** | `findings/snake-mcts-c2-verification-20260526.md` |
| 20260526-062120 | Černý reset-word search | C3 evolutionary hard-automata generator vs random and hill-climb baselines | **partial-positive** | `findings/c3-cerny-hard-automata-20260526.md` |
| (C3b follow-up) | Černý strict hard automata | Strongly connected, no closed-core attachments; strict cycle evolution vs strict random/hill baselines | **partial-positive** | `findings/c3b-cerny-strict-hard-automata-20260526.md` |
| (C3c follow-up) | Černý distance stress | Distance >=4/6 from Černý; far cycle evolution vs far random, hill, and seeded-hill baselines | **clean-negative** | `findings/c3c-cerny-distance-stress-20260526.md` |
| 20260526-062120 | Černý reset-word search | C4 landmark-biased MCTS vs plain MCTS and greedy pair compression | **partial-negative** | `findings/c4-cerny-landmark-mcts-20260526.md` |
| 20260526-205855 | Game-theory equilibrium search | C1 no-u-turn restart dynamics vs tuned no-regret/extragradient baselines | **clean-negative** | `findings/c1-game-theory-no-u-turn-20260526.md` |
| 20260521-023514 | Cache replacement (gate-precision test) | C5 time-to-next-use eviction — **rejected** composition (L3=invalid) verified vs LRU/SIEVE/ARC/S3FIFO | **clean-negative** | `findings/cache-ttnu-c5-rejected-verification-20260529.md` |
| 20260520-053938 | Short-string compression (gate-precision test) | C3 static micro-LM + arithmetic coding — **rejected** composition (L3=invalid) verified vs gzip/brotli/zstd+dict/HPACK | **partial-positive (gate over-fired)** | `findings/short-string-c3-rejected-verification-20260529.md` |
| 20260520-053938 | Short-string compression (gate-precision test) | C2 BPE micro-domain coder — **rejected** on BOTH axes (L3=invalid + purely_testable=False) verified vs all baselines | **clean-positive (gate over-fired; best in project)** | `findings/short-string-c2-rejected-verification-20260529.md` |
| 20260522-215133 | TSP Euclidean (rescued from gate) | C2 Delaunay candidate edges + k-way partition + per-cluster 2-opt + stitch vs global 2-opt / NN | **partial-positive (10× speedup, loses tour quality; L3 seam objection confirmed)** | `findings/tsp-delaunay-c2-rescued-verification-20260529.md` |
| 20260530-065501 | Voice isolation (multi-mic; first new-problem run post gate-fix) | C5 blind AuxIVA demixing prior (+ratio mask) vs delay-sum/MVDR beamformers, SI-SDR on synthetic array mixtures | **partial-positive (blind prior beats beamformers incl. oracle-DOA; mask add-on hurts; anechoic/synthetic only)** | `findings/voice-isolation-c5-verification-20260530.md` |

| 20260702-072744 | Snake-in-the-box (new pack; first flywheel label on a new domain) | C1 exact-pruning + rollout-guided + multimodal-frontier search vs restarted stochastic beam at equal budget, holdout Q6-8 unseen seeds | **positive (transfer-win, auto/medium; +1.0 mean edges, 48.33 vs 47.33)** | `findings/snake-in-box-c1-verification-20260702.md` |

| 20260702-072744 | Snake-in-the-box | C5 SAT restart/look-ahead + hyper-heuristic control + fixed-budget trial policy vs stochastic beam, holdout Q6-8 unseen seeds | **positive (transfer-win, auto/medium; +3.33 mean edges, 50.67 vs 47.33 — strongest snake result)** | `findings/snake-record-campaign-20260702.md` |
| 20260702-072744 | Snake-in-the-box | C2 progressive edge-growth + pheromone-guided construction + decomposed multiobjective vs stochastic beam, holdout Q6-8 | **partial (auto/medium; +0.67 mean edges, below 1-edge min effect)** | `findings/snake-record-campaign-20260702.md` |

_To add a row: write a finding doc, then append a line here with the `run_id` of the originating compose run. Outcomes feed the calibration dashboard's outcome column and the verified-outcomes counts._
