# Rescue sweep — re-judging gate-rejected compositions (2026-05-29)

Re-evaluates previously gate-rejected compositions under the fixed gate (prioritize-not-veto ranking + baseline-relative L4). Code-testable domains only.

- Code-testable rejected compositions examined: **32**
- Rescued by priority ranking alone (L3=invalid, fund=0, now top-K): **21**
- Flipped fund>0 → fund=0 under baseline-relative L4: **6**
- Total rescued (now verify-eligible / testable): **27** (84% of rejected)
- Still rejected: **5**

## Rescued by ranking (rejected on L3 verdict only, now top-K)

| run | cid | priority | L3 | claim |
|---|---|---|---|---|
| 20260519-023154 | C3 | 6.9 | invalid | P3's compressive high-dimensional acquisition + P5's tensor adaptive reconstruction + P17' |
| 20260519-023154 | C1 | 6.6 | invalid | P17's quantics tensor cross interpolation + P13's almost-linear graph-flow primitives coul |
| 20260519-023154 | C2 | 6.3 | invalid | P6's topology-aware tensor-network contraction + P13's almost-linear graph-flow framework  |
| 20260519-023154 | C4 | 6.3 | invalid | P16's tensor-based adaptive consensus graph learning + P17's quantics tensor cross interpo |
| 20260520-065937 | C1 | 5.9 | invalid | P131's explicit multi-track music dependency model + P36's attention-only long-context seq |
| 20260521-060508 | C1 | 5.6 | invalid | P36's spectral scale separation + P73's weighted-nearest-neighbor utility estimation + P62 |
| 20260521-023514 | C1 | 5.4 | invalid | P108's per-instance weighting of heterogeneous signals + P89's one-pass streaming discipli |
| 20260521-023514 | C3 | 5.4 | invalid | P128's instance-dependent solver selection + P64's model comparison framing + P108's local |
| 20260521-060508 | C2 | 5.4 | invalid | P122's compressed-sensing variational reconstruction + P111's sparse explanatory feature s |
| 20260520-065937 | C3 | 5.3 | invalid | P119's compact cross-channel CNN filtering + P47's sequential audio/video modeling + P24's |
| 20260521-060508 | C4 | 5.3 | invalid | P91's weighted least-squares embedding objective + P46's least-squares uncertainty evaluat |
| 20260521-060508 | C5 | 5.3 | invalid | P78's rapidly computed circular fingerprints + P142's memory-reduced high-speed k-mer clas |
| 20260519-162024 | C1 | 5.1 | invalid | P105's sparse random-graph regime theory + P134's exact branch-and-reduce template for an  |
| 20260519-162024 | C2 | 5.1 | invalid | P103's clustering of demand profiles + P100's real-time traffic sensing/control + P114's p |
| 20260519-162024 | C4 | 5.1 | invalid | P131's persistent dynamic 3D scene representation + P114's NP-hard autonomous path-plannin |
| 20260520-065937 | C2 | 5.1 | invalid | P131's shared multi-track latent music model + P103's modality-weighted integration idea + |
| 20260521-023514 | C6 | 5.1 | invalid | P9's adjustable relation framework + P108's per-instance modality weights + P17's broad mo |
| 20260521-043936 | C4 | 4.8 | invalid | P82's explicit observation-geometry thinking + P121's topological boundary-network modelin |
| 20260522-062501 | C4 | 0.5 | invalid | P85's explicit spectral-gap/conductance bounds for Metropolis chains + P76's partial-deact |
| 20260522-062501 | C5 | 0.5 | invalid | P93's spectral-independence criterion for fast mixing + P41's reinforcement-learned adapti |
| 20260522-154341 | C3 | 0.5 | invalid | P25's multilevel partitioning + P31's isoperimetric/spectral-gap analysis of Metropolis ch |

## Flipped by baseline-relative L4 (fund>0 → fund=0)

| run | cid | old_fund | new_fund | new_priority | claim |
|---|---|---|---|---|---|
| 20260522-215133 | C2 | 1 | 0 | 5.9 | P26's Delaunay triangulation gives a sparse geometric candidate edge set for Euclidean TSP |
| 20260526-205855 | C6 | 2 | 0 | 4.8 | P177's compressive wideband sensing analogy + P80's black-box adversarial attack procedure |
| 20260522-215133 | C4 | 1 | 0 | 4.6 | P86 decomposes large TSP/routing instances into local subproblems, P72 supplies almost-lin |
| 20260522-062501 | C1 | 1 | 0 | 0.5 | P80's spectral-informed multilevel differentiable partitioner + P70's global/local neural  |
| 20260522-154341 | C1 | 1 | 0 | 0.5 | P25's fast multilevel k-way partitioning + P23's global-partition/local-construction routi |
| 20260522-154341 | C2 | 3 | 0 | 0.5 | P1's density-based graph filtering + P19's continuous multilinear-extension optimization a |

## Still rejected

| run | cid | old_fund | new_fund | L3 | claim |
|---|---|---|---|---|---|
| 20260519-024943 | C3 | 6 | 3 | invalid | P11's multiple electronic-structure representations + P63's machine-learned interatomic-po |
| 20260521-023514 | C5 | 0 | - | invalid | P8's constrained temporal scheduling + P45's next-event prediction + P21's high-rate trigg |
| 20260521-060508 | C6 | 0 | - | invalid | P121's aggressive undersampling plus reconstruction tradeoff + P122's learned variational  |
| 20260522-062501 | C3 | 0 | - | invalid | P26's differentiable routing relaxation + P57's multilinear-extension optimization and rou |
| 20260522-062501 | C7 | 0 | - | invalid | P31's self-play learning of local mesh operations + P44's removal/insertion operator taxon |
