# Local Run Topic Candidates - 2026-05-23

This note captures promising topics for the next Congruity compose runs. The
filter is practical: each topic should be understandable, locally testable, and
capable of producing either a useful positive result or a clean negative.

## Top Recommendation

Run these three first:

1. **Solver Portfolio For Hard Route Problems**
2. **Adaptive Streaming Sketches**
3. **Structured Log / JSON Compression**

They have the best mix of local testability, strong baselines, product
relevance, and chance of producing a meaningful discovery.

## Candidate Topics

### 1. Solver Portfolio For Hard Route Problems

**Plain-language question:** Can cheap measurements of a route problem tell us
which solving strategy is most likely to work?

**Why it is good:** This builds directly on the newest TSP/Hamiltonian run. It
is also easy to start locally because the first experiment can use synthetic
route instances and simple solver choices.

**Likely baselines:**

- Fixed KNN-pruned solver
- Simple SAT, dynamic-programming, or local-search choices
- Single default solver strategy

**Possible datasets:**

- TSPLIB-style synthetic instances
- ASlib/TSP-LION if available locally

**Discovery test:** Do spectral, topology, or graph-hardness features improve
solver selection over a single default strategy?

### 2. Adaptive Streaming Sketches

**Plain-language question:** Can a tiny memory summary learn where to spend its
memory as the data stream changes?

**Why it is good:** This follows naturally from the recent Count-Min Sketch
work. The low-rank multichannel result was a useful negative; this asks a better
next question about adaptive allocation rather than shared reconstruction.

**Likely baselines:**

- Count-Min Sketch
- CountSketch
- SpaceSaving
- Misra-Gries

**Possible datasets:**

- Synthetic Zipf streams
- Drifting-key streams
- Multichannel streams with changing correlation

**Discovery test:** Does adaptive allocation win under drift or skew without
using extra memory or hidden side information?

### 3. CPU Vector Search Under Memory Caps

**Plain-language question:** Can we search high-dimensional vectors faster or
with less memory by using structure in the data?

**Why it is good:** This is product-relevant, measurable, and local. It maps to
retrieval, semantic search, and embedding infrastructure.

**Likely baselines:**

- Brute force search
- HNSW
- Random projection
- PCA or product quantization

**Possible datasets:**

- Local sentence embeddings
- Synthetic clustered vectors
- ANN-Benchmarks-style data

**Discovery test:** Can a structure-aware quantization or routing method improve
recall, latency, or memory versus simple PCA or HNSW?

### 4. Structured Log / JSON Compression

**Plain-language question:** Can we compress logs and JSON better by first
understanding their structure, then applying reversible transforms?

**Why it is good:** This is easy to explain and potentially marketable. It also
has very clear byte-and-time metrics.

**Likely baselines:**

- gzip
- brotli
- zstd
- zstd with dictionaries
- lz4

**Possible datasets:**

- Synthetic logs
- Synthetic JSON event streams
- Canterbury, Silesia, or Pizza&Chili-style corpora

**Discovery test:** Does schema extraction plus per-column reversible transforms
beat strong general codecs after metadata overhead is included?

### 5. Geometry Optimizer Follow-Up

**Plain-language question:** Can the successful Heilbronn optimization idea
generalize to other hard point-placement problems?

**Why it is good:** Congruity already produced one positive result here, so this
is a good place to ask whether the mechanism transfers.

**Likely baselines:**

- Simulated annealing
- Differential evolution
- CMA-ES
- Basin hopping

**Possible targets:**

- No-three-in-line layouts
- Packing problems
- Max-min distance layouts
- Discrepancy-style point layouts

**Discovery test:** Does smooth surrogate optimization repeatedly beat generic
black-box optimizers on small, reproducible geometry benchmarks?

### 6. SAT / MIP Restart And Branching Policies

**Plain-language question:** Can simple instance features tell a solver when to
restart or which variable to branch on?

**Why it is good:** SAT and MIP have serious benchmark cultures and very clear
runtime outcomes. A small local pilot can still be meaningful if the stop rules
are strict.

**Likely baselines:**

- Default PySAT or SCIP-style behavior
- Random policies
- Static heuristic policies

**Possible datasets:**

- SATLIB instances
- Small MIPLIB instances
- Synthetic SAT or integer-programming instances

**Discovery test:** Does a learned or feature-guided restart/branching rule
improve solved count or runtime under fixed time limits?

### 7. Learned Bloom / Cuckoo Filter Variants

**Plain-language question:** Can a membership filter use learned structure in
the keys to reduce false alarms without using more memory?

**Why it is good:** The benchmark can be compact and clean. It is also easy for
laypeople to understand: "Does the filter say something is present when it is
not?"

**Likely baselines:**

- Bloom filter
- Cuckoo filter
- Sandwiched learned Bloom filter

**Possible datasets:**

- Synthetic skewed-key sets
- URL-like keys
- Log-derived identifiers
- Drifting positive/negative query distributions

**Discovery test:** Does a learned or partitioned filter reduce false positives
at equal memory, including under distribution drift?

### 8. Dynamic Vehicle Routing Decomposition

**Plain-language question:** Can a big changing delivery-route problem be split
into smaller pieces without losing too much route quality?

**Why it is good:** The value proposition is understandable, and it connects the
TSP work to a practical logistics story.

**Likely baselines:**

- OR-Tools routing
- Nearest-neighbor route construction
- K-means decomposition plus local routing

**Possible datasets:**

- Synthetic vehicle-routing instances
- Synthetic vehicle-routing-with-time-window instances
- VRP-REP-style benchmark instances

**Discovery test:** Does decomposition improve runtime while preserving route
quality when orders or travel times change?

## Selection Criteria

Prefer topics with:

- Named baselines before implementation
- Numeric success and failure metrics
- Smoke tests under 10 minutes
- Full pilots under 4 hours
- No paid APIs
- CPU-only feasibility
- Clear explanation for laypeople
- A result that can feed back into the web dashboard as evidence

## Sources To Use When Framing Runs

- ANN-Benchmarks: <https://ann-benchmarks.com/>
- SATLIB: <https://www.cs.ubc.ca/~hoos/SATLIB/>
- ASlib: <https://arxiv.org/abs/1506.02465>
- MIPLIB: <https://miplib.zib.de/>
- OR-Tools routing: <https://developers.google.com/optimization/routing/routing_tasks>
- Canterbury Corpus: <https://corpus.canterbury.ac.nz/index.html>
- Silesia Corpus: <https://data-compression.info/Corpora/SilesiaCorpus/index.html>
- Cuckoo Filter paper: <https://www.cs.cmu.edu/~dga/papers/cuckoo-conext2014.pdf>
- Learned Bloom Filters: <https://arxiv.org/abs/1802.00884>
