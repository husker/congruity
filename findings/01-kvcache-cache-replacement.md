# Finding 01 — The LLM-serving community has reinvented cache replacement

Date: 2026-05-18
Author: James Gagan
Method: AI-driven cross-subfield bridge discovery (congruity v0 prototype)
Status: PRELIMINARY — single bridge, single AI, partial novelty check

## TL;DR

A 30-minute experiment using GPT-4-class Codex to bridge two contemporary CS sub-literatures — KV-cache management for LLM serving (Corpus A) and cache replacement in databases/systems (Corpus B), both 2025–2026 — produced a specific, well-grounded equivalence claim with 8/8 verified citations:

> KV-cache token/page eviction in LLM serving is operationally equivalent to adaptive cache/page replacement in databases, OS buffer pools, and edge systems, with "token usefulness" playing the role of reuse probability. The features used in classical and learned cache-replacement policies (reuse distance, access frequency, temporal locality, hit/miss cost) map directly onto KV-cache signals (attention reuse, token importance, decoding phase).

A novelty probe on FlashInfer (a major recent LLM-serving paper, 69 references) found **0 references** to classical cache-replacement work — no Belady, no ARC, no LRU-K, no LIRS, no 2Q, no "page replacement." If this pattern holds across the LLM-serving literature, the field has been broadly blind to 60 years of directly relevant work.

This finding is a real research output. It is also a working demo of the bridge-discovery system described in the congruity design doc.

## Why this experiment matters

Two flagship "AI as scientific researcher" systems shipped in the last two years: Google's multi-agent AI co-scientist on Gemini 2.0 and Stanford/CZ Biohub's Virtual Lab. Both industrialized **hypothesis generation** and added model-judging-model critique. Neither solved **verification**. Both routed truth-checking to humans in a wet lab.

The congruity project asks: can a solo builder own the verification half by picking a domain where verification is **executable code, not a wet lab**? CS algorithmic equivalence — where you can implement both algorithms and run differential tests — is the candidate domain. This finding is a first test of whether AI-driven bridge discovery in that domain actually produces useful output.

## The experimental design (and the bug we fixed)

### What we did

1. Selected two CS sub-literatures we believed were genuinely separated despite working on structurally identical problems:
   - **Corpus A**: KV-cache memory management for LLM inference serving (2025–2026 only).
   - **Corpus B**: Cache replacement and buffer-pool work in databases / systems / edge (2025–2026 only).
2. Fetched 8 papers per side via OpenAlex keyword search, restricted to post-2024 publication.
3. Built a single prompt: "Here are 16 abstracts. Identify any operational equivalence between Corpus A and Corpus B. Cite only the papers in the corpus."
4. Ran Codex (GPT-4-class, training cutoff ~late 2024 / early 2025) on the prompt with no web search, high reasoning effort.
5. Verified Codex's citations against the supplied corpus via an automated checker.
6. Probed novelty: fetched references of a major LLM-serving paper (FlashInfer, 69 refs) and checked how many cite classical cache-replacement literature.

### What we learned the hard way (process bug worth documenting)

An earlier experiment (session-types ↔ Singularity OS) appeared to show Codex confabulating citations. On inspection, the bug was ours: the fetch script silently substituted off-topic papers when title-search didn't find an exact match, and OpenAlex had attached a chemotherapy-paper abstract to the Mach OS paper's title. Codex's "confabulated" citations were actually correct references to the (wrong) papers we'd given it.

Fix: require ≥0.4 word-Jaccard similarity between requested and returned title, never silently substitute, drop papers with corrupted abstracts. After the fix, the corpus is clean and the Codex citations land cleanly on real corpus papers.

The lesson: **the verification harness for AI-driven bridge discovery is at least as much about data hygiene as it is about model output quality.** Garbage in, garbage out.

### Why post-cutoff papers matter

A previous experiment used pre-Singularity papers (1993–2004) on session types and SPIN. Codex correctly surfaced the well-documented session-types-↔-Singularity bridge — but we couldn't tell if it derived the bridge from the abstracts or recited it from training data. Aggressive vocabulary paraphrase failed to degrade Codex's output, which is what memorization looks like.

Using papers from 2025–2026 fixes that. Codex's training cutoff is ~late 2024; these papers post-date that. Whatever bridge Codex finds cannot be memorized from the literature, because Codex has never read the literature.

## The finding

Codex's verbatim claim:

> **Equivalence claim:** KV-cache token/page pruning in LLM serving is operationally equivalent to adaptive cache/page replacement in backend, database, edge, and processor systems, with "token/block usefulness" playing the role of reuse probability.
>
> **Mechanism transferred:** Reuse learned or adaptive replacement policies from Corpus B to decide which KV tokens, blocks, or pages in Corpus A should be retained, evicted, compressed, or offloaded. Features such as reuse distance, access frequency, temporal locality, query/context similarity, hit/miss cost, and dynamic workload state map directly onto KV-cache signals such as attention reuse, token semantic importance, decoding phase, sequence position, transfer cost, and accuracy degradation risk.
>
> **Confidence: 9/10.** The abstraction is strong because both corpora optimize bounded memory under uncertain future reuse with eviction decisions, and several papers on both sides explicitly discuss page/block-level replacement, learned policies, semantic/contextual importance, and latency-throughput tradeoffs.

### Citations Codex provided

| Corpus | Paper | Year | In corpus? |
|---|---|---|---|
| A | Zipage: Maintain High Request Concurrency for LLM Reasoning through Compressed PagedAttention | 2026 | ✅ |
| A | IceCache: Memory-efficient KV-cache Management for Long-Sequence LLMs | 2026 | ✅ |
| A | PagedEviction: Structured Block-wise KV Cache Pruning | 2026 | ✅ |
| B | Improving Cache Performance using a Learned Replacement Policy | 2025 | ✅ |
| B | Machine Learning-Enhanced Database Cache Management | 2026 | ✅ |
| B | Learning-based page replacement scheme for efficient I/O processing | 2025 | ✅ |
| B | Adaptive Contextual Caching for Mobile-Edge Large Language Model Service | 2025 | ✅ |
| B | A Deep Q-Network Approach to Intelligent Cache Management in Dynamic Backend Environments | 2025 | ✅ |

All 8 citations are in the supplied corpus. Automated citation checker passes (`scripts/step3_citation_check.py`).

### Novelty probe

For the bridge to be a useful finding, the LLM-serving community needs to be currently failing to cite classical cache-replacement work. A complete novelty probe would check several representative LLM-serving papers' references for citations to Belady, ARC, LRU-K, LIRS, 2Q, CAR, page-replacement theory, working-set theory, etc.

Partial result (Semantic Scholar's free tier rate-limited us out of 3 of 4 attempts):
- **FlashInfer** (Ye et al., a serious LLM-serving inference engine paper): **0 of 69 references** match any classical cache-replacement pattern. No Belady, no ARC, no working-set theory, no virtual-memory paging literature.

Single-paper data point, but striking. A real systems paper with 69 carefully-chosen references contains zero direct engagement with the decades of cache-replacement work that addresses its exact core problem (which entries to keep in fast memory under skewed access patterns).

This needs to be replicated across more representative LLM-serving papers (PagedAttention/vLLM, SGLang, vAttention, IceCache, PagedEviction, Zipage) before strong claims about field-wide blindness. The rate-limiting on Semantic Scholar's free tier was the actual blocker; a paid API key or local PDF parsing would resolve it.

## What this proves, and what it doesn't

### What it proves

1. **AI-driven cross-subfield bridge discovery is possible.** Codex, given only post-training-cutoff abstracts, produced a specific, mechanistically-described bridge with verified citations in under a minute. The architecture sketched in the congruity design doc works for at least this case.
2. **Citation grounding is mechanically verifiable.** The automated check ruled out confabulation: every cited paper is in the supplied corpus.
3. **There is at least one major LLM-serving paper with zero engagement with classical cache-replacement literature.** Sample of one, but suggestive.

### What it doesn't prove

1. **It doesn't prove the bridge is novel.** A single-paper novelty probe is not enough. Some 2025–2026 KV-cache papers may cite classical work; we haven't checked broadly.
2. **It doesn't prove the bridge is correct.** "Plausible" ≠ "true." The strongest validation would be the differential-test oracle from § 10 of the design doc — implement ARC on a vLLM KV-cache, run on real serving workloads, compare to PagedEviction. That's a research project, not a 30-minute test.
3. **It doesn't prove the system generalizes.** One bridge in one domain pair is one data point. A second test (speculative decoding ↔ CPU branch prediction) is queued.

### What's between proven and not

The redaction-ablation experiment from the earlier session-types test showed Codex pulling "session" from training data even when redacted out of the prompt. Even with post-cutoff papers, we can't fully prove Codex's reasoning is purely over the supplied content — it may be applying general knowledge of "cache replacement" as a concept that pre-dates its training cutoff, even though the specific KV-cache papers don't. This is an inherent limit of training-aware LLM evaluation; mitigations exist (forcing verbatim quote extraction from the prompt; differential-test oracles; multi-model agreement) but none fully eliminate the question.

## Implications

If this generalizes (next experiment will tell):

1. **AI-driven bridge discovery is a real research-acceleration tool** for finding cross-community equivalences that the communities themselves miss because vocabulary differs. The mechanism is independent of the AI being trained on the answer — what matters is whether the AI was trained on the supplied abstracts.

2. **The verification half of "AI as scientific researcher" is ownable by solo builders** in domains where verification is executable code. Differential testing on cache replacement is a 1-week project, not a 1-year wet-lab study. The same architectural pattern doesn't work for biomedical discovery (verification needs a lab) but works for computational equivalence claims.

3. **The LLM-serving community may benefit from reading their own discipline's predecessor literature.** Specifically: ARC's adaptive split between recency and frequency, LIRS's inter-reference recency, 2Q's two-queue insight, Belady's optimal lower bound. These are exactly the kinds of structured policies the KV-cache literature is currently rediscovering empirically. Whether any of them ports productively requires the differential test; whether the field finds the connection worth following is a sociological question.

## Reproducibility

All scripts and data at `~/Projects/congruity/`:
- `scripts/step3_kvcache_experiment.py` — fetches corpus, builds prompt
- `data/step3_experiment/corpus.json` — the 16 papers
- `data/step3_experiment/prompt.txt` — the literal prompt sent to Codex
- `data/step3_experiment/codex_response.txt` — Codex's full output
- `scripts/step3_citation_check.py` — automated citation verifier
- `data/step3_experiment/citation_check.json` — citation-check result
- `scripts/step3_novelty_check_foundational.py` — partial novelty probe
- `data/step3_experiment/novelty_check_foundational.json` — FlashInfer ref scan

To reproduce, `uv run scripts/step3_kvcache_experiment.py` then pipe the prompt to `codex exec`. Citation check is mechanical. Novelty check requires patience with Semantic Scholar's rate limits or a paid API key.

## Open work

1. Complete the novelty probe across 5–10 representative LLM-serving papers. If the FlashInfer pattern (0/69) holds broadly, the bridge is publishable as a position paper without further verification. If most papers cite classical work, the bridge is folklore.
2. Run the differential-test oracle on a single concrete claim. The cleanest candidate: implement ARC-style adaptive split between recency and frequency on a vLLM KV-cache; measure hit rate and serving throughput against PagedEviction's published numbers.
3. Test generalization on a second domain pair (speculative decoding ↔ CPU branch prediction is queued).
4. Replicate the bridge claim with a different model family (Claude rather than Codex) to detect any single-model artifacts in the result.
