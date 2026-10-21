"""Synthetic LLM-serving workload generator.

Models cache access patterns characteristic of multi-tenant LLM inference:
- A shared system-prompt prefix that all requests read (the "hot prefix").
- Per-request unique prompt + generation blocks.
- During generation, each new token attends over all prior blocks in the
  same request (modeled as one read per prior block per new generated block).

The workload is a sequence of `AccessEvent`s. A cache policy decides which
keys to evict on miss. This is the input to `oracle.runner.differential_test`.

For v0 differential testing this is sufficient. A real KV-cache benchmark
would replace this with traces captured from vLLM running production prompts
(see `oracle/policies.py::PagedEvictionStub` for the integration TODO).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class AccessEvent:
    """One read of a single cache block."""
    request_id: int
    block_id: int

    @property
    def key(self) -> str:
        """Cache-content-hash analogue: shared-prefix blocks (request_id=-1)
        collapse to the same key across all requests, modeling prefix caching."""
        if self.request_id < 0:
            return f"shared_b{self.block_id}"
        return f"r{self.request_id}_b{self.block_id}"


@dataclass
class WorkloadConfig:
    n_requests: int = 200
    avg_prompt_blocks: int = 30
    avg_generate_blocks: int = 20
    shared_prefix_blocks: int = 10
    prompt_jitter: float = 4.0
    generate_jitter: float = 3.0
    seed: int = 42


def generate_workload(cfg: WorkloadConfig) -> Iterator[AccessEvent]:
    """Yield `AccessEvent`s for an LLM-serving workload.

    Each request:
    1. Reads the shared prefix (cfg.shared_prefix_blocks blocks).
    2. Reads its unique prompt blocks (~cfg.avg_prompt_blocks).
    3. Generates tokens; each generated block triggers a re-read over all
       prior blocks in the same request (attention), then writes the new
       block. We model this as: for each new generated block, emit reads
       on all prior request-local blocks, then emit the read of the new
       block itself.
    """
    rng = random.Random(cfg.seed)
    next_block_id = 0  # request-local blocks numbered from 0 per request

    for req in range(cfg.n_requests):
        # Per-request lengths jittered around averages
        prompt_blocks = max(1, int(rng.gauss(cfg.avg_prompt_blocks, cfg.prompt_jitter)))
        gen_blocks = max(1, int(rng.gauss(cfg.avg_generate_blocks, cfg.generate_jitter)))

        # Shared prefix reads (cache-key shared across requests)
        for b in range(cfg.shared_prefix_blocks):
            yield AccessEvent(request_id=-1, block_id=b)

        # Per-request unique block IDs
        req_blocks = list(range(next_block_id, next_block_id + prompt_blocks + gen_blocks))
        next_block_id += prompt_blocks + gen_blocks

        # Prefill: read each prompt block once
        for b in req_blocks[:prompt_blocks]:
            yield AccessEvent(request_id=req, block_id=b)

        # Decode: each new block re-reads prior blocks then itself
        for i in range(prompt_blocks, prompt_blocks + gen_blocks):
            for prior in req_blocks[:i]:
                yield AccessEvent(request_id=req, block_id=prior)
            yield AccessEvent(request_id=req, block_id=req_blocks[i])
