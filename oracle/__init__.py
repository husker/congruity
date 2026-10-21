"""congruity oracle — differential testing for cache-replacement equivalence claims.

This package operationalizes the bridge claim from finding 01:
KV-cache eviction policies in LLM serving are operationally equivalent to
adaptive cache replacement in databases/systems. If true, applying ARC
(Megiddo & Modha, 2003) to a KV-cache should be competitive with or beat
current LLM-serving eviction strategies (PagedEviction, Zipage, IceCache).

v0 scope: pure-Python differential testing on synthetic LLM-serving workloads.
Real-system validation (drop ARC into vLLM, measure on production-like traces)
is a future-work item; the synthetic test is the cheap fail-fast gate.
"""

__version__ = "0.0.1"
