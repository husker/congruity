# Cache Replacement Gotchas

- Zipf-only traces are not enough; include loop, scan, and scan-with-hot traces.
- Check capacity after every access.
- Unless a policy is explicitly an oracle baseline, it cannot inspect future
  accesses.
- Successor prediction is usually a prefetching signal, not an eviction signal.
- Compare against SIEVE, ARC, and S3FIFO before claiming a meaningful win.
