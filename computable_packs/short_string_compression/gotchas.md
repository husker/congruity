# Short String Compression Gotchas

- Round-trip validation is mandatory for every evaluated record.
- Include dictionary or model side information in the byte budget unless the
  benchmark explicitly declares it pre-shared.
- Do not train dictionaries, tokenizers, or templates on held-out records.
- HPACK is the strongest specialist bar for HTTP-like streams; a method that
  beats Brotli or Zstd but loses to HPACK is only a partial result.
- Compression ratio and latency are different metrics. Do not claim one from the
  other.
