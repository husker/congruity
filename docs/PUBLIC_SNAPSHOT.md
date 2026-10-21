# Public snapshot contents

This candidate is based on the default branch at
`84baf8a350bc15d03470b138a6eb7752b52311bc` (July 8, 2026), with documentation,
publication hygiene and portability changes prepared on October 20, 2026.
Later experimental branches and uncommitted research are not included.

## Included

- Pipeline, evaluation packs, experiment implementations and tests.
- Dated findings, including negative and inconclusive results.
- Selected mathematical certificates, numerical checkpoints and record coordinates.
  Existing source attribution is retained beside externally sourced coordinates.
- A synthetic offline dashboard example, clearly labelled as unverified.
- A dependency lockfile and CI configuration.

## Excluded

- The original Git history, branches, commit email metadata and local working state.
- Personal agent configuration, memory and session handoff notes.
- Raw prompts, responses, stderr, generated candidate runs and fetched paper corpora.
- Captured HTTP response headers, embedded document caches and downloaded web pages.
- Generated dashboards, which can repeat private run content.

This is intentionally a code-and-selected-evidence snapshot, not a full archive of
the experiments described in the findings. Some historical scripts require local
inputs that are not bundled. Do not invent replacements or claim exact historical
reproduction from the example data.

The optional HTTP-header collector now excludes cookies and session identifiers.
Its output differs from the historical corpus, so historical compression scores
must not be attributed to a new capture without rerunning the experiment.

Record tables contain dated references, not live claims of current records.
Original project code and documentation are licensed under Apache 2.0. See
`LICENSE` and `NOTICE`; third-party sources retain their own rights and attribution.
