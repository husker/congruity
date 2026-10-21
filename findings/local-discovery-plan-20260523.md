# Plan: Turn Congruity Into A Local Discovery Engine

Date: 2026-05-23

## Goal

Make Congruity capable of producing real, computer-testable discoveries on a
local machine.

The target loop is:

> compose ideas -> adversarially review them -> select locally testable
> candidates -> build the cheapest benchmark -> run it -> record the result ->
> feed the result back into the next run.

The project should not optimize for impressive ideas. It should optimize for
ideas that can be settled by code on this computer.

## Definition Of A Discovery

A "discovery" in this project means one of the following:

1. A generated method beats a named baseline on a reproducible benchmark.
2. A generated method partially beats a baseline and reveals which ingredient
   mattered.
3. A generated method fails, but cheaply falsifies an assumption that looked
   plausible.
4. A benchmark reveals that a cross-domain analogy does not transfer.
5. An ablation identifies a smaller, more useful mechanism than the original
   generated claim.

This definition is deliberately empirical. A claim is not a discovery until a
local benchmark says what happened.

## Constraints

Every candidate promoted into the discovery queue must pass a local-testability
screen:

- Runs on this computer.
- CPU-only preferred; GPU optional but not required.
- No paid APIs for the verification step.
- No wet-lab, human-subject, or subjective evaluation dependency.
- Dataset is already local, easy to download, or small enough to cache.
- Benchmark runtime target: smoke test under 10 minutes, full pilot under 4
  hours.
- Baselines are named before implementation.
- Stop rule is named before implementation.
- Final metric is numeric and falsifiable.

## Product Principle

The product value is not "Congruity generated an idea."

The product value is:

> Congruity found a testable research bet, named the baseline it must beat,
> and produced the cheapest experiment that can prove or kill it.

## Phase 1: Add A Local Verification Queue

### Objective

Create a first-class queue of candidates that are ready to test locally.

### Implementation

Add a module:

```text
pipeline/verification_queue.py
```

It should scan:

```text
data/pipeline/compositions/*/state.json
findings/INDEX.md
```

and emit:

```text
findings/verification-queue.json
findings/verification-queue.md
```

Each queue item should include:

- `run_id`
- `composition_id`
- plain-English title
- original synthesis claim
- L3 verdict
- L4 testability score
- reconciled status
- whether already verified
- candidate domain
- named baselines
- benchmark dataset
- primary metric
- minimum effect size
- stop rule
- expected runtime
- likely failure mode
- local-testability score
- implementation-effort score
- expected-value score
- priority score
- recommended next action

### Scoring

Use a simple transparent scoring model first.

```text
priority =
  3 * local_testability
  + 2 * baseline_clarity
  + 2 * expected_value
  + 1 * cheap_negative_value
  - 2 * implementation_effort
  - 2 * external_dependency_risk
```

Scores should be visible in the markdown output. Do not hide the judgment.

### Local-Testability Rubric

Score 0-3:

- `3`: can be smoke-tested locally with existing code or a small harness.
- `2`: needs a new harness but uses standard Python packages and small data.
- `1`: possible locally, but needs heavy solver setup, large data, or GPU.
- `0`: not suitable for this computer.

### Acceptance Criteria

- Queue includes only unverified `actionable` or `sound` candidates by default.
- Already verified candidates are shown separately or marked done.
- The latest run `20260523-003811` appears with C2, C4, and C10 near the top.
- The queue is regenerated with one command.
- Tests cover queue parsing, scoring, and verified-item exclusion.

### Command

```bash
uv run python -m pipeline.verification_queue
```

## Phase 2: Add Queue To The Web UI

### Objective

Make the dashboard guide the user toward the next local experiment.

### Implementation

Extend:

```text
pipeline/web_ui.py
```

Add a dashboard section:

```text
Next Local Tests
```

Each item should show:

- plain-English candidate title
- why it is worth testing
- cheapest test
- baseline
- estimated runtime
- stop rule
- link to originating run page

Add a detail page or section for each queue item:

```text
findings/web/queue/<run_id>-<composition_id>.html
```

### UX Rule

A layperson should understand:

- what the idea is
- what would count as success
- what would count as failure
- why testing it matters

An advanced user should be able to drill into:

- raw claim
- L3 objections
- L4 classification
- recipe
- planned benchmark details

### Acceptance Criteria

- `findings/web/index.html` shows a top-ranked "Next Local Tests" section.
- `20260523-003811 C2`, `C4`, and `C10` are understandable without jargon.
- Existing run pages link to queue items when a candidate is unverified and
  locally testable.
- Browser check passes on desktop and mobile.

## Phase 3: Standardize Experiment Harnesses

### Objective

Make it easy to convert a queue item into code.

### Directory Template

Every new experiment should follow:

```text
experiments/<domain>_<short_claim>/
  __init__.py
  bench.py
  baselines.py
  method.py
  evaluator.py
  README.md
```

Optional:

```text
  data.py
  report.py
  fixtures/
```

### Required `bench.py` Behavior

Every benchmark runner should support:

```bash
uv run python -m experiments.<name>.bench --smoke
uv run python -m experiments.<name>.bench
```

The smoke test should run in under 10 minutes.

The full pilot should target under 4 hours.

### Required Output

Every benchmark should print:

- run configuration
- baseline scores
- candidate scores
- primary metric
- effect size
- whether the stop rule fired
- verdict: clean-positive, partial-positive, clean-negative, partial-negative

If practical, it should also write:

```text
findings/<experiment>-verification-<date>.md
```

### Shared Utilities

Add:

```text
experiments/common/
  __init__.py
  reporting.py
  stats.py
  timing.py
  seeds.py
```

Keep this minimal. Do not overbuild before there are 2-3 new harnesses.

### Acceptance Criteria

- New experiment template is documented.
- Existing experiments are not refactored unless needed.
- At least one new experiment uses the template.

## Phase 4: Implement First New Local Discovery Sprint

### Candidate

Start with:

```text
run: 20260523-003811
candidate: C2
plain title: Hard-route solver portfolio test
```

C2 asks whether spectral or graph-complexity features improve solver selection
for TSP-like problems.

### Why C2 First

- It does not require building a new TSP solver.
- It can be tested with algorithm-selection data.
- The output is numeric.
- It is likely to produce a clear positive or clean negative.
- It tests an actual discovery claim: do imported spectral features add
  predictive value beyond normal graph features?

### Benchmark Plan

Create:

```text
experiments/tsp_portfolio/
```

Implement:

- dataset loader for ASlib TSP-LION2015, or a small local fixture if ASlib
  download is unavailable
- feature extraction:
  - cheap graph features
  - Laplacian spectral features
  - optional Krylov-style approximations if cheap
- model:
  - random forest or gradient boosting selector
  - cross-validation
- baselines:
  - single-best solver
  - graph-feature selector without spectral features
  - spectral-feature selector

### Metrics

Primary:

```text
PAR10 runtime reduction vs best non-spectral selector
```

Secondary:

- oracle gap closed
- solver-selection accuracy
- feature extraction time

### Success Criteria

Clean positive:

- spectral features improve PAR10 by at least 5 percent over the best
  non-spectral selector
- feature extraction time is not more than 25 percent of median solver runtime

Partial positive:

- spectral features improve one subset or close part of the oracle gap, but not
  the full pre-registered threshold

Clean negative:

- spectral features improve PAR10 by less than 2 percent, or feature extraction
  overhead dominates

### Smoke Test

If ASlib is unavailable, build a local synthetic smoke test:

- generate graph/TSP instances
- run 2-3 simple solvers or heuristics
- extract graph and spectral features
- test whether spectral features predict which heuristic wins

This is weaker than ASlib but good enough to validate the harness.

### Acceptance Criteria

- `uv run python -m experiments.tsp_portfolio.bench --smoke` runs locally.
- Full benchmark writes a finding doc.
- `findings/INDEX.md` is updated only after the result is real.
- Web dashboard refresh shows the verified result.

## Phase 5: Implement Second Sprint From Existing TSP Code

### Candidate

```text
run: 20260523-003811
candidate: C4
plain title: Candidate-edge pruning test
```

C4 asks whether a proposal/filter/solve pipeline can shrink the TSP search
space without deleting edges needed for the optimum.

### Current Status

There is already a harness:

```text
experiments/tsp_knn_recall/bench.py
```

It tests a cheap premise:

> Do K-nearest-neighbor candidate edges preserve optimal-tour edges on small
> exact Euclidean TSP instances?

Recent local pilot:

- K=3: mean recall 93.28 percent, many failures
- K=5: mean recall 99.06 percent, but 7 of 50 below 99 percent
- K=8: mean recall 99.89 percent, 1 of 50 below 99 percent
- K=10: 100 percent recall on all 50 instances

C4 is not falsified at this recall gate. The next test is whether pruning helps
solve time or solution quality.

### Next Benchmark

Create or extend:

```text
experiments/tsp_candidate_pruning/
```

Compare:

- full exact Held-Karp on small n
- full heuristic local search
- KNN-pruned candidate search
- random-pruned candidate search at same edge budget
- Delaunay-pruned candidate search if SciPy is available

### Metrics

- optimal-edge recall
- solved-to-optimality rate
- runtime
- candidate edge count
- optimality gap if exact optimum is unavailable

### Success Criteria

Clean positive:

- KNN or learned-style pruning preserves optimality and reduces runtime by at
  least 25 percent over full search on the pilot set.

Clean negative:

- pruning either misses required edges or does not reduce runtime after overhead.

### Acceptance Criteria

- Uses existing exact small-n evaluator.
- Has a smoke command.
- Produces a finding doc.

## Phase 6: Add Feedback Into Compose

### Objective

Make future compose runs prefer locally testable ideas.

### Implementation

Add a "local verification preference" block to the compose problem prompt or a
post-processing filter:

```text
Prefer candidates that:
- can be tested in Python on one workstation
- have named baselines
- use small public datasets or synthetic generators
- have a smoke test under 10 minutes
- have a full pilot under 4 hours
- can fail cleanly
```

Add a local-testability classifier:

```text
pipeline/local_testability.py
```

Fields:

- local_runtime_estimate
- package_risk
- data_risk
- baseline_clarity
- implementation_complexity
- recommended_smoke_test
- recommended_full_test

### Acceptance Criteria

- New compose digests identify "best local tests" directly.
- Dashboard queue quality improves.
- Fewer attractive-but-impractical candidates reach the top.

## Phase 7: Operational Loop

### Weekly Discovery Routine

1. Pick one narrow domain.
2. Harvest broadly, but phrase the problem around executable verification.
3. Run compose with recipes.
4. Generate the verification queue.
5. Pick the top candidate with a smoke test under 10 minutes.
6. Build the harness.
7. Run smoke.
8. If smoke is promising, run full pilot.
9. Write finding doc.
10. Update `findings/INDEX.md`.
11. Refresh web UI.
12. Capture what was learned in memory or a findings note.

### Stop Discipline

Abandon a candidate when its stop rule fires. Do not keep investing because the
idea is aesthetically interesting.

The point is not to rescue ideas. The point is to discover which ideas deserve
more work.

## Initial Backlog

### Highest Priority

1. `pipeline/verification_queue.py`
2. web UI "Next Local Tests" section
3. `experiments/tsp_portfolio/` for run `20260523-003811` C2

### Next Priority

4. `experiments/tsp_candidate_pruning/` for run `20260523-003811` C4
5. standardized experiment template
6. local-testability classifier

### Later

7. C10 densest-subgraph + SAT preprocessing
8. Heilbronn C2 metadynamics basin-escape
9. automated finding-doc generation from benchmark output
10. customer/report export mode for the web UI

## What Success Looks Like

In the next few sessions, success is not a polished product.

Success is:

- a ranked local verification queue
- one new benchmark harness from a previously untested candidate
- one new finding doc with a clean verdict
- the dashboard showing the result

In the longer run, success is a project where opening the dashboard answers:

1. What should I test next?
2. Why is it worth testing?
3. What command do I run?
4. What would count as success?
5. What did the last experiments prove or kill?

That is the shape of a local discovery engine.

