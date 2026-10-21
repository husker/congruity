# Congruity

**An AI research workflow that turns connections between papers into experiments you can test.**

Congruity searches across research literature, asks a coding agent to propose
combinations of existing methods, challenges those proposals, and tests selected
ideas against executable benchmarks. A plausible explanation is a starting point;
the useful output is a measured result, including a result that disproves the idea.

I built Congruity to explore a practical question: how much of the path from
reading papers to testing an idea can an agent carry out, while keeping the
evidence inspectable?

This is a **local research prototype**, built in Python. It includes a CLI,
agent prompts and tool integration, resumable workflows, benchmark contracts,
automated candidate search, and a static evidence dashboard.

## How it works

```text
Research question
       ↓
OpenAlex corpus → paper scoring → multi-paper proposals
                                         ↓
                   consistency · source grounding · adversarial review · testability
                                         ↓
                           ranked experiment plans
                                         ↓
                candidate code → fixed benchmark → recorded outcome
                                         ↓
                             evidence dashboard
```

- **Research intake:** gathers papers, checks corpus coverage and scores relevance
  to a specific problem.
- **Agent orchestration:** uses the locally authenticated Codex CLI, with structured
  outputs, retries, saved stage artifacts and resume support. Optional API-based
  relevance backends are available.
- **Evaluation:** domain packs specify candidate interfaces, datasets, baselines,
  metrics and validity checks. Candidate generation does not get to invent its
  own measuring stick.
- **Bounded search:** `autosearch` iterates on the training split and reserves a
  separate holdout confirmation. Timeouts stop runaway benchmark processes.
- **Evidence:** findings distinguish model critiques, benchmark outcomes and
  record claims. The dashboard presents saved runs without needing a server.

The selection policy is **prioritize, don't veto**. Model reviews provide questions
to test; their verdicts do not automatically discard a candidate. This policy came
from the project's own calibration work, where some rejected proposals performed
well in subsequent experiments. It is a project-specific observation, not a general
claim about model reliability.

## Try the dashboard without an AI account

Prerequisites: Python **3.13** (the version used for validation and CI),
[uv](https://docs.astral.sh/uv/), and Git. No model credentials are needed for
the example or unit tests. Installing dependencies requires internet access.

Clone the public repository, then install its locked dependencies:

```bash
git clone https://github.com/husker/congruity.git
cd congruity
uv sync --locked
uv run python -m pipeline.web_ui \
  --root examples/demo/compositions \
  --index examples/demo/INDEX.md \
  --out .local/demo-site
```

Open `.local/demo-site/index.html` in your browser. The included example is an
explicitly labelled, synthetic fixture showing the saved-run format; it is not a
research result. For recorded experiments, start with the findings linked below.

```bash
uv run pytest
uv run ruff check .
uv run python -m pipeline pack doctor --no-smoke
```

Some numerical tests compile a small C extension and can take several minutes;
a C compiler is needed for those tests. Optional experiments have additional
dependencies documented in their modules.

## Run a research workflow

Use an authenticated `codex` CLI for composition. Copy `.env.example` to `.env`
and configure OpenAlex access for your account; optional relevance providers use
the API keys listed there. Keep credentials local.

```bash
cp .env.example .env

uv run python -m pipeline.topics harvest \
  --queries "graph coloring,tabu search,spectral graph partitioning" \
  --papers-per-query 100 \
  --year-min 2010 \
  --save data/pipeline/topic_lists/coloring.json

uv run python -m pipeline compose \
  --topic-list data/pipeline/topic_lists/coloring.json \
  --problem "Can spectral partitioning improve graph-coloring local search? Compare against a fixed tabu-search baseline under the same evaluation budget." \
  --max-papers 200 \
  --year-min 2010 \
  --target 6 \
  --relevance-mode auto \
  --with-recipes
```

This contacts external services and uses your model account. Inspect the run's
digest and recipe before starting an executable experiment. Costs depend on corpus
size, model settings and search budget; usage artifacts are estimates, not a bill.

Once you have a run ID and candidate ID, supported packs can run one experiment
or a bounded search:

```bash
uv run python -m pipeline autoverify --run <run_id> --cid <candidate_id> --pack graph_coloring
uv run python -m pipeline autosearch --run <run_id> --cid <candidate_id> --pack graph_coloring \
  --population 4 --generations 3 --elites 2
```

**Generated benchmark code executes locally.** Timeouts and prompt instructions
are not a security sandbox. Use a disposable VM or container with restricted
access, and review generated code before execution. See [data and execution
boundaries](docs/DATA_AND_EXECUTION.md) and the [runbook](docs/RUNBOOK.md).

## Results and limits

The project explores compression, caching, graph colouring, geometric optimization
and combinatorial search. The useful evidence includes both successful experiments
and falsifications:

- [Verification index](findings/INDEX.md): dated experiment write-ups and outcomes.
- [Short-string experiment](findings/short-string-c2-rejected-verification-20260529.md):
  a candidate revisited after a negative model review.
- [Graph-colouring evaluation](findings/graph-coloring-domain-20260611.md): early
  autonomous benchmark outcomes and the importance of valid baseline comparisons.
- [Outcome calibration](findings/outcome-calibration-2026-05-29.md): comparison
  of model signals with recorded experimental outcomes.

These are historical development reports. A win against a configured baseline
does not establish a new state-of-the-art result. A model-assisted literature search
does not establish novelty. Published record tables are dated snapshots that need
independent verification before any record claim.

The public snapshot includes code, selected numerical fixtures and findings. Raw
agent sessions, captured HTTP headers, downloaded paper text and personal working
notes are excluded. It is not a complete archive for reproducing every historical
result; see [snapshot contents](docs/PUBLIC_SNAPSHOT.md).

## Code map

| Area | What to look at |
|---|---|
| [`pipeline/compose.py`](pipeline/compose.py) | Stage orchestration and resume behaviour |
| [`pipeline/llm_io.py`](pipeline/llm_io.py) | Codex calls, retries and per-stage artifacts |
| [`pipeline/autoverify.py`](pipeline/autoverify.py) | Candidate generation and benchmark grading |
| [`pipeline/autosearch.py`](pipeline/autosearch.py) | Bounded search and holdout confirmation |
| [`computable_packs/`](computable_packs/) | Domain interfaces, metrics and baseline contracts |
| [`experiments/`](experiments/) | Domain implementations and verification harnesses |
| [`pipeline/web_ui.py`](pipeline/web_ui.py) | Static evidence dashboard |
| [`tests/`](tests/) | Pipeline, evaluation and regression tests |

For implementation details, see [the architecture](OVERVIEW.md),
[pipeline documentation](pipeline/README.md) and [agent instructions](AGENTS.md).

## License

Original project code and documentation are licensed under the
[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution and the
treatment of third-party material.

Built by [James Gagan](https://husker.github.io/).
