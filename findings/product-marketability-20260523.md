# Congruity Product Marketability Memo

Date: 2026-05-23

## Short Answer

Yes, there is a marketable product here, but the product is not "an AI scientist
that invents breakthroughs."

The stronger product is:

> Congruity is a verification-first research scout that turns a technical
> problem and a paper corpus into ranked, testable research bets, then shows
> the cheapest way to prove or kill each bet.

People are unlikely to pay much for another tool that generates plausible
hypotheses. Hypotheses are abundant now. The paid wedge is decision support:
helping an R&D team decide which technical ideas deserve engineering time.

## Why This Could Be Valuable

The expensive part of R&D is not writing down ideas. It is choosing which ideas
to chase, setting the right baselines, and discovering early that a plausible
idea does not survive contact with a benchmark.

Congruity already has the beginnings of that loop:

- It composes cross-paper hypotheses from a technical corpus.
- It runs adversarial review before celebration.
- It classifies whether objections are empirically testable.
- It emits benchmark recipes with named baselines, metrics, effect sizes, and
  stop rules.
- It tracks clean positives, partial positives, and clean negatives in a
  customer-readable evidence board.

That last part is important. The negatives are not a weakness. They are part of
the product. A system that can cheaply kill bad ideas is more credible than a
system that only produces impressive-sounding proposals.

## Market Positioning

The product should be positioned as:

> R&D decision support for computational research bets.

or:

> A due-diligence engine for technical hypotheses.

or, more bluntly:

> Before your engineers spend two weeks chasing a research idea, Congruity tells
> you whether it is testable, what would falsify it, and what baseline it must
> beat.

Avoid positioning it as:

- a general AI scientist
- a literature review assistant
- a chat-with-papers product
- a generic hypothesis generator
- a fully autonomous research lab

Those categories are crowded and invite comparisons Congruity does not need.

## Who Would Pay

The best early buyers are teams with expensive technical uncertainty and cheap
or medium-cost computational verification:

- AI/ML infrastructure teams
- search and retrieval teams
- compression teams
- optimization and solver teams
- database and systems teams
- robotics or logistics optimization teams
- venture studios evaluating technical bets
- technical diligence teams
- research labs that already have benchmarks and engineers

These buyers already understand that a single wrong technical bet can cost
weeks or months.

The weakest early buyers are:

- individual academics with low budgets
- people who only want paper summaries
- wet-lab-heavy biomedical teams, unless Congruity becomes deeply specialized
- broad "AI scientist" buyers expecting magic discoveries on demand

## Competitive Context

There are already credible products and systems around AI-assisted research:

- [Google AI Co-Scientist](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist)
  focuses on hypothesis generation and research proposals.
- [Sakana AI Scientist](https://sakana.ai/ai-scientist) explores automated
  scientific discovery workflows.
- [FutureHouse](https://www.futurehouse.org/research-announcements/launching-futurehouse-platform-ai-agents)
  offers scientific discovery agents.
- [Elicit](https://elicit.com) is well known for literature review and evidence
  extraction workflows.
- [Iris.ai](https://iris.ai/pricing) and [PatSnap](https://www.patsnap.com/pricing)
  show that enterprises pay for R&D and innovation intelligence workflows.
- [BenchSci ASCEND](https://www.benchsci.com/ascend) shows that verticalized
  scientific decision-support platforms can command serious enterprise budgets.

The opportunity is not to out-hypothesize these systems. The opportunity is to
own the step after ideation:

1. Is this claim testable?
2. What is the cheapest falsification path?
3. What baseline must it beat?
4. What did the benchmark actually show?
5. Does the track record justify more investment?

That is a narrower and more defensible wedge.

## Product Shape

The first sellable product should probably be a productized service, not pure
self-serve SaaS.

### Offer

**Research Bet Audit**

Input:

- a technical problem statement
- target baselines or success metrics
- a paper corpus, patent corpus, internal notes, or curated reading list

Output:

- an evidence dashboard
- 5-10 ranked research bets
- adversarial objections for each bet
- L3/L4 testability classification
- benchmark recipes with baselines, datasets, metrics, effect sizes, and stop
  rules
- optionally, 1-2 executable prototype harnesses

### Why Service First

The product needs trust before it needs scale. A service-led pilot lets the
team learn:

- which domains produce useful testable candidates
- which buyers understand the value
- how much setup is needed per customer
- how often a benchmark can be built cheaply
- whether customers act on the evidence

Once repeated patterns emerge, the repeatable parts can become SaaS.

## Pricing Hypotheses

These are not final prices; they are market-test anchors.

### Lightweight Pilot: 5k-10k

Deliver:

- one domain run
- customer-facing dashboard
- ranked candidate list
- adversarial review
- test recipes

No custom benchmark harness.

### Verification Pilot: 15k-30k

Deliver:

- everything in the lightweight pilot
- one working benchmark harness
- findings memo with positive, negative, or partial verdict

This is likely the best initial offer.

### Team Subscription: 50k-150k/year

For teams that want:

- recurring runs
- private corpora
- persistent evidence board
- benchmark harness archive
- integrations with internal repos or document stores
- support for repeated research triage

This tier should come later, after paid pilots prove repeatability.

## Core ROI Story

The strongest ROI claim is not:

> Congruity will discover breakthroughs.

It is:

> Congruity can save weeks of engineering time by killing weak research bets
> early and focusing effort on the few ideas with concrete, benchmarkable upside.

A believable sales example:

> Your team was about to spend two engineers for two weeks exploring a plausible
> optimization idea. Congruity identifies the cheapest falsification test,
> runs or specifies the benchmark, and shows in hours whether the idea is dead,
> partial, or worth deeper investment.

If a single avoided dead-end saves 40-80 engineering hours, even a 15k pilot can
be rational for the right team.

## Evidence Congruity Already Has

The current project has a useful early track record:

- A clean positive in the Heilbronn geometry run.
- A partial positive in short-string compression.
- Multiple clean negatives in cache replacement, multitrack audio, embedding
  quantization, and TSP/Delaunay candidate filtering.
- A growing set of recipe-ready candidates where each one has named baselines,
  metrics, and stop rules.

The most important signal is not the positive rate alone. It is that Congruity
can produce a falsifiable queue, then turn some candidates into actual
verification artifacts.

## Risks

### Trust Risk

Customers may not trust generated hypotheses or generated benchmark recipes.

Mitigation:

- expose provenance
- show adversarial review
- keep raw artifacts
- make every conclusion reproducible
- show clean negatives, not only wins

### Scope Risk

The system may work only in domains where cheap computational verification is
available.

Mitigation:

- embrace that constraint
- start with algorithms, systems, search, compression, solvers, and ML
  infrastructure
- avoid wet-lab or ambiguous qualitative domains early

### Novelty Risk

Some generated ideas are merely recombinations of known techniques.

Mitigation:

- sell prioritization, not guaranteed novelty
- include novelty checks as a workflow component
- evaluate whether a composition beats baselines, not whether it sounds clever

### Integration Risk

Real customer value may depend on private corpora, internal baselines, and local
engineering constraints.

Mitigation:

- service-led pilots
- clear input template
- customer-provided benchmark constraints
- private deployment path later

## Validation Plan

The next step is not more product polish. It is customer discovery.

Run 10-15 interviews with:

- ML infrastructure leads
- optimization engineers
- search/retrieval leads
- applied research managers
- technical diligence investors
- CTOs at research-heavy startups

Ask:

1. What research ideas did your team explore in the last quarter?
2. Which ones were dead ends?
3. How long did you spend before knowing they were dead?
4. What baselines decided the question?
5. Would a ranked testability dashboard have changed what you did?
6. Would you pay for a one-week research bet audit?
7. What would make you trust or distrust the result?
8. Who owns the budget for this?

The buying signal to look for:

> "Can you run this on our problem?"

Not:

> "That's interesting."

## Recommended Next Product Work

1. Add a "customer mode" dashboard export that hides internal IDs and leads with:
   problem, claim, baseline, cheapest test, verdict, next action.
2. Add a `verification_queue.md` or UI queue that lists the top untested
   candidates by expected value and implementation cost.
3. Build one more benchmark harness from the newest route-planning run, ideally
   C2 or C4, to show repeatability beyond the original examples.
4. Package a sample customer report as a PDF or polished HTML export.
5. Write a one-page landing pitch around "Research Bet Audit."

## Bottom Line

Someone would pay for Congruity if it is sold as R&D decision support with
executable evidence.

Someone probably would not pay much if it is sold as a generic AI idea
generator.

The marketable wedge is:

> Generate less noise. Kill bad bets early. Put engineering time behind the few
> research ideas that survive adversarial review and have a cheap path to
> verification.

