# Reproduction judge rubric

## Purpose

Use this rubric to estimate whether a Trackio logbook demonstrates independent reproduction. It is a conservative preflight heuristic, not the private challenge judge.

## Claim evidence levels

| Score | Class | Required evidence | Common non-examples |
|---:|---|---|---|
| 0 | Source-confirmed / blocked | The logbook establishes what the paper or release says but produces no independent measurement of the claim. | PDF transcription, table lookup, paper-value plot, source audit, unreleased-code notice. |
| 1 | Partial / proxy | Executable new evidence tests part of the mechanism or direction, but materially changes scale, data, model, task, metric, or claim scope. | Synthetic corpus, toy network, hash embedding, formula unit test without downstream training. |
| 2 | Independently reproduced | A new run directly tests the required claim using an appropriate implementation and comparable protocol, with raw measurements and reproducible provenance. | Re-evaluation of released checkpoints, faithful retraining subset with declared scale, matched baseline experiment. |

For a compound claim, require every essential component for score 2. Example: “method A improves accuracy without increasing cost” needs both accuracy and cost evidence.

## Evidence packet expected per claim

- Exact required claim, copied verbatim.
- Verdict: `REPRODUCED`, `PARTIAL/PROXY`, `INCONCLUSIVE/BLOCKED`, or `FALSIFIED`.
- What was executed and where.
- Model/data/task/seed/sample scale relative to the paper.
- Newly measured metric with uncertainty or repeated runs where meaningful.
- Exact command/configuration and successful exit.
- Raw machine-readable output and figure data.
- Job URL for substantive feasible GPU work.
- Artifact or repository location sufficient to rerun.
- Limitations and any version boundary.

## Scoring and risk

Let `E = sum(claim scores)` and `M = 2 × number of required claims`.

- **High scientific risk:** `E/M < 0.60`, or fewer than half the claims score 2.
- **Borderline scientific risk:** `0.60 ≤ E/M < 0.75`, especially when evidence is proxy-heavy.
- **Lower scientific risk:** `E/M ≥ 0.75` and most claims score 2.

These bands estimate judge risk only. A private judge may use a different threshold or weight claims unequally.

## Mechanical contract

Treat the following as submission blockers when required by the current challenge guide:

1. Missing discovery tags or paper metadata.
2. Missing claim pages or exact claim wording.
3. No Bucket-backed reproduction bundle.
4. No pinned Executive summary or wrong pin order.
5. Missing literal Scope & cost schema.
6. Missing pinned Posterly figure.
7. Local/unpromoted dashboard presented as published evidence.
8. Artifact cells still using local paths or `trackio-artifact://`.

Treat these as quality warnings unless the live guide says otherwise:

- failed exploratory runs retained without a resolution note;
- duplicate cells or artifacts;
- paper summaries dominating claim pages;
- verdict language such as “verified” without a provenance qualifier;
- a poster that emphasizes process or blockers more than reproduced measurements.

## False-positive controls

- Do not award score 2 because a result numerically matches the paper when the number came from the paper.
- Do not award score 2 because official code ran if the evaluated claim was not measured.
- Do not award score 2 for a synthetic proxy that the logbook itself says is not evidence for the paper setup.
- Do not fail a scientifically strong logbook solely for honest negative or falsifying results. Independent falsification is reproduction evidence.
- Do not hide blocked claims. Score them zero and preserve the release limitation.

## Recommended report table

| Claim | Verdict | Provenance | Scale match | New measurement | Score |
|---|---|---|---|---|---:|

After the table, report the estimated total as `E/M` and explain the two largest score constraints.
