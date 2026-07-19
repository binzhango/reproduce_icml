---
name: reproduction-logbook-judge
description: Audit a local or published Trackio reproduction logbook before submission. Use when Codex must estimate why a reproduction may fail a judge, distinguish paper-derived source confirmation from independent experiments, check ICML reproduction challenge metadata/pins/artifacts/poster requirements, compare claim evidence, or produce a prioritized preflight report for a `.trackio/logbook`.
---

# Reproduction Logbook Judge

Act as a skeptical scientific reproduction reviewer. Separate scientific evidence quality from mechanical logbook compliance. Never turn source confirmation into reproduction credit.

## Workflow

1. Locate the target logbook. Prefer the dedicated reproduction directory over an unrelated root `.trackio` directory.
2. Read the agent view first:

   ```bash
   trackio logbook read <workspace-or-space-id>
   ```

   For a public Space with a broken cached token, retry with `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`. For local work, avoid network access.
3. Run the deterministic preflight:

   ```bash
   python <skill-dir>/scripts/audit_logbook.py <workspace>
   python <skill-dir>/scripts/audit_logbook.py <workspace> --strict --json
   ```

   Treat its findings as contract and hygiene checks, not a scientific verdict.
4. Read `references/judge-rubric.md` completely before assigning evidence scores.
5. Inspect every claim page and any raw figure/code/artifact needed to establish provenance. Do not infer independent evidence from a figure alone.
6. Produce the report described below. Stay read-only unless the user explicitly asks for repairs.

## Evidence rules

- Give **0 — source-confirmed** when a result is copied, transcribed, digitized, recalculated from paper values, or merely located in official text/code.
- Give **1 — partial/proxy** for an executable mechanism check, toy setup, synthetic data, reduced substitute, or materially different model/task.
- Give **2 — independently reproduced** only when an experiment directly tests the claim with appropriate data/model/method, reports newly measured metrics, and exposes reproducible code plus raw outputs.
- A successful code cell proves execution, not claim validity.
- An official-code smoke test proves operability, not the paper result.
- A GPU Job, dashboard, plot, or artifact improves provenance but earns no claim credit without a direct result.
- Unreleased assets are a legitimate blocker, not reproduction evidence.
- Score compound claims on their weakest essential component. State any narrower subclaim that does reproduce.

Use an evidence denominator of `2 × number of required claims`. Label the score an estimate, never an official judge score. Treat a score below 60% or fewer than half of claims at level 2 as high risk.

## Contract checks

Verify at minimum:

- descriptive `Repro - ...` title;
- paper metadata and both discovery tags;
- one page per required claim with the exact claim text and an explicit verdict;
- successful executable runs, raw figure data, Job links, and honest scale labels where applicable;
- Bucket-backed reproduction-bundle artifact and a download explanation;
- Hub collection link;
- Conclusion page with the first pinned cell titled `Executive summary`;
- literal `## Scope & cost` table with `This reproduction` and `Full replication` columns and `Scope`, `Hardware`, `Compute time`, `Cost`, and `Outcome` rows;
- second pinned cell titled `Reproduction poster`, stored as a figure;
- no judge-facing local dashboards, unresolved artifact URIs, or prominent failed runs without a later resolution.

If current requirements matter and network access is authorized, refresh the challenge guide before judging. Report any difference between the bundled rubric and the live guide.

## Report format

Lead with one of: `Likely pass`, `Borderline`, or `Likely fail`, followed by confidence and a one-sentence reason.

Then include:

1. **Claim scorecard** — exact claim, evidence class, measured evidence, missing evidence, score.
2. **Contract audit** — failures first, then warnings.
3. **What will not fix the score** — cosmetic edits, relabeling, or additional paper summaries when experimental evidence is missing.
4. **Minimum path to pass** — the smallest scientific experiments first, then structural cleanup.

Quote logbook wording sparingly and cite cell ids or local page paths so every finding is traceable. Distinguish observed facts from judge-risk inferences.
