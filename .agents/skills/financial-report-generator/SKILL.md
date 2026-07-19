---
name: financial-report-generator
description: Create validated monthly management financial reports from structured JSON, including an executive summary, income statement, balance sheet, cash-flow statement, KPIs, commentary, provenance, HTML, and print-ready PDF. Use when Codex must turn monthly finance data into a repeatable management-report package, refresh an existing report with a new period, or diagnose financial reconciliation and report-layout failures. Do not use for audited opinions, tax filings, regulatory submissions, or investment advice without qualified human review.
---

# Financial Report Generator

Build an original, data-backed monthly management report. Keep calculations deterministic and use the language model only for organization, explanation, and explicitly sourced commentary.

## Required workflow

1. Inspect the supplied data and identify the reporting entity, current period, comparison period, currency, scale, sources, and sign convention.
2. Read [references/data-contract.md](references/data-contract.md) before changing the input schema or mapping spreadsheet columns.
3. Normalize the source into the canonical JSON contract. Never silently replace missing values with zero.
4. Run `scripts/build_report.py`. Treat every hard-gate failure as a release blocker.
5. Inspect `GATE_REPORT.json`, `DATA_MANIFEST.json`, and `report.html`.
6. Run `scripts/render_report.py` to produce the PDF. If Playwright or Chromium is unavailable, report that dependency clearly; do not claim the HTML is a verified PDF.
7. Run `scripts/verify_output.py` on the HTML and PDF.
8. Visually inspect every rendered page. Check table splits, repeated headings, clipped text, chart labels, negative-number formatting, and source notes.
9. Deliver the HTML, PDF, gate report, manifest, and any unresolved warnings together.

## Quick start

```bash
python scripts/build_report.py assets/example-input.json --out-dir report-output
python scripts/render_report.py report-output/report.html --output report-output/report.pdf
python scripts/verify_output.py report-output/report.html --pdf report-output/report.pdf --output report-output/VERIFY_REPORT.json
```

Resolve paths relative to this skill directory when invoking the bundled scripts from elsewhere.

## Control rules

- Preserve source values separately from derived values.
- Use `Decimal`; never use binary floating-point for financial calculations.
- Require assets to equal liabilities plus equity within the declared tolerance.
- Require opening cash plus operating, investing, and financing cash flow to equal closing cash.
- Require cash-flow closing cash to agree with balance-sheet cash.
- Require a source inventory and stable source identifiers.
- Require commentary `source_refs` to resolve to canonical fields. Warn on numeric commentary without source references.
- Display the reporting period, comparison period, currency, scale, generation time, and validation status.
- Keep the report self-contained: no remote scripts, fonts, images, or chart services.
- Fail closed. Do not render a release report when a hard data gate fails.

Read [references/control-framework.md](references/control-framework.md) when interpreting failures, adding controls, or preparing an externally distributed report.

## Input handling

- JSON already matching the contract: build directly.
- CSV or spreadsheet: map it into the contract first; retain the original file and record its hash in `sources` when possible.
- Database or API: export a dated snapshot before building so the report is reproducible.
- Missing prior period: stop and ask whether to produce a current-only variant; do not invent comparison values.
- Multiple entities or currencies: consolidate upstream with documented eliminations and FX assumptions. The MVP contract represents one reporting entity and one presentation currency.

## Commentary policy

Prefer concise observations generated from validated values: direction, magnitude, margin movement, liquidity, and cash conversion. Distinguish:

- `data observation`: directly supported by report fields;
- `management explanation`: supplied by management and attributed as such;
- `forecast or judgment`: clearly labeled and never presented as historical fact.

Do not infer causality from two changing values. Do not add market, tax, audit, or legal conclusions unless the user supplies an authoritative source and requests them.

## Design policy

Use the bundled original template as a starting point. It is a restrained multi-page management document, not a poster. Modify its design tokens and components rather than adding one-off inline styles. Preserve print CSS, semantic tables, repeating table headers, page-break controls, accessibility labels, and the provenance appendix.

## Release boundary

This skill produces management information, not an audit opinion. Reports intended for lenders, investors, regulators, tax authorities, or public filing require review by the appropriate accounting, legal, and compliance professionals. Never label a report “audited” or “reviewed” unless that status is supplied and verified by the user.

