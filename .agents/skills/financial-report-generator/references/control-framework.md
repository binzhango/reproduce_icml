# Financial report control framework

## Contents

1. Hard gates
2. Warnings and review points
3. Provenance
4. External distribution

## 1. Hard gates

The release build must stop for:

- missing or malformed required fields;
- non-decimal, non-finite, or boolean financial values;
- balance-sheet mismatch beyond tolerance;
- cash-flow roll-forward mismatch beyond tolerance;
- cash-flow closing cash mismatch with balance-sheet cash;
- missing source inventory;
- unresolved commentary source references;
- unresolved template placeholders or remote resources;
- absent required report sections.

Tolerance is expressed in the report's presentation scale. A tolerance of `1` in a report presented in thousands means one thousand currency units. Choose it deliberately; never enlarge it simply to force a pass.

## 2. Warnings and review points

Warnings do not automatically block an internal draft, but must be resolved or disclosed:

- negative revenue or expense magnitudes under the canonical sign convention;
- negative equity;
- numeric commentary without source references;
- source dates later than the reporting-period end;
- unusual margin or cash movements;
- missing source hashes;
- HTML-only output without verified PDF rendering.

The reviewer must confirm that management explanations are supplied assertions, not model-inferred causes.

## 3. Provenance

`DATA_MANIFEST.json` records the input and template SHA-256 hashes, generation timestamp, reporting period, sources, and formulas. Preserve it beside the report. If source files are available, hash the actual snapshots and store the hashes in `sources`; do not claim a hash was verified when it was merely supplied as text.

## 4. External distribution

Before external distribution, add organization-specific controls for chart of accounts, consolidation, intercompany eliminations, foreign exchange, revenue recognition, materiality, access control, approval signatures, retention, and jurisdiction-specific disclosure. Obtain qualified accounting and legal review. This skill does not create an audit trail by itself and does not issue assurance.

