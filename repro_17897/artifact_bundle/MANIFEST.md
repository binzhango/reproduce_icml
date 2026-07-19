# ML-Agent reproduction artifact manifest

This directory is the self-contained download bundle attached to the Trackio
logbook for ICML 2026 paper #17897 (OpenReview `kcPPWaoegr`, arXiv
`2505.23723`). It intentionally excludes credentials, Trackio's local database,
Git metadata, Python caches, and the failed Cloudflare/OpenReview HTML responses.

## Contents

- `README.md`: scope, limitations, and local smoke-test commands.
- `configs/experiment_scope.json`: declared experiment scope and seed.
- `scripts/`: source audit, mechanism proxy, and figure-rendering code.
- `outputs/`: claim matrix, local and T4 results, and generated figures.
- `source/`: archived arXiv v1 and v2 PDFs plus extracted text.
- `official_release/`: the official README, public shell scripts, requirements,
  and README figures at commit `15932e7525deb99d59f7416bbe8c75077cff3690`.
- `poster/`: all-green Posterly HTML, gate report, interactive embed, preview PNG,
  and verified one-page 60 × 36 inch PDF.

## Reproduction tiers

Claims 1–3 are source-verified but not independently rerun. Claim 4 is a
disclosed mechanism proxy using synthetic ideas and MiniLM embeddings. Claim 5
is an exact executable reproduction of the arXiv v1 reward equation, with the
revised v2 equation tested side by side.

The completed GPU Job is
`https://huggingface.co/jobs/binzhango/6a5830b6b1669a49bf0767d3`.
