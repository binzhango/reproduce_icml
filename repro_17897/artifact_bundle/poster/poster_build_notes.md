# Poster build notes

## Design decisions

- Template: Posterly `landscape_4col_neutral`, 60 × 36 inches. A first strict render of the two-column portrait scaffold was overfull by roughly 40%; the standard ICML landscape gives each of the five claim verdicts and both generated figures legible space without shrinking type.
- Palette seed: the official ML-Agent repository logo uses cyan and deep blue on a pale background. The poster uses `#238FBD` (accent), `#0D4F70` (deep accent), `#E5F5FA` (light accent), and a restrained gold `#D69A2D` only for version-bound warnings and the final verdict.
- Visual assets: both charts are generated directly from `outputs/gpu_mechanism_results.json`. No paper figure is represented as independently reproduced evidence, and the optional Posterly paper-asset gate is intentionally not invoked.
- QR target: the intended public Trackio Space, `https://huggingface.co/spaces/binzhango/repro-ml-agent-autonomous-ml-engineering`.

## Mandatory content audit

| Poster statement | Evidence | Audit result |
| --- | --- | --- |
| 7B beats GPT-4o / GPT-4o-mini / DeepSeek-R1 agents | arXiv v1 Table 1 and surrounding text | Accurate only as a v1-reported result; explicitly labeled source-only |
| 9 training tasks and 10 held-out tasks across three modalities | arXiv v1/v2 Section 4 and task tables | Split and modality statement verified; empirical generalization labeled unreplicated |
| CIFAR-10 68.88 average and 81.45 best | arXiv v1 Table 1 | Exact; poster labels it v1 and source-verified |
| 100+ candidates and top-10 diversity selection | arXiv v1 Section 4 | Exact; proxy experiment separately labels its synthetic 120-idea corpus |
| v1 reward 0 / 0.5 / sigmoid and v2 reward −1 / 0 / linear | arXiv v1 and v2 equations plus executable assertions | Exact; chart and prose distinguish revisions |
| MiniLM selected 0.9647 versus random mean 0.8643 and random maximum 0.9443 | Saved completed Job result, seed 17897, 5,000 random subsets | Exact to four decimals |
| Total T4 attempts cost below USD 0.02 | 138 observed running seconds; T4-small rate USD 0.40/hour; 172 wall-clock seconds as upper bound | Conservative and accurate |

No claim on the poster attributes the proxy mechanism result to the paper authors' unreleased pool, and no claim treats source verification as an independent benchmark rerun.
