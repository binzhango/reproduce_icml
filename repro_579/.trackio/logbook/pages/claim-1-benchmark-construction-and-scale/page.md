# Claim 1: Benchmark construction and scale


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_23cfca003aff", "created_at": "2026-07-16T00:08:39+00:00", "title": "Scope and verification plan"}
-->
This page verifies the reported Squirrel benchmark counts, source-corpus breadth, and structural complexity against the paper, then compares them with a deterministic toy proxy because the official benchmark data and construction code are not publicly available.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_7a748a6feb28", "created_at": "2026-07-16T00:14:15+00:00", "title": "Claim 1 verdict: reported values verified"}
-->
Paper evidence: arXiv:2601.18119v1 reports 469 Squirrel-Syntax and 516 Squirrel-Semantic tasks. Section 3.1 states that the seed corpus contains 1,000+ validated SQL scripts spanning 26 business scenarios, averaging over 120 lines with AST depth >8 and width >12. The final benchmark is later reported at 141.58–163.69 lines, AST depth 8.75–8.93, and AST width 11.12–11.69. These values are verified as paper-reported facts, not independently recomputed, because the official Squirrel dataset and construction code were unavailable. Paper: https://arxiv.org/abs/2601.18119
