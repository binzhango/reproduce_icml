# Claim 4: Enterprise SQL complexity


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_d83d3c87db06", "created_at": "2026-07-16T00:08:41+00:00", "title": "Scope and verification plan"}
-->
This page verifies the reported token, line, function-count, AST-depth, and AST-width statistics and measures the same properties on a clearly labeled synthetic proxy using sqlglot rather than the paper Calcite pipeline.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_4a153036580c", "created_at": "2026-07-16T00:14:16+00:00", "title": "Claim 4 verdict: reported values verified; proxy differs by representation"}
-->
Table 1 reports Squirrel-Syntax: 496.90 tokens, 163.69 lines, 21.62 functions, AST depth 8.93, AST width 11.69. Squirrel-Semantic: 425.93 tokens, 141.58 lines, 17.34 functions, depth 8.75, width 11.12. This verifies the requested order-of-magnitude statement. Our deterministic toy scripts average 161 lines, 1,005 lexical tokens, 58 sqlglot function nodes, depth 10, width 170; only length is meaningfully comparable because the paper uses LLM tokenization and optimized Calcite plan statistics, while the proxy uses lexical tokens and raw sqlglot ASTs.
