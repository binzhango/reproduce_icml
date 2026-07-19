# ML-Agent ICML 2026 reproduction (#17897)

This bundle audits and partially reproduces the five requested claims for
"ML-Agent: Reinforcing LLM Agents for Autonomous Machine Learning Engineering"
(OpenReview `kcPPWaoegr`, arXiv `2505.23723`).

The headline benchmark cannot be independently rerun from the public release:
the official repository still withholds the trained checkpoint, evaluation
code, RL code, and training trajectories. The bundle therefore separates:

- source verification against arXiv v1 and v2;
- release/readiness auditing at official commit `15932e7`;
- a disclosed proxy reproduction of the 100-to-10 diversity-selection mechanism;
- exact executable tests of both the v1 and revised v2 reward equations.

Local smoke test:

```bash
uv run --with numpy --with matplotlib scripts/audit_claims.py
uv run --with numpy --with matplotlib scripts/reproduce_mechanisms.py \
  --backend hash --random-trials 1000 --output-prefix local
```

The GPU run uses `sentence-transformers/all-MiniLM-L6-v2` embeddings in place
of the hash smoke-test backend. All results explicitly label this as a mechanism
reproduction rather than a full ML-Agent replication.
