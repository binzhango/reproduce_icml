# ICML 2026 paper 21218 reproduction

This workspace reproduces the two requested efficiency claims for **Less Token,
More Signal: MoE Expert Pruning via Critical Token Selection** (OpenReview
`4iupzej9nT`).

The official release only includes runnable OLMoE support. The substantive GPU
experiment therefore runs the authors' STEP implementation on
`allenai/OLMoE-1B-7B-0125` with 50% expert sparsity and the full released
128-step calibration schedule. It is a scaled proxy, not a Qwen3-30B
replication. Raw results, the immutable Job URL, and exact limitations are
recorded in the Trackio logbook.
