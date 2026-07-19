# Claim 2: Nine-Task Training and Generalization


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_35737f776585", "created_at": "2026-07-16T01:12:10+00:00", "title": "Verdict: split verified; generalization result blocked"}
-->
Both revisions state that training used 9 tasks: 4 from MLAgentBench and 5 from MLE-bench. They list 10 held-out MLE-bench tasks spanning image, text, and tabular inputs; v2 also reports additional evaluations beyond the core ten. The task split and modality coverage are source-verified. The empirical generalization result is not independently reproduced because neither trained weights nor evaluation trajectories are public.
