# reproduce_icml

This repository is a collection of ICML paper-reproduction workspaces. Each
`repro_<paper-id>/` directory turns a paper claim into auditable source checks,
local smoke tests, managed Hugging Face GPU Jobs, machine-readable results,
Trackio experiment pages, a downloadable artifact bundle, and—where
applicable—a verified research poster.

Start with the [Hugging Face and repository workflow guide](HUGGINGFACE_WORKFLOW_GUIDE.md).
It explains:

- how the `repro_*` directories are organized and how evidence moves through
  the repository;
- how Hugging Face Jobs, dataset repositories, Buckets, and Spaces divide
  responsibilities;
- how Trackio records metrics, artifacts, claim pages, and a published
  logbook;
- how Hugging Face agent skills under `.agents/skills/` help Codex operate the
  CLI and Trackio correctly;
- how to reproduce the same workflow for a new paper without filling the local
  disk with large models and datasets.
