# From Local Script to Reproducible Hugging Face Job, Artifact, and Space

This repository used Hugging Face as a small research platform rather than only
as a model download site:

1. **Hugging Face Jobs** supplied managed CPU/GPU compute.
2. **Hub repositories and Buckets** preserved code, predictions, checkpoints,
   and summaries after the temporary Job container stopped.
3. **Spaces** published Gradio demos and Trackio experiment dashboards.

The useful mental model is:

```text
local script -> HF Job -> durable Hub repo/Bucket -> Space or Trackio dashboard
                     \-> Job URL and logs
```

A Job is temporary compute. A model, dataset, or Space repository is versioned
storage. A Bucket is mutable object storage. A Space is a running application.
Keeping these roles separate makes a run easier to reproduce and much harder to
lose.

## What worked in this repository

The reproductions demonstrate the workflow at three scales:

- [`repro_579/toy_gpu_eval.py`](repro_579/toy_gpu_eval.py) ran a 7B coding model
  on an `a10g-small` Job. The Job finished 12 evaluations in about 13 minutes,
  and the final dataset repository preserved its predictions and summary.
- [`repro_32495/PROTOCOL.md`](repro_32495/PROTOCOL.md) records a successful
  `t4-small` run, its immutable Job URL, and the exact script revision. It also
  records two failed attempts, which made dependency and API incompatibilities
  diagnosable instead of invisible.
- [`repro_21218/outputs/summary.json`](repro_21218/outputs/summary.json) links an
  A100 Job to a Bucket result and records hardware, runtime, estimated cost, and
  scientific scope in one machine-readable file.

The most important negative lesson is in
[`repro_579/outputs/gpu_job_metrics.json`](repro_579/outputs/gpu_job_metrics.json):
the evaluation completed, but its first publishing attempt returned HTTP 403
because the token passed to the Job did not have write access. Compute success
and artifact persistence are two separate success conditions.

## 0. How this repository works

This is not one Python package with one application entry point. It is a
collection of independent reproduction workspaces. A directory such as
`repro_21218/` or `repro_579/` represents one paper and carries its evidence
from source material to a published, auditable result.

The repository's real architecture is an evidence pipeline:

```text
paper PDF / official code
        |
        v
claim audit + declared scope
        |
        v
local smoke test -----> outputs/smoke*
        |
        v
Hugging Face GPU Job -> immutable Job URL and logs
        |
        +-------------> dataset repo or Bucket (raw durable outputs)
        |
        v
analysis scripts ----> JSON / CSV / SVG / HTML summaries
        |
        v
Trackio metrics + artifacts + claim-oriented logbook pages
        |
        +-------------> published Trackio Space and artifact Bucket
        |
        v
artifact_bundle/ + poster/ + manifest + gate reports
```

### The directory contract

Not every reproduction needs every directory, but the mature workspaces follow
the same convention:

| Path | Role | Typical contents |
| --- | --- | --- |
| `repro_<id>/README.md` | Honest entry point | Paper identity, supported claims, limitations, and rerun commands |
| `source/` | Frozen source evidence | Paper PDF/text, OpenReview metadata, extracted figures |
| `official_repo/` or `upstream_*` | Author-code snapshot | Exact upstream implementation or released subset |
| `configs/` | Declared experimental scope | Seeds, model IDs, scale reductions, claim mapping |
| `scripts/` or top-level `*.py` | Executable reproduction | Source audit, smoke test, GPU harness, analysis, artifact logging |
| `outputs/` | Machine-readable evidence | Raw JSON/CSV, predictions, summaries, plots, provenance |
| `.trackio/` | Experiment narrative | Local metadata, claim pages, dashboard references, artifact cells |
| `artifact_bundle/` | Downloadable handoff | Minimal self-contained code, source, results, poster, and manifest |
| `poster/` or top-level poster files | Human-facing summary | HTML, PNG/PDF preview, QR link, style and gate reports |

The `artifact_bundle/` is deliberately not a blind copy of the workspace. For
example, [`repro_17897/artifact_bundle/MANIFEST.md`](repro_17897/artifact_bundle/MANIFEST.md)
states that credentials, Trackio's local database, Git metadata, caches, and
failed HTML responses are excluded. The result is portable evidence rather
than a dump of one developer machine.

### What each current reproduction demonstrates

| Workspace | Scientific shape | HF/Trackio pattern |
| --- | --- | --- |
| `repro_17897/` | Source audit plus disclosed mechanism proxies | T4 Job, claim matrix, Trackio artifact bundle, published claim pages and poster |
| `repro_17897_v2/` | Deeper scaled follow-up | Multiple GPU Jobs, staged data and artifacts in Buckets, compact v2 bundle |
| `repro_21218/` | Released STEP code on a smaller OLMoE configuration | A100 Job, raw Bucket result, bootstrap analysis, Trackio artifact and poster |
| `repro_32495/` | Scaled multi-agent component ablation | T4 Job, post-hoc Trackio import, raw-result artifact, claim logbook |
| `repro_579/` | Source verification plus a deterministic SQL-debugging proxy | A10G Job, dataset-repo persistence, Trackio metrics, failure-recovery lesson |

The scientific labels matter. A GPU Job proves that code ran; it does not by
itself prove a paper claim. These workspaces distinguish source-confirmed
claims, reduced proxies, and direct independent reproductions in their READMEs,
summaries, and logbook pages.

### One complete flow: `repro_32495`

This workspace is a useful template for understanding the moving parts:

1. [`repro_32495/PROTOCOL.md`](repro_32495/PROTOCOL.md) declares the three
   experimental conditions, datasets, seeds, hardware, Job URL, and the limits
   of the scaled proxy.
2. [`repro_32495/run_ablation.py`](repro_32495/run_ablation.py) executes the
   conditions, writes `trials.csv`, `details.json`, and `summary.json`, and logs
   per-trial metrics with Trackio.
3. The real GPU execution runs as a Hugging Face Job. The Job page preserves
   the command, environment, status, and logs independently of the local tree.
4. [`repro_32495/ingest_gpu_results.py`](repro_32495/ingest_gpu_results.py)
   imports the immutable CSV/JSON output into Trackio. This recovery path is
   valuable when the remote Job could write raw results but could not update a
   Trackio Space live.
5. `trackio.log_artifact(...)` registers the raw output directory as a dataset
   artifact. Publishing the logbook promotes that artifact to its HF Bucket.
6. `.trackio/logbook/pages/` organizes evidence into one page per claim plus a
   conclusion. The index is only a table of contents.
7. The poster and `artifact_bundle/` turn the same evidence into a conference
   summary and a downloadable reproduction package; `GATE_REPORT.json` and
   `style_check.json` record mechanical validation.

### Starting a new `repro_<id>` workspace

A practical sequence is:

```bash
mkdir repro_PAPER_ID
cd repro_PAPER_ID

# 1. Create the local experiment narrative before running expensive work.
trackio logbook open --title "Repro - PAPER TITLE" --no-serve
trackio logbook page "Source reproducibility audit"

# 2. Add source, protocol, scripts, and a cheap smoke test.
#    Record the exact command rather than running it invisibly.
trackio logbook run --page "Source reproducibility audit" -- \
  uv run verify_claims.py

# 3. Submit the validated script to HF Jobs and persist raw outputs.
# 4. Import or live-sync metrics into Trackio.
# 5. Add claim verdict pages, conclusion, artifact bundle, and poster.
```

Do not publish the logbook until its pages have been checked for secrets,
private paths, misleading evidence labels, and missing raw outputs.

## 1. One-time setup

Install the current `hf` CLI and log in:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash -s
hf auth login
hf auth whoami
hf version
```

Use `hf`, not the deprecated `huggingface-cli` command. Jobs require an account
or organization with a positive credit balance. Before spending GPU credit,
inspect the currently available hardware and prices:

```bash
hf jobs hardware
hf spaces hardware
```

Create a fine-grained token with only the permissions the workload needs:

- read access for public/private downloads;
- write access to the exact model, dataset, Bucket, or Space that receives
  outputs;
- never place the literal token in source code, shell history, or a committed
  `.env` file.

The CLI can securely forward the token from the authenticated local session to
a Job with `--secrets HF_TOKEN`.

## 2. Download models, datasets, or Space source to your local disk

> **Local disk warning:** every `hf download` command in this section runs on
> your local computer and uses local disk space. Without `--local-dir`, files
> are stored in the Hugging Face cache (normally
> `~/.cache/huggingface/hub`). With `--local-dir`, files are stored in the
> directory you choose. A full model can use tens or hundreds of GB, and a
> dataset can be much larger. Space source code is usually comparatively small.

The same command downloads all three repository types. Check the transfer
before downloading a large repository:

```bash
# Preview which files would be downloaded and their total size
hf download Qwen/Qwen2.5-Coder-1.5B-Instruct --dry-run
hf download USER/large-dataset --repo-type dataset --dry-run
```

Then download only what the local workflow actually needs:

```bash
# Entire public model into the shared HF cache
hf download Qwen/Qwen2.5-Coder-1.5B-Instruct

# Only selected model files into a normal local directory
hf download Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --include "*.json" --include "tokenizer*" \
  --local-dir ./models/qwen-coder-1.5b-metadata

# A dataset revision
hf download USER/experiment-results \
  --repo-type dataset --revision main \
  --local-dir ./downloaded-results

# The source code of a Space
hf download USER/my-demo \
  --repo-type space --local-dir ./my-demo
```

Useful reproducibility options:

```bash
# Pin an immutable commit instead of taking a moving `main`
hf download USER/experiment-results \
  --repo-type dataset --revision COMMIT_SHA \
  --local-dir ./downloaded-results

# Verify the local copy against Hub checksums
hf cache verify USER/experiment-results \
  --repo-type dataset --local-dir ./downloaded-results
```

`--local-dir` creates local Hugging Face metadata so later downloads transfer
only changed files. Omit it when library-managed caching is preferable; this
changes where the files live, but it does not eliminate local disk usage.

Inspect the cache regularly, especially after experimenting with several model
revisions:

```bash
# Show the largest cached repositories first
hf cache list --sort size --limit 20

# Find cached repositories larger than 5 GB
hf cache list --filter "size>5GB"

# Preview removable detached revisions and incomplete downloads
hf cache prune --dry-run
```

Run `hf cache prune` without `--dry-run` only after reviewing the preview. Use
`hf cache rm --help` when a specific cached repository or revision must be
removed. Cache cleanup does not remove files downloaded into a separate
`--local-dir`; manage that directory yourself.

### Avoid a large download on your laptop

If the goal is remote training, do not download the model or dataset locally
first. Submit the small training script and let the remote Job access Hub data.
For example, mount repositories directly into the Job container:

```bash
hf jobs uv run \
  --flavor a10g-small --timeout 4h \
  --secrets HF_TOKEN \
  --volume hf://models/Qwen/Qwen2.5-Coder-1.5B-Instruct:/models/qwen:ro \
  --volume hf://datasets/USER/training-data:/data:ro \
  train.py --model-path /models/qwen --data-path /data
```

This keeps the large files off the local computer; storage and transfer occur
in Hugging Face's remote Job environment. Model, dataset, and Space mounts are
read-only. Use a read/write Bucket mount for checkpoints and generated results.

For row-oriented processing, stream a dataset instead of materializing the
entire dataset locally:

```python
from datasets import load_dataset

rows = load_dataset("USER/large-dataset", split="train", streaming=True)
for row in rows.take(1000):
    process(row)
```

Streaming still transfers the rows that are consumed, but it avoids downloading
the complete dataset up front. Some training algorithms require random access
or repeated passes and therefore still need a materialized dataset or a remote
mounted copy.

## 3. Make a Python workload self-contained

For Python, a UV script is the quickest Job format. Put dependencies directly
in the script with PEP 723 metadata:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface-hub>=1.0",
#   "torch>=2.5",
#   "transformers>=4.51",
# ]
# ///

import argparse
import json
import os
from pathlib import Path

import torch
from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument("--push-repo")
args = parser.parse_args()

print({"cuda": torch.cuda.is_available()})

result = {
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "torch": torch.__version__,
}
output = Path("outputs")
output.mkdir(exist_ok=True)
(output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

# A Job filesystem disappears when the Job ends, so upload real-run results.
if args.push_repo:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN was not supplied to the Job")

    api = HfApi(token=token)
    api.create_repo(args.push_repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=output,
        repo_id=args.push_repo,
        repo_type="dataset",
        path_in_repo="runs/gpu-check",
        commit_message="Persist GPU check results",
    )
    print(f"Saved to https://huggingface.co/datasets/{args.push_repo}")
```

Save it as `gpu_check.py`. Test locally on CPU first:

```bash
uv run gpu_check.py
```

Passing `--push-repo` requires `HF_TOKEN`; omitting it makes the local smoke test
write only to `./outputs`. The repository's
[`repro_579/toy_gpu_eval.py`](repro_579/toy_gpu_eval.py) is a larger example
with `--mode smoke`, `--mode gpu`, and `--push-repo` modes.

## 4. Submit a managed CPU or GPU Job

First create the durable result repository:

```bash
hf repos create USER/gpu-check-results \
  --repo-type dataset --private --exist-ok
```

Run the smallest smoke test:

```bash
hf jobs uv run \
  --name gpu-check-smoke \
  --flavor cpu-basic \
  --timeout 20m \
  --secrets HF_TOKEN \
  gpu_check.py --push-repo USER/gpu-check-results
```

Then change only the hardware for the real run:

```bash
hf jobs uv run \
  --name gpu-check-a10g \
  --flavor a10g-small \
  --timeout 2h \
  --secrets HF_TOKEN \
  gpu_check.py --push-repo USER/gpu-check-results
```

Add `--detach` to return immediately with a Job ID. Local paths work with the
CLI because it uploads the script before starting the container. Arguments
after the script are forwarded to the script, for example:

```bash
hf jobs uv run \
  --detach --flavor a10g-small --timeout 4h \
  --secrets HF_TOKEN \
  repro_579/toy_gpu_eval.py \
  --mode gpu \
  --output-dir outputs/gpu \
  --push-repo USER/paper579-results
```

Use Docker Jobs instead when the environment is not naturally expressed as a
Python UV script:

```bash
hf jobs run \
  --name cuda-check \
  --flavor a10g-small \
  --timeout 30m \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
  python -c "import torch; print(torch.cuda.get_device_name())"
```

### Monitor and debug

```bash
hf jobs list
hf jobs inspect JOB_ID
hf jobs logs --follow JOB_ID
hf jobs stats JOB_ID
hf jobs wait JOB_ID
hf jobs cancel JOB_ID
```

The Job URL is a durable provenance record containing status, configuration,
and logs. Record both the URL and ID in the final `summary.json` or README.

Start with CPU or a small GPU, validate imports and one mini-batch, then scale.
Set an explicit timeout with setup and upload buffer; the default is 30 minutes.
Checkpoint long jobs because a timeout stops the container immediately.

## 5. Choose the correct persistence layer

### Versioned model/dataset repositories

Use these for results that should have commits, revisions, cards, and stable
links:

```bash
hf repos create USER/my-results --repo-type dataset --private --exist-ok

# Upload one file
hf upload USER/my-results ./outputs/summary.json runs/001/summary.json \
  --repo-type dataset \
  --commit-message "Add run 001 summary"

# Upload a directory
hf upload USER/my-results ./outputs runs/001 \
  --repo-type dataset \
  --exclude "*.tmp" \
  --commit-message "Add run 001 artifacts"

# Download the same result later
hf download USER/my-results runs/001/summary.json \
  --repo-type dataset --local-dir ./recovered
```

Use `--create-pr` when contributors should propose artifacts without writing
directly to the main branch.

### Buckets and mounted volumes

Use a Bucket for mutable checkpoints, large intermediate data, or a directory
that a running Job must write repeatedly:

```bash
hf buckets create USER/training-checkpoints --private --exist-ok
hf buckets sync ./checkpoints hf://buckets/USER/training-checkpoints
hf buckets list USER/training-checkpoints --tree --recursive
```

Mount it read/write in a Job:

```bash
hf jobs uv run \
  --flavor a10g-small --timeout 8h \
  --secrets HF_TOKEN \
  --volume hf://buckets/USER/training-checkpoints:/checkpoints:rw \
  train.py --output-dir /checkpoints/run-001
```

Models, datasets, and Spaces mount read-only; Buckets can mount read/write.
Promote the final checkpoint from the Bucket to a versioned model repository
when the run is complete.

## 6. Publish a Gradio Space in minutes

Create a folder called `space/` with three files.

`space/app.py`:

```python
import gradio as gr


def describe_run(job_id: str, score: float) -> str:
    return f"Job `{job_id}` finished with score **{score:.3f}**."


with gr.Blocks() as demo:
    gr.Markdown("# Reproduction result viewer")
    job_id = gr.Textbox(label="Hugging Face Job ID")
    score = gr.Number(label="Score", value=0.9)
    output = gr.Markdown()
    gr.Button("Create summary").click(
        describe_run, inputs=[job_id, score], outputs=output
    )

demo.launch()
```

`space/requirements.txt`:

```text
gradio>=5
```

`space/README.md`:

```yaml
---
title: Reproduction Result Viewer
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Reproduction Result Viewer

A small UI for presenting a completed Hugging Face Job.
```

Create and deploy it:

```bash
hf repos create USER/reproduction-viewer \
  --type space --sdk gradio --exist-ok
hf upload USER/reproduction-viewer ./space . \
  --repo-type space \
  --commit-message "Deploy initial Gradio app"
hf spaces wait USER/reproduction-viewer --timeout 10m
hf spaces logs USER/reproduction-viewer --build
```

Every upload is a Space repository commit and triggers a rebuild. Put public
configuration in variables and credentials in secrets:

```bash
hf spaces variables add USER/reproduction-viewer \
  --env RESULTS_REPO=USER/my-results
hf spaces secrets add USER/reproduction-viewer \
  --secrets HF_TOKEN
```

Do not add `HF_TOKEN` to `README.md`, `app.py`, or Space variables. Secrets are
write-only and their values are not returned by the API.

Useful operations:

```bash
hf spaces info USER/reproduction-viewer
hf spaces logs USER/reproduction-viewer --follow
hf spaces restart USER/reproduction-viewer
hf spaces settings USER/reproduction-viewer --sleep-time 300
```

## 7. How Trackio is used in this repository

Trackio serves three related but different purposes here:

1. **Metrics database and dashboard** — numeric values from a run.
2. **Artifact registry** — versioned result directories that publish to an HF
   Bucket with the logbook.
3. **Reproduction logbook** — claim-oriented Markdown/code/figure/artifact
   pages published as a static HF Space for humans and agents.

Trackio does not submit the GPU Job and does not replace raw result storage. HF
Jobs provide compute; Hub repositories or Buckets preserve raw files; Trackio
connects those files and metrics to the scientific narrative.

### Pattern A: log metrics live from the training or evaluation script

[`repro_32495/run_ablation.py`](repro_32495/run_ablation.py) initializes one
Trackio project, logs every measured condition, and logs final aggregate
metrics. A minimal equivalent is:

```python
import trackio

trackio.init(
    project="icml-PAPER_ID-ablation",
    name="gpu-proxy-seed-0",
    config={
        "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "scope": "scaled-proxy",
        "seed": 0,
    },
    space_id="USER/icml-PAPER_ID-ablation",
    private=True,
)

for step, result in enumerate(run_experiment()):
    trackio.log(
        {
            "balanced_accuracy": result["score"],
            "condition_id": result["condition_id"],
            "seed": result["seed"],
        },
        step=step,
    )

trackio.finish()
```

For an HF Job, add `trackio` to the UV dependencies and pass `HF_TOKEN` as a
secret. `space_id` tells Trackio to sync remotely; without it, metrics remain in
the Job's temporary filesystem and disappear when the worker terminates.
Auto-created Trackio Spaces are public by default, so pass `private=True` or
pre-create the destination with the required visibility.

### Pattern B: import completed Job outputs after the fact

Live logging can fail even when the expensive computation succeeds. The
repository therefore also uses a safer two-stage pattern:

```text
HF Job -> durable trials.csv + summary.json -> local import -> Trackio Space
```

[`repro_32495/ingest_gpu_results.py`](repro_32495/ingest_gpu_results.py) reads
the completed Job output, replays each trial into Trackio, records the original
Job URL in the run configuration, and registers the raw directory as an
artifact. This preserves the distinction between immutable source data and the
dashboard derived from it.

```bash
cd repro_32495
uv run ingest_gpu_results.py \
  --results outputs/gpu \
  --job-url https://huggingface.co/jobs/USER/JOB_ID

trackio list projects --json
trackio list runs --project icml-32495-component-ablation --json
trackio get run \
  --project icml-32495-component-ablation \
  --run qwen-1.5b-gpu-proxy-imported --json
```

Use post-hoc import when:

- the Job token cannot create or update a Space;
- the run has already completed;
- metrics must be reconstructed from authoritative raw files;
- the dashboard schema changed after the GPU work was done.

### Register the downloadable artifact bundle

The `log_bundle.py` scripts use Trackio artifacts rather than treating the
bundle as an anonymous folder. For example,
[`repro_21218/scripts/log_bundle.py`](repro_21218/scripts/log_bundle.py) does:

```python
from pathlib import Path
import trackio

bundle = Path("artifact_bundle")
trackio.init(
    project="step-repro-21218-artifacts",
    name="complete-reproduction-bundle",
    config={"scope": "released OLMoE proxy"},
)
trackio.log_artifact(
    str(bundle),
    name="reproduction-bundle",
    type="dataset",
    aliases=["challenge", "latest"],
)
trackio.finish()
```

When a logbook exists, `trackio.log_artifact()` automatically creates an
artifact cell. On publication, Trackio pushes the artifact payload to the
logbook's HF Bucket and rewrites the cell to the remote URL. Merely letting
`trackio logbook run` detect an output file records a path reference; it does
not upload that file. Use `log_artifact()` for anything that must travel with
the published reproduction.

### Understand the `.trackio/` directory

Each reproduction keeps a local logbook such as:

```text
repro_579/.trackio/
├── metadata.json
└── logbook/
    ├── logbook.json
    ├── pages/index.md
    ├── pages/claim-1-.../page.md
    ├── pages/gpu-toy-debugging-experiment/page.md
    ├── pages/conclusion/page.md
    └── static viewer files
```

- `metadata.json` remembers the published Space, privacy, last page, local
  dashboards, local artifacts, and artifact Bucket.
- `pages/index.md` is a table of contents, not a findings page.
- Each `page.md` contains typed cells encoded as Markdown comments plus their
  human-readable body.
- `logbook.json` is the structured page tree used by the viewer and agent
  reader.

The logbook complements `outputs/`: `outputs/` is authoritative data, while
the logbook explains which claim that data supports, the execution context,
and the verdict.

### Work with the logbook

```bash
cd repro_579

# Read a compact whole-logbook view suitable for an agent.
trackio logbook read

# Create/select a page and append findings.
trackio logbook page "GPU toy debugging experiment"
trackio logbook cell markdown \
  "The A10G run completed; raw predictions are linked below."

# Capture an exact local command, scripts, output, exit code, and duration.
trackio logbook run --page "GPU toy debugging experiment" -- \
  uv run toy_gpu_eval.py --mode smoke --output-dir outputs/toy_smoke

# Preview locally before publishing.
trackio logbook serve
```

Because the logbook stores ordinary files, they can be reviewed and edited
directly. After direct edits to an already published logbook, run:

```bash
trackio logbook sync
```

### Publish with the correct privacy

The first publish creates a public static Space immediately unless `--private`
is supplied:

```bash
# Public challenge/research logbook
trackio logbook publish USER/repro-PAPER_ID

# Internal or sensitive logbook, dashboards, and artifacts
trackio logbook publish USER/repro-PAPER_ID --private
```

After first publication, page/cell/run commands auto-sync. The Space is the
reader-facing notebook; Trackio dashboards embedded in it remain the metric
view; artifacts are stored in the associated Bucket. Always preview and scan
for tokens, local absolute paths, private datasets, and unredacted samples
before the first publish.

### From the logbook to the poster

The posters are another view of the same evidence, not an independent result.
For example, [`repro_21218/poster/poster_build_notes.md`](repro_21218/poster/poster_build_notes.md)
names the local Trackio logbook and raw GPU output as its content authority.
Posterly generates the HTML poster, preview PNG/PDF, navigation hotspots, and
gate reports. The finished poster is then stored as a pinned `Reproduction
poster` figure cell on the Trackio conclusion page, and its QR code points back
to the published logbook Space. This gives a conference reader a compact
summary while preserving a path to claim pages, Job provenance, and raw
artifacts.

### Query metrics without opening a browser

```bash
# Local data
trackio list projects --json
trackio list runs --project PROJECT --json
trackio get metric --project PROJECT --run RUN --metric loss --json

# Published Space (`--space` is a global option and comes before the command)
trackio --space USER/TRACKIO_SPACE list projects --json
trackio --space USER/TRACKIO_SPACE get metric \
  --project PROJECT --run RUN --metric loss --json

# One-off aggregate over Trackio's local data
trackio query project --project PROJECT \
  --sql "SELECT run_name, MAX(step) AS last_step FROM metrics GROUP BY run_name" \
  --json
```

JSON output is especially useful for agents and automated gates. A Trackio
dashboard, artifact, or successful code cell improves provenance but does not
turn a proxy into an independent reproduction; the claim page must still state
scope and evidence honestly.

## 8. How Hugging Face skills help Codex work with this repository

An HF skill is a `SKILL.md` instruction package for an AI coding assistant. It
is not a Python dependency, does not run a Job, and does not grant Hub access.
It teaches the assistant which commands, safety rules, and workflow patterns to
use when a task matches the skill.

Project-local skills live under `.agents/skills/`, so an assistant working in
this repository can discover them automatically. This checkout contains:

| Skill directory | Purpose in this repository |
| --- | --- |
| `.agents/skills/hf-cli/` | Current `hf` syntax for auth, downloads, Jobs, repos, Buckets, Spaces, and skills |
| `.agents/skills/trackio/` | Metric logging, alerts, dashboard sync, querying, artifacts, and logbooks |
| `.agents/skills/reproduction-logbook-judge/` | Repository-specific scientific and submission audit of a Trackio reproduction logbook |
| `.agents/skills/financial-report-generator/` | Unrelated to the ICML reproduction flow |

The `hf-cli` skill is generated from the installed CLI, which reduces stale
command examples. The Trackio skill adds the higher-level experiment workflow
that raw CLI help cannot explain. The reproduction judge is a local extension:
it separates paper transcription from independent evidence and checks the
claim pages, raw outputs, artifacts, conclusion, and poster requirements.

### Install or update skills

Run these commands from the repository root:

```bash
# Generate the project-local hf-cli skill in .agents/skills/.
hf skills add

# Install the current Trackio marketplace skill.
hf skills add huggingface-trackio

# Inspect the marketplace and installed locations.
hf skills list

# Refresh installed marketplace skills and regenerate hf-cli when needed.
hf skills update
hf skills add --force
```

Use `--global` only when the same skill should apply to every repository on the
machine. Project-local installation is preferable here because collaborators
can inspect the exact instructions associated with the reproduction workflow.
Project-local skills are still ordinary files: add `.agents/skills/` to Git if
collaborators should receive them with the repository; otherwise they exist
only in the current checkout.

### How to invoke the skills as a user

Skills normally trigger from the task description, but naming the desired
skill makes the boundary explicit. Good requests include:

```text
Use the hf-cli skill to prepare an A10G Job command for
repro_32495/run_ablation.py. Do not submit it yet.

Use the hugging-face-trackio skill to add loss, GPU utilization, and NaN alerts
to this training script, then show how to query the metrics as JSON.

Use the reproduction-logbook-judge skill to audit repro_21218 read-only and
report which claims are source-confirmed, proxy-level, or independently tested.
```

The request should still specify authorization boundaries such as “prepare but
do not submit,” “use a private Space,” or “audit read-only.” Skills improve how
the agent performs a task; they do not broaden what the agent is allowed to
change or publish.

### Recommended skill-assisted workflow for a new reproduction

1. Ask the agent to use `hf-cli` to inspect authentication, hardware, expected
   storage, and prepare a smoke Job.
2. Ask it to use Trackio to instrument metrics and alerts before submitting
   expensive work.
3. Run locally through `trackio logbook run` so failed and successful attempts
   retain exact commands and outputs.
4. Submit the remote Job only after the smoke path passes; persist raw outputs
   independently of Trackio.
5. Import or sync Trackio metrics, register the artifact bundle, and build one
   claim page per paper claim.
6. Run the reproduction-logbook judge before publication; fix scientific
   evidence gaps before cosmetic issues.

## 9. A reliable production checklist

Before submission:

- run a local smoke test and a small remote smoke Job;
- pin critical package and model revisions;
- select the smallest hardware that fits;
- pass secrets with `--secrets`, never `--env`;
- verify that the token has write access to the exact destination;
- set an explicit timeout;
- decide where every important output will persist.

After submission:

- save the Job ID and URL;
- inspect logs and resource statistics;
- verify that the output commit or Bucket object exists;
- download one persisted file as a recovery test;
- record hardware, model revision, dataset revision, command, dependency
  versions, runtime, cost basis, and scope limitations;
- publish only non-sensitive results in a public Space.

## Common failures

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| HTTP 401 | Missing or invalid token | Log in again and pass `--secrets HF_TOKEN` |
| HTTP 403 during upload | Token lacks write permission to the destination | Grant narrow write access or choose a repository you own |
| Job succeeds but no files remain | Results stayed on the ephemeral filesystem | Upload to a repo/Bucket before exit; checkpoint during long runs |
| Job times out | Default or chosen timeout is too short | Inspect logs, estimate duration, then add setup/upload buffer |
| CUDA out of memory | Model, batch, or context is too large | Reduce batch/context, use quantization, or choose larger hardware |
| Import/API error | Dependency was missing or moved | Add and pin it in the PEP 723 dependency block |
| Space build fails | Missing dependency or invalid README metadata | Read `hf spaces logs SPACE_ID --build` and fix `requirements.txt`/YAML |
| Private metrics become visible | Trackio Space was auto-created public | Pass `private=True` on first creation or pre-create a private Space |

## Official references

- [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs)
- [`hf` CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
- [Upload files](https://huggingface.co/docs/huggingface_hub/en/guides/upload)
- [Spaces overview](https://huggingface.co/docs/hub/en/spaces)
- [Gradio Spaces](https://huggingface.co/docs/hub/en/spaces-sdks-gradio)
- [Trackio documentation](https://huggingface.co/docs/trackio/index)

Commands in this guide were checked against `hf` 1.24.0 and Trackio 0.31.5 on
2026-07-19.
