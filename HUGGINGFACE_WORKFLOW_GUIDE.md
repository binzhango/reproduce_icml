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

## 7. Use a Trackio Space for training metrics

For experiments, Trackio can create or sync a Space dashboard directly. This is
the pattern used by the repository's GPU evaluation code:

```python
import trackio

trackio.init(
    project="my-training-project",
    name="baseline-a10g",
    config={"model": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "lr": 2e-5},
    space_id="USER/my-training-dashboard",
    private=True,
)

for step in range(10):
    trackio.log({"step": step, "train_loss": 1.0 / (step + 1)})

trackio.finish()
```

Add `trackio` to the UV dependencies and submit `HF_TOKEN` as a Job secret.
For remote training, always set `space_id`; otherwise the dashboard database is
only local to the temporary worker. Auto-created Trackio Spaces are public by
default, so pass `private=True` for private metrics.

Trackio is a presentation and tracking layer, not a substitute for preserving
the model and raw result files. Upload those separately to a model/dataset repo
or Bucket.

## 8. A reliable production checklist

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

Commands in this guide were checked against `hf` 1.24.0 on 2026-07-19.
