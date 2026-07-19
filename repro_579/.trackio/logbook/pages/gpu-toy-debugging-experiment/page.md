# GPU toy debugging experiment


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_eaac7906f503", "created_at": "2026-07-16T00:08:41+00:00", "title": "Scope and verification plan"}
-->
This page records a scaled toy experiment on a Hugging Face GPU Job. It evaluates an open code model on deterministic enterprise-style SQL debugging pairs; it does not substitute for evaluation on the unreleased Squirrel benchmark.


---
<!-- trackio-cell
{"type": "code", "id": "cell_c30ffa798684", "created_at": "2026-07-16T00:10:57+00:00", "title": "Run: uv toy_gpu_eval.py (exit 2)", "command": ["uv", "run", "repro_579/toy_gpu_eval.py", "--mode", "smoke", "--output-dir", "repro_579/outputs/toy_smoke"], "exit_code": 2, "duration_s": 0.01}
-->
````bash
$ uv run repro_579/toy_gpu_eval.py --mode smoke --output-dir repro_579/outputs/toy_smoke
````

exit 2 · 0.0s


````python title=toy_gpu_eval.py
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate>=1.6",
#   "huggingface-hub>=0.31",
#   "sqlglot>=26.0",
#   "torch>=2.5",
#   "trackio>=0.3",
#   "transformers>=4.51",
# ]
# ///
"""Deterministic toy enterprise-SQL benchmark and optional GPU model evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, deque
from pathlib import Path

import sqlglot
from sqlglot import exp

DOMAINS = ["finance", "ecommerce", "healthcare", "logistics", "advertising", "travel"]


def make_sql(domain: str, idx: int) -> str:
    measures = ["amount", "quantity", "cost", "revenue", "duration", "score"]
    lines = ["WITH"]
    for stage in range(7):
        name = f"{domain}_stage_{stage}"
        source = f"{domain}_events" if stage == 0 else f"{domain}_stage_{stage - 1}"
        prefix = "" if stage == 0 else ","
        lines.extend([
            f"{prefix}{name} AS (",
            "    SELECT",
            "        t.entity_id,",
            "        t.event_date,",
            "        t.region_id,",
            "        d.segment_name,",
            f"        SUM(t.{measures[stage % len(measures)]}) AS metric_{stage},",
            f"        COUNT(DISTINCT t.event_id) AS events_{stage},",
            f"        AVG(t.{measures[(stage + 1) % len(measures)]}) AS average_{stage},",
            "        ROW_NUMBER() OVER (",
            "            PARTITION BY t.entity_id, t.event_date",
            f"            ORDER BY SUM(t.{measures[stage % len(measures)]}) DESC",
            f"        ) AS rank_{stage}",
            f"    FROM {source} AS t",
            f"    LEFT JOIN {domain}_dimensions AS d",
            "        ON t.entity_id = d.entity_id",
            "       AND t.region_id = d.region_id",
            f"    WHERE t.event_date >= DATE '2026-0{(idx % 6) + 1}-01'",
            "      AND t.is_valid = 1",
            "    GROUP BY t.entity_id, t.event_date, t.region_id, d.segment_name",
            ")",
        ])
    lines.extend([
        "SELECT",
        "    entity_id,",
        "    event_date,",
        "    region_id,",
        "    segment_name,",
        "    metric_6,",
        "    events_6,",
        "    average_6,",
        "    CASE WHEN rank_6 = 1 THEN 'leader' ELSE 'other' END AS cohort",
        f"FROM {domain}_stage_6",
        "WHERE rank_6 <= 10",
        "ORDER BY event_date, region_id, metric_6 DESC",
        ";",
    ])
    return "\n".join(lines)


def inject(sql: str, kind: str) -> tuple[str, str]:
    if kind == "syntax":
        buggy = sql.replace("SUM(t.amount) AS metric_0", "SUM(t.amount AS metric_0", 1)
        return buggy, "Parser error near metric_0: expected a closing parenthesis."
    buggy = sql.replace("LEFT JOIN", "INNER JOIN", 1)
    return buggy, "Rows without a matching dimension record disappeared from the result. Preserve them."


def canonical(sql: str) -> str | None:
    try:
        return sqlglot.parse_one(sql, read="hive").sql(dialect="hive", pretty=False)
    except Exception:
        return None


def tree_stats(sql: str) -> dict[str, float]:
    tree = sqlglot.parse_one(sql, read="hive")
    by_depth: Counter[int] = Counter()
    queue = deque([(tree, 0)])
    while queue:
        node, depth = queue.popleft()
        by_depth[depth] += 1
        for child in node.iter_expressions():
            queue.append((child, depth + 1))
    return {
        "lines": len(sql.splitlines()),
        "tokens": len(re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|<>|<=|>=|!=|\S", sql)),
        "functions": sum(1 for node in tree.walk() if isinstance(node, exp.Func)),
        "ast_depth": max(by_depth),
        "ast_width": max(by_depth.values()),
    }


def build_tasks() -> list[dict]:
    tasks = []
    for idx, domain in enumerate(DOMAINS):
        gold = make_sql(domain, idx)
        for kind in ("syntax", "semantic"):
            buggy, issue = inject(gold, kind)
            tasks.append({
                "id": f"{domain}-{kind}",
                "domain": domain,
                "kind": kind,
                "issue": issue,
                "buggy_sql": buggy,
                "gold_sql": gold,
            })
    return tasks


def extract_sql(text: str) -> str:
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.I | re.S)
    return (match.group(1) if match else text).strip()


def evaluate_prediction(task: dict, prediction: str) -> dict:
    pred_canon = canonical(prediction)
    gold_canon = canonical(task["gold_sql"])
    buggy_canon = canonical(task["buggy_sql"])
    gm = int(pred_canon is not None and pred_canon == gold_canon)
    em = int(prediction.strip() == task["gold_sql"].strip())
    if pred_canon is None:
        mb = 0
    elif buggy_canon is None:
        mb = 1
    else:
        import difflib
        before = difflib.SequenceMatcher(None, buggy_canon, gold_canon).ratio()
        after = difflib.SequenceMatcher(None, pred_canon, gold_canon).ratio()
        mb = int(after > before)
    return {"em": em, "gm_proxy": gm, "modify_better_proxy": mb, "prediction_parseable": int(pred_canon is not None)}


def prompt(task: dict) -> str:
    return f"""You are debugging a long Hive ETL SQL script.
Issue report: {task['issue']}
Make the smallest correction that fixes the issue. Return only the complete corrected SQL.

```sql
{task['buggy_sql']}
```"""


def run_model(tasks: list[dict], model_id: str) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    results = []
    for task in tasks:
        messages = [{"role": "user", "content": prompt(task)}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        started = time.time()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=2800,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        prediction = extract_sql(response)
        row = {**task, "prediction": prediction, "latency_seconds": time.time() - started}
        row.update(evaluate_prediction(task, prediction))
        results.append(row)
        print(json.dumps({k: row[k] for k in ("id", "em", "gm_proxy", "modify_better_proxy", "prediction_parseable", "latency_seconds")}))
    return results


def summarize(results: list[dict], stats: list[dict], mode: str, model_id: str) -> dict:
    summary = {
        "scope": "toy proxy; not the unreleased Squirrel benchmark",
        "mode": mode,
        "model": model_id if mode == "gpu" else "oracle-smoke-test",
        "tasks": len(results),
        "syntax_tasks": sum(r["kind"] == "syntax" for r in results),
        "semantic_tasks": sum(r["kind"] == "semantic" for r in results),
    }
    for key in ("lines", "tokens", "functions", "ast_depth", "ast_width"):
        summary[f"mean_{key}"] = sum(s[key] for s in stats) / len(stats)
    for kind in ("all", "syntax", "semantic"):
        subset = results if kind == "all" else [r for r in results if r["kind"] == kind]
        for metric in ("em", "gm_proxy", "modify_better_proxy", "prediction_parseable"):
            summary[f"{kind}_{metric}_pct"] = 100 * sum(r[metric] for r in subset) / len(subset)
    if mode == "gpu":
        summary["mean_latency_seconds"] = sum(r["latency_seconds"] for r in results) / len(results)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "gpu"), default="smoke")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--output-dir", default="repro_579/outputs/toy")
    parser.add_argument("--space-id")
    parser.add_argument("--push-repo")
    args = parser.parse_args()

    tasks = build_tasks()
    stats = [tree_stats(t["gold_sql"]) for t in tasks]
    if args.mode == "smoke":
        results = []
        for task in tasks[:2]:
            row = {**task, "prediction": task["gold_sql"], "latency_seconds": 0.0}
            row.update(evaluate_prediction(task, row["prediction"]))
            results.append(row)
        stats = stats[:2]
    else:
        results = run_model(tasks, args.model_id)

    summary = summarize(results, stats, args.mode, args.model_id)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    (out / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY " + json.dumps(summary, sort_keys=True))

    import trackio
    run = trackio.init(
        project="paper579-squirrel-toy",
        name=f"{args.mode}-{args.model_id.split('/')[-1]}",
        config={"model": args.model_id, "scope": "toy", "task_count": len(results)},
        space_id=args.space_id,
    )
    for index, row in enumerate(results):
        trackio.log({
            "task_index": index,
            "gm_proxy": row["gm_proxy"],
            "modify_better_proxy": row["modify_better_proxy"],
            "prediction_parseable": row["prediction_parseable"],
            "latency_seconds": row["latency_seconds"],
        })
    trackio.finish()

    if args.push_repo:
        from huggingface_hub import HfApi
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required for --push-repo")
        api = HfApi(token=token)
        api.create_repo(args.push_repo, repo_type="dataset", exist_ok=True)
        api.upload_folder(folder_path=out, repo_id=args.push_repo, repo_type="dataset", path_in_repo="gpu-run")
        api.upload_file(path_or_fileobj=Path(__file__).read_bytes(), path_in_repo="toy_gpu_eval.py", repo_id=args.push_repo, repo_type="dataset")
        print(f"PUSHED https://huggingface.co/datasets/{args.push_repo}")


if __name__ == "__main__":
    main()

````


````output
error: Failed to initialize cache at `/Users/binzhang/.cache/uv`
  Caused by: failed to open file `/Users/binzhang/.cache/uv/sdists-v9/.git`: Operation not permitted (os error 1)

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_1c00423aaea5", "created_at": "2026-07-16T00:11:11+00:00", "title": "Run: env toy_gpu_eval.py (exit 2)", "command": ["env", "UV_CACHE_DIR=/tmp/uv-cache", "uv", "run", "repro_579/toy_gpu_eval.py", "--mode", "smoke", "--output-dir", "repro_579/outputs/toy_smoke"], "exit_code": 2, "duration_s": 5.769}
-->
````bash
$ env UV_CACHE_DIR=/tmp/uv-cache uv run repro_579/toy_gpu_eval.py --mode smoke --output-dir repro_579/outputs/toy_smoke
````

exit 2 · 5.8s


````python title=toy_gpu_eval.py
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate>=1.6",
#   "huggingface-hub>=0.31",
#   "sqlglot>=26.0",
#   "torch>=2.5",
#   "trackio>=0.3",
#   "transformers>=4.51",
# ]
# ///
"""Deterministic toy enterprise-SQL benchmark and optional GPU model evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, deque
from pathlib import Path

import sqlglot
from sqlglot import exp

DOMAINS = ["finance", "ecommerce", "healthcare", "logistics", "advertising", "travel"]


def make_sql(domain: str, idx: int) -> str:
    measures = ["amount", "quantity", "cost", "revenue", "duration", "score"]
    lines = ["WITH"]
    for stage in range(7):
        name = f"{domain}_stage_{stage}"
        source = f"{domain}_events" if stage == 0 else f"{domain}_stage_{stage - 1}"
        prefix = "" if stage == 0 else ","
        lines.extend([
            f"{prefix}{name} AS (",
            "    SELECT",
            "        t.entity_id,",
            "        t.event_date,",
            "        t.region_id,",
            "        d.segment_name,",
            f"        SUM(t.{measures[stage % len(measures)]}) AS metric_{stage},",
            f"        COUNT(DISTINCT t.event_id) AS events_{stage},",
            f"        AVG(t.{measures[(stage + 1) % len(measures)]}) AS average_{stage},",
            "        ROW_NUMBER() OVER (",
            "            PARTITION BY t.entity_id, t.event_date",
            f"            ORDER BY SUM(t.{measures[stage % len(measures)]}) DESC",
            f"        ) AS rank_{stage}",
            f"    FROM {source} AS t",
            f"    LEFT JOIN {domain}_dimensions AS d",
            "        ON t.entity_id = d.entity_id",
            "       AND t.region_id = d.region_id",
            f"    WHERE t.event_date >= DATE '2026-0{(idx % 6) + 1}-01'",
            "      AND t.is_valid = 1",
            "    GROUP BY t.entity_id, t.event_date, t.region_id, d.segment_name",
            ")",
        ])
    lines.extend([
        "SELECT",
        "    entity_id,",
        "    event_date,",
        "    region_id,",
        "    segment_name,",
        "    metric_6,",
        "    events_6,",
        "    average_6,",
        "    CASE WHEN rank_6 = 1 THEN 'leader' ELSE 'other' END AS cohort",
        f"FROM {domain}_stage_6",
        "WHERE rank_6 <= 10",
        "ORDER BY event_date, region_id, metric_6 DESC",
        ";",
    ])
    return "\n".join(lines)


def inject(sql: str, kind: str) -> tuple[str, str]:
    if kind == "syntax":
        buggy = sql.replace("SUM(t.amount) AS metric_0", "SUM(t.amount AS metric_0", 1)
        return buggy, "Parser error near metric_0: expected a closing parenthesis."
    buggy = sql.replace("LEFT JOIN", "INNER JOIN", 1)
    return buggy, "Rows without a matching dimension record disappeared from the result. Preserve them."


def canonical(sql: str) -> str | None:
    try:
        return sqlglot.parse_one(sql, read="hive").sql(dialect="hive", pretty=False)
    except Exception:
        return None


def tree_stats(sql: str) -> dict[str, float]:
    tree = sqlglot.parse_one(sql, read="hive")
    by_depth: Counter[int] = Counter()
    queue = deque([(tree, 0)])
    while queue:
        node, depth = queue.popleft()
        by_depth[depth] += 1
        for child in node.iter_expressions():
            queue.append((child, depth + 1))
    return {
        "lines": len(sql.splitlines()),
        "tokens": len(re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|<>|<=|>=|!=|\S", sql)),
        "functions": sum(1 for node in tree.walk() if isinstance(node, exp.Func)),
        "ast_depth": max(by_depth),
        "ast_width": max(by_depth.values()),
    }


def build_tasks() -> list[dict]:
    tasks = []
    for idx, domain in enumerate(DOMAINS):
        gold = make_sql(domain, idx)
        for kind in ("syntax", "semantic"):
            buggy, issue = inject(gold, kind)
            tasks.append({
                "id": f"{domain}-{kind}",
                "domain": domain,
                "kind": kind,
                "issue": issue,
                "buggy_sql": buggy,
                "gold_sql": gold,
            })
    return tasks


def extract_sql(text: str) -> str:
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.I | re.S)
    return (match.group(1) if match else text).strip()


def evaluate_prediction(task: dict, prediction: str) -> dict:
    pred_canon = canonical(prediction)
    gold_canon = canonical(task["gold_sql"])
    buggy_canon = canonical(task["buggy_sql"])
    gm = int(pred_canon is not None and pred_canon == gold_canon)
    em = int(prediction.strip() == task["gold_sql"].strip())
    if pred_canon is None:
        mb = 0
    elif buggy_canon is None:
        mb = 1
    else:
        import difflib
        before = difflib.SequenceMatcher(None, buggy_canon, gold_canon).ratio()
        after = difflib.SequenceMatcher(None, pred_canon, gold_canon).ratio()
        mb = int(after > before)
    return {"em": em, "gm_proxy": gm, "modify_better_proxy": mb, "prediction_parseable": int(pred_canon is not None)}


def prompt(task: dict) -> str:
    return f"""You are debugging a long Hive ETL SQL script.
Issue report: {task['issue']}
Make the smallest correction that fixes the issue. Return only the complete corrected SQL.

```sql
{task['buggy_sql']}
```"""


def run_model(tasks: list[dict], model_id: str) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    results = []
    for task in tasks:
        messages = [{"role": "user", "content": prompt(task)}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        started = time.time()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=2800,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        prediction = extract_sql(response)
        row = {**task, "prediction": prediction, "latency_seconds": time.time() - started}
        row.update(evaluate_prediction(task, prediction))
        results.append(row)
        print(json.dumps({k: row[k] for k in ("id", "em", "gm_proxy", "modify_better_proxy", "prediction_parseable", "latency_seconds")}))
    return results


def summarize(results: list[dict], stats: list[dict], mode: str, model_id: str) -> dict:
    summary = {
        "scope": "toy proxy; not the unreleased Squirrel benchmark",
        "mode": mode,
        "model": model_id if mode == "gpu" else "oracle-smoke-test",
        "tasks": len(results),
        "syntax_tasks": sum(r["kind"] == "syntax" for r in results),
        "semantic_tasks": sum(r["kind"] == "semantic" for r in results),
    }
    for key in ("lines", "tokens", "functions", "ast_depth", "ast_width"):
        summary[f"mean_{key}"] = sum(s[key] for s in stats) / len(stats)
    for kind in ("all", "syntax", "semantic"):
        subset = results if kind == "all" else [r for r in results if r["kind"] == kind]
        for metric in ("em", "gm_proxy", "modify_better_proxy", "prediction_parseable"):
            summary[f"{kind}_{metric}_pct"] = 100 * sum(r[metric] for r in subset) / len(subset)
    if mode == "gpu":
        summary["mean_latency_seconds"] = sum(r["latency_seconds"] for r in results) / len(results)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "gpu"), default="smoke")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--output-dir", default="repro_579/outputs/toy")
    parser.add_argument("--space-id")
    parser.add_argument("--push-repo")
    args = parser.parse_args()

    tasks = build_tasks()
    stats = [tree_stats(t["gold_sql"]) for t in tasks]
    if args.mode == "smoke":
        results = []
        for task in tasks[:2]:
            row = {**task, "prediction": task["gold_sql"], "latency_seconds": 0.0}
            row.update(evaluate_prediction(task, row["prediction"]))
            results.append(row)
        stats = stats[:2]
    else:
        results = run_model(tasks, args.model_id)

    summary = summarize(results, stats, args.mode, args.model_id)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    (out / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY " + json.dumps(summary, sort_keys=True))

    import trackio
    run = trackio.init(
        project="paper579-squirrel-toy",
        name=f"{args.mode}-{args.model_id.split('/')[-1]}",
        config={"model": args.model_id, "scope": "toy", "task_count": len(results)},
        space_id=args.space_id,
    )
    for index, row in enumerate(results):
        trackio.log({
            "task_index": index,
            "gm_proxy": row["gm_proxy"],
            "modify_better_proxy": row["modify_better_proxy"],
            "prediction_parseable": row["prediction_parseable"],
            "latency_seconds": row["latency_seconds"],
        })
    trackio.finish()

    if args.push_repo:
        from huggingface_hub import HfApi
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required for --push-repo")
        api = HfApi(token=token)
        api.create_repo(args.push_repo, repo_type="dataset", exist_ok=True)
        api.upload_folder(folder_path=out, repo_id=args.push_repo, repo_type="dataset", path_in_repo="gpu-run")
        api.upload_file(path_or_fileobj=Path(__file__).read_bytes(), path_in_repo="toy_gpu_eval.py", repo_id=args.push_repo, repo_type="dataset")
        print(f"PUSHED https://huggingface.co/datasets/{args.push_repo}")


if __name__ == "__main__":
    main()

````


````output
WARN Retry attempt #0. Sleeping 1.320359674s before the next attempt
WARN Retry attempt #0. Sleeping 1.264605478s before the next attempt
WARN Retry attempt #0. Sleeping 1.924979363s before the next attempt
WARN Retry attempt #0. Sleeping 1.395354216s before the next attempt
WARN Retry attempt #0. Sleeping 1.492712865s before the next attempt
WARN Retry attempt #0. Sleeping 1.39161421s before the next attempt
WARN Retry attempt #1. Sleeping 3.56662341s before the next attempt
WARN Retry attempt #1. Sleeping 1.80024393s before the next attempt
WARN Retry attempt #1. Sleeping 3.835587216s before the next attempt
WARN Retry attempt #1. Sleeping 3.698762436s before the next attempt
WARN Retry attempt #1. Sleeping 2.375962514s before the next attempt
WARN Retry attempt #1. Sleeping 3.302109721s before the next attempt
WARN Retry attempt #2. Sleeping 2.59246437s before the next attempt
WARN Retry attempt #2. Sleeping 5.73345311s before the next attempt
WARN Retry attempt #2. Sleeping 1.371216684s before the next attempt
WARN Retry attempt #2. Sleeping 2.909520084s before the next attempt
WARN Retry attempt #2. Sleeping 3.112814321s before the next attempt
WARN Retry attempt #2. Sleeping 4.678134963s before the next attempt
error: Request failed after 3 retries in 5.7s
  Caused by: Failed to fetch: `https://pypi.org/simple/sqlglot/`
  Caused by: error sending request for url (https://pypi.org/simple/sqlglot/)
  Caused by: client error (Connect)
  Caused by: dns error
  Caused by: failed to lookup address information: nodename nor servname provided, or not known

````


---
<!-- trackio-cell
{"type": "dashboard", "id": "cell_6be98ba15d37", "created_at": "2026-07-16T00:12:56+00:00", "title": "Dashboard: paper579-squirrel-toy", "dashboard_project": "paper579-squirrel-toy"}
-->
**🎯 Trackio dashboard** `paper579-squirrel-toy`

trackio-local-dashboard://paper579-squirrel-toy


---
<!-- trackio-cell
{"type": "dashboard", "id": "cell_107b38be6dc9", "created_at": "2026-07-16T00:12:56+00:00", "title": "Dashboard: paper579-squirrel-toy", "dashboard_project": "paper579-squirrel-toy"}
-->
**🎯 Trackio dashboard** `paper579-squirrel-toy`

trackio-local-dashboard://paper579-squirrel-toy


---
<!-- trackio-cell
{"type": "code", "id": "cell_8f9f14b698cd", "created_at": "2026-07-16T00:12:56+00:00", "title": "Run: env toy_gpu_eval.py (exit 0)", "command": ["env", "UV_CACHE_DIR=/tmp/uv-cache", "uv", "run", "repro_579/toy_gpu_eval.py", "--mode", "smoke", "--output-dir", "repro_579/outputs/toy_smoke"], "exit_code": 0, "duration_s": 96.131}
-->
````bash
$ env UV_CACHE_DIR=/tmp/uv-cache uv run repro_579/toy_gpu_eval.py --mode smoke --output-dir repro_579/outputs/toy_smoke
````

exit 0 · 96.1s


````python title=toy_gpu_eval.py
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate>=1.6",
#   "huggingface-hub>=0.31",
#   "sqlglot>=26.0",
#   "torch>=2.5",
#   "trackio>=0.3",
#   "transformers>=4.51",
# ]
# ///
"""Deterministic toy enterprise-SQL benchmark and optional GPU model evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, deque
from pathlib import Path

import sqlglot
from sqlglot import exp

DOMAINS = ["finance", "ecommerce", "healthcare", "logistics", "advertising", "travel"]


def make_sql(domain: str, idx: int) -> str:
    measures = ["amount", "quantity", "cost", "revenue", "duration", "score"]
    lines = ["WITH"]
    for stage in range(7):
        name = f"{domain}_stage_{stage}"
        source = f"{domain}_events" if stage == 0 else f"{domain}_stage_{stage - 1}"
        prefix = "" if stage == 0 else ","
        lines.extend([
            f"{prefix}{name} AS (",
            "    SELECT",
            "        t.entity_id,",
            "        t.event_date,",
            "        t.region_id,",
            "        d.segment_name,",
            f"        SUM(t.{measures[stage % len(measures)]}) AS metric_{stage},",
            f"        COUNT(DISTINCT t.event_id) AS events_{stage},",
            f"        AVG(t.{measures[(stage + 1) % len(measures)]}) AS average_{stage},",
            "        ROW_NUMBER() OVER (",
            "            PARTITION BY t.entity_id, t.event_date",
            f"            ORDER BY SUM(t.{measures[stage % len(measures)]}) DESC",
            f"        ) AS rank_{stage}",
            f"    FROM {source} AS t",
            f"    LEFT JOIN {domain}_dimensions AS d",
            "        ON t.entity_id = d.entity_id",
            "       AND t.region_id = d.region_id",
            f"    WHERE t.event_date >= DATE '2026-0{(idx % 6) + 1}-01'",
            "      AND t.is_valid = 1",
            "    GROUP BY t.entity_id, t.event_date, t.region_id, d.segment_name",
            ")",
        ])
    lines.extend([
        "SELECT",
        "    entity_id,",
        "    event_date,",
        "    region_id,",
        "    segment_name,",
        "    metric_6,",
        "    events_6,",
        "    average_6,",
        "    CASE WHEN rank_6 = 1 THEN 'leader' ELSE 'other' END AS cohort",
        f"FROM {domain}_stage_6",
        "WHERE rank_6 <= 10",
        "ORDER BY event_date, region_id, metric_6 DESC",
        ";",
    ])
    return "\n".join(lines)


def inject(sql: str, kind: str) -> tuple[str, str]:
    if kind == "syntax":
        buggy = sql.replace("SUM(t.amount) AS metric_0", "SUM(t.amount AS metric_0", 1)
        return buggy, "Parser error near metric_0: expected a closing parenthesis."
    buggy = sql.replace("LEFT JOIN", "INNER JOIN", 1)
    return buggy, "Rows without a matching dimension record disappeared from the result. Preserve them."


def canonical(sql: str) -> str | None:
    try:
        return sqlglot.parse_one(sql, read="hive").sql(dialect="hive", pretty=False)
    except Exception:
        return None


def tree_stats(sql: str) -> dict[str, float]:
    tree = sqlglot.parse_one(sql, read="hive")
    by_depth: Counter[int] = Counter()
    queue = deque([(tree, 0)])
    while queue:
        node, depth = queue.popleft()
        by_depth[depth] += 1
        for child in node.iter_expressions():
            queue.append((child, depth + 1))
    return {
        "lines": len(sql.splitlines()),
        "tokens": len(re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|<>|<=|>=|!=|\S", sql)),
        "functions": sum(1 for node in tree.walk() if isinstance(node, exp.Func)),
        "ast_depth": max(by_depth),
        "ast_width": max(by_depth.values()),
    }


def build_tasks() -> list[dict]:
    tasks = []
    for idx, domain in enumerate(DOMAINS):
        gold = make_sql(domain, idx)
        for kind in ("syntax", "semantic"):
            buggy, issue = inject(gold, kind)
            tasks.append({
                "id": f"{domain}-{kind}",
                "domain": domain,
                "kind": kind,
                "issue": issue,
                "buggy_sql": buggy,
                "gold_sql": gold,
            })
    return tasks


def extract_sql(text: str) -> str:
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.I | re.S)
    return (match.group(1) if match else text).strip()


def evaluate_prediction(task: dict, prediction: str) -> dict:
    pred_canon = canonical(prediction)
    gold_canon = canonical(task["gold_sql"])
    buggy_canon = canonical(task["buggy_sql"])
    gm = int(pred_canon is not None and pred_canon == gold_canon)
    em = int(prediction.strip() == task["gold_sql"].strip())
    if pred_canon is None:
        mb = 0
    elif buggy_canon is None:
        mb = 1
    else:
        import difflib
        before = difflib.SequenceMatcher(None, buggy_canon, gold_canon).ratio()
        after = difflib.SequenceMatcher(None, pred_canon, gold_canon).ratio()
        mb = int(after > before)
    return {"em": em, "gm_proxy": gm, "modify_better_proxy": mb, "prediction_parseable": int(pred_canon is not None)}


def prompt(task: dict) -> str:
    return f"""You are debugging a long Hive ETL SQL script.
Issue report: {task['issue']}
Make the smallest correction that fixes the issue. Return only the complete corrected SQL.

```sql
{task['buggy_sql']}
```"""


def run_model(tasks: list[dict], model_id: str) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    results = []
    for task in tasks:
        messages = [{"role": "user", "content": prompt(task)}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        started = time.time()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=2800,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        prediction = extract_sql(response)
        row = {**task, "prediction": prediction, "latency_seconds": time.time() - started}
        row.update(evaluate_prediction(task, prediction))
        results.append(row)
        print(json.dumps({k: row[k] for k in ("id", "em", "gm_proxy", "modify_better_proxy", "prediction_parseable", "latency_seconds")}))
    return results


def summarize(results: list[dict], stats: list[dict], mode: str, model_id: str) -> dict:
    summary = {
        "scope": "toy proxy; not the unreleased Squirrel benchmark",
        "mode": mode,
        "model": model_id if mode == "gpu" else "oracle-smoke-test",
        "tasks": len(results),
        "syntax_tasks": sum(r["kind"] == "syntax" for r in results),
        "semantic_tasks": sum(r["kind"] == "semantic" for r in results),
    }
    for key in ("lines", "tokens", "functions", "ast_depth", "ast_width"):
        summary[f"mean_{key}"] = sum(s[key] for s in stats) / len(stats)
    for kind in ("all", "syntax", "semantic"):
        subset = results if kind == "all" else [r for r in results if r["kind"] == kind]
        for metric in ("em", "gm_proxy", "modify_better_proxy", "prediction_parseable"):
            summary[f"{kind}_{metric}_pct"] = 100 * sum(r[metric] for r in subset) / len(subset)
    if mode == "gpu":
        summary["mean_latency_seconds"] = sum(r["latency_seconds"] for r in results) / len(results)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "gpu"), default="smoke")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--output-dir", default="repro_579/outputs/toy")
    parser.add_argument("--space-id")
    parser.add_argument("--push-repo")
    args = parser.parse_args()

    tasks = build_tasks()
    stats = [tree_stats(t["gold_sql"]) for t in tasks]
    if args.mode == "smoke":
        results = []
        for task in tasks[:2]:
            row = {**task, "prediction": task["gold_sql"], "latency_seconds": 0.0}
            row.update(evaluate_prediction(task, row["prediction"]))
            results.append(row)
        stats = stats[:2]
    else:
        results = run_model(tasks, args.model_id)

    summary = summarize(results, stats, args.mode, args.model_id)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    (out / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY " + json.dumps(summary, sort_keys=True))

    import trackio
    run = trackio.init(
        project="paper579-squirrel-toy",
        name=f"{args.mode}-{args.model_id.split('/')[-1]}",
        config={"model": args.model_id, "scope": "toy", "task_count": len(results)},
        space_id=args.space_id,
    )
    for index, row in enumerate(results):
        trackio.log({
            "task_index": index,
            "gm_proxy": row["gm_proxy"],
            "modify_better_proxy": row["modify_better_proxy"],
            "prediction_parseable": row["prediction_parseable"],
            "latency_seconds": row["latency_seconds"],
        })
    trackio.finish()

    if args.push_repo:
        from huggingface_hub import HfApi
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required for --push-repo")
        api = HfApi(token=token)
        api.create_repo(args.push_repo, repo_type="dataset", exist_ok=True)
        api.upload_folder(folder_path=out, repo_id=args.push_repo, repo_type="dataset", path_in_repo="gpu-run")
        api.upload_file(path_or_fileobj=Path(__file__).read_bytes(), path_in_repo="toy_gpu_eval.py", repo_id=args.push_repo, repo_type="dataset")
        print(f"PUSHED https://huggingface.co/datasets/{args.push_repo}")


if __name__ == "__main__":
    main()

````


````output
WARN Fixing invalid version specifier by removing stray quotes (before: `>= '2.7'`; after: `>= 2.7`)
Downloading torch (106.1MiB)
Downloading pillow (4.6MiB)
Downloading hf-xet (3.7MiB)
Downloading uvloop (1.3MiB)
Downloading pygments (1.2MiB)
Downloading numpy (5.1MiB)
Downloading transformers (11.1MiB)
Downloading tokenizers (2.8MiB)
Downloading sympy (6.0MiB)
Downloading networkx (2.0MiB)
Downloading trackio (1.9MiB)
 Downloaded pygments
 Downloaded uvloop
 Downloaded trackio
 Downloaded networkx
 Downloaded tokenizers
 Downloaded hf-xet
 Downloaded pillow
 Downloaded numpy
 Downloaded sympy
 Downloaded transformers
 Downloaded torch
Installed 50 packages in 247ms
SUMMARY {"all_em_pct": 100.0, "all_gm_proxy_pct": 100.0, "all_modify_better_proxy_pct": 100.0, "all_prediction_parseable_pct": 100.0, "mean_ast_depth": 10.0, "mean_ast_width": 170.0, "mean_functions": 58.0, "mean_lines": 161.0, "mean_tokens": 1005.0, "mode": "smoke", "model": "oracle-smoke-test", "scope": "toy proxy; not the unreleased Squirrel benchmark", "semantic_em_pct": 100.0, "semantic_gm_proxy_pct": 100.0, "semantic_modify_better_proxy_pct": 100.0, "semantic_prediction_parseable_pct": 100.0, "semantic_tasks": 1, "syntax_em_pct": 100.0, "syntax_gm_proxy_pct": 100.0, "syntax_modify_better_proxy_pct": 100.0, "syntax_prediction_parseable_pct": 100.0, "syntax_tasks": 1, "tasks": 2}
* Trackio project initialized: paper579-squirrel-toy
* Trackio metrics logged to: /Users/binzhang/.cache/huggingface/trackio
* View dashboard by running in your terminal:
[1m[38;5;208mtrackio show --project "paper579-squirrel-toy"[0m
* or by running in Python: trackio.show(project="paper579-squirrel-toy")
* Apple Silicon detected, enabling automatic GPU/system metrics logging
* psutil detected, enabling automatic CPU/system metrics logging
* Created new run: smoke-Qwen2.5-Coder-7B-Instruct
* Run finished. Uploading logs to Trackio (please wait...)

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_a906f58b7924", "created_at": "2026-07-16T00:12:56+00:00", "title": "Artifact: tasks.jsonl", "path": "repro_579/outputs/toy_smoke/tasks.jsonl", "size": 124514, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `repro_579/outputs/toy_smoke/tasks.jsonl` · dataset · 0.1 MB

https://huggingface.co/buckets/binzhango/repro-beyond-text-to-sql-squirrel-artifacts#logbook-files/repro_579/outputs/toy_smoke/tasks.jsonl


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_80fc9b59a5f9", "created_at": "2026-07-16T00:12:56+00:00", "title": "Artifact: predictions.jsonl", "path": "repro_579/outputs/toy_smoke/predictions.jsonl", "size": 30953, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `repro_579/outputs/toy_smoke/predictions.jsonl` · dataset · 31.0 kB

https://huggingface.co/buckets/binzhango/repro-beyond-text-to-sql-squirrel-artifacts#logbook-files/repro_579/outputs/toy_smoke/predictions.jsonl


---
<!-- trackio-cell
{"type": "code", "id": "cell_c1220b79b533", "created_at": "2026-07-16T00:12:56+00:00", "title": "Run: env toy_gpu_eval.py (exit 0)", "command": ["env", "UV_CACHE_DIR=/tmp/uv-cache", "uv", "run", "repro_579/toy_gpu_eval.py", "--mode", "smoke", "--output-dir", "repro_579/outputs/toy_smoke"], "exit_code": 0, "duration_s": 51.081}
-->
````bash
$ env UV_CACHE_DIR=/tmp/uv-cache uv run repro_579/toy_gpu_eval.py --mode smoke --output-dir repro_579/outputs/toy_smoke
````

exit 0 · 51.1s


````python title=toy_gpu_eval.py
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate>=1.6",
#   "huggingface-hub>=0.31",
#   "sqlglot>=26.0",
#   "torch>=2.5",
#   "trackio>=0.3",
#   "transformers>=4.51",
# ]
# ///
"""Deterministic toy enterprise-SQL benchmark and optional GPU model evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, deque
from pathlib import Path

import sqlglot
from sqlglot import exp

DOMAINS = ["finance", "ecommerce", "healthcare", "logistics", "advertising", "travel"]


def make_sql(domain: str, idx: int) -> str:
    measures = ["amount", "quantity", "cost", "revenue", "duration", "score"]
    lines = ["WITH"]
    for stage in range(7):
        name = f"{domain}_stage_{stage}"
        source = f"{domain}_events" if stage == 0 else f"{domain}_stage_{stage - 1}"
        prefix = "" if stage == 0 else ","
        lines.extend([
            f"{prefix}{name} AS (",
            "    SELECT",
            "        t.entity_id,",
            "        t.event_date,",
            "        t.region_id,",
            "        d.segment_name,",
            f"        SUM(t.{measures[stage % len(measures)]}) AS metric_{stage},",
            f"        COUNT(DISTINCT t.event_id) AS events_{stage},",
            f"        AVG(t.{measures[(stage + 1) % len(measures)]}) AS average_{stage},",
            "        ROW_NUMBER() OVER (",
            "            PARTITION BY t.entity_id, t.event_date",
            f"            ORDER BY SUM(t.{measures[stage % len(measures)]}) DESC",
            f"        ) AS rank_{stage}",
            f"    FROM {source} AS t",
            f"    LEFT JOIN {domain}_dimensions AS d",
            "        ON t.entity_id = d.entity_id",
            "       AND t.region_id = d.region_id",
            f"    WHERE t.event_date >= DATE '2026-0{(idx % 6) + 1}-01'",
            "      AND t.is_valid = 1",
            "    GROUP BY t.entity_id, t.event_date, t.region_id, d.segment_name",
            ")",
        ])
    lines.extend([
        "SELECT",
        "    entity_id,",
        "    event_date,",
        "    region_id,",
        "    segment_name,",
        "    metric_6,",
        "    events_6,",
        "    average_6,",
        "    CASE WHEN rank_6 = 1 THEN 'leader' ELSE 'other' END AS cohort",
        f"FROM {domain}_stage_6",
        "WHERE rank_6 <= 10",
        "ORDER BY event_date, region_id, metric_6 DESC",
        ";",
    ])
    return "\n".join(lines)


def inject(sql: str, kind: str) -> tuple[str, str]:
    if kind == "syntax":
        buggy = sql.replace("SUM(t.amount) AS metric_0", "SUM(t.amount AS metric_0", 1)
        return buggy, "Parser error near metric_0: expected a closing parenthesis."
    buggy = sql.replace("LEFT JOIN", "INNER JOIN", 1)
    return buggy, "Rows without a matching dimension record disappeared from the result. Preserve them."


def canonical(sql: str) -> str | None:
    try:
        return sqlglot.parse_one(sql, read="hive").sql(dialect="hive", pretty=False)
    except Exception:
        return None


def tree_stats(sql: str) -> dict[str, float]:
    tree = sqlglot.parse_one(sql, read="hive")
    by_depth: Counter[int] = Counter()
    queue = deque([(tree, 0)])
    while queue:
        node, depth = queue.popleft()
        by_depth[depth] += 1
        for child in node.iter_expressions():
            queue.append((child, depth + 1))
    return {
        "lines": len(sql.splitlines()),
        "tokens": len(re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|<>|<=|>=|!=|\S", sql)),
        "functions": sum(1 for node in tree.walk() if isinstance(node, exp.Func)),
        "ast_depth": max(by_depth),
        "ast_width": max(by_depth.values()),
    }


def build_tasks() -> list[dict]:
    tasks = []
    for idx, domain in enumerate(DOMAINS):
        gold = make_sql(domain, idx)
        for kind in ("syntax", "semantic"):
            buggy, issue = inject(gold, kind)
            tasks.append({
                "id": f"{domain}-{kind}",
                "domain": domain,
                "kind": kind,
                "issue": issue,
                "buggy_sql": buggy,
                "gold_sql": gold,
            })
    return tasks


def extract_sql(text: str) -> str:
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.I | re.S)
    return (match.group(1) if match else text).strip()


def evaluate_prediction(task: dict, prediction: str) -> dict:
    pred_canon = canonical(prediction)
    gold_canon = canonical(task["gold_sql"])
    buggy_canon = canonical(task["buggy_sql"])
    gm = int(pred_canon is not None and pred_canon == gold_canon)
    em = int(prediction.strip() == task["gold_sql"].strip())
    if pred_canon is None:
        mb = 0
    elif buggy_canon is None:
        mb = 1
    else:
        import difflib
        before = difflib.SequenceMatcher(None, buggy_canon, gold_canon).ratio()
        after = difflib.SequenceMatcher(None, pred_canon, gold_canon).ratio()
        mb = int(after > before)
    return {"em": em, "gm_proxy": gm, "modify_better_proxy": mb, "prediction_parseable": int(pred_canon is not None)}


def prompt(task: dict) -> str:
    return f"""You are debugging a long Hive ETL SQL script.
Issue report: {task['issue']}
Make the smallest correction that fixes the issue. Return only the complete corrected SQL.

```sql
{task['buggy_sql']}
```"""


def run_model(tasks: list[dict], model_id: str) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    results = []
    for task in tasks:
        messages = [{"role": "user", "content": prompt(task)}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        started = time.time()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=2800,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        prediction = extract_sql(response)
        row = {**task, "prediction": prediction, "latency_seconds": time.time() - started}
        row.update(evaluate_prediction(task, prediction))
        results.append(row)
        print(json.dumps({k: row[k] for k in ("id", "em", "gm_proxy", "modify_better_proxy", "prediction_parseable", "latency_seconds")}))
    return results


def summarize(results: list[dict], stats: list[dict], mode: str, model_id: str) -> dict:
    summary = {
        "scope": "toy proxy; not the unreleased Squirrel benchmark",
        "mode": mode,
        "model": model_id if mode == "gpu" else "oracle-smoke-test",
        "tasks": len(results),
        "syntax_tasks": sum(r["kind"] == "syntax" for r in results),
        "semantic_tasks": sum(r["kind"] == "semantic" for r in results),
    }
    for key in ("lines", "tokens", "functions", "ast_depth", "ast_width"):
        summary[f"mean_{key}"] = sum(s[key] for s in stats) / len(stats)
    for kind in ("all", "syntax", "semantic"):
        subset = results if kind == "all" else [r for r in results if r["kind"] == kind]
        for metric in ("em", "gm_proxy", "modify_better_proxy", "prediction_parseable"):
            summary[f"{kind}_{metric}_pct"] = 100 * sum(r[metric] for r in subset) / len(subset)
    if mode == "gpu":
        summary["mean_latency_seconds"] = sum(r["latency_seconds"] for r in results) / len(results)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "gpu"), default="smoke")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--output-dir", default="repro_579/outputs/toy")
    parser.add_argument("--space-id")
    parser.add_argument("--push-repo")
    args = parser.parse_args()

    tasks = build_tasks()
    stats = [tree_stats(t["gold_sql"]) for t in tasks]
    if args.mode == "smoke":
        results = []
        for task in tasks[:2]:
            row = {**task, "prediction": task["gold_sql"], "latency_seconds": 0.0}
            row.update(evaluate_prediction(task, row["prediction"]))
            results.append(row)
        stats = stats[:2]
    else:
        results = run_model(tasks, args.model_id)

    summary = summarize(results, stats, args.mode, args.model_id)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    (out / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY " + json.dumps(summary, sort_keys=True))

    import trackio
    run = trackio.init(
        project="paper579-squirrel-toy",
        name=f"{args.mode}-{args.model_id.split('/')[-1]}",
        config={"model": args.model_id, "scope": "toy", "task_count": len(results)},
        space_id=args.space_id,
    )
    for index, row in enumerate(results):
        trackio.log({
            "task_index": index,
            "gm_proxy": row["gm_proxy"],
            "modify_better_proxy": row["modify_better_proxy"],
            "prediction_parseable": row["prediction_parseable"],
            "latency_seconds": row["latency_seconds"],
        })
    trackio.finish()

    if args.push_repo:
        from huggingface_hub import HfApi
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required for --push-repo")
        api = HfApi(token=token)
        api.create_repo(args.push_repo, repo_type="dataset", exist_ok=True)
        api.upload_folder(folder_path=out, repo_id=args.push_repo, repo_type="dataset", path_in_repo="gpu-run")
        api.upload_file(path_or_fileobj=Path(__file__).read_bytes(), path_in_repo="toy_gpu_eval.py", repo_id=args.push_repo, repo_type="dataset")
        print(f"PUSHED https://huggingface.co/datasets/{args.push_repo}")


if __name__ == "__main__":
    main()

````


````output
SUMMARY {"all_em_pct": 100.0, "all_gm_proxy_pct": 100.0, "all_modify_better_proxy_pct": 100.0, "all_prediction_parseable_pct": 100.0, "mean_ast_depth": 10.0, "mean_ast_width": 170.0, "mean_functions": 58.0, "mean_lines": 161.0, "mean_tokens": 1005.0, "mode": "smoke", "model": "oracle-smoke-test", "scope": "toy proxy; not the unreleased Squirrel benchmark", "semantic_em_pct": 100.0, "semantic_gm_proxy_pct": 100.0, "semantic_modify_better_proxy_pct": 100.0, "semantic_prediction_parseable_pct": 100.0, "semantic_tasks": 1, "syntax_em_pct": 100.0, "syntax_gm_proxy_pct": 100.0, "syntax_modify_better_proxy_pct": 100.0, "syntax_prediction_parseable_pct": 100.0, "syntax_tasks": 1, "tasks": 2}
* Trackio project initialized: paper579-squirrel-toy
* Trackio metrics logged to: /Users/binzhang/.cache/huggingface/trackio
* View dashboard by running in your terminal:
[1m[38;5;208mtrackio show --project "paper579-squirrel-toy"[0m
* or by running in Python: trackio.show(project="paper579-squirrel-toy")
* Apple Silicon detected, enabling automatic GPU/system metrics logging
* psutil detected, enabling automatic CPU/system metrics logging
* Created new run: smoke-Qwen2.5-Coder-7B-Instruct
* Run finished. Uploading logs to Trackio (please wait...)

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_87153eca973b", "created_at": "2026-07-16T00:12:56+00:00", "title": "Artifact: tasks.jsonl", "path": "repro_579/outputs/toy_smoke/tasks.jsonl", "size": 124514, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `repro_579/outputs/toy_smoke/tasks.jsonl` · dataset · 0.1 MB

https://huggingface.co/buckets/binzhango/repro-beyond-text-to-sql-squirrel-artifacts#logbook-files/repro_579/outputs/toy_smoke/tasks.jsonl


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_13d18ead8b4b", "created_at": "2026-07-16T00:12:56+00:00", "title": "Artifact: predictions.jsonl", "path": "repro_579/outputs/toy_smoke/predictions.jsonl", "size": 30953, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `repro_579/outputs/toy_smoke/predictions.jsonl` · dataset · 31.0 kB

https://huggingface.co/buckets/binzhango/repro-beyond-text-to-sql-squirrel-artifacts#logbook-files/repro_579/outputs/toy_smoke/predictions.jsonl


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_68b82a80ad35", "created_at": "2026-07-16T00:23:39+00:00", "title": "GPU Job configuration and live artifacts"}
-->
Hugging Face GPU Job: https://huggingface.co/jobs/binzhango/6a58221cb1669a49bf07668b. Hardware: 1× Nvidia A10G-small (24 GB), Qwen/Qwen2.5-Coder-7B-Instruct in bfloat16, greedy decoding, max 2,800 new tokens, 12/985 tasks (1.2% scale), two-hour safety timeout. Official pricing is $1.00/hour; final cost is computed from actual running time. Outputs are persisted at https://huggingface.co/datasets/binzhango/paper579-squirrel-toy-repro and metrics at https://huggingface.co/spaces/binzhango/paper579-squirrel-toy-trackio.


---
<!-- trackio-cell
{"type": "code", "id": "cell_18859fa0f3ec", "created_at": "2026-07-16T00:37:19+00:00", "title": "Run: python ingest_gpu_metrics.py (exit 0)", "command": [".venv/bin/python", "repro_579/ingest_gpu_metrics.py"], "exit_code": 0, "duration_s": 0.245}
-->
````bash
$ .venv/bin/python repro_579/ingest_gpu_metrics.py
````

exit 0 · 0.2s


````python title=ingest_gpu_metrics.py
#!/usr/bin/env python3
"""Ingest the completed GPU Job metrics into local Trackio for logbook publication."""

import json
from pathlib import Path

import trackio

path = Path(__file__).parent / "outputs" / "gpu_job_metrics.json"
data = json.loads(path.read_text())
trackio.init(
    project="paper579-squirrel-toy",
    name="gpu-Qwen2.5-Coder-7B-Instruct-recovered",
    config={
        "job_id": data["job_id"],
        "model": data["model"],
        "scope": "toy",
        "hardware": data["hardware"],
    },
)
for index, row in enumerate(data["task_metrics"]):
    trackio.log({
        "task_index": index,
        "gm_proxy": row["gm_proxy"],
        "modify_better_proxy": row["modify_better_proxy"],
        "prediction_parseable": row["prediction_parseable"],
        "latency_seconds": row["latency_seconds"],
    })
trackio.finish()
print(json.dumps({k: data[k] for k in ("job_url", "tasks", "all_gm_proxy_pct", "mean_latency_seconds", "running_seconds", "estimated_cost_usd")}, indent=2))

````


````output
/Users/binzhang/vibe_coding_repo/reproduce_icml/.venv/lib/python3.13/site-packages/trackio/utils.py:27: UserWarning: trackio.init() could not inspect existing runs for project 'paper579-squirrel-toy': unable to open database file. Continuing without resume metadata.
  warnings.warn(message, *args, **kwargs)
/Users/binzhang/vibe_coding_repo/reproduce_icml/.venv/lib/python3.13/site-packages/trackio/utils.py:27: UserWarning: trackio.init() could not recover the previous step for run 'gpu-Qwen2.5-Coder-7B-Instruct-recovered': unable to open database file. Continuing from step 0.
  warnings.warn(message, *args, **kwargs)
/Users/binzhang/vibe_coding_repo/reproduce_icml/.venv/lib/python3.13/site-packages/trackio/utils.py:27: UserWarning: trackio failed to flush metric logs for run 'gpu-Qwen2.5-Coder-7B-Instruct-recovered': [Errno 1] Operation not permitted: '/Users/binzhang/.cache/huggingface/trackio/paper579-squirrel-toy.lock'. User code will continue, but this batch could not be persisted.
  warnings.warn(message, *args, **kwargs)
* Trackio project initialized: paper579-squirrel-toy
* Trackio metrics logged to: /Users/binzhang/.cache/huggingface/trackio
* View dashboard by running in your terminal:
[1m[38;5;208mtrackio show --project "paper579-squirrel-toy"[0m
* or by running in Python: trackio.show(project="paper579-squirrel-toy")
* Created new run: gpu-Qwen2.5-Coder-7B-Instruct-recovered
* Run finished. Uploading logs to Trackio (please wait...)
{
  "job_url": "https://huggingface.co/jobs/binzhango/6a58221cb1669a49bf07668b",
  "tasks": 12,
  "all_gm_proxy_pct": 100.0,
  "mean_latency_seconds": 51.42013074954351,
  "running_seconds": 780,
  "estimated_cost_usd": 0.22
}

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_0e89ca9ac0b2", "created_at": "2026-07-16T00:37:34+00:00", "title": "Run: python ingest_gpu_metrics.py (exit 0)", "command": [".venv/bin/python", "repro_579/ingest_gpu_metrics.py"], "exit_code": 0, "duration_s": 0.442}
-->
````bash
$ .venv/bin/python repro_579/ingest_gpu_metrics.py
````

exit 0 · 0.4s


````python title=ingest_gpu_metrics.py
#!/usr/bin/env python3
"""Ingest the completed GPU Job metrics into local Trackio for logbook publication."""

import json
from pathlib import Path

import trackio

path = Path(__file__).parent / "outputs" / "gpu_job_metrics.json"
data = json.loads(path.read_text())
trackio.init(
    project="paper579-squirrel-toy",
    name="gpu-Qwen2.5-Coder-7B-Instruct-recovered",
    config={
        "job_id": data["job_id"],
        "model": data["model"],
        "scope": "toy",
        "hardware": data["hardware"],
    },
)
for index, row in enumerate(data["task_metrics"]):
    trackio.log({
        "task_index": index,
        "gm_proxy": row["gm_proxy"],
        "modify_better_proxy": row["modify_better_proxy"],
        "prediction_parseable": row["prediction_parseable"],
        "latency_seconds": row["latency_seconds"],
    })
trackio.finish()
print(json.dumps({k: data[k] for k in ("job_url", "tasks", "all_gm_proxy_pct", "mean_latency_seconds", "running_seconds", "estimated_cost_usd")}, indent=2))

````


````output
* Trackio project initialized: paper579-squirrel-toy
* Trackio metrics logged to: /Users/binzhang/.cache/huggingface/trackio
* View dashboard by running in your terminal:
[1m[38;5;208mtrackio show --project "paper579-squirrel-toy"[0m
* or by running in Python: trackio.show(project="paper579-squirrel-toy")
* Created new run: gpu-Qwen2.5-Coder-7B-Instruct-recovered
* Run finished. Uploading logs to Trackio (please wait...)
{
  "job_url": "https://huggingface.co/jobs/binzhango/6a58221cb1669a49bf07668b",
  "tasks": 12,
  "all_gm_proxy_pct": 100.0,
  "mean_latency_seconds": 51.42013074954351,
  "running_seconds": 780,
  "estimated_cost_usd": 0.22
}

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_f27c5a17ab7b", "created_at": "2026-07-16T00:37:47+00:00", "title": "A10G result and upload correction"}
-->
**Completed result.** Qwen/Qwen2.5-Coder-7B-Instruct repaired all 12/12 easy synthetic cases exactly: 100% syntax GM proxy and 100% semantic GM proxy. Mean latency was 51.42 seconds per task on one A10G-small; the Job ran for 780 seconds, with an estimated cost of about USD 0.22.

**Scope guardrail.** This is a deterministic pipeline sanity check, not an estimate of performance on the unavailable official Squirrel benchmark.

**Persistence correction.** The evaluation completed, but its post-run Hub upload failed with HTTP 403 because the token injected into the Job lacked repository-creation rights. All per-task metrics were recovered from the Job logs, ingested into this logbook locally, and included in the final reproduction bundle.

[Open the completed Hugging Face Job](https://huggingface.co/jobs/binzhango/6a58221cb1669a49bf07668b)
