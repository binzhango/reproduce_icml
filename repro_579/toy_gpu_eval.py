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
        try:
            api.create_repo(args.push_repo, repo_type="dataset", exist_ok=True)
            api.upload_folder(folder_path=out, repo_id=args.push_repo, repo_type="dataset", path_in_repo="gpu-run")
            api.upload_file(path_or_fileobj=Path(__file__).read_bytes(), path_in_repo="toy_gpu_eval.py", repo_id=args.push_repo, repo_type="dataset")
            print(f"PUSHED https://huggingface.co/datasets/{args.push_repo}")
        except Exception as exc:
            print(f"PERSISTENCE_WARNING {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
