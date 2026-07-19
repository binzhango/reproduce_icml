#!/usr/bin/env python3
"""Deterministic structural preflight for a local Trackio reproduction logbook."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


CELL_RE = re.compile(r"<!--\s*trackio-cell\s*\n(\{.*?\})\s*\n-->", re.DOTALL)
URL_RE = re.compile(r"https://huggingface\.co/[^\s)>]+")


def locate_logbook(start: Path) -> Path:
    candidates = []
    if start.name == "logbook" and (start / "logbook.json").is_file():
        return start
    candidates.extend(
        [
            start / ".trackio" / "logbook",
            start / "logbook",
        ]
    )
    for candidate in candidates:
        if (candidate / "logbook.json").is_file():
            return candidate
    found = sorted(start.glob("**/.trackio/logbook/logbook.json"))
    if len(found) == 1:
        return found[0].parent
    if len(found) > 1:
        choices = "\n".join(f"  - {p.parent}" for p in found)
        raise SystemExit(f"Multiple logbooks found; pass the intended workspace:\n{choices}")
    raise SystemExit(f"No .trackio/logbook/logbook.json found under {start}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a JSON object in {path}")
    return value


def parse_cells(page: Path) -> list[dict[str, Any]]:
    text = page.read_text(encoding="utf-8")
    cells: list[dict[str, Any]] = []
    matches = list(CELL_RE.finditer(text))
    for index, match in enumerate(matches):
        try:
            meta = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        cells.append({"meta": meta, "body": text[body_start:body_end].strip(), "page": str(page)})
    return cells


def cell_title(cell: dict[str, Any]) -> str:
    return str(cell["meta"].get("title", "")).strip()


def check(name: str, passed: bool, severity: str, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "detail": detail}


def audit(workspace: Path) -> dict[str, Any]:
    logbook = locate_logbook(workspace.resolve())
    manifest = load_json(logbook / "logbook.json")
    metadata_path = logbook.parent / "metadata.json"
    metadata = load_json(metadata_path) if metadata_path.is_file() else {}

    pages = sorted((logbook / "pages").glob("**/*.md"))
    page_cells = {str(page.relative_to(logbook)): parse_cells(page) for page in pages}
    cells = [cell for group in page_cells.values() for cell in group]
    all_text = "\n".join(cell["body"] for cell in cells)

    title = str(manifest.get("title", ""))
    paper = metadata.get("paper") or manifest.get("paper") or {}
    tags = set(metadata.get("tags") or manifest.get("tags") or [])
    conclusion_key = next((key for key in page_cells if key.endswith("conclusion/page.md")), None)
    conclusion = page_cells.get(conclusion_key or "", [])
    claims = [key for key in page_cells if "/claim-" in f"/{key.lower()}" or key.lower().startswith("pages/claim-")]

    pinned = sorted(
        [cell for cell in cells if cell["meta"].get("pinned") is True],
        key=lambda cell: str(cell["meta"].get("pinned_at") or cell["meta"].get("created_at") or ""),
    )
    executive = next((cell for cell in conclusion if cell_title(cell).lower() == "executive summary"), None)
    poster = next((cell for cell in conclusion if cell_title(cell).lower() == "reproduction poster"), None)
    exec_body = executive["body"] if executive else ""

    required_rows = ["Scope", "Hardware", "Compute time", "Cost", "Outcome"]
    row_presence = {
        row: bool(re.search(rf"^\|\s*{re.escape(row)}\s*\|", exec_body, re.MULTILINE | re.IGNORECASE))
        for row in required_rows
    }

    artifact_cells = [cell for cell in cells if cell["meta"].get("type") == "artifact"]
    bucket_artifacts = [cell for cell in artifact_cells if "https://huggingface.co/buckets/" in cell["body"]]
    dashboards = [cell for cell in cells if cell["meta"].get("type") == "dashboard"]
    local_dashboards = [cell for cell in dashboards if re.search(r"\blocal\b", cell["body"], re.IGNORECASE)]
    failed_cells = [
        cell
        for cell in cells
        if cell["meta"].get("type") == "code"
        and re.search(r"(?:exit(?:_code)?\s*[:= ]\s*|\(exit\s+)([1-9]\d*)", cell["body"] + " " + cell_title(cell), re.IGNORECASE)
    ]
    failed_conclusion = [cell for cell in failed_cells if conclusion_key and cell["page"].endswith(conclusion_key)]

    checks = [
        check("descriptive_title", title.startswith("Repro - "), "error", f"title={title!r}"),
        check("paper_arxiv_id", bool(isinstance(paper, dict) and paper.get("arxiv_id")), "error", f"paper={paper!r}"),
        check("discovery_tag", "icml2026-repro" in tags, "error", f"tags={sorted(tags)!r}"),
        check("paper_tag", any(str(tag).startswith("paper-") for tag in tags), "error", f"tags={sorted(tags)!r}"),
        check("claim_pages", bool(claims), "error", f"count={len(claims)}"),
        check("conclusion_page", conclusion_key is not None, "error", conclusion_key or "missing"),
        check("executive_summary", executive is not None, "error", cell_title(executive) if executive else "missing"),
        check("executive_pinned", bool(executive and executive["meta"].get("pinned") is True), "error", "must be pinned"),
        check("scope_cost_heading", "## Scope & cost" in exec_body, "error", "literal heading required"),
        check("scope_cost_columns", "This reproduction" in exec_body and "Full replication" in exec_body, "error", "both comparison columns required"),
        check("scope_cost_rows", all(row_presence.values()), "error", f"rows={row_presence}"),
        check("poster_figure", bool(poster and poster["meta"].get("type") == "figure"), "error", "Conclusion figure titled Reproduction poster"),
        check("poster_pinned", bool(poster and poster["meta"].get("pinned") is True), "error", "must be pinned"),
        check(
            "pin_order",
            len(pinned) >= 2 and cell_title(pinned[0]).lower() == "executive summary" and cell_title(pinned[1]).lower() == "reproduction poster",
            "error",
            f"first_pins={[cell_title(cell) for cell in pinned[:4]]}",
        ),
        check("bucket_artifact", bool(bucket_artifacts), "error", f"bucket_artifacts={len(bucket_artifacts)}"),
        check("bundle_download_note", bool(re.search(r"\b(download|browse)\b.*\bbundle\b|\bbundle\b.*\b(download|browse)\b", all_text, re.IGNORECASE | re.DOTALL)), "error", "describe bundle contents and access"),
        check("collection_link", "https://huggingface.co/collections/" in all_text, "error", "public collection URL required"),
        check("gpu_job_link", "https://huggingface.co/jobs/" in all_text, "warning", "required when substantive GPU work is feasible"),
        check("no_local_dashboards", not local_dashboards, "warning", f"local_dashboards={len(local_dashboards)}"),
        check("no_failed_conclusion_runs", not failed_conclusion, "warning", f"failed_conclusion_runs={len(failed_conclusion)}"),
    ]

    source_signals = re.compile(r"source[- ]verified|reported in|paper reports|not independently|cannot be independently|rerun blocked|unreleased", re.IGNORECASE)
    proxy_signals = re.compile(r"\btoy\b|\bproxy\b|synthetic|reduced scale", re.IGNORECASE)
    claim_cues = []
    for key in claims:
        body = "\n".join(cell["body"] for cell in page_cells[key])
        claim_cues.append(
            {
                "page": key,
                "cells": len(page_cells[key]),
                "successful_code_cells": sum(
                    1
                    for cell in page_cells[key]
                    if cell["meta"].get("type") == "code" and cell not in failed_cells
                ),
                "figure_cells": sum(1 for cell in page_cells[key] if cell["meta"].get("type") == "figure"),
                "source_only_language": bool(source_signals.search(body)),
                "proxy_language": bool(proxy_signals.search(body)),
                "job_urls": sorted(set(URL_RE.findall(body))) if "huggingface.co/jobs/" in body else [],
            }
        )

    return {
        "logbook": str(logbook),
        "title": title,
        "claim_pages": len(claims),
        "checks": checks,
        "summary": {
            "errors": sum(1 for item in checks if not item["passed"] and item["severity"] == "error"),
            "warnings": sum(1 for item in checks if not item["passed"] and item["severity"] == "warning"),
            "passed": sum(1 for item in checks if item["passed"]),
            "total": len(checks),
        },
        "claim_cues": claim_cues,
        "note": "Claim cues are discovery aids only; apply the scientific rubric manually.",
    }


def render_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"Logbook: {report['logbook']}",
        f"Contract: {summary['passed']}/{summary['total']} checks passed; {summary['errors']} errors; {summary['warnings']} warnings",
        "",
    ]
    for item in report["checks"]:
        mark = "PASS" if item["passed"] else item["severity"].upper()
        lines.append(f"[{mark}] {item['name']}: {item['detail']}")
    lines.extend(["", "Claim evidence cues (manual review required):"])
    for cue in report["claim_cues"]:
        lines.append(
            f"- {cue['page']}: code_ok={cue['successful_code_cells']} figures={cue['figure_cells']} "
            f"source_language={cue['source_only_language']} proxy_language={cue['proxy_language']}"
        )
    lines.append("")
    lines.append(report["note"])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", nargs="?", default=".", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--strict", action="store_true", help="exit 1 when contract errors are present")
    args = parser.parse_args()
    report = audit(args.workspace)
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))
    return 1 if args.strict and report["summary"]["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
