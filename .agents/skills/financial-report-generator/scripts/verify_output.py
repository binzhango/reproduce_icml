#!/usr/bin/env python3
"""Verify structural HTML requirements and optional PDF output."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

REQUIRED_SECTIONS = {
    "executive-summary", "income-statement", "balance-sheet",
    "cash-flow", "controls-and-provenance",
}


def gate(items: list[dict[str, str]], name: str, passed: bool, detail: str) -> None:
    items.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-pages", type=int, default=5)
    args = parser.parse_args()
    items: list[dict[str, str]] = []
    text = args.html.read_text(encoding="utf-8") if args.html.is_file() else ""
    gate(items, "html-exists", bool(text), str(args.html))
    unresolved = sorted(set(re.findall(r"\{\{[^{}]+\}\}", text)))
    gate(items, "template-complete", not unresolved, "no unresolved placeholders" if not unresolved else ", ".join(unresolved))
    sections = set(re.findall(r'data-report-section="([^"]+)"', text))
    missing = sorted(REQUIRED_SECTIONS - sections)
    gate(items, "required-sections", not missing, "all five sections present" if not missing else "missing: " + ", ".join(missing))
    remote = re.findall(r'(?:src|href)=["\']https?://', text, flags=re.I)
    gate(items, "offline-resources", not remote, "no remote resources")
    gate(items, "print-css", "@page" in text and "@media print" in text, "paged-media rules present")
    if args.pdf:
        pdf_ok = args.pdf.is_file() and args.pdf.stat().st_size > 10_000
        gate(items, "pdf-exists", pdf_ok, f"{args.pdf} ({args.pdf.stat().st_size if args.pdf.is_file() else 0} bytes)")
        pdfinfo = shutil.which("pdfinfo")
        if pdf_ok and pdfinfo:
            proc = subprocess.run([pdfinfo, str(args.pdf)], capture_output=True, text=True, check=False)
            match = re.search(r"^Pages:\s+(\d+)", proc.stdout, flags=re.M)
            pages = int(match.group(1)) if match else 0
            gate(items, "pdf-page-count", pages == args.expected_pages, f"pages={pages}; expected={args.expected_pages}")
        elif pdf_ok:
            gate(items, "pdf-page-count", False, "pdfinfo unavailable; page count not verified")
    overall = "PASS" if all(item["status"] == "PASS" for item in items) else "FAIL"
    result = {"overall": overall, "gates": items}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{overall}: wrote {args.output}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

