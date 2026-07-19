#!/usr/bin/env python3
"""Render a self-contained report HTML file to PDF with Chromium."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()
    source = args.html.resolve()
    if not source.is_file():
        print(f"ERROR: HTML file not found: {source}", file=sys.stderr)
        return 2
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: Playwright is required. Install it and its Chromium browser before rendering.", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path = args.diagnostics or args.output.with_name("RENDER_REPORT.json")
    try:
        with sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=1)
            page.goto(source.as_uri(), wait_until="networkidle")
            page.emulate_media(media="print")
            page.evaluate("document.fonts && document.fonts.ready")
            diagnostics = page.evaluate("""() => {
              const sections = [...document.querySelectorAll('[data-report-section]')];
              const overflow = [...document.querySelectorAll('body *')].filter(el => {
                const r = el.getBoundingClientRect();
                return r.right > document.documentElement.clientWidth + 2 || r.left < -2;
              }).slice(0, 20).map(el => ({tag: el.tagName, className: el.className, text: (el.textContent || '').trim().slice(0, 80)}));
              return {sections: sections.map(x => x.dataset.reportSection), horizontal_overflow: overflow};
            }""")
            if diagnostics["horizontal_overflow"]:
                diagnostics["status"] = "FAIL"
                diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
                print(f"FAIL: horizontal overflow detected; see {diagnostics_path}", file=sys.stderr)
                browser.close()
                return 1
            page.pdf(path=str(args.output.resolve()), print_background=True, prefer_css_page_size=True)
            browser.close()
    except Exception as exc:
        print(f"ERROR: Chromium render failed: {exc}", file=sys.stderr)
        return 2
    diagnostics["status"] = "PASS"
    diagnostics["pdf"] = str(args.output.resolve())
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

