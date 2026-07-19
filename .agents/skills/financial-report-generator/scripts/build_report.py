#!/usr/bin/env python3
"""Build a deterministic monthly management report from canonical JSON."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

INCOME_FIELDS = ("revenue", "cost_of_revenue", "operating_expenses", "interest_expense", "tax_expense")
BALANCE_FIELDS = (
    "cash", "accounts_receivable", "inventory", "other_current_assets",
    "property_plant_equipment", "other_assets", "accounts_payable",
    "short_term_debt", "other_current_liabilities", "long_term_debt",
    "other_liabilities", "equity",
)
CASH_FIELDS = ("opening_cash", "operating_cash_flow", "investing_cash_flow", "financing_cash_flow", "closing_cash")
REQUIRED_META = (
    "company_name", "report_title", "current_period", "current_period_end",
    "prior_period", "prior_period_end", "currency", "scale", "page_size", "tolerance",
)
SCALE_LABELS = {"units": "units", "thousands": "000s", "millions": "millions"}
PAGE_SIZES = {"A4", "Letter"}
PAGE_HEIGHTS = {"A4": "270mm", "Letter": "252mm"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def dec(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or value is None or value == "":
        raise ValueError(f"{field}: expected a decimal value")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field}: invalid decimal {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"{field}: value must be finite")
    return result


def derive(period: dict[str, Any], prefix: str) -> dict[str, dict[str, Decimal]]:
    income = {key: dec(period["income"][key], f"{prefix}.income.{key}") for key in INCOME_FIELDS}
    balance = {key: dec(period["balance"][key], f"{prefix}.balance.{key}") for key in BALANCE_FIELDS}
    cash = {key: dec(period["cash_flow"][key], f"{prefix}.cash_flow.{key}") for key in CASH_FIELDS}
    income["gross_profit"] = income["revenue"] - income["cost_of_revenue"]
    income["operating_income"] = income["gross_profit"] - income["operating_expenses"]
    income["pretax_income"] = income["operating_income"] - income["interest_expense"]
    income["net_income"] = income["pretax_income"] - income["tax_expense"]
    income["gross_margin"] = income["gross_profit"] / income["revenue"] if income["revenue"] else Decimal(0)
    income["operating_margin"] = income["operating_income"] / income["revenue"] if income["revenue"] else Decimal(0)
    income["net_margin"] = income["net_income"] / income["revenue"] if income["revenue"] else Decimal(0)
    balance["total_current_assets"] = sum((balance[k] for k in ("cash", "accounts_receivable", "inventory", "other_current_assets")), Decimal(0))
    balance["total_assets"] = balance["total_current_assets"] + balance["property_plant_equipment"] + balance["other_assets"]
    balance["total_current_liabilities"] = sum((balance[k] for k in ("accounts_payable", "short_term_debt", "other_current_liabilities")), Decimal(0))
    balance["total_liabilities"] = balance["total_current_liabilities"] + balance["long_term_debt"] + balance["other_liabilities"]
    balance["liabilities_and_equity"] = balance["total_liabilities"] + balance["equity"]
    cash["net_change"] = cash["operating_cash_flow"] + cash["investing_cash_flow"] + cash["financing_cash_flow"]
    cash["calculated_closing_cash"] = cash["opening_cash"] + cash["net_change"]
    return {"income": income, "balance": balance, "cash_flow": cash}


def add_gate(gates: list[dict[str, Any]], name: str, passed: bool, detail: str, severity: str = "hard") -> None:
    gates.append({"name": name, "status": "PASS" if passed else ("WARN" if severity == "warning" else "FAIL"), "severity": severity, "detail": detail})


def validate(data: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    gates: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []
    meta = data.get("metadata")
    missing = [key for key in REQUIRED_META if not isinstance(meta, dict) or key not in meta]
    if missing:
        errors.append("missing metadata fields: " + ", ".join(missing))
        add_gate(gates, "data-contract", False, errors[-1])
        return {}, gates, errors
    if meta["scale"] not in SCALE_LABELS:
        errors.append("metadata.scale must be units, thousands, or millions")
    if meta["page_size"] not in PAGE_SIZES:
        errors.append("metadata.page_size must be A4 or Letter")
    if meta["current_period"] == meta["prior_period"]:
        errors.append("current and prior periods must be distinct")
    parsed_dates: dict[str, Any] = {}
    for when in ("current_period_end", "prior_period_end"):
        try:
            parsed_dates[when] = datetime.fromisoformat(str(meta[when])).date()
        except ValueError:
            errors.append(f"metadata.{when} must be an ISO date")
    periods = data.get("periods", {})
    try:
        current = derive(periods["current"], "current")
        prior = derive(periods["prior"], "prior")
        tolerance = abs(dec(meta["tolerance"], "metadata.tolerance"))
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
        add_gate(gates, "data-contract", False, "; ".join(errors))
        return {}, gates, errors
    add_gate(gates, "data-contract", not errors, "canonical fields and decimal values are valid" if not errors else "; ".join(errors))
    for name, values in (("current", current), ("prior", prior)):
        bdiff = values["balance"]["total_assets"] - values["balance"]["liabilities_and_equity"]
        add_gate(gates, f"balance-sheet-{name}", abs(bdiff) <= tolerance, f"difference={bdiff}; tolerance={tolerance}")
        cdiff = values["cash_flow"]["calculated_closing_cash"] - values["cash_flow"]["closing_cash"]
        add_gate(gates, f"cash-rollforward-{name}", abs(cdiff) <= tolerance, f"difference={cdiff}; tolerance={tolerance}")
        bridge = values["cash_flow"]["closing_cash"] - values["balance"]["cash"]
        add_gate(gates, f"cash-to-balance-{name}", abs(bridge) <= tolerance, f"difference={bridge}; tolerance={tolerance}")
        for field in INCOME_FIELDS:
            if values["income"][field] < 0:
                warning = f"{name}.income.{field} is negative under a positive-magnitude convention"
                warnings.append(warning)
                add_gate(gates, f"sign-{name}-income-{field}", False, warning, "warning")
        if values["balance"]["equity"] < 0:
            warning = f"{name}.balance.equity is negative"
            warnings.append(warning)
            add_gate(gates, f"negative-equity-{name}", False, warning, "warning")
    sources = data.get("sources")
    source_ok = isinstance(sources, list) and bool(sources) and all(isinstance(s, dict) and s.get("id") and s.get("label") and s.get("as_of") for s in sources)
    if source_ok and len({str(s["id"]) for s in sources}) != len(sources):
        source_ok = False
    add_gate(gates, "source-inventory", source_ok, "source identifiers, labels, and dates are present" if source_ok else "at least one complete source is required")
    if not source_ok:
        errors.append("invalid or missing source inventory")
    else:
        period_end = parsed_dates.get("current_period_end")
        for index, source in enumerate(sources):
            try:
                source_date = datetime.fromisoformat(str(source["as_of"])).date()
                if period_end and source_date > period_end:
                    warning = f"sources[{index}].as_of is later than the reporting-period end"
                    warnings.append(warning)
                    add_gate(gates, f"source-date-{index}", False, warning, "warning")
            except ValueError:
                errors.append(f"sources[{index}].as_of must be an ISO date")
                add_gate(gates, f"source-date-{index}", False, errors[-1])
            if not source.get("sha256"):
                warning = f"sources[{index}] has no verified snapshot hash"
                warnings.append(warning)
                add_gate(gates, f"source-hash-{index}", False, warning, "warning")
    all_refs = {f"{section}.{key}" for section in ("income", "balance", "cash_flow") for key in current[section]}
    unresolved: list[str] = []
    for index, item in enumerate(data.get("commentary", [])):
        refs = item.get("source_refs", []) if isinstance(item, dict) else []
        unresolved.extend(f"commentary[{index}]:{ref}" for ref in refs if ref not in all_refs)
        if isinstance(item, dict) and re.search(r"\d", str(item.get("text", ""))) and not refs:
            warning = f"commentary[{index}] contains numbers without source_refs"
            warnings.append(warning)
            add_gate(gates, f"numeric-commentary-{index}", False, warning, "warning")
        if isinstance(item, dict) and item.get("kind") not in {"data observation", "management explanation", "forecast or judgment"}:
            errors.append(f"commentary[{index}].kind is unsupported")
            add_gate(gates, f"commentary-kind-{index}", False, errors[-1])
    add_gate(gates, "commentary-references", not unresolved, "all commentary references resolve" if not unresolved else ", ".join(unresolved))
    if unresolved:
        errors.append("unresolved commentary references")
    for index, item in enumerate(data.get("kpis", [])):
        if not isinstance(item, dict) or not item.get("label") or item.get("format") not in {"number", "percent", "money"}:
            errors.append(f"kpis[{index}] is incomplete or has an unsupported format")
            add_gate(gates, f"kpi-contract-{index}", False, errors[-1])
            continue
        try:
            dec(item.get("current"), f"kpis[{index}].current")
            dec(item.get("prior"), f"kpis[{index}].prior")
        except ValueError as exc:
            errors.append(str(exc))
            add_gate(gates, f"kpi-contract-{index}", False, errors[-1])
    hard_failed = [g for g in gates if g["severity"] == "hard" and g["status"] == "FAIL"]
    if hard_failed and "one or more hard gates failed" not in errors:
        errors.append("one or more hard gates failed")
    return {"current": current, "prior": prior, "tolerance": tolerance}, gates, errors


def money(value: Decimal, currency: str) -> str:
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CNY": "¥"}
    sign = "-" if value < 0 else ""
    symbol = symbols.get(currency.upper(), currency.upper() + " ")
    return f"{sign}{symbol}{abs(value):,.0f}"


def pct(value: Decimal) -> str:
    return f"{value * 100:.1f}%"


def delta(current: Decimal, prior: Decimal, percent: bool = False) -> str:
    change = current - prior
    direction = "up" if change >= 0 else "down"
    shown = pct(abs(change)) if percent else f"{abs(change):,.0f}"
    arrow = "▲" if change >= 0 else "▼"
    return f'<span class="{direction}">{arrow} {shown} vs prior</span>'


def metric_card(label: str, value: str, change: str) -> str:
    return f'<div class="card metric"><div class="label">{html.escape(label)}</div><div class="value">{value}</div><div class="delta">{change}</div></div>'


def table(rows: list[tuple[str, Decimal, Decimal, str]], meta: dict[str, Any]) -> str:
    out = [f'<table><thead><tr><th>Line item</th><th>{html.escape(str(meta["current_period"]))}</th><th>{html.escape(str(meta["prior_period"]))}</th><th>Change</th></tr></thead><tbody>']
    for label, cur, prior, kind in rows:
        css = f' class="{kind}"' if kind else ""
        out.append(f'<tr{css}><td>{html.escape(label)}</td><td>{money(cur, meta["currency"])}</td><td>{money(prior, meta["currency"])}</td><td>{money(cur-prior, meta["currency"])}</td></tr>')
    out.append("</tbody></table>")
    return "".join(out)


def bar_chart(items: list[tuple[str, Decimal, Decimal]], current_label: str, prior_label: str) -> str:
    maximum = max((abs(v) for _, a, b in items for v in (a, b)), default=Decimal(1)) or Decimal(1)
    parts = ['<svg viewBox="0 0 620 240" role="img" aria-label="Current and prior period performance comparison">']
    parts.append('<style>.a{fill:#174f66}.b{fill:#9bb4bf}.t{font:12px Arial;fill:#667085}.v{font:bold 11px Arial;fill:#172033}</style>')
    for i, (label, cur, prior) in enumerate(items):
        y = 25 + i * 68
        cw = float(abs(cur) / maximum) * 380
        pw = float(abs(prior) / maximum) * 380
        parts.append(f'<text class="t" x="0" y="{y+13}">{html.escape(label)}</text><rect class="a" x="125" y="{y}" width="{cw:.1f}" height="18" rx="3"/><rect class="b" x="125" y="{y+24}" width="{pw:.1f}" height="18" rx="3"/><text class="v" x="{min(585,130+cw):.1f}" y="{y+13}">{cur:,.0f}</text><text class="v" x="{min(585,130+pw):.1f}" y="{y+37}">{prior:,.0f}</text>')
    parts.append(f'<text class="t" x="125" y="232">■ {html.escape(current_label)}   ▪ {html.escape(prior_label)}</text></svg>')
    return "".join(parts)


def header(meta: dict[str, Any], compact: bool = False) -> str:
    title = html.escape(str(meta["company_name"])) if compact else html.escape(str(meta["report_title"]))
    subtitle = html.escape(str(meta["report_title"])) if compact else html.escape(str(meta["company_name"]))
    return f'<header class="report-head"><div><div class="eyebrow">{subtitle}</div><h1>{title}</h1></div><div class="meta"><strong>{html.escape(str(meta["current_period"]))}</strong>Compared with {html.escape(str(meta["prior_period"]))}<br>{html.escape(str(meta["currency"]))} · {html.escape(SCALE_LABELS[meta["scale"]])}<br><span class="status">Validated internal report</span></div></header>'


def footer(meta: dict[str, Any], generated: str) -> str:
    return f'<footer class="footer"><span>{html.escape(str(meta["company_name"]))} - Confidential</span><span>Generated {html.escape(generated[:10])} UTC</span></footer>'


def render(data: dict[str, Any], derived: dict[str, Any], template: str, input_hash: str, template_hash: str, generated: str, gates: list[dict[str, Any]]) -> str:
    meta = data["metadata"]
    cur, prior = derived["current"], derived["prior"]
    summary = "".join((
        metric_card("Revenue", money(cur["income"]["revenue"], meta["currency"]), delta(cur["income"]["revenue"], prior["income"]["revenue"])),
        metric_card("Gross margin", pct(cur["income"]["gross_margin"]), delta(cur["income"]["gross_margin"], prior["income"]["gross_margin"], True)),
        metric_card("Operating income", money(cur["income"]["operating_income"], meta["currency"]), delta(cur["income"]["operating_income"], prior["income"]["operating_income"])),
        metric_card("Closing cash", money(cur["balance"]["cash"], meta["currency"]), delta(cur["balance"]["cash"], prior["balance"]["cash"])),
    ))
    kpis = []
    for item in data.get("kpis", []):
        c, p = dec(item["current"], f'kpi.{item.get("label", "")}.current'), dec(item["prior"], f'kpi.{item.get("label", "")}.prior')
        fmt = item.get("format", "number")
        shown = pct(c) if fmt == "percent" else money(c, meta["currency"]) if fmt == "money" else f"{c:,.0f}"
        kpis.append(metric_card(str(item["label"]), shown, delta(c, p, fmt == "percent")))
    commentary = []
    for item in data.get("commentary", []):
        commentary.append(f'<div class="commentary"><div class="kind">{html.escape(str(item.get("kind", "management commentary")))}</div><h3>{html.escape(str(item.get("title", "Observation")))}</h3><p>{html.escape(str(item.get("text", "")))}</p></div>')
    income_rows = [
        ("Revenue", cur["income"]["revenue"], prior["income"]["revenue"], ""),
        ("Cost of revenue", cur["income"]["cost_of_revenue"], prior["income"]["cost_of_revenue"], ""),
        ("Gross profit", cur["income"]["gross_profit"], prior["income"]["gross_profit"], "subtotal"),
        ("Operating expenses", cur["income"]["operating_expenses"], prior["income"]["operating_expenses"], ""),
        ("Operating income", cur["income"]["operating_income"], prior["income"]["operating_income"], "subtotal"),
        ("Interest expense", cur["income"]["interest_expense"], prior["income"]["interest_expense"], ""),
        ("Pretax income", cur["income"]["pretax_income"], prior["income"]["pretax_income"], "subtotal"),
        ("Tax expense", cur["income"]["tax_expense"], prior["income"]["tax_expense"], ""),
        ("Net income", cur["income"]["net_income"], prior["income"]["net_income"], "total"),
    ]
    balance_rows = [
        ("Cash", cur["balance"]["cash"], prior["balance"]["cash"], ""),
        ("Accounts receivable", cur["balance"]["accounts_receivable"], prior["balance"]["accounts_receivable"], ""),
        ("Inventory", cur["balance"]["inventory"], prior["balance"]["inventory"], ""),
        ("Other current assets", cur["balance"]["other_current_assets"], prior["balance"]["other_current_assets"], ""),
        ("Total current assets", cur["balance"]["total_current_assets"], prior["balance"]["total_current_assets"], "subtotal"),
        ("Property, plant and equipment", cur["balance"]["property_plant_equipment"], prior["balance"]["property_plant_equipment"], ""),
        ("Other assets", cur["balance"]["other_assets"], prior["balance"]["other_assets"], ""),
        ("Total assets", cur["balance"]["total_assets"], prior["balance"]["total_assets"], "total"),
        ("Accounts payable", cur["balance"]["accounts_payable"], prior["balance"]["accounts_payable"], ""),
        ("Short-term debt", cur["balance"]["short_term_debt"], prior["balance"]["short_term_debt"], ""),
        ("Other current liabilities", cur["balance"]["other_current_liabilities"], prior["balance"]["other_current_liabilities"], ""),
        ("Total current liabilities", cur["balance"]["total_current_liabilities"], prior["balance"]["total_current_liabilities"], "subtotal"),
        ("Long-term debt", cur["balance"]["long_term_debt"], prior["balance"]["long_term_debt"], ""),
        ("Other liabilities", cur["balance"]["other_liabilities"], prior["balance"]["other_liabilities"], ""),
        ("Total liabilities", cur["balance"]["total_liabilities"], prior["balance"]["total_liabilities"], "subtotal"),
        ("Equity", cur["balance"]["equity"], prior["balance"]["equity"], ""),
        ("Liabilities and equity", cur["balance"]["liabilities_and_equity"], prior["balance"]["liabilities_and_equity"], "total"),
    ]
    cash_rows = [
        ("Opening cash", cur["cash_flow"]["opening_cash"], prior["cash_flow"]["opening_cash"], ""),
        ("Operating cash flow", cur["cash_flow"]["operating_cash_flow"], prior["cash_flow"]["operating_cash_flow"], ""),
        ("Investing cash flow", cur["cash_flow"]["investing_cash_flow"], prior["cash_flow"]["investing_cash_flow"], ""),
        ("Financing cash flow", cur["cash_flow"]["financing_cash_flow"], prior["cash_flow"]["financing_cash_flow"], ""),
        ("Net change in cash", cur["cash_flow"]["net_change"], prior["cash_flow"]["net_change"], "subtotal"),
        ("Closing cash", cur["cash_flow"]["closing_cash"], prior["cash_flow"]["closing_cash"], "total"),
    ]
    controls = '<div class="controls-grid">' + "".join(
        f'<div class="control"><div class="control-mark">OK</div><div><h3>{html.escape(g["name"])}</h3><p>{html.escape(g["detail"])}</p></div></div>'
        for g in gates if g["severity"] == "hard"
    ) + "</div>"
    warning_items = [g for g in gates if g["status"] == "WARN"]
    warnings_panel = (f'<div class="warnings"><h3>Review warnings</h3><p>{len(warning_items)} warning(s) are recorded in GATE_REPORT.json and require review before release.</p></div>') if warning_items else ""
    sources = "".join(f'<div class="source"><div class="source-id">{html.escape(str(s["id"]))}</div><strong>{html.escape(str(s["label"]))}</strong><div class="note">As of {html.escape(str(s["as_of"]))} · {html.escape(str(s.get("path") or s.get("uri") or "location not disclosed"))}</div></div>' for s in data["sources"])
    values = {
        "DOCUMENT_TITLE": f'{meta["company_name"]} - {meta["report_title"]}', "PAGE_SIZE": meta["page_size"], "PAGE_HEIGHT": PAGE_HEIGHTS[meta["page_size"]],
        "HEADER": header(meta), "HEADER_COMPACT": header(meta, True), "FOOTER": footer(meta, generated),
        "SUMMARY_METRICS": summary, "PERFORMANCE_CHART": bar_chart([("Revenue", cur["income"]["revenue"], prior["income"]["revenue"]), ("Gross profit", cur["income"]["gross_profit"], prior["income"]["gross_profit"]), ("Net income", cur["income"]["net_income"], prior["income"]["net_income"])], meta["current_period"], meta["prior_period"]),
        "COMMENTARY": "".join(commentary) or '<p class="note">No management commentary supplied.</p>', "KPI_CARDS": "".join(kpis) or '<div class="card">No operating KPIs supplied.</div>',
        "INCOME_TABLE": table(income_rows, meta), "BALANCE_TABLE": table(balance_rows, meta), "CASH_FLOW_TABLE": table(cash_rows, meta),
        "CASH_CHART": bar_chart([("Operating", cur["cash_flow"]["operating_cash_flow"], prior["cash_flow"]["operating_cash_flow"]), ("Investing", cur["cash_flow"]["investing_cash_flow"], prior["cash_flow"]["investing_cash_flow"]), ("Financing", cur["cash_flow"]["financing_cash_flow"], prior["cash_flow"]["financing_cash_flow"])], meta["current_period"], meta["prior_period"]),
        "CURRENCY": html.escape(meta["currency"]), "SCALE": html.escape(SCALE_LABELS[meta["scale"]]),
        "GROSS_MARGIN_NOTE": f'<p>{pct(cur["income"]["gross_margin"])} compared with {pct(prior["income"]["gross_margin"])} in the prior period.</p>',
        "OPERATING_MARGIN_NOTE": f'<p>{pct(cur["income"]["operating_margin"])} compared with {pct(prior["income"]["operating_margin"])} in the prior period.</p>',
        "CONTROL_RESULTS": controls, "WARNINGS_PANEL": warnings_panel, "SOURCE_LIST": sources, "INPUT_HASH": input_hash, "TEMPLATE_HASH": template_hash,
    }
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    unresolved = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", template)))
    if unresolved:
        raise ValueError("unresolved template placeholders: " + ", ".join(unresolved))
    return template


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=Path(__file__).resolve().parents[1] / "assets" / "monthly-report.html")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    derived, gates, errors = validate(data)
    warnings = [g["detail"] for g in gates if g["status"] == "WARN"]
    overall = "FAIL" if errors else "PASS"
    report = {"overall": overall, "generated_at": generated, "gates": gates, "warnings": warnings, "errors": errors}
    (args.out_dir / "GATE_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if errors:
        print(f"FAIL: {len(errors)} error(s); see {args.out_dir / 'GATE_REPORT.json'}", file=sys.stderr)
        return 1
    template = args.template.read_text(encoding="utf-8")
    input_hash, template_hash = sha256(args.input), sha256(args.template)
    output = render(data, derived, template, input_hash, template_hash, generated, gates)
    if re.search(r"(?:src|href)=[\"']https?://", output, re.I):
        print("FAIL: remote resources are forbidden", file=sys.stderr)
        return 1
    html_path = args.out_dir / "report.html"
    html_path.write_text(output, encoding="utf-8")
    formulas = {
        "gross_profit": "revenue - cost_of_revenue", "operating_income": "gross_profit - operating_expenses",
        "pretax_income": "operating_income - interest_expense", "net_income": "pretax_income - tax_expense",
        "total_assets": "current assets + property_plant_equipment + other_assets",
        "total_liabilities": "current liabilities + long_term_debt + other_liabilities",
        "net_change": "operating_cash_flow + investing_cash_flow + financing_cash_flow",
    }
    manifest = {"generated_at": generated, "company": data["metadata"]["company_name"], "period_end": data["metadata"]["current_period_end"], "input": {"path": str(args.input), "sha256": input_hash}, "template": {"path": str(args.template), "sha256": template_hash}, "sources": data["sources"], "formulas": formulas}
    (args.out_dir / "DATA_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: wrote {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
