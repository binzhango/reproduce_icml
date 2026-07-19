# Monthly management report data contract

## Contents

1. Top-level object
2. Period values
3. Commentary and sources
4. Sign and scaling conventions
5. Canonical field references

## 1. Top-level object

The input is UTF-8 JSON:

```json
{
  "metadata": {
    "company_name": "Example Manufacturing, Inc.",
    "report_title": "Monthly Management Report",
    "current_period": "2026-06",
    "current_period_end": "2026-06-30",
    "prior_period": "2026-05",
    "prior_period_end": "2026-05-31",
    "currency": "USD",
    "scale": "thousands",
    "page_size": "Letter",
    "tolerance": "1"
  },
  "periods": {
    "current": { "income": {}, "balance": {}, "cash_flow": {} },
    "prior": { "income": {}, "balance": {}, "cash_flow": {} }
  },
  "kpis": [],
  "commentary": [],
  "sources": []
}
```

Required page sizes are `A4` or `Letter`. Supported scales are `units`, `thousands`, and `millions`. Values are expressed in the selected scale; the builder does not rescale inputs.

## 2. Period values

Each `current` and `prior` period requires these fields.

### Income

- `revenue`
- `cost_of_revenue`
- `operating_expenses`
- `interest_expense`
- `tax_expense`

The builder derives gross profit, operating income, pretax income, net income, and margins.

### Balance sheet

- `cash`
- `accounts_receivable`
- `inventory`
- `other_current_assets`
- `property_plant_equipment`
- `other_assets`
- `accounts_payable`
- `short_term_debt`
- `other_current_liabilities`
- `long_term_debt`
- `other_liabilities`
- `equity`

The builder derives total assets, total liabilities, and liabilities plus equity.

### Cash flow

- `opening_cash`
- `operating_cash_flow`
- `investing_cash_flow`
- `financing_cash_flow`
- `closing_cash`

Cash-flow categories use signed values. Outflows are negative.

JSON numbers or decimal strings are accepted. Decimal strings are preferred when source precision matters. `null`, blank strings, booleans, NaN, and infinity are rejected.

## 3. Commentary and sources

KPIs have `label`, `current`, `prior`, and `format`. Supported formats are `number`, `percent`, and `money`.

Commentary items have:

```json
{
  "title": "Revenue",
  "text": "Revenue increased following higher shipment volume.",
  "kind": "management explanation",
  "source_refs": ["income.revenue"]
}
```

Supported kinds are `data observation`, `management explanation`, and `forecast or judgment`.

Sources have a stable `id`, human-readable `label`, `as_of` ISO date, and optional `path`, `uri`, and `sha256`. Do not put credentials or expiring signed URLs into the report.

## 4. Sign and scaling conventions

- Revenue, cost of revenue, operating expenses, interest expense, and tax expense are positive magnitudes.
- Balance-sheet values are positive magnitudes. Negative equity is allowed but flagged for review.
- Cash-flow inflows are positive and outflows are negative.
- Currency and scale apply to every monetary statement value.
- Percent KPIs use decimal representation: `0.245` renders as `24.5%`.

## 5. Canonical field references

Commentary references use `<section>.<field>`, such as:

- `income.revenue`
- `income.net_income`
- `balance.cash`
- `balance.total_assets`
- `cash_flow.operating_cash_flow`
- `cash_flow.net_change`

Derived references are valid after calculation. Unknown references fail the commentary-reference gate.

