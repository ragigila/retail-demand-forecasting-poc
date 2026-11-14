# Retail demand forecasting - batch pipeline proof of concept

A reproducible pipeline that turns an ERP **item ledger** into weekly demand series per **style x location**, benchmarks forecasting models with **rolling-origin evaluation**, and measures which input features matter. It was built as a proof of concept for a master's-level AI course project on multi-location demand forecasting (batch pipeline, weekly cadence, hierarchy of SKU / location / location type).

> **This repository contains code and methodology only.** It does not contain any company data, extracts, intermediate datasets, trained models or result numbers. Everything company-specific (company id, brand codes, location codes, custom ERP field names, paths, credentials) lives in git-ignored local files; see *Configuration*.

## What it does

| Stage | Script | Output (all under `workdir/`, git-ignored) |
|---|---|---|
| 0. Extract (read-only) | `extract/ile-pull.js`, `extract/item-pull.js` | raw ledger and item-attribute NDJSON |
| 1. Profile | `src/profile_data.py` | data profile (entry types, transfer structure, returns, series length, intermittency classes) |
| 2. Dataset | `src/build_dataset.py` | pseudonymised weekly parquet files for stores and warehouses, with features and a reconciliation of units |
| 3. Experiment | `src/experiment.py` | cached forecasts per fold (resumable) |
| 4. Evaluate | `src/evaluate.py` | metric tables, ablation table, figures, results summary |

Models compared: seasonal naive (with documented fallback), rolling mean, Croston, TSB, ETS, and one global LightGBM per location type. See [METHODOLOGY.md](METHODOLOGY.md) for definitions, protocol and limitations.

## Privacy by design

- ERP access is **read-only** and credentials come only from the environment or an untracked `.env`.
- Item numbers, location codes and style keys are **replaced by random IDs at load time**; the id mapping stays in `workdir/private/`. Reports and parquet files only contain pseudonymised IDs.
- Raw rows are never printed. `.gitignore` blocks data, parquet, NDJSON, `.env` and all `*.local.*` configuration.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                      # versions are pinned
cp config.example.yaml config.local.yaml             # then fill in your values
cp location_types.example.yaml location_types.local.yaml
cp bc_fields.example.json bc_fields.local.json
```

### Configuration (all git-ignored)

- `config.local.yaml`: company id, brand codes, date range, cut-off week, item attribute names, modelling parameters (seed, folds, hyper-parameters).
- `location_types.local.yaml`: explicit mapping of ERP location codes to `store`, `warehouse` or `exclude`, plus the transit-parking location. An explicit mapping always wins over the share-based fallback rule, so no location is left ambiguous.
- `bc_fields.local.json`: names of the web services and of any custom item attributes in your ERP.
- `.env`: `BC_TENANT_ID`, `BC_CLIENT_ID`, `BC_CLIENT_SECRET`, `BC_SCOPE`. The target environment is set with `BC_TARGET_ENV` (default `Production`) and is intentionally not read from a shared `.env`.

## Run

```bash
node extract/ile-pull.js BRAND_A 2025-01 2025-12 COMPANY_ID
node extract/ile-pull.js BRAND_B 2025-01 2025-12 COMPANY_ID
node extract/item-pull.js COMPANY_ID
python src/profile_data.py
python src/build_dataset.py
python src/experiment.py            # resumable; --tag smoke --origins <w> --cap 5000 for a quick test
python src/evaluate.py
```

## Assumptions about the ERP

Written against Microsoft Dynamics 365 Business Central OData v4 (Item Ledger Entry page). It assumes Global Dimension 2 carries the brand, that returns are posted as sale entries with positive quantity, and that transfers between locations pass through an intermediary transit location. Adapt `extract/` and the location mapping if your ERP differs.

## License

MIT, see [LICENSE](LICENSE).
