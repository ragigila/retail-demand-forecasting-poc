# Methodology

Result numbers are intentionally **not** published here. This document defines what the code computes and why, so that the pipeline can be audited and re-run on other data.

## 1. Scope

Forecast weekly demand per **style x location** for two location types, evaluated separately because the same goods are counted at both:

- **store**: demand = sales to end customers;
- **warehouse**: demand = goods issued from the warehouse (online-channel sales plus transfers out).

Store and warehouse totals are never summed across location types.

## 2. Source data

Item ledger entries (one row per inventory movement: entry type, posting date, item, location, quantity, document type, order number, sales amount, dimensions) plus a small set of item attributes (description, item type, department, class) from the item master. Only fields needed for the analysis are extracted; customer, vendor and document-level detail are not.

## 3. Privacy handling

1. Item numbers and location codes are mapped to random identifiers (seeded permutation) immediately after loading; documents and orders become integer codes.
2. The mapping file is kept in a private, git-ignored directory and is never printed.
3. Style keys (see 5) are pseudonymised the same way before any dataset is written.
4. A leakage-style check verifies that no real identifier appears in the written datasets.

## 4. Location handling

- An explicit mapping (local file) assigns every location to `store`, `warehouse` or `exclude`. Locations that are not listed must satisfy a share-based fallback rule (share of sale entries among all entries); anything else aborts the build instead of being guessed.
- Intermediary **transit** locations are not demand locations and are never series. In the ledger a transfer leaves entries at the sending location, at intermediary locations and at the receiving location; counting all negative transfer quantities would count the same goods several times.
- Sample, rejected-goods, event and administrative locations are excluded.
- A warehouse shipment whose destination is the transit-parking location is still warehouse demand (it is the only way goods leave the warehouse), but shipments to excluded sites are not.

## 5. Items and aggregation level

- Only inventory-type items are kept; packaging and other non-merchandise departments are excluded.
- The modelling level is **style x location**: colour and size are collapsed. The style key is the item description combined with the brand.
- Time grid: ISO weeks (Monday-Sunday). Rows after the last complete week are dropped because ledger posting lags.
- Each series starts at the first ledger entry of any type for that style and location (stock is present from then on). Weeks between that start and the last week with no sales are filled with 0; weeks before the start are not demand.

## 6. Demand definition

- **Store**: gross sales out = sales shipment entries with negative quantity, per week. Return entries are ignored, so demand is never negative and no clipping is needed. Residual bias: an exchange appears as one sale of the original item and one sale of the replacement.
- **Warehouse**: sales out plus transfer-shipment entries leaving the warehouse (see 4).

## 7. Features

All features for week *t* use only information from weeks before *t*, or the calendar:

- lags of demand (1, 2, 4, 8 and 52 weeks) and rolling means (4, 8, 13 weeks), weeks since last sale, series age;
- calendar: ISO week, month, public-holiday days in the week and the following week, payday week;
- last observed average price per unit (from sales amount / quantity);
- stock features: previous-week stock-out flag and previous-week balance (below);
- static: location, brand, item department, item class.

Missing values mean "not computable yet" (for example lag 52 before 52 weeks of history) and are never replaced by 0.

**Stock-out proxy.** The running balance per style and location is the cumulative ledger quantity. A week is flagged when the end-of-week balance is zero or negative. The balance is trusted from the start of the series until it first goes negative (a negative balance proves incomplete history or an unknown opening balance); afterwards the flag is missing. The flag cannot distinguish a real stock-out from a style that is simply no longer replenished.

**Leakage test.** The build replaces demand, price and balance from a chosen week onwards by arbitrary values and asserts that every model feature of earlier weeks is unchanged. Same-week flags kept for analysis are excluded from this test and from the model inputs.

**Unit reconciliation.** The build checks that total gross sales units equal excluded locations + non-merchandise + warehouse + modelled store units, so every filter is accounted for.

## 8. Evaluation protocol

- **Rolling origin**: horizon 4 weeks, step 4 weeks, non-overlapping test windows; training uses all weeks before the origin. First origin and step are parameters chosen from the history length, not tuned on results.
- **Metrics**: WAPE = sum |forecast - actual| / sum actual; bias = (sum forecast - sum actual) / sum actual. Pooled over folds by summing errors (not averaging ratios). Also WAPE relative to the seasonal-naive reference on the same series, the median per-series WAPE (WAPE is dominated by large series), and WAPE after aggregating forecasts to location x week and to location type x week.
- **Series classes**: Syntetos-Boylan-Cardoso classes (smooth, erratic, intermittent, lumpy; ADI 1.32 and CV^2 0.49) plus "single demand week" and "no demand so far", computed from each fold's **training window only**.
- **Series that never sell** are reported separately (every positive forecast there is a false positive). The split uses the full period and is therefore a selection on outcome; this is stated wherever it is used.
- **Cold start**: series that do not yet exist at the origin cannot be forecast by any model; their test volume is reported as coverage.

## 9. Models

| Model | Notes |
|---|---|
| Seasonal naive | value of the same week one season earlier when it exists for the series, otherwise the 13-week rolling mean. Deviation from "fall back when history < 2 seasons": the available history is shorter than two seasons, so that rule would never apply. |
| Rolling mean | 13-week mean, plain baseline |
| Croston, TSB | fixed smoothing constants (literature defaults), not tuned |
| ETS | additive error, no seasonal term, no-trend vs damped-trend chosen by AIC; fitted only for smooth/erratic series with enough non-zero weeks |
| Class-wise statistical | ETS for smooth/erratic, TSB for the rest; mapping fixed in advance, not chosen from test scores |
| LightGBM | one global model per location type, direct multi-horizon (horizon is a feature), Tweedie objective, fixed hyper-parameters (see `config.example.yaml`), trained on a seeded random subset of series, all series forecast |

All models are reported for all classes so the data can contradict the pre-set mapping. Nothing is tuned on test data.

## 10. Feature ablation

LightGBM is retrained on the same folds and the same training series without, in turn, (a) the calendar features, (b) the price feature and (c) the stock-out features, and compared with the full model on WAPE and bias.

## 11. Limitations

- History is short relative to a yearly season; yearly features and seasonal models cannot be assessed properly.
- Demand at style x location x week level is very sparse, so SKU-level WAPE values are inherently high; aggregated levels are more informative.
- No promotion data; the price feature is a weak proxy.
- Online-channel demand is only visible to the extent it is posted at the warehouse.
- Stock-out is a proxy (see 7).
- Hyper-parameters are fixed, a single seed is used and there is no uncertainty interval, so small differences between models or features should not be read as effects.
- Prophet, hierarchical reconciliation (MinT), SHAP explanations and an LSTM are part of the original design but were not implemented here.
