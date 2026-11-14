"""Stage 2: build pseudonymised weekly datasets (store x style, warehouse x style) as parquet.
usage: python build_dataset.py [config.yaml]"""
import json
import os
import sys

import holidays
import numpy as np
import pandas as pd
import yaml

from common import load_config, load_pseudonymised, load_items, md_table
from features import make_features

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
S2, P = cfg["stage2"], cfg["profile"]
out_dir = os.path.join(cfg["paths"]["out_dir"], "datasets")
os.makedirs(out_dir, exist_ok=True)
md = []
L = md.append

df, info, id_map = load_pseudonymised(cfg)
items = load_items(cfg, id_map)
assert items is not None, "item master missing"

cut = pd.Timestamp(cfg["period"]["last_full_week_end"])
df = df[df["date"] <= cut].copy()
df["week_start"] = df["date"].dt.to_period("W-SUN").dt.start_time
w0 = df["week_start"].min()
last_idx = int((cut.to_period("W-SUN").start_time - w0).days // 7)
df["widx"] = ((df["week_start"] - w0).dt.days // 7).astype(int)

# ---- location types (explicit mapping, share rule only as asserted fallback) ---------------------------
lt_file = yaml.safe_load(open(S2["location_types_file"], encoding="utf-8"))
code2pseudo = id_map["Location_Code"]
ltype = {}
for typ in ("warehouse", "store", "exclude"):
    for code in lt_file.get(typ, []):
        if code in code2pseudo:
            ltype[code2pseudo[code]] = "excluded" if typ == "exclude" else typ
prop = pd.read_csv(os.path.join(cfg["paths"]["out_dir"], "location_type_proposal.csv"), keep_default_na=False)
for _, r in prop.iterrows():
    if r["loc"] not in ltype:
        assert r["proposed_type"] == "store", f"unmapped non-store location {r['loc']} ({r['proposed_type']})"
        ltype[r["loc"]] = "store"
df["ltype"] = df["loc"].map(ltype)
assert df["ltype"].notna().all()

# ---- Transfer Order destination (needed for warehouse outflow) ----------------------------------------
tr = df[df["entry_type"] == "Transfer"]
# destination = receipt-side location that is not the blank in-transit leg
blank_pseudo = code2pseudo.get("")
# The transit-parking location is excluded as a series location but is a legitimate destination of warehouse outflow; other excluded sites (rejected/sample/...) are not
transit_pseudo = {code2pseudo[c] for c in lt_file.get("transit_parking", []) if c in code2pseudo} | {blank_pseudo}
dest_loc = tr[(tr["qty"] > 0) & (tr["loc"] != blank_pseudo)].groupby("order")["loc"].first().rename("dest_loc")

# ---- item side: merchandise filter + style key ----------------------------------------------------------
brand_of = df.groupby("sku")["brand"].agg(lambda s: s.mode().iat[0])
items = items.set_index("sku")
items["brand"] = brand_of.reindex(items.index).fillna("UNK")  # items only seen after the cutoff
desc = items["description"].fillna("").astype(str).str.strip()
fallback = items["item_no_raw"].astype(str).str.split().str[0]
items["style_raw"] = items["brand"].astype(str) + "|" + np.where(desc != "", desc, fallback)
rng = np.random.default_rng(cfg["seed"] + 1)
uniq = np.array(sorted(items["style_raw"].unique()))
perm = rng.permutation(len(uniq))
style_map = {u: f"STYLE_{int(o) + 1:05d}" for u, o in zip(uniq, perm)}
items["style"] = items["style_raw"].map(style_map)
id_map["Style"] = style_map
with open(os.path.join(cfg["paths"]["private_dir"], "id_map.json"), "w", encoding="utf-8") as fh:
    json.dump(id_map, fh)
merch = (items["item_type"] == S2["require_item_type"]) & (~items["department"].fillna("").astype(str).isin(S2["excluded_item_departments"]))
df["is_merch"] = df["sku"].map(merch).fillna(False).astype(bool)
df["style"] = df["sku"].map(items["style"])
style_dept = items.loc[merch].groupby("style")["department"].agg(lambda s: s.mode().iat[0])
style_class = items.loc[merch].groupby("style")["class_code"].agg(lambda s: s.mode().iat[0])
style_brand = items.loc[merch].groupby("style")["brand"].first()

# ---- demand definitions + reconciliation -------------------------------------------------------------
sale_out = (df["entry_type"] == "Sale") & (df["doc_type"] == "Sales Shipment") & (df["qty"] < 0)
df["gross_sale_units"] = np.where(sale_out, -df["qty"], 0.0)
tr_ship = (df["entry_type"] == "Transfer") & (df["qty"] < 0)
df = df.join(dest_loc, on="order")
df["dest_is_excluded_site"] = df["dest_loc"].map(lambda l: ltype.get(l) == "excluded" and l not in transit_pseudo).astype(bool)
wh_out = tr_ship & (df["ltype"] == "warehouse") & (~df["dest_is_excluded_site"])
df["wh_transfer_units"] = np.where(wh_out, -df["qty"], 0.0)

rec = []
tot_sale = df["gross_sale_units"].sum()
rec.append(("Gross sale units, all locations in extract", tot_sale))
rec.append(("  - at excluded locations", df.loc[df.ltype == "excluded", "gross_sale_units"].sum()))
rec.append(("  - non-merchandise items (excluded department / non-inventory) at kept locations", df.loc[(df.ltype != "excluded") & (~df.is_merch), "gross_sale_units"].sum()))
rec.append(("  - warehouse locations (online channel)", df.loc[(df.ltype == "warehouse") & df.is_merch, "gross_sale_units"].sum()))
rec.append(("  - store locations, merchandise (modelled)", df.loc[(df.ltype == "store") & df.is_merch, "gross_sale_units"].sum()))

d = df[(df.ltype != "excluded") & df.is_merch].copy()

# ---- weekly aggregation + grid --------------------------------------------------------------------------
cal_weeks = pd.DataFrame({"widx": np.arange(last_idx + 1)})
cal_weeks["week_start"] = w0 + pd.to_timedelta(cal_weeks["widx"] * 7, unit="D")
hol = holidays.country_holidays(S2["holiday_country"], years=sorted({w0.year, cut.year, cut.year + 1}))
hol_dates = set(pd.Timestamp(k) for k in hol.keys())
def week_hols(ws, shift=0):
    return sum(((ws + pd.Timedelta(days=shift * 7 + i)) in hol_dates) for i in range(7))
cal = pd.DataFrame({
    "iso_week": cal_weeks["week_start"].dt.isocalendar().week.astype(int).to_numpy(),
    "month": (cal_weeks["week_start"] + pd.Timedelta(days=3)).dt.month.to_numpy(),
    "holiday_days": [week_hols(w) for w in cal_weeks["week_start"]],
    "holiday_days_next_week": [week_hols(w, 1) for w in cal_weeks["week_start"]],
    "payday_week": [int(any(((w + pd.Timedelta(days=i)).day in S2["payday_days"]) for i in range(7))) for w in cal_weeks["week_start"]],
}, index=cal_weeks["widx"])


def build(kind):
    x = d[d["ltype"] == kind].copy()
    x["sid_key"] = x["style"] + "|" + x["loc"]
    codes, uniq_keys = pd.factorize(x["sid_key"])
    x["sid"] = codes
    meta = pd.DataFrame({"sid_key": uniq_keys})
    meta["style"] = meta["sid_key"].str.split("|").str[0]
    meta["loc"] = meta["sid_key"].str.split("|").str[1]
    start = x.groupby("sid")["widx"].min()
    n = (last_idx - start.to_numpy() + 1)
    rep = np.repeat(start.index.to_numpy(), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    g = pd.DataFrame({"sid": rep, "widx": start.to_numpy()[np.searchsorted(start.index.to_numpy(), rep)] + offs})
    # demand
    if kind == "store":
        x["dem"] = x["gross_sale_units"]
    else:
        x["dem"] = x["gross_sale_units"] + x["wh_transfer_units"]
    wk = x.groupby(["sid", "widx"]).agg(y=("dem", "sum")).reset_index()
    pr = x[(x["gross_sale_units"] > 0) & (x["sales_amt"] > 0)].groupby(["sid", "widx"]).agg(amt=("sales_amt", "sum"), u=("gross_sale_units", "sum")).reset_index()
    pr["price"] = pr["amt"] / pr["u"]
    # running balance at style x location (all entry types)
    x = x.sort_values(["sid", "entry_no"])
    x["cum"] = x.groupby("sid")["qty"].cumsum()
    bal = x.groupby(["sid", "widx"]).agg(bal_end=("cum", "last"), bal_min=("cum", "min")).reset_index()
    rel = (x.groupby("sid")["cum"].min() >= -1e-9).rename("stock_reliable")
    g = g.merge(wk, on=["sid", "widx"], how="left").merge(pr[["sid", "widx", "price"]], on=["sid", "widx"], how="left").merge(bal, on=["sid", "widx"], how="left")
    g = g.sort_values(["sid", "widx"]).reset_index(drop=True)
    g["y"] = g["y"].fillna(0.0)
    g["bal_end"] = g.groupby("sid")["bal_end"].ffill()
    g["bal_end"] = g["bal_end"].fillna(0.0)
    g["bal_min"] = g["bal_min"].fillna(g["bal_end"])
    feats = make_features(g, start, cal, S2)
    # leakage check: perturb everything from week t0 on; features up to and including t0 must not move
    t0 = S2["leakage_check_week"]
    g2 = g.copy(); m = g2["widx"] >= t0
    g2.loc[m, "y"] += 7.0; g2.loc[m, "price"] = 999.0; g2.loc[m, "bal_end"] = -5.0; g2.loc[m, "bal_min"] = -5.0
    f2 = make_features(g2, start, cal, S2)
    chk = (g["widx"] <= t0).to_numpy()
    NOT_FEATURES = {"reliable_asof", "stockout_contemp"}  # same-week analysis columns, never given to a model
    diff = [c for c in feats.columns if c not in NOT_FEATURES and not np.allclose(feats.loc[chk, c].to_numpy(dtype=float), f2.loc[chk, c].to_numpy(dtype=float), equal_nan=True)]
    assert not diff, f"LEAKAGE: features changed when future data changed: {diff}"
    out = pd.concat([g[["sid", "widx", "y"]], feats], axis=1)
    out["series_id"] = out["sid"].map(meta["sid_key"].str.replace("|", "__", regex=False))
    out["style"] = out["sid"].map(meta["style"]); out["loc"] = out["sid"].map(meta["loc"])
    out["brand"] = out["style"].map(style_brand); out["department"] = out["style"].map(style_dept); out["class"] = out["style"].map(style_class)
    out["location_type"] = kind
    out["week_start"] = w0 + pd.to_timedelta(out["widx"] * 7, unit="D")
    out["stock_never_negative"] = out["sid"].map(rel).fillna(False).astype(bool)  # series-level, uses whole history: reporting/analysis only
    out = out.drop(columns=["sid"])
    return out, x


results = {}
for kind in ("store", "warehouse"):
    out, xk = build(kind)
    results[kind] = out
    out.to_parquet(os.path.join(out_dir, f"weekly_{kind}.parquet"), index=False)
    if kind == "store":
        modelled_units = xk["gross_sale_units"].sum()
    else:
        modelled_units = xk["gross_sale_units"].sum()
rec.append(("Check: store y total in parquet", results["store"]["y"].sum()))
rec.append(("Check: warehouse y total in parquet (online sale + transfers out)", results["warehouse"]["y"].sum()))
rec.append(("  of which warehouse transfers out (incl. to transit parking, excl. rejected/sample/etc sites)", d.loc[d.ltype == "warehouse", "wh_transfer_units"].sum()))
assert abs(results["store"]["y"].sum() - df.loc[(df.ltype == "store") & df.is_merch, "gross_sale_units"].sum()) < 1e-6, "store reconciliation failed"
assert abs(results["warehouse"]["y"].sum() - (d.loc[d.ltype == "warehouse", "gross_sale_units"].sum() + d.loc[d.ltype == "warehouse", "wh_transfer_units"].sum())) < 1e-6, "warehouse reconciliation failed"

# ---- summary ------------------------------------------------------------------------------------------
L("# Dataset summary - Stage 2 (pseudonymised weekly datasets)\n")
L(f"Weeks: {w0.date()} .. {(w0 + pd.Timedelta(days=7 * last_idx)).date()} ({last_idx + 1} ISO weeks). Series start at the first ILE entry of any type for that style x location; zeros are filled only from there to the last week.\n")
L("## Unit reconciliation (gross Sales Shipment units; every number from the code)\n")
L(md_table(pd.DataFrame(rec, columns=["Item", "Units"]), floatfmt="{:,.0f}") + "\n")
L(f"Modelled store units = total - excluded locations - non-merch - warehouse: {tot_sale - rec[1][1] - rec[2][1] - rec[3][1]:,.0f} (parquet: {results['store']['y'].sum():,.0f}).\n")
for kind, o in results.items():
    L(f"## {kind}\n")
    ser = o.groupby("series_id").agg(n=("y", "size"), nz=("y", lambda s: int((s > 0).sum())), s1=("y", "sum"), rel=("stock_never_negative", "first"))
    L(md_table(pd.DataFrame([
        ("rows (series x week)", len(o)), ("series (style x location)", len(ser)), ("styles", o["style"].nunique()), ("locations", o["loc"].nunique()),
        ("series length min / median / max (weeks)", f"{ser.n.min()} / {ser.n.median():.0f} / {ser.n.max()}"),
        ("series with < %d weeks" % P["short_history_weeks"], int((ser.n < P["short_history_weeks"]).sum())),
        ("share of weeks with y = 0 (%)", f"{100 * (o.y == 0).mean():.1f}"),
        ("series with zero total demand", int((ser.s1 == 0).sum())),
        ("series whose balance never goes negative (%)", f"{100 * ser.rel.mean():.1f}"),
        ("weeks with a trusted balance (as-of, before it first goes negative) (%)", f"{100 * o['reliable_asof'].mean():.1f}"),
        ("weeks flagged stockout (end-of-week balance <= 0), among trusted weeks (%)", f"{100 * o['stockout_contemp'].mean():.1f}"),
    ], columns=["Metric", "Value"])) + "\n")
    ser["adi"] = ser.n / ser.nz.replace(0, np.nan)
    sub = o[o.y > 0].groupby("series_id")["y"].agg(["mean", "std", "count"])
    sub["cv2"] = (sub["std"].fillna(0) * np.sqrt((sub["count"] - 1).clip(lower=0) / sub["count"].replace(0, np.nan)) / sub["mean"]) ** 2
    ser = ser.join(sub["cv2"])
    cls = np.select([ser.nz == 0, ser.nz < 2, (ser.adi < P["sbc_adi_cut"]) & (ser.cv2 < P["sbc_cv2_cut"]), ser.adi < P["sbc_adi_cut"], ser.cv2 < P["sbc_cv2_cut"]],
                    ["no_demand", "insufficient(1 nonzero week)", "smooth", "erratic", "intermittent"], default="lumpy")
    cc = pd.Series(cls).value_counts().rename_axis("class").reset_index(name="series")
    cc["share_%"] = 100 * cc["series"] / cc["series"].sum()
    L("SBC class (series x week as built; ADI/CV2 per series):\n")
    L(md_table(cc) + "\n")
    feat_cols = [c for c in o.columns if c.startswith(("lag_", "roll_", "weeks_since", "price_", "stockout_lag", "stock_bal"))]
    nn = pd.DataFrame({"feature": feat_cols, "missing_%": [100 * o[c].isna().mean() for c in feat_cols]})
    L("Feature missingness (NaN = not computable yet, e.g. lag_52 before 52 weeks of history):\n")
    L(md_table(nn) + "\n")
L("## Leakage check\n\nPassed: for both datasets, replacing y, price and stock balance by arbitrary values from week %d onward leaves every feature of weeks <= %d unchanged.\n" % (S2["leakage_check_week"], S2["leakage_check_week"]))
L("`stockout_contemp` (same-week flag) is stored for analysis and for the stockout-aware evaluation only; the model feature is `stockout_lag1`.\n")
with open(os.path.join(cfg["paths"]["out_dir"], "dataset_summary.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(md))
print("done")
