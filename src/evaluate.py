"""Stage 3/4: turn cached predictions into metric tables, figures and results.md. All numbers come from the cached predictions.
usage: python evaluate.py [--tag full|smoke]"""
import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import models as M
from common import load_config, md_table

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="full")
args = ap.parse_args()
cfg = load_config()
S3 = cfg["stage3"]
H = S3["horizon"]
base_out = cfg["paths"]["out_dir"]
out_dir = base_out if args.tag == "full" else os.path.join(base_out, args.tag)
pred_dir = os.path.join(out_dir, "predictions")
fig_dir = os.path.join(out_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

MODELS = ["snaive_fb", "rollmean13", "croston", "tsb", "ets", "stat_by_class", "lgbm", "lgbm_no_calendar", "lgbm_no_price", "lgbm_no_stockout"]
CLASSES = M.CLASSES


def load_meta(kind):
    d = pd.read_parquet(os.path.join(base_out, "datasets", f"weekly_{kind}.parquet"), columns=["series_id", "widx", "y", "loc"])
    d = d.sort_values(["series_id", "widx"], kind="stable").reset_index(drop=True)
    codes, uniq = pd.factorize(d["series_id"])
    n, W = len(uniq), int(d["widx"].max()) + 1
    Y = np.full((n, W), np.nan)
    Y[codes, d["widx"].to_numpy()] = d["y"].to_numpy()
    loc = d.groupby(codes)["loc"].first().pipe(lambda s: pd.factorize(s)[0])
    return dict(Y=Y, loc=loc, ever=np.nansum(Y, 1) > 0, ids=np.array(uniq))


rows, agg_rows, cov_rows = [], [], []
per_series = defaultdict(list)
preds_cache = {}
for kind in ["store", "warehouse"]:
    files = sorted(glob.glob(os.path.join(pred_dir, f"{kind}_T*.npz")), key=lambda p: int(re.search(r"_T(\d+)", p).group(1)))
    if not files:
        continue
    meta = load_meta(kind)
    for f in files:
        T = int(re.search(r"_T(\d+)", f).group(1))
        z = np.load(f)
        preds_cache[(kind, T)] = {k: z[k] for k in z.files}
        fc_ok, cls, A = z["forecastable"], z["cls"], z["A"].astype(float)
        Yfull = meta["Y"]
        new_units = np.nansum(Yfull[~fc_ok, T:T + H])
        cov_rows.append(dict(fold=T, type=kind, units_forecastable=np.nansum(A[fc_ok]), units_in_series_not_yet_started=new_units,
                             snaive_true_seasonal_share=float(z["snaive_cov"][fc_ok].mean()), series_forecast=int(fc_ok.sum())))
        for group in ["ever_sold", "never_sold"]:
            gmask = fc_ok & (meta["ever"] if group == "ever_sold" else ~meta["ever"])
            for cname in ["ALL"] + CLASSES:
                cmask = gmask & ((cls == cname) if cname != "ALL" else True)
                for model in MODELS:
                    if model not in z.files:
                        continue
                    Pm = z[model].astype(float)
                    m = cmask & ~np.isnan(Pm[:, 0])
                    if m.sum() == 0:
                        continue
                    a, p = A[m], Pm[m]
                    sn = z["snaive_fb"].astype(float)[m]
                    sa = a.sum()
                    ae = np.abs(p - a).sum(1)
                    sai = a.sum(1)
                    ok = sai > 0
                    r = dict(fold=T, type=kind, group=group, cls=cname, model=model, n_series=int(m.sum()), n_cells=int(a.size),
                             sum_actual=sa, sum_forecast=p.sum(), sum_abs_err=ae.sum(), sum_abs_err_snaive=np.abs(sn - a).sum(),
                             median_series_wape=float(np.median(ae[ok] / sai[ok])) if ok.any() else np.nan)
                    rows.append(r)
                    if group == "ever_sold" and ok.any():
                        per_series[(kind, cname, model)].append(ae[ok] / sai[ok])
        # aggregation levels (ever-sold series): per location x horizon cell, and whole location-type per horizon
        gm = fc_ok & meta["ever"]
        for model in MODELS:
            if model not in z.files:
                continue
            Pm = z[model].astype(float)
            m = gm & ~np.isnan(Pm[:, 0])
            if m.sum() == 0:
                continue
            nl = meta["loc"].max() + 1
            for h in range(H):
                pass
            keyloc = (meta["loc"][m][:, None] * H + np.arange(H)[None, :]).ravel()
            fa = np.bincount(keyloc, weights=A[m].ravel(), minlength=nl * H)
            ff = np.bincount(keyloc, weights=Pm[m].ravel(), minlength=nl * H)
            agg_rows.append(dict(fold=T, type=kind, level="location_x_week", model=model, sum_actual=fa.sum(), sum_forecast=ff.sum(), sum_abs_err=np.abs(ff - fa).sum()))
            ta, tf = A[m].sum(0), Pm[m].sum(0)
            agg_rows.append(dict(fold=T, type=kind, level="type_x_week", model=model, sum_actual=ta.sum(), sum_forecast=tf.sum(), sum_abs_err=np.abs(tf - ta).sum()))

R = pd.DataFrame(rows)
for df_ in (R,):
    df_["wape"] = np.where(df_["sum_actual"] > 0, df_["sum_abs_err"] / df_["sum_actual"].replace(0, np.nan), np.nan)
    df_["bias"] = np.where(df_["sum_actual"] > 0, (df_["sum_forecast"] - df_["sum_actual"]) / df_["sum_actual"].replace(0, np.nan), np.nan)
    df_["wape_snaive"] = np.where(df_["sum_actual"] > 0, df_["sum_abs_err_snaive"] / df_["sum_actual"].replace(0, np.nan), np.nan)
    df_["rel_wape_vs_snaive"] = df_["wape"] / df_["wape_snaive"]
# pooled over folds (sums, not averages of per-fold ratios)
keys = ["type", "group", "cls", "model"]
pool = R.groupby(keys).agg(n_series=("n_series", "sum"), n_cells=("n_cells", "sum"), sum_actual=("sum_actual", "sum"), sum_forecast=("sum_forecast", "sum"),
                           sum_abs_err=("sum_abs_err", "sum"), sum_abs_err_snaive=("sum_abs_err_snaive", "sum")).reset_index()
pool["fold"] = "ALL"
pool["wape"] = np.where(pool["sum_actual"] > 0, pool["sum_abs_err"] / pool["sum_actual"].replace(0, np.nan), np.nan)
pool["bias"] = np.where(pool["sum_actual"] > 0, (pool["sum_forecast"] - pool["sum_actual"]) / pool["sum_actual"].replace(0, np.nan), np.nan)
pool["wape_snaive"] = np.where(pool["sum_actual"] > 0, pool["sum_abs_err_snaive"] / pool["sum_actual"].replace(0, np.nan), np.nan)
pool["rel_wape_vs_snaive"] = pool["wape"] / pool["wape_snaive"]
pool["median_series_wape"] = [np.median(np.concatenate(per_series[(t, c, m)])) if (t, c, m) in per_series and g == "ever_sold" else np.nan
                              for t, g, c, m in zip(pool["type"], pool["group"], pool["cls"], pool["model"])]
cols = ["fold", "type", "group", "cls", "model", "n_series", "n_cells", "sum_actual", "sum_forecast", "sum_abs_err", "wape", "bias", "wape_snaive",
        "rel_wape_vs_snaive", "median_series_wape"]
RT = pd.concat([R[cols], pool[cols]], ignore_index=True).rename(columns={"cls": "series_class"})
RT["fold"] = RT["fold"].astype(str)
RT.to_csv(os.path.join(out_dir, "results_tables.csv"), index=False)

AG = pd.DataFrame(agg_rows)
AGp = AG.groupby(["type", "level", "model"])[["sum_actual", "sum_forecast", "sum_abs_err"]].sum().reset_index()
AGp["fold"] = "ALL"
AG = pd.concat([AG, AGp], ignore_index=True)
AG["fold"] = AG["fold"].astype(str)
AG["wape"] = AG["sum_abs_err"] / AG["sum_actual"]
AG["bias"] = (AG["sum_forecast"] - AG["sum_actual"]) / AG["sum_actual"]
AG.to_csv(os.path.join(out_dir, "results_agg_levels.csv"), index=False)
COV = pd.DataFrame(cov_rows)
COV.to_csv(os.path.join(out_dir, "coverage.csv"), index=False)

# ---- ablation table ------------------------------------------------------------------------------------
ab = RT[(RT["group"] == "ever_sold") & RT["model"].isin(["lgbm", "lgbm_no_calendar", "lgbm_no_price", "lgbm_no_stockout"])].copy()
full = ab[ab["model"] == "lgbm"][["fold", "type", "series_class", "wape", "bias"]].rename(columns={"wape": "wape_full", "bias": "bias_full"})
ab = ab[ab["model"] != "lgbm"].merge(full, on=["fold", "type", "series_class"], how="left")
ab["fold"] = ab["fold"].astype(str)
ab["variant_removed"] = ab["model"].str.replace("lgbm_no_", "", regex=False)
ab["delta_wape_vs_full"] = ab["wape"] - ab["wape_full"]
ab["delta_bias_vs_full"] = ab["bias"] - ab["bias_full"]
ab[["fold", "type", "series_class", "variant_removed", "wape_full", "wape", "delta_wape_vs_full", "bias_full", "bias", "delta_bias_vs_full", "n_series"]].to_csv(
    os.path.join(out_dir, "ablation_table.csv"), index=False)

# ---- figures -------------------------------------------------------------------------------------------
plot_models = ["snaive_fb", "rollmean13", "croston", "tsb", "ets", "stat_by_class", "lgbm"]
for kind in RT["type"].unique():
    sub = RT[(RT["fold"] == "ALL") & (RT["group"] == "ever_sold") & (RT["type"] == kind) & RT["model"].isin(plot_models)]
    for metric, ttl in (("wape", "WAPE"), ("bias", "Bias")):
        piv = sub.pivot_table(index="series_class", columns="model", values=metric).reindex(["ALL"] + CLASSES).dropna(how="all")[[m for m in plot_models if m in sub["model"].unique()]]
        ax = piv.plot(kind="bar", figsize=(11, 4.5), width=0.85)
        ax.set_title(f"{ttl} per model and series class (pooled over folds, {kind}, series with demand)")
        ax.set_xlabel("series class (SBC on training window)"); ax.set_ylabel(ttl)
        if metric == "bias":
            ax.axhline(0, color="k", lw=0.8)
        plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(os.path.join(fig_dir, f"{metric}_by_model_class_{kind}.png"), dpi=130); plt.close()
    # example series: seeded random pick among series with sizeable demand
    meta = load_meta(kind)
    rng = np.random.default_rng(cfg["seed"])
    tot = np.nansum(meta["Y"], 1)
    cand = np.where(tot >= 40)[0]
    pick = rng.choice(cand, size=min(6, len(cand)), replace=False)
    fig, axs = plt.subplots(len(pick), 1, figsize=(11, 2.2 * len(pick)), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, s in zip(axs, pick):
        ax.plot(meta["Y"][s], color="k", lw=1, label="actual")
        for T in sorted(t for (k, t) in preds_cache if k == kind):
            pz = preds_cache[(kind, T)]
            if pz["forecastable"][s]:
                ax.plot(range(T, T + H), pz["lgbm"][s], color="tab:red", marker="o", ms=3, lw=1, label="lgbm" if T == 39 else None)
                ax.plot(range(T, T + H), pz["tsb"][s], color="tab:blue", marker="s", ms=3, lw=1, label="tsb" if T == 39 else None)
        ax.set_ylabel(str(meta["ids"][s]).replace("__", "\n"), fontsize=6)
    axs[0].legend(loc="upper left", fontsize=7); axs[-1].set_xlabel("week index")
    plt.suptitle(f"Actual vs forecast, example series ({kind}, pseudonymised IDs)"); plt.tight_layout(); plt.savefig(os.path.join(fig_dir, f"examples_{kind}.png"), dpi=130); plt.close()

# ---- results.md ----------------------------------------------------------------------------------------
md = []
L = md.append
L("# Results - Stage 3 (rolling-origin, as computed)\n")
folds = sorted(int(x) for x in R["fold"].unique())
L(f"Setup: horizon {H} weeks, origins (week index) {folds}, training = all weeks before the origin, test = the {H} weeks from the origin; series = style x location, "
  f"weekly gross sales. WAPE = sum|forecast-actual| / sum actual; bias = (sum forecast - sum actual) / sum actual. Pooled numbers sum errors over all folds. "
  f"Series classes (SBC) are computed from each fold's training window only. Main table = series that sell at least once in the full period; series that never sell are reported separately below.\n")
L("## 1. All series with demand, pooled over folds\n")
for kind in RT["type"].unique():
    t = RT[(RT["fold"] == "ALL") & (RT["group"] == "ever_sold") & (RT["type"] == kind) & (RT["series_class"] == "ALL")]
    t = t[["model", "n_series", "sum_actual", "wape", "bias", "rel_wape_vs_snaive", "median_series_wape"]].rename(columns={"n_series": "series-folds"})
    L(f"### {kind}\n"); L(md_table(t.sort_values("wape"), floatfmt="{:,.3f}") + "\n")
    L("`rel_wape_vs_snaive` = model WAPE / seasonal-naive-with-fallback WAPE on the same series (`ets` is only defined for smooth/erratic series with enough history). "
      "`median_series_wape` = median over series-folds of each series' own WAPE (series with zero test demand excluded).\n")
L("## 2. Per series class, pooled over folds (WAPE | bias)\n")
for kind in RT["type"].unique():
    sub = RT[(RT["fold"] == "ALL") & (RT["group"] == "ever_sold") & (RT["type"] == kind) & RT["model"].isin(plot_models)]
    w = sub.pivot_table(index="series_class", columns="model", values="wape").reindex(CLASSES)
    b = sub.pivot_table(index="series_class", columns="model", values="bias").reindex(CLASSES)
    n = sub[sub["model"] == "lgbm"].set_index("series_class")["n_series"].reindex(CLASSES)
    w.insert(0, "series-folds", n)
    L(f"### {kind} - WAPE\n"); L(md_table(w.reset_index(), floatfmt="{:,.3f}") + "\n")
    L(f"### {kind} - bias\n"); L(md_table(b.reset_index(), floatfmt="{:,.3f}") + "\n")
L("## 3. Per fold: WAPE of lgbm, tsb, seasonal-naive-with-fallback (all series with demand)\n")
for kind in RT["type"].unique():
    sub = RT[(RT["fold"] != "ALL") & (RT["group"] == "ever_sold") & (RT["type"] == kind) & (RT["series_class"] == "ALL") & RT["model"].isin(["lgbm", "tsb", "snaive_fb", "rollmean13"])]
    pv = sub.pivot_table(index="fold", columns="model", values="wape")
    pb = sub.pivot_table(index="fold", columns="model", values="bias").add_suffix("_bias")
    L(f"### {kind}\n"); L(md_table(pv.join(pb).reset_index(), floatfmt="{:,.3f}") + "\n")
L("## 4. Aggregated levels (forecasts summed before scoring), pooled over folds\n")
a = AG[(AG["fold"] == "ALL") & AG["model"].isin(["snaive_fb", "rollmean13", "tsb", "stat_by_class", "lgbm"])]
L(md_table(a.pivot_table(index=["type", "level"], columns="model", values="wape").reset_index(), floatfmt="{:,.3f}") + "\n")
L("## 5. Feature ablation (lgbm; same folds, same training series; pooled over folds, all series with demand)\n")
for kind in ab["type"].unique():
    t = ab[(ab["fold"] == "ALL") & (ab["type"] == kind) & (ab["series_class"] == "ALL")][["variant_removed", "wape_full", "wape", "delta_wape_vs_full", "bias_full", "bias", "delta_bias_vs_full"]]
    L(f"### {kind}\n"); L(md_table(t, floatfmt="{:,.4f}") + "\n")
    pf = ab[(ab["fold"] != "ALL") & (ab["type"] == kind) & (ab["series_class"] == "ALL")].pivot_table(index="fold", columns="variant_removed", values="delta_wape_vs_full")
    L(f"Per-fold change in WAPE when the group is removed ({kind}); positive = worse without it:\n"); L(md_table(pf.reset_index(), floatfmt="{:,.4f}") + "\n")
L("## 6. Series that never sell (separate report)\n")
ns = RT[(RT["fold"] == "ALL") & (RT["group"] == "never_sold") & (RT["series_class"] == "ALL")][["type", "model", "n_series", "sum_forecast"]]
tot = RT[(RT["fold"] == "ALL") & (RT["group"] == "ever_sold") & (RT["series_class"] == "ALL") & (RT["model"] == "snaive_fb")].set_index("type")["sum_actual"]
ns["forecast_units_as_share_of_actual_units_of_selling_series_%"] = [100 * s / tot.get(t, np.nan) for t, s in zip(ns["type"], ns["sum_forecast"])]
L("Actual demand of these series is 0 in every week, so every forecast unit is a false positive.\n")
L(md_table(ns.sort_values(["type", "sum_forecast"]), floatfmt="{:,.1f}") + "\n")
L("## 7. Coverage\n")
L(md_table(COV, floatfmt="{:,.3f}") + "\n")
L("`snaive_true_seasonal_share` = share of forecast cells where the seasonal-naive model could use y[t-52]; elsewhere it falls back to the 13-week rolling mean. "
  "`units_in_series_not_yet_started` = test-window demand of style x location series that did not exist at the origin (no forecast possible).\n")
L("## 8. Descriptive winners (lowest pooled WAPE per class; not a claim of significance)\n")
for kind in RT["type"].unique():
    sub = RT[(RT["fold"] == "ALL") & (RT["group"] == "ever_sold") & (RT["type"] == kind) & RT["model"].isin(plot_models)]
    sub = sub[~((sub["series_class"] == "ALL") & (sub["model"] == "ets"))]  # ETS covers only a few smooth/erratic series: not comparable on 'ALL'
    best = sub.loc[sub.groupby("series_class")["wape"].idxmin()][["series_class", "model", "wape"]].set_index("series_class").reindex(["ALL"] + CLASSES).dropna()
    lg = sub[sub["model"] == "lgbm"].set_index("series_class")["wape"]
    sn = sub[sub["model"] == "snaive_fb"].set_index("series_class")["wape"]
    best["lgbm_wape"] = lg; best["snaive_fb_wape"] = sn
    best["lgbm_beats_snaive_fb"] = best["lgbm_wape"] < best["snaive_fb_wape"]
    L(f"### {kind}\n"); L(md_table(best.reset_index(), floatfmt="{:,.3f}") + "\n")
L("## 9. Limitations found (numbers computed here or in dataset_summary.md)\n")
tot_u = COV.groupby("type")[["units_forecastable", "units_in_series_not_yet_started"]].sum()
for kind, r in tot_u.iterrows():
    L(f"- {kind}: {100 * r.units_in_series_not_yet_started / (r.units_forecastable + r.units_in_series_not_yet_started):.1f}% of test-window units belong to style x location series that did not exist at the forecast origin and cannot be forecast by any model here (cold start).")
for kind, r in COV.groupby("type")["snaive_true_seasonal_share"].mean().items():
    L(f"- {kind}: the seasonal-naive baseline could use a true same-week-last-year value in {100 * r:.1f}% of forecast cells (mean over folds); elsewhere it is the 13-week rolling mean, because history is at most 71 weeks (< 2 seasons).")
L("- The LightGBM model is global per location type, trained on a seeded random subset of series (see config.yaml `train_series_cap`) with fixed, untuned hyper-parameters; all series are forecast. Statistical models use fixed smoothing constants. Nothing was tuned on test data.")
L("- ETS is fitted only for smooth/erratic series with enough non-zero weeks (no seasonal term; history < 2 seasons). Prophet and MinT reconciliation (both optional in the plan) were not run; the warehouse hierarchy is trivial (2 locations).")
L("- The split into 'series with demand' and 'series that never sell' uses the full 71 weeks (selection on outcome); the SBC class used in all class tables uses the training window only.")
L("- No promotion data. Price feature = last observed average price per unit from the ledger (missing until a series' first sale). The stockout proxy cannot separate a true stock-out from a style that is simply not replenished, and is only available while the running balance has never been negative.")
L("- Demand is gross sales out (returns ignored); exchanges are slightly double-counted (decisions C6).")
with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(md))
print("evaluation written to", out_dir)
