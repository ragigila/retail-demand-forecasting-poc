"""Stage 3: rolling-origin experiment. usage:
  python experiment.py [--tag full|smoke] [--origins 39 43 ...] [--cap N] [--types store warehouse] [--no-ablation]
Predictions are cached per (type, origin) in <out>/predictions/*.npz, so a rerun resumes. Metrics: see evaluate.py"""
import argparse
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import models as M
from common import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="full")
ap.add_argument("--origins", type=int, nargs="*")
ap.add_argument("--cap", type=int)
ap.add_argument("--types", nargs="*", default=["store", "warehouse"])
ap.add_argument("--no-ablation", action="store_true")
args = ap.parse_args()

cfg = load_config()
S3, P = cfg["stage3"], cfg["profile"]
H = S3["horizon"]
base_out = cfg["paths"]["out_dir"]
out_dir = base_out if args.tag == "full" else os.path.join(base_out, args.tag)
pred_dir = os.path.join(out_dir, "predictions")
os.makedirs(pred_dir, exist_ok=True)

ORIGIN_COLS = ["age_weeks", "lag_1", "lag_2", "lag_4", "lag_8", "roll_mean_4", "roll_mean_8", "roll_mean_13",
               "weeks_since_last_sale", "price_lag1", "stockout_lag1", "stock_bal_lag1"]   # known at the forecast origin
TARGET_COLS = ["lag_52", "iso_week", "month", "holiday_days", "holiday_days_next_week", "payday_week"]  # lag_52 of the target week + calendar
STATIC_COLS = ["loc_c", "brand_c", "dept_c", "class_c"]
CATS = ["loc_c", "brand_c", "dept_c", "class_c"]
FEATS = ORIGIN_COLS + TARGET_COLS + ["horizon"] + STATIC_COLS
VARIANTS = {"lgbm": []}
if not args.no_ablation:
    VARIANTS.update({"lgbm_no_calendar": S3["ablation_groups"]["calendar"], "lgbm_no_price": S3["ablation_groups"]["price"],
                     "lgbm_no_stockout": S3["ablation_groups"]["stockout"]})


def load_kind(kind):
    d = pd.read_parquet(os.path.join(base_out, "datasets", f"weekly_{kind}.parquet"))
    d = d.sort_values(["series_id", "widx"], kind="stable").reset_index(drop=True)
    codes, uniq = pd.factorize(d["series_id"])
    n, W = len(uniq), int(d["widx"].max()) + 1
    Y = np.full((n, W), np.nan)
    Y[codes, d["widx"].to_numpy()] = d["y"].to_numpy()
    row_of = np.full((n, W), -1, dtype=np.int64)
    row_of[codes, d["widx"].to_numpy()] = np.arange(len(d))
    start = d.groupby(codes)["widx"].min().to_numpy()
    first = d.groupby(codes).first()
    stat = np.column_stack([pd.factorize(first[c])[0] for c in ["loc", "brand", "department", "class"]]).astype(np.float32)
    return dict(d=d, n=n, W=W, Y=Y, row_of=row_of, start=start, stat=stat, ids=np.array(uniq), loc=pd.factorize(first["loc"])[0],
                Fo=d[ORIGIN_COLS].to_numpy(np.float32), Ft=d[TARGET_COLS].to_numpy(np.float32))


def make_X(D, s, o, h):
    ro = D["row_of"][s, o]
    rt = D["row_of"][s, o + h - 1]
    return np.hstack([D["Fo"][ro], D["Ft"][rt], np.full((len(s), 1), h, np.float32), D["stat"][s]])


def train_set(D, T, sel):
    Xs, ys = [], []
    for h in range(1, H + 1):
        omax = T - h
        cols = np.arange(omax + 1)
        mask = (D["row_of"][sel][:, :omax + 1] >= 0) & (cols[None, :] >= (D["start"][sel][:, None] + 1))
        si, oi = np.nonzero(mask)
        s = sel[si]
        Xs.append(make_X(D, s, oi, h))
        ys.append(D["Y"][s, oi + h - 1])
    return np.vstack(Xs), np.concatenate(ys)


def fit_predict(D, T, sel, drop, seed):
    keep = [i for i, f in enumerate(FEATS) if f not in drop]
    names = [FEATS[i] for i in keep]
    X, y = train_set(D, T, sel)
    L = S3["lgbm"]
    params = dict(objective=L["objective"], tweedie_variance_power=L["tweedie_variance_power"], num_leaves=L["num_leaves"],
                  learning_rate=L["learning_rate"], min_child_samples=L["min_child_samples"], feature_fraction=L["feature_fraction"],
                  bagging_fraction=L["bagging_fraction"], bagging_freq=L["bagging_freq"], reg_lambda=L["reg_lambda"],
                  seed=seed, deterministic=True, force_row_wise=True, verbosity=-1)
    t0 = time.time()
    ds = lgb.Dataset(X[:, keep], y, feature_name=names, categorical_feature=[c for c in CATS if c in names], free_raw_data=True)
    booster = lgb.train(params, ds, num_boost_round=L["n_estimators"])
    allp = np.where(D["start"] <= T - 1)[0]
    pred = np.full((D["n"], H), np.nan, np.float32)
    for h in range(1, H + 1):
        Xp = make_X(D, allp, np.full(len(allp), T), h)
        pred[allp, h - 1] = np.clip(booster.predict(Xp[:, keep]), 0, None)
    return pred, len(y), time.time() - t0


def run_fold(kind, D, T, rng_seed):
    path = os.path.join(pred_dir, f"{kind}_T{T}.npz")
    if os.path.exists(path):
        print(f"[{kind} T={T}] cached", flush=True)
        return
    Y, n = D["Y"], D["n"]
    fc_ok = D["start"] <= T - 1
    A = np.full((n, H), np.nan)
    A[:] = Y[:, T:T + H]
    Ytr = Y[:, :T]
    cls, nz, n_w = M.sbc_classes(Ytr, P["sbc_adi_cut"], P["sbc_cv2_cut"])
    out = {"forecastable": fc_ok, "cls": cls.astype("U20"), "A": A.astype(np.float32), "nz": nz, "n_w": n_w}
    sn, cov = M.seasonal_naive(Y, T, H, S3["season"], S3["rollmean_window"])
    out["snaive_fb"] = sn
    out["snaive_cov"] = cov
    out["rollmean13"] = M.rollmean(Y, T, H, S3["rollmean_window"])
    out["croston"] = M.croston(Y, T, H, S3["croston_alpha"])
    out["tsb"] = M.tsb(Y, T, H, S3["tsb_alpha"], S3["tsb_beta"])
    idx = np.where(fc_ok & np.isin(cls, ["smooth", "erratic"]) & (nz >= S3["ets_min_nonzero_weeks"]))[0]
    t0 = time.time()
    out["ets"] = M.ets(Y, T, H, idx, out["tsb"])
    sbc = out["tsb"].copy()
    has = ~np.isnan(out["ets"][:, 0])
    sbc[has] = out["ets"][has]
    out["stat_by_class"] = sbc
    print(f"[{kind} T={T}] stat models done (ETS on {len(idx)} series, {time.time() - t0:.0f}s)", flush=True)
    rng = np.random.default_rng(cfg["seed"] + T)
    avail = np.where(D["start"] <= T - 2)[0]
    cap = args.cap or S3["train_series_cap"][kind]
    sel = np.sort(rng.choice(avail, size=min(cap, len(avail)), replace=False))
    for name, drop in VARIANTS.items():
        p, nrows, sec = fit_predict(D, T, sel, set(drop), cfg["seed"])
        out[name] = p
        print(f"[{kind} T={T}] {name}: {len(sel)} series, {nrows:,} training rows, {sec:.0f}s", flush=True)
    for k in out:
        if isinstance(out[k], np.ndarray) and out[k].dtype == np.float64:
            out[k] = out[k].astype(np.float32)
    np.savez_compressed(path, **out)


origins = args.origins or list(range(S3["first_origin"], int(1e9), S3["step"]))
for kind in args.types:
    D = load_kind(kind)
    valid = [T for T in origins if T + H <= D["W"]]
    print(f"== {kind}: {D['n']:,} series, {D['W']} weeks, origins {valid}", flush=True)
    for T in valid:
        run_fold(kind, D, T, cfg["seed"])
np.save(os.path.join(pred_dir, "_done.npy"), np.array([1]))
print("experiment done", flush=True)
