"""Feature construction on a contiguous weekly grid.

Input grid columns: sid, widx (int week index, contiguous inside each sid), y, price, bal_end, bal_min.
Every feature at week t uses only y / price / balance of weeks < t, or the calendar (known in advance).
"""
import numpy as np
import pandas as pd


def make_features(g, start_idx, cal, cfg2):
    """g: grid sorted by (sid, widx). start_idx: Series (index = sid) with each series' first week index.
    cal: DataFrame indexed by widx with calendar columns. Returns a new DataFrame of feature columns."""
    sid = g["sid"]
    y = g["y"]
    f = pd.DataFrame(index=g.index)
    f["age_weeks"] = g["widx"] - g["sid"].map(start_idx)  # prior weeks available (0 at first week)
    for k in cfg2["lags"]:
        f[f"lag_{k}"] = y.groupby(sid, sort=False).shift(k)
    cs = y.groupby(sid, sort=False).cumsum()
    s_prev = cs.groupby(sid, sort=False).shift(1, fill_value=0.0)  # sum of y over weeks < t
    for w in cfg2["rolling_windows"]:
        s_old = cs.groupby(sid, sort=False).shift(w + 1, fill_value=0.0)  # sum over weeks < t-w
        cnt = np.minimum(w, f["age_weeks"]).astype(float)
        f[f"roll_mean_{w}"] = np.where(cnt > 0, (s_prev - s_old) / cnt.replace(0, np.nan), np.nan)
    lp = pd.Series(np.where(y > 0, g["widx"].astype(float), np.nan), index=g.index)
    lp = lp.groupby(sid, sort=False).ffill().groupby(sid, sort=False).shift(1)
    f["weeks_since_last_sale"] = g["widx"] - lp
    f["price_lag1"] = g["price"].groupby(sid, sort=False).ffill().groupby(sid, sort=False).shift(1)
    # a balance is "trusted" from the start until it first goes negative (evidence the opening balance / history is incomplete)
    f["reliable_asof"] = (g["bal_min"].groupby(sid, sort=False).cummin() >= -1e-9)
    so = pd.Series(np.where(f["reliable_asof"], (g["bal_end"] <= 0).astype(float), np.nan), index=g.index)
    f["stockout_contemp"] = so  # same-week flag: analysis only, NOT a model feature
    f["stockout_lag1"] = so.groupby(sid, sort=False).shift(1)
    f["stock_bal_lag1"] = g["bal_end"].groupby(sid, sort=False).shift(1)
    for c in cal.columns:
        f[c] = g["widx"].map(cal[c]).to_numpy()
    return f
