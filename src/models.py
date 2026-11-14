"""Forecasting models on a matrix Y[series, week] (NaN before a series starts).
All functions forecast weeks T .. T+H-1 using only Y[:, :T]. Returns arrays of shape (n_series, H)."""
import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")

import numpy as np

CLASSES = ["smooth", "erratic", "intermittent", "lumpy", "single_demand", "no_demand_so_far"]


def sbc_classes(Ytr, adi_cut, cv2_cut):
    n_w = (~np.isnan(Ytr)).sum(1).astype(float)
    pos = np.nan_to_num(Ytr) > 0
    pv = np.where(pos, np.nan_to_num(Ytr), 0.0)
    nz = pos.sum(1).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = pv.sum(1) / nz
        var = np.maximum((pv ** 2).sum(1) / nz - mean ** 2, 0.0)
        cv2 = var / mean ** 2
        adi = n_w / nz
    cls = np.select(
        [nz == 0, nz == 1, (adi < adi_cut) & (cv2 < cv2_cut), adi < adi_cut, cv2 < cv2_cut],
        ["no_demand_so_far", "single_demand", "smooth", "erratic", "intermittent"], default="lumpy")
    return cls, nz, n_w


def rollmean(Y, T, H, window):
    m = np.nanmean(Y[:, max(T - window, 0):T], axis=1)
    return np.repeat(m[:, None], H, axis=1)


def seasonal_naive(Y, T, H, season, window):
    """y[t-season] when that week exists for the series, otherwise the rolling mean (fallback, documented)."""
    fb = rollmean(Y, T, H, window)
    out = fb.copy()
    covered = np.zeros(Y.shape[0], dtype=float)
    for h in range(H):
        src = T + h - season
        if src >= 0:
            v = Y[:, src]
            ok = ~np.isnan(v)
            out[ok, h] = v[ok]
            covered += ok
    return out, covered / H


def tsb(Y, T, H, alpha, beta):
    Ytr = Y[:, :T]
    obs = ~np.isnan(Ytr)
    y = np.nan_to_num(Ytr)
    pos = y > 0
    nz = pos.sum(1)
    n = obs.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, nz / n, 0.0)  # initial demand probability = share of non-zero weeks in the training window
        z = np.where(nz > 0, (y * pos).sum(1) / nz, 0.0)  # initial size = mean non-zero demand
    for t in range(T):
        o = obs[:, t]
        i = pos[:, t]
        p = np.where(o, p + beta * (i - p), p)
        z = np.where(o & i, z + alpha * (y[:, t] - z), z)
    f = p * z
    return np.repeat(f[:, None], H, axis=1)


def croston(Y, T, H, alpha):
    Ytr = Y[:, :T]
    obs = ~np.isnan(Ytr)
    y = np.nan_to_num(Ytr)
    pos = y > 0
    nz = pos.sum(1)
    n = obs.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(nz > 0, (y * pos).sum(1) / nz, 0.0)
        q = np.where(nz > 0, n / nz, 1.0)
    c = np.zeros(Y.shape[0])
    for t in range(T):
        o = obs[:, t]
        c = np.where(o, c + 1, c)
        i = o & pos[:, t]
        z = np.where(i, z + alpha * (y[:, t] - z), z)
        q = np.where(i, q + alpha * (c - q), q)
        c = np.where(i, 0.0, c)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(nz > 0, z / q, 0.0)
    return np.repeat(f[:, None], H, axis=1)


def ets(Y, T, H, idx, fallback):
    """ETS(A,N,N) vs damped ETS(A,Ad,N), picked by AIC, on the given series indices. No seasonal term:
    the longest history is 71 weeks (< 2 seasons). Falls back to `fallback` rows on failure."""
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    out = np.full((Y.shape[0], H), np.nan)
    for s in idx:
        y = Y[s, :T]
        y = y[~np.isnan(y)]
        best, best_aic = None, np.inf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for kw in ({"trend": None}, {"trend": "add", "damped_trend": True}):
                try:
                    r = ETSModel(y, error="add", **kw).fit(disp=False)
                    if np.isfinite(r.aic) and r.aic < best_aic:
                        best, best_aic = r, r.aic
                except Exception:
                    continue
        if best is None:
            out[s] = fallback[s]
        else:
            out[s] = np.clip(np.asarray(best.forecast(H)), 0, None)
    return out
