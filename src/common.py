"""Shared loading + pseudonymisation for the ILE PoC. Real Item No / Location Code never leave this module
except into the private id_map.json."""
import glob, json, os
import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


ROOT = os.path.dirname(HERE)


def load_config(path=None):
    """config.local.yaml (git-ignored, your real values) wins over config.example.yaml (placeholders)."""
    if path is None:
        for cand in ("config.local.yaml", "config.example.yaml"):
            if os.path.exists(os.path.join(ROOT, cand)):
                path = os.path.join(ROOT, cand)
                break
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(path))
    for k, v in cfg["paths"].items():
        if not os.path.isabs(v):
            cfg["paths"][k] = os.path.normpath(os.path.join(base, v))
    lt = cfg.get("stage2", {}).get("location_types_file")
    if lt and not os.path.isabs(lt):
        cfg["stage2"]["location_types_file"] = os.path.normpath(os.path.join(base, lt))
    return cfg


def _read_ndjson(path):
    return pd.read_json(path, lines=True, dtype=False, convert_dates=False)


def load_pseudonymised(cfg):
    """Read raw ILE NDJSON, pseudonymise IDs, drop unneeded columns. Returns (df, load_info)."""
    files = sorted(glob.glob(os.path.join(cfg["paths"]["raw_dir"], f"*_{cfg['extract']['company']}_*.ndjson")))
    files = [f for f in files if os.path.exists(f + ".done") and os.path.basename(f).split("_")[0] in cfg["extract"]["brands"]]
    raw = pd.concat([_read_ndjson(f) for f in files], ignore_index=True)
    info = {"files": len(files), "rows_raw": len(raw)}
    info["duplicate_entry_no"] = int(raw["Entry_No"].duplicated().sum())
    raw = raw.drop_duplicates("Entry_No")
    raw["Posting_Date"] = pd.to_datetime(raw["Posting_Date"])

    dim1_eq_loc = (raw["Global_Dimension_1_Code"] == raw["Location_Code"]).to_numpy()
    rng = np.random.default_rng(cfg["seed"])
    id_map = {}
    for col, prefix, width in (("Item_No", "SKU_", 5), ("Location_Code", "LOC_", 2)):
        uniq = np.array(sorted(raw[col].astype(str).unique()))
        order = rng.permutation(len(uniq))
        pseudo = {u: f"{prefix}{int(o) + 1:0{width}d}" for u, o in zip(uniq, order)}
        id_map[col] = pseudo
        raw[col] = raw[col].astype(str).map(pseudo)
    os.makedirs(cfg["paths"]["private_dir"], exist_ok=True)
    with open(os.path.join(cfg["paths"]["private_dir"], "id_map.json"), "w", encoding="utf-8") as f:
        json.dump(id_map, f)

    df = pd.DataFrame({
        "entry_no": raw["Entry_No"].astype("int64"),
        "date": raw["Posting_Date"],
        "entry_type": raw["Entry_Type"],
        "doc_type": raw["Document_Type"].replace("", "(none)").fillna("(none)").str.strip().replace("", "(none)"),
        "doc": pd.factorize(raw["Document_No"])[0].astype("int64"),
        "order": np.where(raw["Order_No"].fillna("") == "", -1, pd.factorize(raw["Order_No"])[0]).astype("int64"),
        "sku": raw["Item_No"],
        "loc": raw["Location_Code"],
        "brand": raw["Global_Dimension_2_Code"],
        "dim1_eq_loc": dim1_eq_loc,
        "qty": raw["Quantity"].astype("float64"),
        "sales_amt": raw["Sales_Amount_Actual"].astype("float64"),
    })
    # Item master (category), keyed by the same pseudonym; only for items that appear in the ledger.
    return df.sort_values(["entry_no"]).reset_index(drop=True), info, id_map


def load_items(cfg, id_map):
    path = cfg["paths"]["items_file"]
    if not os.path.exists(path):
        return None
    it = _read_ndjson(path)
    it["sku"] = it["No"].astype(str).map(id_map["Item_No"])
    it = it.dropna(subset=["sku"]).rename(columns={"No": "item_no_raw"})  # raw No kept in memory only
    # canonical names for the (possibly company-specific) Item Card attributes configured in bc_item_fields
    it = it.rename(columns={bc: canon for canon, bc in cfg["bc_item_fields"].items()})
    return it.drop_duplicates("sku")


def md_table(df, floatfmt="{:,.2f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append(floatfmt.format(v))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
