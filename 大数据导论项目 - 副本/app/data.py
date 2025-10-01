from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd

# ---- Column mapping (Chinese aliases -> standardized names) ----
COLUMN_ALIASES = {
    "价格": "price",
    "售价": "price",
    "成交价": "price",
    "功率": "power",
    "最大功率": "power",
    "里程": "kilometer",
    "公里数": "kilometer",
    "行驶里程": "kilometer",
    "上牌时间": "regDate",
    "注册日期": "regDate",
    "品牌": "brand",
    "车身类型": "bodyType",
    "燃料类型": "fuelType",
    "燃油类型": "fuelType",
    "变速箱": "gearbox",
}


NUMERIC_CANDIDATES = [
    "price",
    "power",
    "kilometer",
]

CATEGORICAL_CANDIDATES = [
    "brand",
    "bodyType",
    "fuelType",
    "gearbox",
]

DATE_CANDIDATES = [
    "regDate",
]


def _detect_excel_sheet(path: str) -> str:
    xls = pd.ExcelFile(path)
    # pick the first sheet with >= 3 columns of non-empty header row
    for sheet in xls.sheet_names:
        try:
            preview = pd.read_excel(path, sheet_name=sheet, header=None, nrows=10)
            if preview.shape[1] >= 3:
                return sheet
        except Exception:
            continue
    return xls.sheet_names[0]


def _detect_header_row(path: str, sheet_name: str, max_rows: int = 10) -> int:
    # return 0-based header row index
    preview = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=max_rows)
    best_row = 0
    best_score = -1
    for i in range(preview.shape[0]):
        row = preview.iloc[i]
        # score: count non-null string-like cells
        score = int(sum((isinstance(v, str) and v.strip() != "") for v in row.values))
        if score > best_score:
            best_score = score
            best_row = i
    return best_row


def load_dataset(path: str, sheet_name: str = None, header_row_index: int = None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        sn = sheet_name or _detect_excel_sheet(path)
        hri = header_row_index
        if hri is None:
            try:
                hri = _detect_header_row(path, sn)
            except Exception:
                hri = 0
        df = pd.read_excel(path, sheet_name=sn, header=hri)
    else:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols: List[str] = []
    for c in df.columns:
        c_str = str(c).strip()
        mapped = COLUMN_ALIASES.get(c_str, None)
        if mapped is None:
            # also try lower case no spaces
            c_norm = c_str.replace(" ", "").lower()
            # simple heuristics
            if c_norm in ["price", "售价", "价格", "成交价"]:
                mapped = "price"
            elif c_norm in ["power", "功率", "最大功率"]:
                mapped = "power"
            elif c_norm in ["kilometer", "km", "里程", "公里数", "行驶里程"]:
                mapped = "kilometer"
            elif c_norm in ["regdate", "上牌时间", "注册日期"]:
                mapped = "regDate"
            elif c_norm in ["brand", "品牌"]:
                mapped = "brand"
            elif c_norm in ["bodytype", "车身类型"]:
                mapped = "bodyType"
            elif c_norm in ["fueltype", "燃料类型", "燃油类型"]:
                mapped = "fuelType"
            elif c_norm in ["gearbox", "变速箱"]:
                mapped = "gearbox"
            else:
                mapped = c_str
        new_cols.append(mapped)
    df.columns = new_cols
    return df


def _clean_numeric_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
    s = s.str.replace("¥", "", regex=False).str.replace("￥", "", regex=False).str.replace("$", "", regex=False)
    # handle chinese '万' if appears: convert e.g., 12万 -> 120000
    def to_num(x: str):
        if x is None or x == "" or x.lower() == "nan":
            return np.nan
        try:
            if "万" in x:
                return float(x.replace("万", "")) * 10000.0
            return float(x)
        except Exception:
            return np.nan
    return s.apply(to_num)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in DATE_CANDIDATES:
        if col in df.columns:
            # try yyyymmdd
            df[col] = pd.to_datetime(df[col], errors="coerce")
            mask_na = df[col].isna()
            if mask_na.any():
                # try parse as int-like yyyymmdd
                try:
                    alt = pd.to_datetime(df.loc[mask_na, col].astype(str), format="%Y%m%d", errors="coerce")
                    df.loc[mask_na, col] = alt
                except Exception:
                    pass

    for col in NUMERIC_CANDIDATES:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = _clean_numeric_series(df[col])
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_CANDIDATES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Numeric: median impute
    for col in df.select_dtypes(include=[np.number]).columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Datetime: fill with median date
    for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
        median_ts = df[col].dropna().median()
        df[col] = df[col].fillna(median_ts)

    # Categorical: fill with mode
    for col in df.select_dtypes(include=["category", "object"]).columns:
        mode_val = df[col].mode(dropna=True)
        fill_val = mode_val.iloc[0] if not mode_val.empty else "unknown"
        df[col] = df[col].fillna(fill_val)
    return df


from typing import Optional


def cap_outliers_iqr(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None, whisker: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    if numeric_cols is None:
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - whisker * iqr
        upper = q3 + whisker * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    present_numeric = [c for c in NUMERIC_CANDIDATES if c in df.columns and c != "price"]
    present_cats = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]

    feature_cols = present_numeric + present_cats

    df_features = df.copy()
    # One-hot encode categoricals
    df_features = pd.get_dummies(df_features, columns=present_cats, drop_first=True)
    used_cols = [c for c in df_features.columns if c != "price"]
    df_features = df_features[used_cols]
    # Keep strictly numeric columns; coerce where reasonable
    for col in df_features.columns:
        if df_features[col].dtype == object:
            df_features[col] = pd.to_numeric(df_features[col], errors="coerce")
    df_features = df_features.select_dtypes(include=[np.number])
    return df_features, present_numeric, present_cats


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    df = coerce_types(df)
    df = handle_missing_values(df)
    # cap outliers for core numeric columns including price, power, kilometer
    numeric_cols = [c for c in ["price", "power", "kilometer"] if c in df.columns]
    if numeric_cols:
        df = cap_outliers_iqr(df, numeric_cols=numeric_cols)
    return df


