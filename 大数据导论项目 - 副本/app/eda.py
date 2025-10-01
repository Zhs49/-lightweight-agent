from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager

# try set a font that supports Chinese
def _setup_chinese_font():
    candidates = [
        "SimHei",  # Windows 常见
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    for name in candidates:
        try:
            plt.rcParams['font.sans-serif'] = [name]
            plt.rcParams['axes.unicode_minus'] = False
            return
        except Exception:
            continue

_setup_chinese_font()


def eda_summary(df: pd.DataFrame) -> Dict[str, pd.DataFrame | pd.Series]:
    numeric_df = df.select_dtypes(include=[np.number])
    categorical_df = df.select_dtypes(include=["object", "category"])

    if numeric_df.shape[1] == 0:
        numeric_desc = pd.DataFrame({"note": ["无数值字段可供统计"]})
    else:
        numeric_desc = numeric_df.describe().T

    if categorical_df.shape[1] == 0:
        categorical_counts: Dict[str, pd.Series] = {}
    else:
        categorical_counts = {col: categorical_df[col].value_counts().head(30) for col in categorical_df.columns}

    return {
        "numeric_desc": numeric_desc,
        "categorical_counts": categorical_counts,
    }


def correlation_analysis(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    numeric_df = df.select_dtypes(include=[np.number])
    pearson = numeric_df.corr(method="pearson")
    spearman = numeric_df.corr(method="spearman")
    return pearson, spearman


def save_heatmap(corr: pd.DataFrame, title: str, out_png: str, out_svg: str) -> None:
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(10, 8))
    if corr is None or corr.empty:
        plt.text(0.5, 0.5, "无足够数值字段以计算相关性", ha="center", va="center", fontsize=12)
        plt.title(title + "（无数据）")
        plt.axis('off')
    else:
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.savefig(out_svg)
    plt.close()


