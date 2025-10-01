from __future__ import annotations

import os
from typing import Dict, Any

import pandas as pd


def write_markdown(
    out_path: str,
    eda_desc: pd.DataFrame,
    cat_counts: Dict[str, pd.Series],
    pearson_png: str,
    spearman_png: str,
    elbow_png: str,
    pca_png: str,
    cluster_profiles_md: str,
    llm_insights: str,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    def rel(p: str) -> str:
        return os.path.relpath(p, start=os.path.dirname(out_path) or ".")

    lines = []
    lines.append("# 二手车商数据深度分析报告")
    lines.append("")
    lines.append("## 1 数据概览")
    lines.append("")
    lines.append("数值字段统计：")
    lines.append("")
    lines.append(eda_desc.to_markdown())
    lines.append("")
    lines.append("类别字段Top值计数（最多展示前30）：")
    for col, s in cat_counts.items():
        lines.append("")
        lines.append(f"- {col}")
        lines.append(s.to_frame(name="count").to_markdown())

    lines.append("")
    lines.append("## 2 相关性分析（含热力图）")
    lines.append("")
    lines.append(f"![Pearson相关性热力图]({rel(pearson_png)})")
    lines.append(f"![Spearman相关性热力图]({rel(spearman_png)})")

    lines.append("")
    lines.append("## 3 聚类分析（含肘部图、PCA 2D 散点图、簇画像）")
    lines.append("")
    lines.append(f"![KMeans肘部法则]({rel(elbow_png)})")
    lines.append(f"![PCA 2D 散点图]({rel(pca_png)})")
    lines.append("")
    lines.append("### 簇画像摘要")
    lines.append("")
    lines.append(cluster_profiles_md)

    lines.append("")
    lines.append("## 4 LLM 自动洞察与业务建议")
    lines.append("")
    lines.append(llm_insights)

    lines.append("")
    lines.append("## 5 结论与后续迭代方向")
    lines.append("")
    lines.append("- 可进一步引入车辆年限、保养记录、事故记录等特征，提升预测力。")
    lines.append("- 尝试GMM/DBSCAN等不同聚类方法，以及更细粒度的特征工程。")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

