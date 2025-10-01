from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from langgraph.graph import StateGraph, END

from .config import settings
from .data import load_dataset, clean_data, prepare_features
from .eda import eda_summary, correlation_analysis, save_heatmap
from .clustering import (
    scale_features,
    run_kmeans_with_elbow,
    save_elbow_plot,
    save_pca_scatter,
    build_cluster_profiles,
)
from .llm import generate_insights
from .report import write_markdown


@dataclass
class PipelineState:
    raw_path: str
    output_md: str
    img_dir: str
    random_seed: int

    df_raw: Optional[pd.DataFrame] = None
    df: Optional[pd.DataFrame] = None
    eda_desc: Optional[pd.DataFrame] = None
    cat_counts: Optional[Dict[str, pd.Series]] = None
    pearson_path: Optional[Tuple[str, str]] = None
    spearman_path: Optional[Tuple[str, str]] = None
    elbow_path: Optional[Tuple[str, str]] = None
    pca_path: Optional[Tuple[str, str]] = None
    cluster_profiles_md: Optional[str] = None
    llm_text: Optional[str] = None


def node_load_clean(state: PipelineState) -> PipelineState:
    # Use optional sheet/header from settings
    df = load_dataset(state.raw_path, sheet_name=settings.input_sheet, header_row_index=settings.header_row_index)
    df = clean_data(df)
    state.df_raw = df.copy()
    state.df = df
    return state


def node_eda(state: PipelineState) -> PipelineState:
    assert state.df is not None
    eda = eda_summary(state.df)
    state.eda_desc = eda["numeric_desc"]
    state.cat_counts = eda["categorical_counts"]

    # 相关性：当无数值字段时返回空矩阵
    try:
        pearson, spearman = correlation_analysis(state.df)
    except Exception:
        pearson, spearman = (pd.DataFrame(), pd.DataFrame())
    pearson_png = os.path.join(state.img_dir, "corr_pearson.png")
    pearson_svg = os.path.join(state.img_dir, "corr_pearson.svg")
    spearman_png = os.path.join(state.img_dir, "corr_spearman.png")
    spearman_svg = os.path.join(state.img_dir, "corr_spearman.svg")
    save_heatmap(pearson, "Pearson相关性", pearson_png, pearson_svg)
    save_heatmap(spearman, "Spearman相关性", spearman_png, spearman_svg)
    state.pearson_path = (pearson_png, pearson_svg)
    state.spearman_path = (spearman_png, spearman_svg)
    return state


def node_cluster(state: PipelineState) -> PipelineState:
    assert state.df is not None
    df_feat, _, _ = prepare_features(state.df)
    if df_feat.shape[1] == 0:
        # 无特征可聚类：生成空的占位结果与提示
        elbow_png = os.path.join(state.img_dir, "kmeans_elbow.png")
        elbow_svg = os.path.join(state.img_dir, "kmeans_elbow.svg")
        pca_png = os.path.join(state.img_dir, "pca_scatter.png")
        pca_svg = os.path.join(state.img_dir, "pca_scatter.svg")
        # 生成占位图
        import matplotlib.pyplot as plt
        os.makedirs(state.img_dir, exist_ok=True)
        for p in [elbow_png, elbow_svg, pca_png, pca_svg]:
            plt.figure(figsize=(6,4))
            plt.text(0.5,0.5,"无足够特征进行聚类", ha='center', va='center')
            plt.axis('off')
            if p.endswith('.png'):
                plt.savefig(p, dpi=150)
            else:
                plt.savefig(p)
            plt.close()
        state.elbow_path = (elbow_png, elbow_svg)
        state.pca_path = (pca_png, pca_svg)
        state.cluster_profiles_md = "当前数据缺少可用特征，无法进行聚类。"
        return state
    X, scaler, feature_cols = scale_features(df_feat)
    if X is None or X.size == 0 or X.shape[1] == 0:
        # 再次兜底：预处理后仍无特征
        elbow_png = os.path.join(state.img_dir, "kmeans_elbow.png")
        elbow_svg = os.path.join(state.img_dir, "kmeans_elbow.svg")
        pca_png = os.path.join(state.img_dir, "pca_scatter.png")
        pca_svg = os.path.join(state.img_dir, "pca_scatter.svg")
        import matplotlib.pyplot as plt
        os.makedirs(state.img_dir, exist_ok=True)
        for p in [elbow_png, elbow_svg, pca_png, pca_svg]:
            plt.figure(figsize=(6,4))
            plt.text(0.5,0.5,"预处理后无可用数值特征", ha='center', va='center')
            plt.axis('off')
            if p.endswith('.png'):
                plt.savefig(p, dpi=150)
            else:
                plt.savefig(p)
            plt.close()
        state.elbow_path = (elbow_png, elbow_svg)
        state.pca_path = (pca_png, pca_svg)
        state.cluster_profiles_md = "预处理后无可用数值特征，聚类跳过。"
        return state
    model, inertias, best_k, best_sil = run_kmeans_with_elbow(
        X, random_state=state.random_seed
    )
    labels = model.labels_

    elbow_png = os.path.join(state.img_dir, "kmeans_elbow.png")
    elbow_svg = os.path.join(state.img_dir, "kmeans_elbow.svg")
    save_elbow_plot(inertias, elbow_png, elbow_svg)

    pca_png = os.path.join(state.img_dir, "pca_scatter.png")
    pca_svg = os.path.join(state.img_dir, "pca_scatter.svg")
    save_pca_scatter(X, labels, pca_png, pca_svg)

    prof_df = build_cluster_profiles(pd.concat([state.df, df_feat], axis=1), labels, feature_cols)
    prof_md = prof_df.to_markdown()
    prof_md += f"\n\nSilhouette Score (best k={best_k}): {best_sil:.3f}"

    state.elbow_path = (elbow_png, elbow_svg)
    state.pca_path = (pca_png, pca_svg)
    state.cluster_profiles_md = prof_md
    return state


def node_llm(state: PipelineState) -> PipelineState:
    eda_brief = {
        "numeric_desc_cols": list(state.eda_desc.index) if state.eda_desc is not None else [],
        "num_rows": int(len(state.df)) if state.df is not None else 0,
    }
    corr_brief = {
        "pearson_img": state.pearson_path[0] if state.pearson_path else "",
        "spearman_img": state.spearman_path[0] if state.spearman_path else "",
    }
    cluster_brief = {
        "has_profiles": bool(state.cluster_profiles_md),
    }
    text = generate_insights(
        api_key=settings.openai_api_key,
        eda_brief=eda_brief,
        corr_brief=corr_brief,
        cluster_brief=cluster_brief,
        language="zh",
    )
    state.llm_text = text
    return state


def node_report(state: PipelineState) -> PipelineState:
    assert state.eda_desc is not None
    assert state.cat_counts is not None
    assert state.pearson_path is not None and state.spearman_path is not None
    assert state.elbow_path is not None and state.pca_path is not None
    assert state.cluster_profiles_md is not None

    write_markdown(
        out_path=state.output_md,
        eda_desc=state.eda_desc,
        cat_counts=state.cat_counts,
        pearson_png=state.pearson_path[0],
        spearman_png=state.spearman_path[0],
        elbow_png=state.elbow_path[0],
        pca_png=state.pca_path[0],
        cluster_profiles_md=state.cluster_profiles_md,
        llm_insights=state.llm_text or "",
    )
    return state


def build_workflow(raw_path: str, out_md: str, img_dir: str, random_seed: int) -> StateGraph:
    graph = StateGraph(PipelineState)
    graph.add_node("load_clean", node_load_clean)
    graph.add_node("eda", node_eda)
    graph.add_node("cluster", node_cluster)
    graph.add_node("llm", node_llm)
    graph.add_node("report", node_report)

    graph.set_entry_point("load_clean")
    graph.add_edge("load_clean", "eda")
    graph.add_edge("eda", "cluster")
    graph.add_edge("cluster", "llm")
    graph.add_edge("llm", "report")
    graph.add_edge("report", END)

    return graph

