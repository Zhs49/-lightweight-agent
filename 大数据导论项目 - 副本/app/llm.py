from __future__ import annotations

import json
from typing import Any, Dict

from openai import OpenAI
import os
import json as _json
import httpx


def generate_insights(
    api_key: str,
    eda_brief: Dict[str, Any],
    corr_brief: Dict[str, Any],
    cluster_brief: Dict[str, Any],
    language: str = "zh",
) -> str:
    # Azure OpenAI configuration
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not api_key and not (azure_endpoint and azure_key and azure_deployment):
        return (
            "[提示] 未配置 OPENAI_API_KEY，已跳过LLM解读。\n"
            "- 影响保值与高价的因素通常包括：较新上牌、低里程、高功率、热门车身/品牌与自动挡等。\n"
            "- 建议车商优先选择低里程、保值率高的品牌/车型，并结合热度与季节进行定价。"
        )
    # If Azure is configured, use Azure REST API directly
    if azure_endpoint and azure_key and azure_deployment:
        system = (
            "你是一名精通二手车市场的资深数据科学顾问，擅长用中文撰写简明、可执行的商业洞察。"
        )
        user = (
            "基于以下EDA摘要、相关性结果以及聚类画像，请用中文给出深度洞察，包含：\n"
            "1) 哪些簇最保值（高价且低折旧风险）；\n"
            "2) 影响高价的关键特征组合（用要点说明，避免过度术语）；\n"
            "3) 对车商的进货/定价策略建议（操作性强）。\n\n"
            f"EDA摘要: {json.dumps(eda_brief, ensure_ascii=False)[:4000]}\n\n"
            f"相关性结果概览: {json.dumps(corr_brief, ensure_ascii=False)[:4000]}\n\n"
            f"聚类画像摘要: {json.dumps(cluster_brief, ensure_ascii=False)[:4000]}\n\n"
            "请以分点形式输出，语言精炼。"
        )
        try:
            headers = {"api-key": azure_key, "Content-Type": "application/json"}
            payload = {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.3,
                "max_tokens": 800,
            }
            url = f"{azure_endpoint.rstrip('/')}/openai/deployments/{azure_deployment}/chat/completions?api-version={azure_api_version}"
            with httpx.Client(timeout=30.0) as s:
                r = s.post(url, headers=headers, json=payload)
            if r.status_code == 200:
                data = r.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            else:
                return (
                    "[提示] Azure OpenAI 调用失败。已跳过LLM解读。\n"
                    f"- 状态码：{r.status_code}\n- 响应：{r.text[:500]}\n"
                    "- 请检查 AZURE_OPENAI_ENDPOINT/DEPLOYMENT/API_VERSION 与 Key 是否正确。"
                )
        except Exception as e:
            return (
                "[提示] Azure OpenAI 调用异常。已跳过LLM解读。\n"
                f"- 错误：{type(e).__name__} {e}\n"
                "- 建议：网络/代理设置、API 版本与部署名。"
            )

    # Public OpenAI path
    # 避免某些 httpx/openai 版本组合对 client kwargs 兼容性问题，改为走环境变量
    os.environ["OPENAI_API_KEY"] = api_key
    try:
        client = OpenAI()
    except Exception:
        client = None

    system = (
        "你是一名精通二手车市场的资深数据科学顾问，擅长用中文撰写简明、可执行的商业洞察。"
    )
    user = (
        "基于以下EDA摘要、相关性结果以及聚类画像，请用中文给出深度洞察，包含：\n"
        "1) 哪些簇最保值（高价且低折旧风险）；\n"
        "2) 影响高价的关键特征组合（用要点说明，避免过度术语）；\n"
        "3) 对车商的进货/定价策略建议（操作性强）。\n\n"
        f"EDA摘要: {json.dumps(eda_brief, ensure_ascii=False)[:4000]}\n\n"
        f"相关性结果概览: {json.dumps(corr_brief, ensure_ascii=False)[:4000]}\n\n"
        f"聚类画像摘要: {json.dumps(cluster_brief, ensure_ascii=False)[:4000]}\n\n"
        "请以分点形式输出，语言精炼。"
    )

    # First try SDK if client is available
    if client is not None:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
                max_tokens=800,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            pass

    # Fallback: direct REST call via httpx (public OpenAI)
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
            "max_tokens": 800,
        }
        with httpx.Client(timeout=30.0) as s:
            r = s.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if r.status_code == 200:
            data = r.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        else:
            return (
                "[提示] LLM 调用失败（REST）。已跳过LLM解读。\n"
                f"- 状态码：{r.status_code}\n- 响应：{r.text[:500]}"
            )
    except Exception as e:
        return (
            "[提示] LLM 调用失败，已跳过LLM解读。\n"
            f"- 错误：{type(e).__name__} {e}\n"
            "- 建议：检查网络/代理；或升级 httpx/openai；Key 是否有效。"
        )

