from __future__ import annotations

from typing import Any

from src.llm.client import call_llm
from src.summ.prompts import build_summary_system_prompt
from src.summ.schemas import PreparedSummaryChunk


def generate_targeted_summary(
    prepared_chunk: PreparedSummaryChunk,
    model_config: dict[str, Any] | None = None,
    max_retry_times: int = 2,
) -> str:
    """
    对单个 PreparedSummaryChunk 生成摘要。

    参数:
        prepared_chunk: 第 5 步得到的摘要输入块
        model_config: 透传给 call_llm 的模型配置
        max_retry_times: 摘要失败时的最大重试次数

    返回:
        str: 摘要文本
    """
    if not isinstance(prepared_chunk, PreparedSummaryChunk):
        return ""

    source_text = prepared_chunk.text.strip()
    if not source_text:
        return ""

    system_prompt = build_summary_system_prompt()
    last_output = ""

    for _ in range(max_retry_times + 1):
        raw_output = call_llm(
            user_input=source_text,
            system_prompt=system_prompt,
            model_config=model_config,
        )

        summary = normalize_summary_output(raw_output)
        if summary:
            return summary

        last_output = raw_output

    return normalize_summary_output(last_output)


def normalize_summary_output(text: str) -> str:
    """
    对摘要输出做最基础的清洗。

    处理逻辑保持简单：
    - 去首尾空白
    - 去掉空行
    - 合并多余空白
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    summary = " ".join(lines).strip()

    return summary
