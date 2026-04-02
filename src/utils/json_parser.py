from __future__ import annotations

import json
import re
from typing import Any

from json_repair import repair_json


def extract_json_block(text: str) -> str:
    """尽量从模型输出文本中提取 JSON 片段。"""
    if not text or not str(text).strip():
        return ""

    text = str(text).strip()

    # 优先提取 Markdown 代码块中的 JSON 内容
    fenced_match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced_match:
        return fenced_match.group(1).strip()

    # 优先尝试提取 JSON 对象
    object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if object_match:
        return object_match.group(0).strip()

    # 再尝试提取 JSON 数组
    array_match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if array_match:
        return array_match.group(0).strip()

    # 没有明显 JSON 结构时，原样返回
    return text


def parse_json_safe(text: str, default: Any = None) -> Any:
    """
    对 LLM 输出执行尽量稳健的 JSON 解析。

    解析失败时不抛异常，返回 default。
    默认 default 为 None。
    """
    if default is None:
        default = None

    if not text or not str(text).strip():
        return default

    candidate = extract_json_block(text)
    if not candidate:
        return default

    try:
        return json.loads(candidate)
    except Exception:
        pass

    try:
        repaired = repair_json(candidate)
        if not repaired or not str(repaired).strip():
            return default
        return json.loads(repaired)
    except Exception:
        return default


def ensure_list(data: Any) -> list[Any]:
    """
    将解析结果统一转换成列表。

    宽松策略：
    - list -> 原样返回
    - dict -> 包装成单元素列表
    - None / 其他异常类型 -> 返回空列表
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []
