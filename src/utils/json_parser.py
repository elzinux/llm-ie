from __future__ import annotations

import json
import re
from typing import Any

from json_repair import repair_json


def _is_valid_json(text: str) -> bool:
    """判断文本是否为合法 JSON。"""
    if not isinstance(text, str) or not text.strip():
        return False

    try:
        json.loads(text)
        return True
    except Exception:
        return False


def _extract_balanced_block(text: str, opener: str, closer: str) -> str:
    """
    提取第一个括号配对完整且可解析的 JSON 子串。
    支持忽略字符串中的括号字符。
    """
    for start_idx, ch in enumerate(text):
        if ch != opener:
            continue

        depth = 0
        in_string = False
        escape = False

        for idx in range(start_idx, len(text)):
            c = text[idx]

            if in_string:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_string = False
                continue

            if c == '"':
                in_string = True
                continue

            if c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start_idx : idx + 1].strip()
                    if _is_valid_json(candidate):
                        return candidate
                    break

    return ""


def extract_json_block(text: str) -> str:
    """尽量从模型输出文本中提取 JSON 片段。"""
    if not text or not str(text).strip():
        return ""

    text = str(text).strip()

    fenced_match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced_match:
        fenced_text = fenced_match.group(1).strip()
        if fenced_text:
            return fenced_text

    if _is_valid_json(text):
        return text

    for opener, closer in (("[", "]"), ("{", "}")):
        candidate = _extract_balanced_block(text, opener, closer)
        if candidate:
            return candidate

    return text


def parse_json_safe(text: str, default: Any = None) -> Any:
    """
    对文本执行尽量稳健的 JSON 解析。
    解析失败时不抛异常，返回 default。
    """
    if not text or not str(text).strip():
        return default

    text = str(text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        repaired = repair_json(text)
        if repaired and str(repaired).strip():
            return json.loads(repaired)
    except Exception:
        pass

    candidate = extract_json_block(text)
    if not candidate:
        return default

    try:
        return json.loads(candidate)
    except Exception:
        pass

    try:
        repaired = repair_json(candidate)
        if repaired and str(repaired).strip():
            return json.loads(repaired)
    except Exception:
        pass

    return default


def ensure_list(data: Any) -> list[Any]:
    """
    将解析结果统一转换成列表。

    宽松策略：
    - list -> 原样返回
    - dict -> 包装成单元素列表
    - None / 其他类型 -> 返回空列表
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []
