from __future__ import annotations

from typing import Any

from src.llm.client import call_llm
from src.summ.prompts import build_extraction_system_prompt
from src.summ.schemas import ExtractionRecord
from src.utils.json_parser import ensure_list, extract_json_block, parse_json_safe


def extract_key_elements(
    text: str,
    model_config: dict[str, Any] | None = None,
    max_retries: int = 2,
) -> ExtractionRecord:
    """
    调用大模型，从输入文本中提取目标信息线索。

    当前 schema 下，这一步不做严格分类抽取，
    只返回：
    - keywords
    - key_sentences

    参数:
        text: 输入文本
        model_config: 透传给 call_llm 的模型配置
        max_retries: 调用失败或解析失败时的最大重试次数

    返回:
        ExtractionRecord
    """
    if not isinstance(text, str) or not text.strip():
        return ExtractionRecord()

    system_prompt = build_extraction_system_prompt()
    last_raw_output = ""

    for _ in range(max_retries + 1):
        raw_output = call_llm(
            user_input=text,
            system_prompt=system_prompt,
            model_config=model_config,
        )
        last_raw_output = raw_output

        record = parse_extraction_output(raw_output)
        record = normalize_extraction_record(record)

        if record.keywords or record.key_sentences:
            return record

    # 多次失败后返回尽量解析得到的结果；如果还是空，则返回空记录
    fallback_record = parse_extraction_output(last_raw_output)
    return normalize_extraction_record(fallback_record)


def parse_extraction_output(raw_output: str) -> ExtractionRecord:
    """
    解析 LLM 原始输出，转为 ExtractionRecord。

    期望输出格式:
    {
      "keywords": [...],
      "key_sentences": [...]
    }

    为提高鲁棒性，解析顺序为：
    1. 直接整体 parse_json_safe
    2. 提取 json block 后 parse_json_safe
    3. 若得到的是 list，则尝试取第一个 dict
    4. 若结构异常，则返回空记录
    """
    if not isinstance(raw_output, str) or not raw_output.strip():
        return ExtractionRecord()

    parsed = parse_json_safe(raw_output, default=None)

    if parsed is None:
        json_block = extract_json_block(raw_output)
        if json_block:
            parsed = parse_json_safe(json_block, default=None)

    if parsed is None:
        return ExtractionRecord()

    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else {}

    if not isinstance(parsed, dict):
        return ExtractionRecord()

    keywords = ensure_list(parsed.get("keywords", []))
    key_sentences = ensure_list(parsed.get("key_sentences", []))

    return ExtractionRecord(
        keywords=_clean_text_list(keywords),
        key_sentences=_clean_text_list(key_sentences),
    )


def normalize_extraction_record(record: ExtractionRecord) -> ExtractionRecord:
    """
    清洗和规范化 ExtractionRecord。

    处理内容：
    - 去除空字符串
    - 去重
    - 去除明显无意义的占位内容
    - 保持原始顺序
    """
    if not isinstance(record, ExtractionRecord):
        return ExtractionRecord()

    keywords = _normalize_keywords(record.keywords)
    key_sentences = _normalize_key_sentences(record.key_sentences)

    return ExtractionRecord(
        keywords=keywords,
        key_sentences=key_sentences,
    )


def _clean_text_list(items: list[Any]) -> list[str]:
    """
    把任意列表清洗成字符串列表。
    """
    cleaned: list[str] = []

    for item in items:
        if item is None:
            continue

        text = str(item).strip()
        if not text:
            continue

        cleaned.append(text)

    return cleaned


def _normalize_keywords(keywords: list[str]) -> list[str]:
    """
    清洗关键词列表。
    """
    invalid_values = {
        "none",
        "null",
        "n/a",
        "na",
        "unknown",
        "not mentioned",
        "not provided",
    }

    results: list[str] = []
    seen: set[str] = set()

    for keyword in keywords:
        text = keyword.strip()
        if not text:
            continue

        lowered = text.lower()
        if lowered in invalid_values:
            continue

        # 关键词不应过长；如果模型返回整句，这里不强删，只保留
        # 但通过去重降低噪声
        dedup_key = lowered
        if dedup_key in seen:
            continue

        seen.add(dedup_key)
        results.append(text)

    return results


def _normalize_key_sentences(key_sentences: list[str]) -> list[str]:
    """
    清洗关键句/关键片段列表。
    """
    invalid_values = {
        "none",
        "null",
        "n/a",
        "na",
        "unknown",
        "not mentioned",
        "not provided",
    }

    results: list[str] = []
    seen: set[str] = set()

    for sentence in key_sentences:
        text = sentence.strip()
        if not text:
            continue

        lowered = text.lower()
        if lowered in invalid_values:
            continue

        dedup_key = " ".join(lowered.split())
        if dedup_key in seen:
            continue

        seen.add(dedup_key)
        results.append(text)

    return results


if __name__ == "__main__":
    TEXT = ""
    res = extract_key_elements(TEXT)
