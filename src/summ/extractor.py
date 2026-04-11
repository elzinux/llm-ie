from __future__ import annotations

from typing import Any

from src.llm.client import call_llm
from src.summ.extractor_checker import extractor_check
from src.summ.prompts import build_extraction_system_prompt
from src.summ.schemas import ExtractionCandidate, ExtractionRecord
from src.utils.json_parser import parse_json_safe

_ALLOWED_CANDIDATE_TYPES = {"field_candidate", "evidence_span"}
_ALLOWED_TARGET_FIELDS = {
    "intervention_or_policy",
    "location",
    "crop",
    "impact",
}
_ALLOWED_RELEVANCE = {"high", "medium", "low", "none"}


def extract_key_elements(
    text: str,
    model_config: dict[str, Any] | None = None,
    max_retries: int = 2,
) -> ExtractionRecord:
    """
    调用大模型抽取候选，并交由 extractor_check 完成审核后返回 ExtractionRecord。

    参数:
        text: 输入文本
        model_config: 传给 call_llm 的模型配置
        max_retries: 调用失败或解析失败时的最大重试次数

    返回:
        ExtractionRecord
    """
    if not isinstance(text, str) or not text.strip():
        return ExtractionRecord()

    system_prompt = build_extraction_system_prompt()
    last_candidates: list[ExtractionCandidate] = []

    for _ in range(max_retries + 1):
        raw_output = call_llm(
            user_input=text,
            system_prompt=system_prompt,
            model_config=model_config,
        )

        candidates = parse_extraction_candidates(raw_output)
        last_candidates = candidates

        checked_candidates = extractor_check(text, candidates)
        record = build_extraction_record(checked_candidates)
        record = normalize_extraction_record(record)

        if record.keywords or record.key_sentences:
            return record

    fallback_checked_candidates = extractor_check(text, last_candidates)
    return normalize_extraction_record(build_extraction_record(fallback_checked_candidates))


def parse_extraction_output(raw_output: str) -> ExtractionRecord:
    """
    兼容保留的解析接口。 

    当前仅负责把抽取模型输出的候选列表整理为 ExtractionRecord，
    不承担审核逻辑。
    """
    candidates = parse_extraction_candidates(raw_output)
    return normalize_extraction_record(build_extraction_record(candidates))


def parse_extraction_candidates(raw_output: str) -> list[ExtractionCandidate]:
    """
    解析抽取模型输出的候选列表。

    期望输出格式为候选数组，每项包含：
    - text
    - candidate_type
    - target_fields
    - relevance
    - reason
    """
    if not isinstance(raw_output, str) or not raw_output.strip():
        return []

    parsed = parse_json_safe(raw_output, default=None)
    if not isinstance(parsed, list):
        return []

    results: list[ExtractionCandidate] = []
    seen: set[str] = set()

    for item in parsed:
        if not isinstance(item, dict):
            continue

        text = str(item.get("text") or "").strip()
        if not text:
            continue

        candidate_type = str(item.get("candidate_type") or "").strip()
        if candidate_type not in _ALLOWED_CANDIDATE_TYPES:
            continue

        target_fields = _normalize_target_fields(item.get("target_fields", []))
        relevance = _normalize_relevance(item.get("relevance", "none"))
        reason = str(item.get("reason") or "").strip()

        dedup_key = (candidate_type, text)
        if str(dedup_key) in seen:
            continue
        seen.add(str(dedup_key))

        results.append(
            ExtractionCandidate(
                text=text,
                candidate_type=candidate_type,
                target_fields=target_fields,
                relevance=relevance,
                reason=reason,
            )
        )

    return results


def build_extraction_record(
    candidates: list[ExtractionCandidate],
) -> ExtractionRecord:
    """
    将候选列表整理为 ExtractionRecord。

    映射规则：
    - field_candidate -> keywords / keyword_relevance
    - evidence_span -> key_sentences / key_sentence_relevance
    """
    keywords: list[str] = []
    key_sentences: list[str] = []
    keyword_relevance: list[str] = []
    key_sentence_relevance: list[str] = []

    for candidate in candidates:
        if not isinstance(candidate, ExtractionCandidate):
            continue

        text = str(candidate.text or "").strip()
        relevance = _normalize_relevance(candidate.relevance)
        if not text:
            continue

        if candidate.candidate_type == "field_candidate":
            keywords.append(text)
            keyword_relevance.append(relevance)
            continue

        if candidate.candidate_type == "evidence_span":
            key_sentences.append(text)
            key_sentence_relevance.append(relevance)

    return ExtractionRecord(
        keywords=keywords,
        key_sentences=key_sentences,
        keyword_relevance=keyword_relevance,
        key_sentence_relevance=key_sentence_relevance,
    )


def normalize_extraction_record(record: ExtractionRecord) -> ExtractionRecord:
    """
    清洗和规范化 ExtractionRecord。

    处理内容：
    - 去除空字符串
    - 去重
    - 去除明显无意义的占位内容
    - 保持文本与 relevance 的索引对应关系
    """
    if not isinstance(record, ExtractionRecord):
        return ExtractionRecord()

    keywords, keyword_relevance = _normalize_text_relevance_pairs(
        record.keywords,
        record.keyword_relevance,
    )
    key_sentences, key_sentence_relevance = _normalize_text_relevance_pairs(
        record.key_sentences,
        record.key_sentence_relevance,
    )

    return ExtractionRecord(
        keywords=keywords,
        key_sentences=key_sentences,
        keyword_relevance=keyword_relevance,
        key_sentence_relevance=key_sentence_relevance,
    )


def _normalize_text_relevance_pairs(
    texts: list[str],
    relevances: list[str],
) -> tuple[list[str], list[str]]:
    """
    对文本及其 relevance 做成对清洗和去重。
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

    normalized_texts: list[str] = []
    normalized_relevances: list[str] = []
    seen: set[str] = set()

    for idx, raw_text in enumerate(texts):
        text = str(raw_text or "").strip()
        if not text:
            continue

        lowered = text.lower()
        if lowered in invalid_values:
            continue

        dedup_key = " ".join(lowered.split())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        relevance = relevances[idx] if idx < len(relevances) else "none"
        normalized_texts.append(text)
        normalized_relevances.append(_normalize_relevance(relevance))

    return normalized_texts, normalized_relevances


def _normalize_target_fields(value: Any) -> list[str]:
    """
    规范化 target_fields 并保序去重。
    """
    if isinstance(value, list):
        raw_items = value
    elif value is None:
        raw_items = []
    else:
        raw_items = [value]

    results: list[str] = []
    seen: set[str] = set()

    for item in raw_items:
        normalized_item = str(item or "").strip().lower()
        if normalized_item not in _ALLOWED_TARGET_FIELDS:
            continue
        if normalized_item in seen:
            continue
        seen.add(normalized_item)
        results.append(normalized_item)

    return results


def _normalize_relevance(value: Any) -> str:
    """
    规范化 relevance，非法值统一回退为 none。
    """
    normalized_value = str(value or "").strip().lower()
    if normalized_value in _ALLOWED_RELEVANCE:
        return normalized_value
    return "none"


if __name__ == "__main__":
    TEXT = ""
    res = extract_key_elements(TEXT)
