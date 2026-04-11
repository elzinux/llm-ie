from __future__ import annotations

import json
import re
from typing import Any

from src.summ.prompts import (
    build_evidence_span_check_prompt,
    build_field_candidate_check_prompt,
)
from src.summ.schemas import ExtractionCandidate
from src.utils.json_parser import parse_json_safe
from src.utils.normalize_text import normalize_text

_ALLOWED_CANDIDATE_TYPES = {"field_candidate", "evidence_span"}
_ALLOWED_TARGET_FIELDS = {
    "intervention_or_policy",
    "location",
    "crop",
    "impact",
}
_ALLOWED_RELEVANCE = {"high", "medium", "low", "none"}

_MONTH_OR_SEASON_SET = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "spring",
    "summer",
    "autumn",
    "fall",
    "winter",
}


def extractor_check(
    source_text: str,
    candidates: list[ExtractionCandidate],
) -> list[ExtractionCandidate]:
    """
    运行 extractor_check 阶段的主流程。

    该函数负责统一调度第二阶段检查逻辑：
    1. 对 candidates 字段调用 rule_filter_candidates 函数进行初步筛选；
    2. 基于 candidate.text 去重，重复文本直接跳过；
    3. 将输入 candidates 按 candidate_type 拆分为 field_candidate 和 evidence_span；
    4. 分别调用对应的检查函数；
    5. 合并检查结果并返回。

    输入：
        source_text: str
            原始文本，用于 evidence_span 检查阶段提供上下文。
        candidates: list[ExtractionCandidate]
            第一阶段规则过滤后的候选列表，可能同时包含 field_candidate
            和 evidence_span。

    输出：
        list[ExtractionCandidate]
            经过 extractor_check 第二阶段审核后的候选列表。
    """
    if not candidates:
        return []

    prepared_candidates = [
        _sanitize_candidate(candidate)
        for candidate in candidates
        if isinstance(candidate, ExtractionCandidate)
    ]
    filtered_candidates = rule_filter_candidates(prepared_candidates)
    filtered_candidates = _deduplicate_candidates_by_text(filtered_candidates)
    if not filtered_candidates:
        return []

    field_candidates = []
    evidence_candidates = []

    for candidate in filtered_candidates:
        if candidate.candidate_type == "field_candidate":
            field_candidates.append(candidate)
        elif candidate.candidate_type == "evidence_span":
            evidence_candidates.append(candidate)

    checked_field_candidates = check_field_candidates(field_candidates)
    checked_evidence_spans = check_evidence_spans(source_text, evidence_candidates)

    field_iter = iter(checked_field_candidates)
    evidence_iter = iter(checked_evidence_spans)
    merged_candidates: list[ExtractionCandidate] = []

    for candidate in filtered_candidates:
        if candidate.candidate_type == "field_candidate":
            merged_candidates.append(next(field_iter, candidate))
            continue

        if candidate.candidate_type == "evidence_span":
            merged_candidates.append(next(evidence_iter, candidate))
            continue

        merged_candidates.append(candidate)

    return _filter_checked_candidates(merged_candidates)


def check_field_candidates(
    candidates: list[ExtractionCandidate],
) -> list[ExtractionCandidate]:
    """
    检查 field_candidate 类型的候选信息列表。

    该函数负责将 field_candidate candidates 送入大模型进行审核，
    判断它们是否与目标关键信息强相关，并检查当前 relevance 是否合理。

    输入：
        candidates: list[ExtractionCandidate]
            待检查的 field_candidate 候选列表。调用方应保证这些 candidate
            的 candidate_type 为 "field_candidate"。

    输出：
        list[ExtractionCandidate]
            大模型审核后的 field_candidate 列表。
            返回结果仍保持 ExtractionCandidate 结构，不新增字段。
    """
    prepared_candidates = [
        candidate
        for candidate in candidates
        if isinstance(candidate, ExtractionCandidate)
        and candidate.candidate_type == "field_candidate"
        and str(candidate.text).strip()
    ]
    if not prepared_candidates:
        return []

    prompt = build_field_candidate_check_prompt()
    user_input = json.dumps(
        [
            _candidate_to_payload(candidate, candidate_id=str(idx))
            for idx, candidate in enumerate(prepared_candidates)
        ],
        ensure_ascii=False,
        indent=2,
    )

    return _check_candidates_with_llm(
        original_candidates=prepared_candidates,
        system_prompt=prompt,
        user_input=user_input,
        allow_text_update=False,
    )


def check_evidence_spans(
    source_text: str,
    candidates: list[ExtractionCandidate],
) -> list[ExtractionCandidate]:
    """
    检查 evidence_span 类型的候选信息列表。

    该函数负责将 evidence_span candidates 送入大模型进行审核，
    结合原文或局部上下文，判断它们是否与目标关键信息强相关，
    当前 relevance 是否合理，以及文本边界是否过长。

    输入：
        source_text: str
            原始文本，用于为 evidence_span 提供必要的上下文支持。
        candidates: list[ExtractionCandidate]
            待检查的 evidence_span 候选列表。调用方应保证这些 candidate
            的 candidate_type 为 "evidence_span"。

    输出：
        list[ExtractionCandidate]
            大模型审核后的 evidence_span 列表。
            返回结果仍保持 ExtractionCandidate 结构，不新增字段。
    """
    prepared_candidates = [
        candidate
        for candidate in candidates
        if isinstance(candidate, ExtractionCandidate)
        and candidate.candidate_type == "evidence_span"
        and str(candidate.text).strip()
    ]
    if not prepared_candidates:
        return []

    payload: list[dict[str, Any]] = []
    for idx, candidate in enumerate(prepared_candidates):
        payload.append(
            {
                **_candidate_to_payload(candidate, candidate_id=str(idx)),
                "context": get_evidence_span_context(source_text, candidate.text),
            }
        )

    prompt = build_evidence_span_check_prompt()
    user_input = json.dumps(payload, ensure_ascii=False, indent=2)

    return _check_candidates_with_llm(
        original_candidates=prepared_candidates,
        system_prompt=prompt,
        user_input=user_input,
        allow_text_update=True,
        source_text=source_text,
    )


def find_span_sentence_window(
    source_text: str,
    span_text: str,
    window_size: int = 1,
) -> str:
    """
    为 evidence_span 提取句子级局部上下文。
    优先基于 spaCy 分句和字符串匹配定位。
    若定位失败，返回空字符串或交给上层兜底。
    """
    if not isinstance(source_text, str) or not source_text.strip():
        return ""

    span_bounds = _find_span_bounds(source_text, span_text)
    if span_bounds is None:
        return ""

    sentence_spans = _get_sentence_spans(source_text)
    if not sentence_spans:
        return ""

    span_start, span_end = span_bounds
    matched_indices = [
        idx
        for idx, (sent_start, sent_end, _) in enumerate(sentence_spans)
        if sent_start < span_end and span_start < sent_end
    ]
    if not matched_indices:
        return ""

    resolved_window_size = _coerce_non_negative_int(window_size, default=1)
    start_idx = max(0, matched_indices[0] - resolved_window_size)
    end_idx = min(len(sentence_spans), matched_indices[-1] + resolved_window_size + 1)

    window_text = " ".join(
        sentence_text.strip()
        for _, _, sentence_text in sentence_spans[start_idx:end_idx]
        if sentence_text.strip()
    )
    return window_text.strip()


def find_span_char_window(
    source_text: str,
    span_text: str,
    char_window: int = 300,
) -> str:
    """
    为 evidence_span 提取字符级局部上下文。
    作为句子级定位失败时的兜底方案。
    """
    if not isinstance(source_text, str) or not source_text.strip():
        return ""

    span_bounds = _find_span_bounds(source_text, span_text)
    if span_bounds is None:
        return ""

    span_start, span_end = span_bounds
    resolved_char_window = _coerce_non_negative_int(char_window, default=300)
    window_start = max(0, span_start - resolved_char_window)
    window_end = min(len(source_text), span_end + resolved_char_window)
    return source_text[window_start:window_end].strip()


def get_evidence_span_context(
    source_text: str,
    span_text: str,
    sentence_window: int = 1,
    char_window: int = 300,
) -> str:
    """
    为 evidence_span 提取局部上下文片段，供 extractor_check 第二阶段使用。

    处理策略：
    1. 优先使用 spaCy 分句后进行句子级定位
    2. 若成功定位，则返回命中句及其前后若干句
    3. 若句子级定位失败，则退化为字符级窗口截取
    4. 若仍失败，则返回 span_text 本身

    设计目标：
    - 为大模型提供足够的局部语境，用于判断 evidence_span 是否强相关
    - 支持判断 evidence_span 是否过长、是否包含明显无关前后缀
    - 保持实现轻量、稳健、易维护
    """
    cleaned_span = str(span_text or "").strip()
    if not cleaned_span:
        return ""

    sentence_context = find_span_sentence_window(
        source_text=source_text,
        span_text=cleaned_span,
        window_size=sentence_window,
    )
    if sentence_context:
        return sentence_context

    char_context = find_span_char_window(
        source_text=source_text,
        span_text=cleaned_span,
        char_window=char_window,
    )
    if char_context:
        return char_context

    return cleaned_span


def light_filter_remove(text: str) -> bool:
    """
    轻量化去除规则。

    输入:
        text: 一个字符串

    返回:
        True: 需要去除
        False: 不需要去除

    去除规则:
    - 数字，如 2002 或 3.3
    - 全大写缩写，如 RDI
    - 大写字母和数字混合代码，如 RCP4.5
    - 单独月份、季节，如 April
    """
    if not text:
        return False

    s = text.strip()
    if not s:
        return False

    if s.lower() in _MONTH_OR_SEASON_SET:
        return True

    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        return True

    if re.fullmatch(r"[A-Z]+", s):
        return True

    if (
        re.fullmatch(r"[A-Z0-9.\-]+", s)
        and re.search(r"[A-Z]", s)
        and re.search(r"\d", s)
    ):
        return True

    return False


def rule_filter_candidates(
    candidates: list["ExtractionCandidate"],
    deduplicate: bool = True,
) -> list["ExtractionCandidate"]:
    """
    对 ExtractionCandidate 做第一阶段的轻量规则过滤。

    处理逻辑：
    1. 删除 text 为空或仅包含空白的 candidate
    2. 使用 light_filter_remove 删除明显噪声
    3. 可选按 normalize_text(text) + candidate_type 去重

    说明：
    - 不修改 candidate 的其他字段
    - 不改写原始 text
    - normalize_text 仅用于去重，不用于覆盖原文
    """
    results: list["ExtractionCandidate"] = []
    seen: set[tuple[str, str]] = set()

    for candidate in candidates:
        text = candidate.text.strip()
        if not text:
            continue

        if light_filter_remove(text):
            continue

        if deduplicate:
            dedup_key = (normalize_text(text), candidate.candidate_type)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

        results.append(candidate)

    return results


def _check_candidates_with_llm(
    original_candidates: list[ExtractionCandidate],
    system_prompt: str,
    user_input: str,
    allow_text_update: bool,
    source_text: str = "",
) -> list[ExtractionCandidate]:
    """
    调用审核模型检查候选列表。

    处理策略：
    1. 懒加载 LLM 客户端，避免模块导入阶段受环境依赖影响；
    2. 审核失败时直接回退为原候选，保证流程稳健；
    3. 使用当前批次内的临时 id 建立映射关系，不依赖返回顺序回填结果；
    4. 未匹配到审核结果的原候选直接保留。
    """
    if not original_candidates:
        return []

    try:
        from src.llm.client import call_llm
    except Exception:
        return list(original_candidates)

    try:
        raw_output = call_llm(
            user_input=user_input,
            system_prompt=system_prompt,
        )
    except Exception:
        return list(original_candidates)

    parsed_items = _parse_candidate_items(raw_output)
    if not parsed_items:
        return list(original_candidates)

    original_candidate_map = {
        str(idx): candidate for idx, candidate in enumerate(original_candidates)
    }
    parsed_item_map: dict[str, dict[str, Any]] = {}

    for parsed_item in parsed_items:
        parsed_id = str(parsed_item.get("id") or "").strip()
        if not parsed_id:
            continue
        if parsed_id not in original_candidate_map:
            continue
        if parsed_id in parsed_item_map:
            continue
        parsed_item_map[parsed_id] = parsed_item

    checked_candidates: list[ExtractionCandidate] = []
    for candidate_id, candidate in original_candidate_map.items():
        parsed_item = parsed_item_map.get(candidate_id)
        if parsed_item is None:
            checked_candidates.append(candidate)
            continue

        checked_candidates.append(
            _merge_checked_candidate(
                original_candidate=candidate,
                parsed_item=parsed_item,
                allow_text_update=allow_text_update,
                source_text=source_text,
            )
        )

    return checked_candidates


def _parse_candidate_items(raw_output: str) -> list[dict[str, Any]]:
    """
    将 LLM 原始输出解析为候选字典列表。

    JSON 解析完全复用项目内统一的 parse_json_safe，
    当前函数只负责对解析结果做轻量结构整理。
    """
    parsed = parse_json_safe(raw_output, default=None)
    if isinstance(parsed, dict):
        for key in ("candidates", "items", "results", "data"):
            nested = parsed.get(key)
            if isinstance(nested, (list, dict)):
                parsed = nested
                break

    items = _ensure_list(parsed)
    results: list[dict[str, Any]] = []

    for item in items:
        if isinstance(item, dict):
            results.append(item)

    return results


def _merge_checked_candidate(
    original_candidate: ExtractionCandidate,
    parsed_item: dict[str, Any] | None,
    allow_text_update: bool,
    source_text: str = "",
) -> ExtractionCandidate:
    """
    将单条审核结果与原始 candidate 合并。

    说明：
    - field_candidate 仅允许更新标签与原因；
    - evidence_span 可在能对齐原文时更新 text；
    - 若审核后字段为空，则统一降为 none。
    """
    merged_text = original_candidate.text.strip()
    target_fields = _normalize_target_fields(original_candidate.target_fields)
    relevance = _normalize_relevance(original_candidate.relevance)
    reason = str(original_candidate.reason or "").strip()

    if isinstance(parsed_item, dict):
        target_fields = _normalize_target_fields(
            parsed_item.get("target_fields", target_fields)
        )
        relevance = _normalize_relevance(parsed_item.get("relevance", relevance))
        reason = str(parsed_item.get("reason", reason) or "").strip()

        if allow_text_update:
            checked_text = str(parsed_item.get("text", merged_text) or "").strip()
            merged_text = _resolve_span_text_from_source(
                source_text=source_text,
                original_text=original_candidate.text,
                checked_text=checked_text,
            )

    if relevance == "none" or not target_fields:
        target_fields = []
        relevance = "none"

    return ExtractionCandidate(
        text=merged_text,
        candidate_type=original_candidate.candidate_type,
        target_fields=target_fields,
        relevance=relevance,
        reason=reason,
    )


def _candidate_to_payload(
    candidate: ExtractionCandidate,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    """
    将 dataclass candidate 转成可发送给 LLM 的基础字典。

    当 candidate_id 不为空时，会附带临时 id，
    仅用于本次审核请求的往返匹配。
    """
    payload = {
        "text": candidate.text,
        "candidate_type": candidate.candidate_type,
        "target_fields": list(candidate.target_fields),
        "relevance": candidate.relevance,
        "reason": candidate.reason,
    }
    if candidate_id is not None:
        payload["id"] = str(candidate_id)
    return payload


def _sanitize_candidate(candidate: ExtractionCandidate) -> ExtractionCandidate:
    """
    对输入 candidate 做轻量规范化。

    仅清洗：
    - 首尾空白
    - candidate_type / target_fields / relevance 的合法值
    - reason 的空白
    """
    text = str(candidate.text or "").strip()
    candidate_type = str(candidate.candidate_type or "").strip()
    if candidate_type not in _ALLOWED_CANDIDATE_TYPES:
        candidate_type = "field_candidate"

    return ExtractionCandidate(
        text=text,
        candidate_type=candidate_type,
        target_fields=_normalize_target_fields(candidate.target_fields),
        relevance=_normalize_relevance(candidate.relevance),
        reason=str(candidate.reason or "").strip(),
    )


def _deduplicate_candidates_by_text(
    candidates: list[ExtractionCandidate],
) -> list[ExtractionCandidate]:
    """
    按 candidate.text.strip() 去重，重复文本直接跳过。
    """
    results: list[ExtractionCandidate] = []
    seen: set[str] = set()

    for candidate in candidates:
        candidate_key = str(candidate.text or "").strip()
        if not candidate_key:
            continue
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        results.append(candidate)

    return results


def _filter_checked_candidates(
    candidates: list[ExtractionCandidate],
) -> list[ExtractionCandidate]:
    """
    对审核后的候选做最终过滤。

    删除：
    - 空文本
    - relevance 为 none 的项
    - target_fields 为空的项
    - 规则过滤阶段识别出的明显噪声
    """
    filtered_candidates = rule_filter_candidates(candidates)
    results: list[ExtractionCandidate] = []

    for candidate in filtered_candidates:
        sanitized_candidate = _sanitize_candidate(candidate)
        if not sanitized_candidate.text:
            continue
        if sanitized_candidate.relevance == "none":
            continue
        if not sanitized_candidate.target_fields:
            continue
        results.append(sanitized_candidate)

    return results


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


def _resolve_span_text_from_source(
    source_text: str,
    original_text: str,
    checked_text: str,
) -> str:
    """
    将审核后的 evidence span 尽量对齐回原文连续片段。

    处理策略：
    1. 优先使用审核后的 text，但前提是它能在原文中定位；
    2. 若审核后的 text 无法定位，则回退到原始 text；
    3. 若原始 text 也无法定位，则返回原始 text 的清洗结果；
    4. 不保留无法在原文中定位的模型自由改写文本。
    """
    cleaned_original_text = str(original_text or "").strip()
    cleaned_checked_text = str(checked_text or "").strip()

    if not isinstance(source_text, str) or not source_text.strip():
        return cleaned_original_text or cleaned_checked_text

    if cleaned_checked_text:
        checked_bounds = _find_span_bounds(source_text, cleaned_checked_text)
        if checked_bounds is not None:
            start, end = checked_bounds
            return source_text[start:end].strip()

    if cleaned_original_text:
        original_bounds = _find_span_bounds(source_text, cleaned_original_text)
        if original_bounds is not None:
            start, end = original_bounds
            return source_text[start:end].strip()

    return cleaned_original_text


def _find_span_bounds(
    source_text: str,
    span_text: str,
) -> tuple[int, int] | None:
    """
    在原文中鲁棒定位 span 的字符边界。

    定位顺序：
    1. 原样匹配
    2. 忽略大小写匹配
    3. 基于轻量归一化文本匹配
    """
    if not isinstance(source_text, str) or not source_text.strip():
        return None

    cleaned_span = str(span_text or "").strip()
    if not cleaned_span:
        return None

    direct_idx = source_text.find(cleaned_span)
    if direct_idx >= 0:
        return direct_idx, direct_idx + len(cleaned_span)

    source_lower = source_text.lower()
    span_lower = cleaned_span.lower()
    lower_idx = source_lower.find(span_lower)
    if lower_idx >= 0:
        return lower_idx, lower_idx + len(cleaned_span)

    source_norm, source_map = _build_normalized_text_map(source_text)
    span_norm, _ = _build_normalized_text_map(cleaned_span)
    if not source_norm or not span_norm:
        return None

    norm_idx = source_norm.find(span_norm)
    if norm_idx < 0:
        return None

    span_start = source_map[norm_idx]
    span_end = source_map[norm_idx + len(span_norm) - 1] + 1
    return span_start, span_end


def _build_normalized_text_map(text: str) -> tuple[str, list[int]]:
    """
    构造归一化文本及其到原文字符索引的映射。

    用于在忽略大小写、标点和多余空白后，
    仍能把命中的归一化片段映射回原文位置。
    """
    normalized_chars: list[str] = []
    index_map: list[int] = []
    pending_space = False

    for idx, char in enumerate(str(text or "").lower()):
        if char.isalnum():
            if pending_space and normalized_chars:
                normalized_chars.append(" ")
                index_map.append(idx)
                pending_space = False
            normalized_chars.append(char)
            index_map.append(idx)
            continue

        if normalized_chars:
            pending_space = True

    return "".join(normalized_chars), index_map


def _get_sentence_spans(source_text: str) -> list[tuple[int, int, str]]:
    """
    返回原文的句子边界与句子文本。

    优先使用项目内 spaCy 资源；
    若不可用，则退回简单正则分句。
    """
    if not isinstance(source_text, str) or not source_text.strip():
        return []

    try:
        from src.summ.config import MODEL_RESOURCE_CONFIG
        from src.utils.model_resources import get_spacy_model

        nlp = get_spacy_model(
            ensure_sentence_boundaries=True,
            resource_config=MODEL_RESOURCE_CONFIG,
        )
        doc = nlp(source_text)
        sentence_spans = [
            (
                sent.start_char,
                sent.end_char,
                source_text[sent.start_char : sent.end_char],
            )
            for sent in doc.sents
            if source_text[sent.start_char : sent.end_char].strip()
        ]
        if sentence_spans:
            return sentence_spans
    except Exception:
        pass

    sentence_spans: list[tuple[int, int, str]] = []
    for match in re.finditer(
        r".+?(?:[.!?]+(?=\s|$)|\n+|$)",
        source_text,
        flags=re.DOTALL,
    ):
        segment = match.group(0)
        stripped_segment = segment.strip()
        if not stripped_segment:
            continue

        start_offset = len(segment) - len(segment.lstrip())
        end_offset = len(segment.rstrip())
        sent_start = match.start() + start_offset
        sent_end = match.start() + end_offset
        sentence_spans.append((sent_start, sent_end, source_text[sent_start:sent_end]))

    return sentence_spans


def _coerce_non_negative_int(value: Any, default: int) -> int:
    """
    将输入安全转换为非负整数。
    """
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return max(default, 0)


def _ensure_list(data: Any) -> list[Any]:
    """
    将解析结果宽松转换为列表。
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


if __name__ == "__main__":
    a = build_evidence_span_check_prompt()
    b = build_field_candidate_check_prompt()
