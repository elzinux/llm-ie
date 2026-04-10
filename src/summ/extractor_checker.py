import re

from schemas import ExtractionCandidate

from src.summ.prompts import (
    build_evidence_span_check_prompt,
    build_field_candidate_check_prompt,
)
from src.utils.normalize_text import normalize_text

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
    1. 将输入 candidates 按 candidate_type 拆分为 field_candidate 和 evidence_span；
    2. 对 field_candidate 字段需要调用 rule_filter_candidates 函数进行初步筛选；
    3. 分别调用对应的检查函数；
    4. 合并检查结果并返回。

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
    ...


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
    ...


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
    ...


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
    ...


def find_span_char_window(
    source_text: str,
    span_text: str,
    char_window: int = 300,
) -> str:
    """
    为 evidence_span 提取字符级局部上下文。
    作为句子级定位失败时的兜底方案。
    """
    ...


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
    ...


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


if __name__ == "__main__":
    a = build_evidence_span_check_prompt()
    b = build_field_candidate_check_prompt()
