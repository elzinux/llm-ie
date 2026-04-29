from __future__ import annotations

import logging
from pathlib import Path

from rapidfuzz import fuzz
from sentence_transformers import util

from src.summ.auxiliary_scorer import compute_auxiliary_score
from src.summ.config import MODEL_RESOURCE_CONFIG, CheckConfig
from src.summ.schemas import CheckResult, ExtractionRecord, PreparedSummaryChunk
from src.summ.splitter import split_paragraph_into_sentences
from src.utils.model_resources import get_embedding_model
from src.utils.normalize_text import normalize_text

_NLI_LOG_FILE = Path("logs") / "summ_nli_scores.txt"


def _get_nli_score_logger() -> logging.Logger:
    """
    仅用于记录 NLI 打分分析结果。
    只写文件，不输出到终端。
    """
    logger = logging.getLogger(f"{__name__}.nli_score")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    _NLI_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(_NLI_LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


nli_score_logger = _get_nli_score_logger()


def _ensure_nli_score_log_header() -> None:
    """
    如果日志文件为空，则写入表头。
    """
    if not _NLI_LOG_FILE.exists() or _NLI_LOG_FILE.stat().st_size == 0:
        nli_score_logger.info(
            "prepared_chunk_id\tcontradiction\tentailment\tneutral\tauxiliary_score"
        )


_DEFAULT_EMBEDDING_MODEL_NAME = MODEL_RESOURCE_CONFIG.embedding_model_name
_RELEVANCE_WEIGHTS = {"high": 3, "medium": 2, "low": 1, "none": 0}


def check_summary_quality(
    source_chunk: PreparedSummaryChunk,
    summary: str,
    config: CheckConfig,
) -> CheckResult:
    """
    检查摘要质量，返回 CheckResult。

    当前检查逻辑保持简单：
    1. 用关键词和关键句估计 information_coverage
    2. 计算摘要对应的 auxiliary_score
    3. 根据阈值判断是否通过
    """
    if not isinstance(source_chunk, PreparedSummaryChunk):
        return CheckResult(
            passed=False,
            information_coverage=0.0,
            auxiliary_score=0.0,
            overall_score=0.0,
        )

    summary = summary.strip()
    if not summary:
        return CheckResult(
            passed=False,
            information_coverage=0.0,
            auxiliary_score=0.0,
            overall_score=0.0,
        )

    extraction_records = source_chunk.extraction_records
    information_coverage = 1.0
    if extraction_records:
        information_coverage = compute_information_coverage(
            extraction_records=extraction_records,
            summary=summary,
            keyword_fuzzy_threshold=config.keyword_fuzzy_threshold,
            sentence_semantic_threshold=config.sentence_semantic_threshold,
        )

    auxiliary_score, nli_score = compute_auxiliary_score(
        source_text=source_chunk.text,
        summary=summary,
    )

    _ensure_nli_score_log_header()

    contradiction, entailment, neutral = nli_score

    nli_score_logger.info(
        "%s\t%.4f\t%.4f\t%.4f\t%.4f",
        source_chunk.prepared_chunk_id,
        contradiction,
        entailment,
        neutral,
        auxiliary_score,
    )

    overall_score = _compute_overall_score(
        information_coverage=information_coverage,
        auxiliary_score=auxiliary_score,
        config=config,
    )

    passed = (
        information_coverage >= config.coverage_threshold
        and auxiliary_score >= config.auxiliary_score_threshold
    ) or (
        overall_score >= config.overall_threshold
        and information_coverage >= config.minimum_coverage_floor
        and auxiliary_score >= config.minimum_auxiliary_floor
    )

    return CheckResult(
        passed=passed,
        information_coverage=round(information_coverage, 4),
        auxiliary_score=round(auxiliary_score, 4),
        overall_score=round(overall_score, 4),
    )


def compute_information_coverage(
    extraction_records: list[ExtractionRecord],
    summary: str,
    keyword_fuzzy_threshold: int,
    sentence_semantic_threshold: float,
) -> float:
    """
    计算摘要对目标信息的覆盖程度。

    先分别计算 keywords 和 key_sentences 的加权覆盖率，
    再按固定比例汇总：
    information_coverage = 0.4 * keyword_coverage + 0.6 * key_sentence_coverage
    """
    keywords, keyword_relevance = _collect_keywords_with_relevance(extraction_records)
    key_sentences, key_sentence_relevance = _collect_key_sentences_with_relevance(
        extraction_records
    )

    keyword_coverage = _compute_keyword_coverage(
        keywords=keywords,
        summary=summary,
        fuzzy_threshold=keyword_fuzzy_threshold,
        keyword_relevance=keyword_relevance,
    )
    key_sentence_coverage = _compute_key_sentence_coverage(
        key_sentences=key_sentences,
        summary=summary,
        semantic_threshold=sentence_semantic_threshold,
        key_sentence_relevance=key_sentence_relevance,
    )

    return 0.4 * keyword_coverage + 0.6 * key_sentence_coverage


def _compute_keyword_coverage(
    keywords: list[str],
    summary: str,
    fuzzy_threshold: int,
    keyword_relevance: list[str] | None = None,
) -> float:
    """
    计算关键词覆盖率（加权覆盖率）。

    逻辑：
    - 先对关键词和摘要做归一化
    - 关键词先做词面包含判断
    - 若未命中，再做模糊匹配
    - 达到阈值则视为该关键词被覆盖
    - 使用 relevance 权重计算加权覆盖率
    """
    if not keywords:
        return 0.0

    summary_norm = normalize_text(summary)
    if not summary_norm:
        return 0.0

    if keyword_relevance is None:
        keyword_relevance = []

    total = 0
    covered = 0

    for i, keyword in enumerate(keywords):
        keyword_norm = normalize_text(keyword)
        if not keyword_norm:
            continue

        relevance = keyword_relevance[i] if i < len(keyword_relevance) else "low"
        weight = _RELEVANCE_WEIGHTS.get(relevance, 1)

        total += weight

        if keyword_norm in summary_norm:
            covered += weight
            continue

        fuzzy_score = fuzz.partial_ratio(keyword_norm, summary_norm)
        if fuzzy_score >= fuzzy_threshold:
            covered += weight

    if total == 0:
        return 0.0

    return covered / total


def _compute_key_sentence_coverage(
    key_sentences: list[str],
    summary: str,
    semantic_threshold: float,
    key_sentence_relevance: list[str] | None = None,
) -> float:
    """
    计算关键句覆盖率（加权覆盖率）。

    逻辑：
    - 先对关键句和摘要句做归一化
    - 每条关键句先做词面包含判断
    - 若未命中，再与摘要句集合计算语义相似度
    - 关键句与任一摘要句的最大相似度达到阈值，则视为被覆盖
    - 使用 relevance 权重计算加权覆盖率
    """
    if not key_sentences:
        return 0.0

    summary_sentences = split_paragraph_into_sentences(summary)
    if not summary_sentences:
        summary_sentences = [summary]
    # 对摘要中的长句子进行切分
    summary_sentences = _build_summary_matching_units(summary_sentences, key_sentences)

    if key_sentence_relevance is None:
        key_sentence_relevance = []

    normalized_key_sentences: list[str] = []
    key_sentence_weights: list[int] = []
    for idx, sentence in enumerate(key_sentences):
        norm = normalize_text(sentence)
        if not norm:
            continue
        normalized_key_sentences.append(norm)
        relevance = (
            key_sentence_relevance[idx] if idx < len(key_sentence_relevance) else "low"
        )
        key_sentence_weights.append(_RELEVANCE_WEIGHTS.get(relevance, 1))

    if not normalized_key_sentences:
        return 0.0

    normalized_summary_sentences = []
    for s in summary_sentences:
        norm = normalize_text(s)
        if norm:
            normalized_summary_sentences.append(norm)

    if not normalized_summary_sentences:
        return 0.0

    try:
        model = get_embedding_model(
            _DEFAULT_EMBEDDING_MODEL_NAME, resource_config=MODEL_RESOURCE_CONFIG
        )
    except RuntimeError:
        return _compute_literal_key_sentence_coverage(
            normalized_key_sentences=normalized_key_sentences,
            normalized_summary_sentences=normalized_summary_sentences,
            key_sentence_weights=key_sentence_weights,
        )

    key_embeddings = model.encode(
        normalized_key_sentences,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    summary_embeddings = model.encode(
        normalized_summary_sentences,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    similarity_matrix = util.cos_sim(key_embeddings, summary_embeddings)

    total = sum(key_sentence_weights)
    covered = 0
    summary_norm = " ".join(normalized_summary_sentences)

    for i, key_sentence in enumerate(normalized_key_sentences):
        weight = key_sentence_weights[i]
        if key_sentence in summary_norm:
            covered += weight
            continue

        best_similarity = float(similarity_matrix[i].max().item())
        if best_similarity >= semantic_threshold:
            covered += weight

    if total == 0:
        return 0.0

    return covered / total


def _compute_literal_key_sentence_coverage(
    normalized_key_sentences: list[str],
    normalized_summary_sentences: list[str],
    key_sentence_weights: list[int] | None = None,
) -> float:
    """
    当句向量模型不可用时，仅基于词面包含计算关键句覆盖率。
    """
    if not normalized_key_sentences or not normalized_summary_sentences:
        return 0.0

    if key_sentence_weights is None:
        key_sentence_weights = [1] * len(normalized_key_sentences)

    summary_norm = " ".join(normalized_summary_sentences)
    total = sum(key_sentence_weights)
    covered = 0

    for idx, key_sentence in enumerate(normalized_key_sentences):
        if key_sentence in summary_norm:
            covered += key_sentence_weights[idx]

    if total == 0:
        return 0.0

    return covered / total


def _collect_keywords_with_relevance(
    extraction_records: list[ExtractionRecord],
) -> tuple[list[str], list[str]]:
    """
    收集并去重所有关键词及其 relevance。
    """
    keywords: list[str] = []
    keyword_relevance: list[str] = []
    seen: set[str] = set()

    for record in extraction_records:
        for idx, keyword in enumerate(record.keywords):
            text = keyword.strip()
            if not text:
                continue

            norm = " ".join(text.lower().split())
            if norm in seen:
                continue

            seen.add(norm)
            keywords.append(text)
            if idx < len(record.keyword_relevance):
                relevance = record.keyword_relevance[idx]
            else:
                relevance = "low"
            keyword_relevance.append(str(relevance or "low").strip().lower() or "low")

    return keywords, keyword_relevance


def _collect_key_sentences_with_relevance(
    extraction_records: list[ExtractionRecord],
) -> tuple[list[str], list[str]]:
    """
    收集并去重所有关键句/关键片段及其 relevance。
    """
    key_sentences: list[str] = []
    key_sentence_relevance: list[str] = []
    seen: set[str] = set()

    for record in extraction_records:
        for idx, sentence in enumerate(record.key_sentences):
            text = sentence.strip()
            if not text:
                continue

            norm = " ".join(text.lower().split())
            if norm in seen:
                continue

            seen.add(norm)
            key_sentences.append(text)
            if idx < len(record.key_sentence_relevance):
                relevance = record.key_sentence_relevance[idx]
            else:
                relevance = "low"
            key_sentence_relevance.append(
                str(relevance or "low").strip().lower() or "low"
            )

    return key_sentences, key_sentence_relevance


def _simple_split_sentences(text: str) -> list[str]:
    """
    简单切句，不做复杂处理。
    """
    text = text.strip()
    if not text:
        return []

    separators = [". ", "? ", "! ", "\n"]
    sentences = [text]

    for sep in separators:
        new_sentences: list[str] = []
        for sentence in sentences:
            parts = sentence.split(sep)
            for part in parts:
                part = part.strip()
                if part:
                    new_sentences.append(part)
        sentences = new_sentences

    return sentences


def _compute_overall_score(
    information_coverage: float,
    auxiliary_score: float,
    config: CheckConfig,
) -> float:
    total_weight = config.coverage_weight + config.auxiliary_score_weight
    if total_weight <= 0:
        return 0.0

    score = (
        information_coverage * config.coverage_weight
        + auxiliary_score * config.auxiliary_score_weight
    ) / total_weight
    return round(score, 4)


# 切分 summary 中的句子，使得长度能匹配 key_sentences
def _is_fragment_boundary_char(c: str) -> bool:
    """
    判断当前字符是否适合作为切分边界。

    这里采用较宽松的规则：
    - 只要不是字母或数字，就认为可以作为边界
    - 典型包括：空格、标点、括号、连字符等
    """
    return not c.isalnum()


def _find_safe_split_pos(text: str, target_pos: int, search_radius: int = 10) -> int:
    """
    在目标切分点附近寻找更安全的切分位置。

    规则：
    1. 先向后找，最多 search_radius 个字符
    2. 再向前找，最多 search_radius 个字符
    3. 如果都找不到，就返回 target_pos 直接硬切

    返回值是切分位置，满足：
    - text[:pos] 为左侧片段
    - text[pos:] 为右侧片段
    """
    n = len(text)
    if n == 0:
        return 0

    target_pos = max(1, min(target_pos, n - 1))

    # 先向后找
    end = min(n - 1, target_pos + search_radius)
    for pos in range(target_pos, end + 1):
        if _is_fragment_boundary_char(text[pos]):
            return pos + 1

    # 再向前找
    start = max(1, target_pos - search_radius)
    for pos in range(target_pos - 1, start - 1, -1):
        if _is_fragment_boundary_char(text[pos]):
            return pos + 1

    # 都找不到则硬切
    return target_pos


def _split_long_sentence_for_matching(
    sentence: str,
    window_len: int,
    overlap_len: int,
    search_radius: int = 10,
) -> list[str]:
    """
    对单个长句做滑动切分，生成适合相似度匹配的较短片段。

    规则：
    - 若句长不超过 window_len，则直接返回原句
    - 若句长超过 window_len，则按窗口长度切分
    - 相邻窗口保留 overlap_len 重叠
    - 每次切分时优先在 target_pos 附近寻找安全边界
    - 保留去重后的非空片段
    """
    sentence = sentence.strip()
    if not sentence:
        return []

    if len(sentence) <= window_len:
        return [sentence]

    results: list[str] = []
    seen: set[str] = set()

    start = 0
    n = len(sentence)

    while start < n:
        raw_end = start + window_len
        if raw_end >= n:
            fragment = sentence[start:].strip()
            if fragment and fragment not in seen:
                results.append(fragment)
                seen.add(fragment)
            break

        split_pos = _find_safe_split_pos(
            sentence,
            target_pos=raw_end,
            search_radius=search_radius,
        )

        # 防御：避免切分点没有前进，导致死循环
        if split_pos <= start:
            split_pos = min(n, raw_end)

        fragment = sentence[start:split_pos].strip()
        if fragment and fragment not in seen:
            results.append(fragment)
            seen.add(fragment)

        next_start = max(start + 1, split_pos - overlap_len)
        if next_start <= start:
            next_start = split_pos

        start = next_start

    return results


def _build_summary_matching_units(
    summary_sentences: list[str],
    key_sentences: list[str],
) -> list[str]:
    """
    为 summary 构造用于匹配 key_sentences 的比较单元。

    逻辑：
    1. 保留所有原始 summary 句子
    2. 根据 key_sentences 的长度统计，确定长句切分参数
    3. 对明显过长的句子，额外切出若干较短片段
    4. 最终返回“原句 + 切分片段”的去重结果

    参数:
        summary_sentences: spaCy 分句后的 summary 句子列表
        key_sentences: 当前记录中的关键片段列表

    返回:
        可用于匹配的 summary 单元列表
    """
    cleaned_summary_sentences = [
        s.strip() for s in summary_sentences if s and s.strip()
    ]
    cleaned_key_sentences = [s.strip() for s in key_sentences if s and s.strip()]

    if not cleaned_summary_sentences:
        return []

    # 没有 key_sentences 时，退化为直接返回原句
    if not cleaned_key_sentences:
        return cleaned_summary_sentences

    key_lengths = sorted(len(s) for s in cleaned_key_sentences)
    if not key_lengths:
        return cleaned_summary_sentences

    # 使用稳健一点的代表长度，而不是 min/max
    p50_idx = int((len(key_lengths) - 1) * 0.50)
    p75_idx = int((len(key_lengths) - 1) * 0.75)

    p50 = key_lengths[p50_idx]
    p75 = key_lengths[p75_idx]

    # 切分参数
    window_len = max(p75 * 2, 80)
    overlap_len = max(p50 // 2, 20)

    results: list[str] = []
    seen: set[str] = set()

    for sentence in cleaned_summary_sentences:
        # 始终保留原句
        if sentence not in seen:
            results.append(sentence)
            seen.add(sentence)

        # 对过长句子再额外切分
        if len(sentence) > window_len:
            fragments = _split_long_sentence_for_matching(
                sentence=sentence,
                window_len=window_len,
                overlap_len=overlap_len,
                search_radius=10,
            )
            for fragment in fragments:
                if fragment not in seen:
                    results.append(fragment)
                    seen.add(fragment)

    return results
