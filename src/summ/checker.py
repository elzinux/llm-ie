from __future__ import annotations

from rapidfuzz import fuzz
from sentence_transformers import util

from src.eval.semantic_eval.scorer import score_semantic_eval
from src.summ.config import MODEL_RESOURCE_CONFIG, CheckConfig
from src.summ.schemas import CheckResult, ExtractionRecord, PreparedSummaryChunk
from src.summ.splitter import split_paragraph_into_sentences
from src.utils.model_resources import get_embedding_model
from src.utils.normalize_text import normalize_text

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
    2. 用摘要与原文的语义覆盖估计 semantic_eval_score
    3. 根据阈值判断是否通过
    """
    if not isinstance(source_chunk, PreparedSummaryChunk):
        return CheckResult(
            passed=False,
            information_coverage=0.0,
            semantic_eval_score=0.0,
            overall_score=0.0,
        )

    summary = summary.strip()
    if not summary:
        return CheckResult(
            passed=False,
            information_coverage=0.0,
            semantic_eval_score=0.0,
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

    semantic_eval_score = score_semantic_eval(
        candidate_text=summary,
        reference_text=source_chunk.text,
    )

    overall_score = _compute_overall_score(
        information_coverage=information_coverage,
        semantic_eval_score=semantic_eval_score,
        config=config,
    )

    passed = (
        information_coverage >= config.coverage_threshold
        and overall_score >= config.overall_threshold
    )

    return CheckResult(
        passed=passed,
        information_coverage=round(information_coverage, 4),
        semantic_eval_score=round(semantic_eval_score, 4),
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
    semantic_eval_score: float,
    config: CheckConfig,
) -> float:
    total_weight = config.coverage_weight + config.semantic_eval_weight
    if total_weight <= 0:
        return 0.0

    score = (
        information_coverage * config.coverage_weight
        + semantic_eval_score * config.semantic_eval_weight
    ) / total_weight
    return round(score, 4)
