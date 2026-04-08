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
        information_coverage=information_coverage,
        semantic_eval_score=semantic_eval_score,
        overall_score=overall_score,
    )


def compute_information_coverage(
    extraction_records: list[ExtractionRecord],
    summary: str,
    keyword_fuzzy_threshold: int,
    sentence_semantic_threshold: float,
) -> float:
    """
    计算摘要对目标信息的覆盖程度。

    组成：
    - keyword coverage
    - key sentence coverage

    最终分数取两者平均。
    """
    keywords = _collect_keywords(extraction_records)
    key_sentences = _collect_key_sentences(extraction_records)

    keyword_coverage = _compute_keyword_coverage(
        keywords=keywords,
        summary=summary,
        fuzzy_threshold=keyword_fuzzy_threshold,
    )

    key_sentence_coverage = _compute_key_sentence_coverage(
        key_sentences=key_sentences,
        summary=summary,
        semantic_threshold=sentence_semantic_threshold,
    )

    if keywords and key_sentences:
        return (keyword_coverage + key_sentence_coverage) / 2.0
    if keywords:
        return keyword_coverage
    if key_sentences:
        return key_sentence_coverage

    return 1.0


def _compute_keyword_coverage(
    keywords: list[str],
    summary: str,
    fuzzy_threshold: int,
) -> float:
    """
    计算关键词覆盖率。

    逻辑：
    - 先对关键词和摘要做归一化
    - 关键词先做词面包含判断
    - 若未命中，再做模糊匹配
    - 达到阈值则视为该关键词被覆盖
    """
    if not keywords:
        return 0.0

    summary_norm = normalize_text(summary)
    if not summary_norm:
        return 0.0

    normalized_keywords = []
    for keyword in keywords:
        keyword_norm = normalize_text(keyword)
        if keyword_norm:
            normalized_keywords.append(keyword_norm)

    if not normalized_keywords:
        return 0.0

    covered = 0

    for keyword_norm in normalized_keywords:
        if keyword_norm in summary_norm:
            covered += 1
            continue

        fuzzy_score = fuzz.partial_ratio(keyword_norm, summary_norm)
        if fuzzy_score >= fuzzy_threshold:
            covered += 1

    return covered / len(normalized_keywords)


def _compute_key_sentence_coverage(
    key_sentences: list[str],
    summary: str,
    semantic_threshold: float,
) -> float:
    """
    计算关键句覆盖率。

    逻辑：
    - 先对关键句和摘要句做归一化
    - 每条关键句先做词面包含判断
    - 若未命中，再与摘要句集合计算语义相似度
    - 关键句与任一摘要句的最大相似度达到阈值，则视为被覆盖
    """
    if not key_sentences:
        return 0.0

    summary_sentences = split_paragraph_into_sentences(summary)
    if not summary_sentences:
        summary_sentences = [summary]

    normalized_key_sentences = []
    for s in key_sentences:
        norm = normalize_text(s)
        if norm:
            normalized_key_sentences.append(norm)

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

    covered = 0
    summary_norm = " ".join(normalized_summary_sentences)

    for i, key_sentence in enumerate(normalized_key_sentences):
        if key_sentence in summary_norm:
            covered += 1
            continue

        best_similarity = float(similarity_matrix[i].max().item())
        if best_similarity >= semantic_threshold:
            covered += 1

    return covered / len(normalized_key_sentences)


def _compute_literal_key_sentence_coverage(
    normalized_key_sentences: list[str],
    normalized_summary_sentences: list[str],
) -> float:
    """
    当句向量模型不可用时，仅基于词面包含计算关键句覆盖率。
    """
    if not normalized_key_sentences or not normalized_summary_sentences:
        return 0.0

    summary_norm = " ".join(normalized_summary_sentences)
    covered = sum(
        1 for key_sentence in normalized_key_sentences if key_sentence in summary_norm
    )
    return covered / len(normalized_key_sentences)


def _collect_keywords(extraction_records: list[ExtractionRecord]) -> list[str]:
    """
    收集并去重所有关键词。
    """
    results: list[str] = []
    seen: set[str] = set()

    for record in extraction_records:
        for keyword in record.keywords:
            text = keyword.strip()
            if not text:
                continue

            norm = " ".join(text.lower().split())
            if norm in seen:
                continue

            seen.add(norm)
            results.append(text)

    return results


def _collect_key_sentences(extraction_records: list[ExtractionRecord]) -> list[str]:
    """
    收集并去重所有关键句/关键片段。
    """
    results: list[str] = []
    seen: set[str] = set()

    for record in extraction_records:
        for sentence in record.key_sentences:
            text = sentence.strip()
            if not text:
                continue

            norm = " ".join(text.lower().split())
            if norm in seen:
                continue

            seen.add(norm)
            results.append(text)

    return results


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
