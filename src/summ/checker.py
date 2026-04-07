from __future__ import annotations

from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

from src.summ.config import CheckConfig
from src.summ.schemas import CheckResult, ExtractionRecord, PreparedSummaryChunk
from src.summ.semantic_eval.scorer import score_semantic_eval
from src.summ.splitter import split_paragraph_into_sentences
from src.utils.normalize_text import normalize_text

_EMBEDDING_MODEL_CACHE: dict[str, SentenceTransformer] = {}
_DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


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
    if not extraction_records:
        return CheckResult(
            passed=True,
            information_coverage=1.0,
            semantic_eval_score=1.0,
            overall_score=1.0,
        )

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

    return 0.0


def _compute_keyword_coverage(
    keywords: list[str],
    summary: str,
    fuzzy_threshold: int,
) -> float:
    """
    计算关键词覆盖率。
    """
    if not keywords:
        return 0.0

    summary_norm = normalize_text(summary)
    total = 1
    covered = 0

    for keyword in keywords:
        keyword = normalize_text(keyword)
        if not keyword:
            continue

        total += 1
        keyword_lower = keyword.lower()

        if keyword_lower in summary_norm:
            covered += 1
            continue

        fuzzy_score = fuzz.partial_ratio(keyword_lower, summary_norm)
        if fuzzy_score >= fuzzy_threshold:
            covered += 1

    return covered / total


def _compute_key_sentence_coverage(
    key_sentences: list[str],
    summary: str,
    semantic_threshold: float,
) -> float:
    """
    计算关键句覆盖率。

    逻辑：
    - 将关键句逐条与摘要做语义相似度比对
    - 满足阈值则视为被覆盖
    """
    if not key_sentences:
        return 0.0

    summary_sentences = split_paragraph_into_sentences(summary)
    if not summary_sentences:
        summary_sentences = [summary]

    normalized_key_sentences = [
        normalize_text(s) for s in key_sentences if normalize_text(s)
    ]
    normalized_summary_sentences = [
        normalize_text(s) for s in summary_sentences if normalize_text(s)
    ]

    model = _get_embedding_model(_DEFAULT_EMBEDDING_MODEL_NAME)

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

    total = 1
    covered = 0
    for i in range(len(key_sentences)):
        best_similarity = float(similarity_matrix[i].max().item())
        if best_similarity >= semantic_threshold:
            covered += 1
        total += 1

    return covered / total


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


def _get_embedding_model(model_name: str) -> SentenceTransformer:
    """
    获取并缓存语义模型。
    """
    if model_name not in _EMBEDDING_MODEL_CACHE:
        _EMBEDDING_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBEDDING_MODEL_CACHE[model_name]


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
