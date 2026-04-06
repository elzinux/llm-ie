from __future__ import annotations

from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

from src.summ.config import CheckConfig
from src.summ.schemas import CheckResult, ExtractionRecord, PreparedSummaryChunk
from src.summ.semantic_eval.scorer import score_semantic_eval

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
    2. 用关键句与摘要的整体语义相似度估计 relation_score
    3. 根据阈值判断是否通过
    """
    if not isinstance(source_chunk, PreparedSummaryChunk):
        return CheckResult(
            passed=False,
            information_coverage=0.0,
            relation_score=0.0,
        )

    summary = summary.strip()
    if not summary:
        return CheckResult(
            passed=False,
            information_coverage=0.0,
            relation_score=0.0,
        )

    extraction_records = source_chunk.extraction_records
    if not extraction_records:
        return CheckResult(
            passed=True,
            information_coverage=1.0,
            relation_score=1.0,
        )

    information_coverage = compute_information_coverage(
        extraction_records=extraction_records,
        summary=summary,
        keyword_fuzzy_threshold=config.keyword_fuzzy_threshold,
        sentence_semantic_threshold=config.sentence_semantic_threshold,
    )

    relation_score = compute_relation_score(
        extraction_records=extraction_records,
        summary=summary,
    )

    semantic_eval_score = score_semantic_eval(
        candidate_text=summary, reference_text=source_chunk.text
    )

    overall_score = _compute_overall_score(
        information_coverage=information_coverage,
        relation_score=relation_score,
        semantic_eval_score=semantic_eval_score,
        config=config,
    )

    passed = (
        information_coverage >= config.coverage_threshold
        and relation_score >= config.relation_threshold
        and overall_score >= config.overall_threshold
    )

    return CheckResult(
        passed=passed,
        information_coverage=information_coverage,
        relation_score=relation_score,
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


def compute_relation_score(
    extraction_records: list[ExtractionRecord],
    summary: str,
) -> float:
    """
    估计摘要对重要信息关系的保持程度。

    当前 schema 没有显式的结构化关系字段，
    因此这里只做一个简单近似：
    - 将所有 key_sentences 合并为参考文本
    - 计算参考文本与摘要的整体语义相似度

    返回 0~1 之间的 float。
    """
    key_sentences = _collect_key_sentences(extraction_records)
    if not key_sentences:
        return 0.0

    reference_text = " ".join(key_sentences).strip()
    if not reference_text or not summary.strip():
        return 0.0

    model = _get_embedding_model(_DEFAULT_EMBEDDING_MODEL_NAME)

    embeddings = model.encode(
        [reference_text, summary],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    score = float(util.cos_sim(embeddings[0], embeddings[1]).item())
    return _clamp_score(score)


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

    summary_lower = summary.lower()
    covered = 0

    for keyword in keywords:
        keyword = keyword.strip()
        if not keyword:
            continue

        keyword_lower = keyword.lower()

        if keyword_lower in summary_lower:
            covered += 1
            continue

        fuzzy_score = fuzz.partial_ratio(keyword_lower, summary_lower)
        if fuzzy_score >= fuzzy_threshold:
            covered += 1

    return covered / len(keywords)


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

    summary_sentences = _simple_split_sentences(summary)
    if not summary_sentences:
        summary_sentences = [summary]

    model = _get_embedding_model(_DEFAULT_EMBEDDING_MODEL_NAME)

    key_embeddings = model.encode(
        key_sentences,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    summary_embeddings = model.encode(
        summary_sentences,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    similarity_matrix = util.cos_sim(key_embeddings, summary_embeddings)

    covered = 0
    for i in range(len(key_sentences)):
        best_similarity = float(similarity_matrix[i].max().item())
        if best_similarity >= semantic_threshold:
            covered += 1

    return covered / len(key_sentences)


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


def _get_embedding_model(model_name: str) -> SentenceTransformer:
    """
    获取并缓存语义模型。
    """
    if model_name not in _EMBEDDING_MODEL_CACHE:
        _EMBEDDING_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBEDDING_MODEL_CACHE[model_name]


def _clamp_score(score: float) -> float:
    """
    将分数限制到 [0, 1]。
    """
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _compute_overall_score(
    information_coverage: float,
    relation_score: float,
    semantic_eval_score: float,
    config: CheckConfig,
) -> float:
    score = (
        information_coverage * config.coverage_weight
        + relation_score * config.relation_weight
        + semantic_eval_score * config.semantic_eval_weight
    )
    return round(score, 4)
