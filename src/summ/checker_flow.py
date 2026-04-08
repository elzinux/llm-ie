from __future__ import annotations

from sentence_transformers import util

from src.summ.checker import (
    _collect_key_sentences,
    _collect_keywords,
    check_summary_quality,
)
from src.summ.config import MODEL_RESOURCE_CONFIG, CheckConfig
from src.summ.extractor import normalize_extraction_record
from src.summ.schemas import CheckResult, ExtractionRecord, PreparedSummaryChunk
from src.summ.selector import _compute_direct_match_score
from src.summ.splitter import split_paragraph_into_sentences
from src.utils.model_resources import get_embedding_model
from src.utils.normalize_text import normalize_text

_DEFAULT_EMBEDDING_MODEL_NAME = MODEL_RESOURCE_CONFIG.embedding_model_name


def check_summary_quality_with_flow(
    prepared_chunk: PreparedSummaryChunk,
    summary: str,
    config: CheckConfig,
    chapter_chunk_count: int = 1,
) -> CheckResult:
    """
    对单个 summary chunk 执行 chunk-aware 检查。

    规则：
    1. 未切块章节，直接复用基础 checker
    2. 切块章节，先构造 local_records，再复用基础 checker
    """
    if chapter_chunk_count <= 1:
        return check_summary_quality(
            source_chunk=prepared_chunk,
            summary=summary,
            config=config,
        )

    local_records = build_local_records_for_chunk(
        prepared_chunk=prepared_chunk,
        config=config,
    )

    local_chunk = PreparedSummaryChunk(
        prepared_chunk_id=prepared_chunk.prepared_chunk_id,
        text=prepared_chunk.text,
        token_count=prepared_chunk.token_count,
        extraction_records=local_records,
        source_block_indices=list(prepared_chunk.source_block_indices),
    )
    return check_summary_quality(
        source_chunk=local_chunk,
        summary=summary,
        config=config,
    )


def build_local_records_for_chunk(
    prepared_chunk: PreparedSummaryChunk,
    config: CheckConfig,
) -> list[ExtractionRecord]:
    """
    从章节级全局参考信号中，为当前 chunk 构造 local_records。
    """
    global_records = list(getattr(prepared_chunk, "extraction_records", None) or [])
    if not global_records:
        return []

    global_keywords = _collect_keywords(global_records)
    global_key_sentences = _collect_key_sentences(global_records)

    local_keywords = filter_local_keywords(
        keywords=global_keywords,
        chunk_text=prepared_chunk.text,
        fuzzy_threshold=config.keyword_fuzzy_threshold,
    )
    local_key_sentences = filter_local_key_sentences(
        key_sentences=global_key_sentences,
        chunk_text=prepared_chunk.text,
        semantic_threshold=config.sentence_semantic_threshold,
    )

    local_record = normalize_extraction_record(
        ExtractionRecord(
            keywords=local_keywords,
            key_sentences=local_key_sentences,
        )
    )
    if not local_record.keywords and not local_record.key_sentences:
        return []

    return [local_record]


def filter_local_keywords(
    keywords: list[str],
    chunk_text: str,
    fuzzy_threshold: int,
) -> list[str]:
    """
    从章节级关键词中筛选当前 chunk 应负责的 local_keywords。
    """
    if not keywords or not chunk_text.strip():
        return []

    results: list[str] = []
    seen: set[str] = set()

    for keyword in keywords:
        match_score = _compute_direct_match_score(chunk_text, [keyword])
        if match_score < fuzzy_threshold:
            continue

        dedup_key = " ".join(keyword.lower().split())
        if dedup_key in seen:
            continue

        seen.add(dedup_key)
        results.append(keyword.strip())

    return results


def filter_local_key_sentences(
    key_sentences: list[str],
    chunk_text: str,
    semantic_threshold: float,
) -> list[str]:
    """
    从章节级关键句中筛选当前 chunk 应负责的 local_key_sentences。

    逻辑：
    - 先对关键句和 chunk 句子做归一化，并过滤空值
    - 对每条关键句，计算其与 chunk 内各句子的最大语义相似度
    - 达到阈值则保留
    - 对结果做归一化去重，但返回原始关键句文本（strip 后）
    """
    if not key_sentences:
        return []

    chunk_text = chunk_text.strip()
    if not chunk_text:
        return []

    chunk_sentences = split_paragraph_into_sentences(chunk_text)
    if not chunk_sentences:
        chunk_sentences = [chunk_text]

    filtered_key_sentences: list[str] = []
    normalized_key_sentences: list[str] = []
    for s in key_sentences:
        raw = s.strip()
        norm = normalize_text(raw)
        if raw and norm:
            filtered_key_sentences.append(raw)
            normalized_key_sentences.append(norm)

    if not normalized_key_sentences:
        return []

    filtered_chunk_sentences: list[str] = []
    normalized_chunk_sentences: list[str] = []
    for s in chunk_sentences:
        raw = s.strip()
        norm = normalize_text(raw)
        if raw and norm:
            filtered_chunk_sentences.append(raw)
            normalized_chunk_sentences.append(norm)

    if not normalized_chunk_sentences:
        return []

    model = get_embedding_model(_DEFAULT_EMBEDDING_MODEL_NAME, resource_config=MODEL_RESOURCE_CONFIG)

    key_embeddings = model.encode(
        normalized_key_sentences,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    chunk_embeddings = model.encode(
        normalized_chunk_sentences,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    similarity_matrix = util.cos_sim(key_embeddings, chunk_embeddings)

    results: list[str] = []
    seen: set[str] = set()

    for idx, raw_key_sentence in enumerate(filtered_key_sentences):
        best_similarity = float(similarity_matrix[idx].max().item())
        if best_similarity < semantic_threshold:
            continue

        dedup_key = normalized_key_sentences[idx]
        if dedup_key in seen:
            continue

        seen.add(dedup_key)
        results.append(raw_key_sentence)

    return results

