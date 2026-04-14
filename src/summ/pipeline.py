from __future__ import annotations

import logging
from typing import Any

from src.summ.checker_flow import check_summary_quality_with_flow
from src.summ.chunk_builder import build_summary_chunks
from src.summ.config import PipelineConfig
from src.summ.connector import connect_selected_sentences, merge_connected_chunks_text
from src.summ.extractor_flow import extract_key_elements_with_flow
from src.summ.schemas import (
    BlockProcessResult,
    CheckResult,
    ExtractionRecord,
    InputDocument,
    InputItem,
    PreparedSummaryChunk,
    SummaryChunkResult,
    SummaryPipelineResult,
)
from src.summ.selector import select_key_sentences
from src.summ.splitter import split_text_into_sentences
from src.summ.summarizer import generate_targeted_summary

logger = logging.getLogger(__name__)


def run_summary_pipeline(
    input_document: InputDocument,
    config: PipelineConfig,
    extraction_model_config: dict[str, Any] | None = None,
    summary_model_config: dict[str, Any] | None = None,
    extraction_max_retries: int = 2,
) -> SummaryPipelineResult:
    """
    对单个 InputDocument 执行完整摘要流程。

    流程：
    1. 对每个 InputItem 执行提取、切分、筛选、连接
    2. 构建最终摘要输入块
    3. 对每个摘要输入块生成摘要并检查
    4. 拼接得到 final_summary
    """
    if not isinstance(input_document, InputDocument):
        logger.warning("Invalid input_document type: %s", type(input_document))
        return SummaryPipelineResult(
            final_summary="",
            block_results=[],
            prepared_chunks=[],
            chunk_results=[],
        )

    block_results: list[BlockProcessResult] = []

    for block_idx, item in enumerate(input_document.items):
        block_result = process_input_item(
            item=item,
            block_idx=block_idx,
            config=config,
            extraction_model_config=extraction_model_config,
            extraction_max_retries=extraction_max_retries,
        )
        block_results.append(block_result)

    prepared_chunks = build_summary_chunks(
        block_results=block_results,
        config=config.chunk,
    )
    chapter_chunk_counts = _build_chapter_chunk_counts(prepared_chunks)

    chunk_results: list[SummaryChunkResult] = []
    final_summaries: list[str] = []

    for prepared_chunk in prepared_chunks:
        location_label = _build_chunk_location_label(
            prepared_chunk=prepared_chunk,
            input_document=input_document,
        )

        chunk_result = summarize_and_check_chunk(
            prepared_chunk=prepared_chunk,
            config=config,
            summary_model_config=summary_model_config,
            location_label=location_label,
            chapter_chunk_counts=chapter_chunk_counts,
        )
        chunk_results.append(chunk_result)

        if chunk_result.check_result.passed and chunk_result.summary.strip():
            final_summaries.append(chunk_result.summary.strip())
        elif chunk_result.summary.strip():
            logger.warning(
                "Skip failed summary chunk in final_summary: %s",
                location_label,
            )

    final_summary = "\n\n".join(final_summaries).strip()

    return SummaryPipelineResult(
        final_summary=final_summary,
        block_results=block_results,
        prepared_chunks=prepared_chunks,
        chunk_results=chunk_results,
    )


def process_input_item(
    item: InputItem,
    block_idx: int,
    config: PipelineConfig,
    extraction_model_config: dict[str, Any] | None = None,
    extraction_max_retries: int = 2,
) -> BlockProcessResult:
    """
    对单个 InputItem 执行第1~4步处理。
    """
    if not isinstance(item, InputItem):
        logger.warning("Invalid item type at block_idx=%s: %s", block_idx, type(item))
        return BlockProcessResult(block_idx=block_idx, original_text="")

    original_text = item.text or ""

    extraction_record = extract_key_elements_with_flow(
        text=original_text,
        chunk_config=config.chunk,
        split_config=config.split,
        model_config=extraction_model_config,
        max_retries=extraction_max_retries,
    )
    extraction_record = _coerce_extraction_record(extraction_record)

    sentences = split_text_into_sentences(
        text=original_text,
        spacy_model_name=config.split.spacy_model_name,
        tokenizer_name=config.split.tokenizer_name,
        min_sentence_chars=config.split.min_sentence_chars,
    )

    sentences = select_key_sentences(
        sentences=sentences,
        extraction_record=extraction_record,
        embedding_model_name=config.selection.embedding_model_name,
        fuzzy_threshold=config.selection.fuzzy_threshold,
        semantic_threshold=config.selection.semantic_threshold,
    )

    connected_chunks = connect_selected_sentences(
        sentences=sentences,
        config=config.connection,
    )

    merged_text, merged_token_count = merge_connected_chunks_text(connected_chunks)

    return BlockProcessResult(
        block_idx=block_idx,
        original_text=original_text,
        extraction_records=_normalize_pipeline_extraction_records(extraction_record),
        sentences=sentences,
        connected_chunks=connected_chunks,
        merged_text=merged_text,
        merged_token_count=merged_token_count,
    )


def summarize_and_check_chunk(
    prepared_chunk: PreparedSummaryChunk,
    config: PipelineConfig,
    summary_model_config: dict[str, Any] | None = None,
    location_label: str = "",
    chapter_chunk_counts: dict[tuple[int, ...], int] | None = None,
) -> SummaryChunkResult:
    """
    对单个 PreparedSummaryChunk 执行摘要和检查。
    """
    if not isinstance(prepared_chunk, PreparedSummaryChunk):
        logger.warning("Invalid prepared_chunk type: %s", type(prepared_chunk))
        return SummaryChunkResult(
            prepared_chunk_id=-1,
            source_text="",
            summary="",
            attempts=0,
            check_result=build_failed_check_result(status="invalid_chunk"),
            retry_reasons=[],
        )

    if not _has_effective_extraction_records(prepared_chunk):
        return SummaryChunkResult(
            prepared_chunk_id=prepared_chunk.prepared_chunk_id,
            source_text=prepared_chunk.text,
            summary="",
            attempts=0,
            check_result=build_failed_check_result(status="no_extraction"),
            retry_reasons=[],
        )

    chapter_chunk_count = _get_chapter_chunk_count(
        prepared_chunk=prepared_chunk,
        chapter_chunk_counts=chapter_chunk_counts,
    )

    max_retry_times = config.summary.max_retry_times
    last_summary = ""
    last_check_result = build_failed_check_result(status="summary_check_failed")
    retry_reasons: list[list[str]] = []

    for attempt in range(1, max_retry_times + 2):
        raw_summary = generate_targeted_summary(
            prepared_chunk=prepared_chunk,
            model_config=summary_model_config,
            max_retry_times=0,
        )
        summary = raw_summary if isinstance(raw_summary, str) else ""
        last_summary = summary

        check_result = check_summary_quality_with_flow(
            prepared_chunk=prepared_chunk,
            summary=summary,
            config=config.check,
            chapter_chunk_count=chapter_chunk_count,
        )
        check_result.status = (
            "passed" if check_result.passed else "summary_check_failed"
        )
        last_check_result = check_result

        if not check_result.passed:
            failed_scores = _collect_failed_score_names(check_result, config)
            retry_reasons.append(failed_scores)
            logger.warning(
                "Summary check failed on attempt %s: %s; reasons: %s",
                attempt,
                location_label
                or f"prepared_chunk_id={prepared_chunk.prepared_chunk_id}",
                _format_check_failure_reasons(check_result, config),
            )

        if check_result.passed:
            return SummaryChunkResult(
                prepared_chunk_id=prepared_chunk.prepared_chunk_id,
                source_text=prepared_chunk.text,
                summary=summary,
                attempts=attempt,
                check_result=check_result,
                retry_reasons=retry_reasons,
            )

    return SummaryChunkResult(
        prepared_chunk_id=prepared_chunk.prepared_chunk_id,
        source_text=prepared_chunk.text,
        summary=last_summary,
        attempts=max_retry_times + 1,
        check_result=last_check_result,
        retry_reasons=retry_reasons,
    )


def build_failed_check_result(status: str = "summary_check_failed") -> CheckResult:
    """
    构造一个默认失败的检查结果。
    """
    return CheckResult(
        passed=False,
        information_coverage=0.0,
        auxiliary_score=0.0,
        overall_score=0.0,
        status=status,
    )


def _coerce_extraction_record(extraction_record: Any) -> ExtractionRecord:
    if isinstance(extraction_record, ExtractionRecord):
        return extraction_record

    keywords = getattr(extraction_record, "keywords", None) or []
    key_sentences = getattr(extraction_record, "key_sentences", None) or []

    return ExtractionRecord(
        keywords=list(keywords),
        key_sentences=list(key_sentences),
    )


def _normalize_pipeline_extraction_records(
    extraction_record: ExtractionRecord,
) -> list[ExtractionRecord]:
    if extraction_record.keywords or extraction_record.key_sentences:
        return [extraction_record]
    return []


def _has_effective_extraction_records(prepared_chunk: PreparedSummaryChunk) -> bool:
    extraction_records = getattr(prepared_chunk, "extraction_records", None) or []
    return any(
        (getattr(record, "keywords", None) or [])
        or (getattr(record, "key_sentences", None) or [])
        for record in extraction_records
    )


def _collect_failed_score_names(
    check_result: CheckResult,
    config: PipelineConfig,
) -> list[str]:
    failed_scores: list[str] = []

    if check_result.information_coverage < config.check.coverage_threshold:
        failed_scores.append("information_coverage")

    if check_result.overall_score < config.check.overall_threshold:
        failed_scores.append("overall_score")

    if not failed_scores:
        failed_scores.append("unknown")

    return failed_scores


def _format_check_failure_reasons(
    check_result: CheckResult,
    config: PipelineConfig,
) -> str:
    failed_scores = _collect_failed_score_names(check_result, config)
    reason_parts: list[str] = []

    for score_name in failed_scores:
        if score_name == "information_coverage":
            reason_parts.append(
                "information_coverage="
                f"{check_result.information_coverage:.4f} < "
                f"{config.check.coverage_threshold:.4f}"
            )
        elif score_name == "overall_score":
            reason_parts.append(
                f"overall_score={check_result.overall_score:.4f} < "
                f"{config.check.overall_threshold:.4f}; "
                f"auxiliary_score={check_result.auxiliary_score:.4f}"
            )
        else:
            reason_parts.append(
                "passed=False but no threshold violation found; "
                f"auxiliary_score={check_result.auxiliary_score:.4f}"
            )

    return "; ".join(reason_parts)


def _build_chapter_chunk_counts(
    prepared_chunks: list[PreparedSummaryChunk],
) -> dict[tuple[int, ...], int]:
    """
    统计每个章节来源对应的 summary chunk 数量。
    """
    results: dict[tuple[int, ...], int] = {}

    for prepared_chunk in prepared_chunks:
        key = tuple(getattr(prepared_chunk, "source_block_indices", None) or [])
        results[key] = results.get(key, 0) + 1

    return results


def _get_chapter_chunk_count(
    prepared_chunk: PreparedSummaryChunk,
    chapter_chunk_counts: dict[tuple[int, ...], int] | None,
) -> int:
    if not chapter_chunk_counts:
        return 1

    key = tuple(getattr(prepared_chunk, "source_block_indices", None) or [])
    return chapter_chunk_counts.get(key, 1)


def _build_chunk_location_label(
    prepared_chunk: PreparedSummaryChunk,
    input_document: InputDocument,
) -> str:
    file_name = input_document.file_name.strip() or "<unknown_file>"
    block_indices = list(getattr(prepared_chunk, "source_block_indices", None) or [])

    block_idx_text = (
        ",".join(str(idx) for idx in block_indices)
        if block_indices
        else "<unknown_block_idx>"
    )

    return (
        f"file_name={file_name}"
        f"block_indices={block_idx_text}, "
        f"prepared_chunk_id={prepared_chunk.prepared_chunk_id}"
    )


def _find_titles_by_block_indices(
    block_indices: list[int],
    input_document: InputDocument,
) -> list[str]:
    if not block_indices:
        return []

    titles: list[str] = []
    seen: set[str] = set()

    for block_idx in block_indices:
        if block_idx < 0 or block_idx >= len(input_document.items):
            continue

        item = input_document.items[block_idx]
        if not isinstance(item, InputItem):
            continue

        title = item.title.strip() or f"<untitled_block_{block_idx}>"
        if title in seen:
            continue

        seen.add(title)
        titles.append(title)

    return titles
