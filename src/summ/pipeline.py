from __future__ import annotations

from typing import Any

from src.summ.checker import check_summary_quality
from src.summ.chunk_builder import build_summary_chunks
from src.summ.config import PipelineConfig
from src.summ.connector import connect_selected_sentences, merge_connected_chunks_text
from src.summ.extractor import extract_key_elements
from src.summ.schemas import (
    BlockProcessResult,
    CheckResult,
    InputDocument,
    InputItem,
    SummaryChunkResult,
    SummaryPipelineResult,
)
from src.summ.selector import select_key_sentences
from src.summ.splitter import split_text_into_sentences
from src.summ.summarizer import generate_targeted_summary


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

    参数:
        input_document: 统一输入文档结构
        config: 全流程配置
        extraction_model_config: 第1步调用大模型时的配置
        summary_model_config: 第6步调用大模型时的配置
        extraction_max_retries: 第1步最大重试次数

    返回:
        SummaryPipelineResult
    """
    if not isinstance(input_document, InputDocument):
        return SummaryPipelineResult(final_summary="")

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

    chunk_results: list[SummaryChunkResult] = []
    final_summaries: list[str] = []

    for prepared_chunk in prepared_chunks:
        chunk_result = summarize_and_check_chunk(
            prepared_chunk=prepared_chunk,
            config=config,
            summary_model_config=summary_model_config,
        )
        chunk_results.append(chunk_result)

        if chunk_result.summary.strip():
            final_summaries.append(chunk_result.summary.strip())

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
    original_text = item.text if item.text else ""

    extraction_record = extract_key_elements(
        text=original_text,
        model_config=extraction_model_config,
        max_retries=extraction_max_retries,
    )

    sentences = split_text_into_sentences(
        text=original_text,
        spacy_model_name=config.split.spacy_model_name,
        tokenizer_name=config.split.tokenizer_name,
        min_sentence_chars=config.split.min_sentence_chars,
    )

    selected_sentences = select_key_sentences(
        sentences=sentences,
        extraction_record=extraction_record,
        embedding_model_name=config.selection.embedding_model_name,
        fuzzy_threshold=config.selection.fuzzy_threshold,
        semantic_threshold=config.selection.semantic_threshold,
    )

    connected_chunks = connect_selected_sentences(
        sentences=selected_sentences,
        config=config.connection,
    )

    merged_text, merged_token_count = merge_connected_chunks_text(connected_chunks)

    return BlockProcessResult(
        block_idx=block_idx,
        original_text=original_text,
        extraction_records=[extraction_record],
        sentences=selected_sentences,
        connected_chunks=connected_chunks,
        merged_text=merged_text,
        merged_token_count=merged_token_count,
    )


def summarize_and_check_chunk(
    prepared_chunk,
    config: PipelineConfig,
    summary_model_config: dict[str, Any] | None = None,
) -> SummaryChunkResult:
    """
    对单个 PreparedSummaryChunk 执行摘要和检查。
    """
    max_retry_times = config.summary.max_retry_times

    last_summary = ""
    last_check_result = build_failed_check_result()

    for attempt in range(1, max_retry_times + 2):
        summary = generate_targeted_summary(
            prepared_chunk=prepared_chunk,
            model_config=summary_model_config,
            max_retry_times=0,
        )
        last_summary = summary

        check_result = check_summary_quality(
            source_chunk=prepared_chunk,
            summary=summary,
            config=config.check,
        )
        last_check_result = check_result

        if check_result.passed:
            return SummaryChunkResult(
                prepared_chunk_id=prepared_chunk.prepared_chunk_id,
                source_text=prepared_chunk.text,
                summary=summary,
                attempts=attempt,
                check_result=check_result,
            )

    return SummaryChunkResult(
        prepared_chunk_id=prepared_chunk.prepared_chunk_id,
        source_text=prepared_chunk.text,
        summary=last_summary,
        attempts=max_retry_times + 1,
        check_result=last_check_result,
    )


def build_failed_check_result() -> CheckResult:
    """
    构造一个默认失败的检查结果。
    """
    return CheckResult(
        passed=False,
        information_coverage=0.0,
        relation_score=0.0,
        semantic_eval_score=0.0,
        overall_score=0.0,
    )
