from __future__ import annotations

from src.summ.config import ChunkConfig
from src.summ.schemas import (
    BlockProcessResult,
    ConnectedChunk,
    ExtractionRecord,
    PreparedSummaryChunk,
)
from src.summ.splitter import count_tokens


def build_summary_chunks(
    block_results: list[BlockProcessResult],
    config: ChunkConfig,
) -> list[PreparedSummaryChunk]:
    """
    根据 block 级处理结果构建最终的摘要输入块。

    处理逻辑：
    1. 先把每个 block 的 connected_chunks 合并成一个基础块
    2. 若基础块过短，则继续顺序拼接后续基础块
    3. 若拼接后的块过长，则按 token 拆分并保留 overlap

    参数:
        block_results: 每个原始文本块的处理结果
        config: 摘要输入块配置

    返回:
        list[PreparedSummaryChunk]
    """
    if not block_results:
        return []

    base_chunks = _build_base_chunks_from_blocks(block_results)
    if not base_chunks:
        return []

    merged_chunks = _merge_small_chunks(
        base_chunks=base_chunks,
        min_tokens=config.llm_text_min_tokens,
    )

    final_chunks: list[PreparedSummaryChunk] = []
    next_chunk_id = 0

    for chunk in merged_chunks:
        if chunk.token_count <= config.llm_text_max_tokens:
            chunk.prepared_chunk_id = next_chunk_id
            final_chunks.append(chunk)
            next_chunk_id += 1
            continue

        split_chunks = split_oversized_chunk_with_overlap(
            chunk=chunk,
            max_tokens=config.llm_text_max_tokens,
            overlap_tokens=config.overlap_tokens,
        )

        for split_chunk in split_chunks:
            split_chunk.prepared_chunk_id = next_chunk_id
            final_chunks.append(split_chunk)
            next_chunk_id += 1

    return final_chunks


def _build_base_chunks_from_blocks(
    block_results: list[BlockProcessResult],
) -> list[PreparedSummaryChunk]:
    """
    将每个 block 的 connected_chunks 合并为一个基础摘要块。
    """
    results: list[PreparedSummaryChunk] = []

    for block in block_results:
        text = _merge_connected_chunks_text(block.connected_chunks)
        token_count = count_tokens(text)

        if not text.strip():
            continue

        results.append(
            PreparedSummaryChunk(
                prepared_chunk_id=-1,
                text=text,
                token_count=token_count,
                extraction_records=list(block.extraction_records),
            )
        )

    return results


def _merge_connected_chunks_text(chunks: list[ConnectedChunk]) -> str:
    """
    将一个 block 内的多个 connected chunk 合并成文本。
    """
    if not chunks:
        return ""

    parts = [chunk.text.strip() for chunk in chunks if chunk.text.strip()]
    return "\n\n".join(parts).strip()


def _merge_small_chunks(
    base_chunks: list[PreparedSummaryChunk],
    min_tokens: int,
) -> list[PreparedSummaryChunk]:
    """
    将过短的基础块与后续块顺序拼接。
    """
    if not base_chunks:
        return []

    merged_results: list[PreparedSummaryChunk] = []
    i = 0

    while i < len(base_chunks):
        current_texts = [base_chunks[i].text]
        current_records = list(base_chunks[i].extraction_records)
        current_token_count = base_chunks[i].token_count

        j = i + 1
        while current_token_count < min_tokens and j < len(base_chunks):
            current_texts.append(base_chunks[j].text)
            current_records.extend(base_chunks[j].extraction_records)
            current_token_count = count_tokens("\n\n".join(current_texts))
            j += 1

        merged_results.append(
            PreparedSummaryChunk(
                prepared_chunk_id=-1,
                text="\n\n".join(current_texts).strip(),
                token_count=current_token_count,
                extraction_records=_deduplicate_extraction_records(current_records),
            )
        )

        i = j

    return merged_results


def split_oversized_chunk_with_overlap(
    chunk: PreparedSummaryChunk,
    max_tokens: int,
    overlap_tokens: int,
) -> list[PreparedSummaryChunk]:
    """
    将超长摘要块按 token 拆分，并保留一定 overlap。

    拆分方式保持简单：
    - 按空白切词
    - 近似按 token 窗口截取
    - overlap 也按同样方式处理

    注意：
    这里为了避免复杂实现，使用“按词近似 token 切分”的方式。
    最终 token_count 会再次调用 count_tokens 重新统计。
    """
    text = chunk.text.strip()
    if not text:
        return []

    if chunk.token_count <= max_tokens:
        return [chunk]

    words = text.split()
    if not words:
        return []

    results: list[PreparedSummaryChunk] = []

    # 用一个较保守的近似比例：1 word ≈ 1.3 tokens
    approx_tokens_per_word = 1.3
    window_size = max(1, int(max_tokens / approx_tokens_per_word))
    overlap_size = max(0, int(overlap_tokens / approx_tokens_per_word))

    start = 0
    while start < len(words):
        end = min(len(words), start + window_size)
        sub_text = " ".join(words[start:end]).strip()

        if sub_text:
            results.append(
                PreparedSummaryChunk(
                    prepared_chunk_id=-1,
                    text=sub_text,
                    token_count=count_tokens(sub_text),
                    extraction_records=list(chunk.extraction_records),
                )
            )

        if end >= len(words):
            break

        start = max(start + 1, end - overlap_size)

    return results


def _deduplicate_extraction_records(
    records: list[ExtractionRecord],
) -> list[ExtractionRecord]:
    """
    对粗提取结果做简单去重。
    """
    if not records:
        return []

    results: list[ExtractionRecord] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()

    for record in records:
        key = (
            tuple(record.keywords),
            tuple(record.key_sentences),
        )
        if key in seen:
            continue

        seen.add(key)
        results.append(record)

    return results
