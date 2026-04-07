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
    1. 默认以章节为边界，将每个 block 构造成章节块
    2. 大章节只在章节内切分，并保留 overlap
    3. 小章节优先独立保留，仅在满足条件时与下一完整章节合并一次
    4. 不做链式扩展

    参数:
        block_results: 每个原始文本块的处理结果
        config: 摘要输入块配置

    返回:
        list[PreparedSummaryChunk]
    """
    if not block_results:
        return []

    chapter_chunks = _build_base_chunks_from_blocks(block_results)
    if not chapter_chunks:
        return []

    final_chunks = _build_final_chunks_by_chapter_boundary(
        chapter_chunks=chapter_chunks,
        config=config,
    )

    for chunk_id, chunk in enumerate(final_chunks):
        chunk.prepared_chunk_id = chunk_id

    return final_chunks


def _build_base_chunks_from_blocks(
    block_results: list[BlockProcessResult],
) -> list[PreparedSummaryChunk]:
    """
    将每个 block 的 connected_chunks 合并为一个基础章节块。
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
                source_block_indices=[block.block_idx],
            )
        )

    return results


def _build_final_chunks_by_chapter_boundary(
    chapter_chunks: list[PreparedSummaryChunk],
    config: ChunkConfig,
) -> list[PreparedSummaryChunk]:
    """
    以章节为边界构建最终摘要块。
    """
    if not chapter_chunks:
        return []

    results: list[PreparedSummaryChunk] = []
    i = 0

    while i < len(chapter_chunks):
        current_chunk = chapter_chunks[i]

        if current_chunk.token_count > config.llm_text_max_tokens:
            results.extend(
                split_oversized_chunk_with_overlap(
                    chunk=current_chunk,
                    max_tokens=config.llm_text_max_tokens,
                    overlap_tokens=config.overlap_tokens,
                    split_window_tokens=(
                        config.llm_text_min_tokens + config.overlap_tokens
                    ),
                )
            )
            i += 1
            continue

        next_chunk = chapter_chunks[i + 1] if i + 1 < len(chapter_chunks) else None
        if _should_merge_short_chunk_with_next(
            current_chunk=current_chunk,
            next_chunk=next_chunk,
            config=config,
        ):
            results.append(
                _merge_adjacent_chunks_once(
                    left_chunk=current_chunk,
                    right_chunk=next_chunk,  # type: ignore
                )
            )
            i += 2
            continue

        results.append(current_chunk)
        i += 1

    return results


def _merge_connected_chunks_text(chunks: list[ConnectedChunk]) -> str:
    """
    将一个 block 内的多个 connected chunk 合并成文本。
    """
    if not chunks:
        return ""

    parts = [chunk.text.strip() for chunk in chunks if chunk.text.strip()]
    return "\n\n".join(parts).strip()


def _should_merge_short_chunk_with_next(
    current_chunk: PreparedSummaryChunk,
    next_chunk: PreparedSummaryChunk | None,
    config: ChunkConfig,
) -> bool:
    """
    判断过短章节是否允许与下一完整章节合并一次。
    """
    if next_chunk is None:
        return False

    if current_chunk.token_count >= config.llm_text_min_tokens:
        return False

    if next_chunk.token_count > config.llm_text_max_tokens:
        return False

    merged_text = _merge_chunk_texts([current_chunk.text, next_chunk.text])
    merged_token_count = count_tokens(merged_text)

    return merged_token_count <= config.llm_text_max_tokens


def _merge_adjacent_chunks_once(
    left_chunk: PreparedSummaryChunk,
    right_chunk: PreparedSummaryChunk,
) -> PreparedSummaryChunk:
    """
    将两个相邻章节块合并一次，不做链式扩展。
    """
    merged_text = _merge_chunk_texts([left_chunk.text, right_chunk.text])

    return PreparedSummaryChunk(
        prepared_chunk_id=-1,
        text=merged_text,
        token_count=count_tokens(merged_text),
        extraction_records=_deduplicate_extraction_records(
            list(left_chunk.extraction_records) + list(right_chunk.extraction_records)
        ),
        source_block_indices=_deduplicate_block_indices(
            list(left_chunk.source_block_indices)
            + list(right_chunk.source_block_indices)
        ),
    )


def _merge_chunk_texts(parts: list[str]) -> str:
    """
    按章节顺序合并文本块。
    """
    valid_parts = [part.strip() for part in parts if part and part.strip()]
    return "\n\n".join(valid_parts).strip()


def split_oversized_chunk_with_overlap(
    chunk: PreparedSummaryChunk,
    max_tokens: int,
    overlap_tokens: int,
    split_window_tokens: int | None = None,
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

    if chunk.token_count <= max_tokens and split_window_tokens is None:
        return [chunk]

    words = text.split()
    if not words:
        return []

    results: list[PreparedSummaryChunk] = []

    # 用一个较保守的近似比例：1 word ≈ 1.3 tokens
    approx_tokens_per_word = 1.3
    target_window_tokens = split_window_tokens or max_tokens
    target_window_tokens = max(1, min(target_window_tokens, max_tokens))
    window_size = max(1, int(target_window_tokens / approx_tokens_per_word))
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
                    source_block_indices=list(chunk.source_block_indices),
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


def _deduplicate_block_indices(block_indices: list[int]) -> list[int]:
    if not block_indices:
        return []

    results: list[int] = []
    seen: set[int] = set()

    for block_idx in block_indices:
        if block_idx in seen:
            continue

        seen.add(block_idx)
        results.append(block_idx)

    return results
