from __future__ import annotations

from src.summ.config import ConnectionConfig
from src.summ.schemas import ConnectedChunk, SentenceInfo


def connect_selected_sentences(
    sentences: list[SentenceInfo],
    config: ConnectionConfig,
) -> list[ConnectedChunk]:
    """
    将已筛选的关键句按原文顺序连接为片段。

    连接规则：
    1. 先找出所有 is_selected=True 的句子
    2. 若两个相邻已选句子之间间隔不大，则补全中间句并连接为同一片段
    3. 否则断开，形成新的片段

    参数:
        sentences: 句子列表，要求已完成筛选并设置 is_selected
        config: 连接配置

    返回:
        list[ConnectedChunk]
    """
    if not sentences:
        return []

    sorted_sentences = sorted(sentences, key=lambda x: x.global_idx)
    selected_indices = [
        i for i, sentence in enumerate(sorted_sentences) if sentence.is_selected
    ]

    if not selected_indices:
        return []

    grouped_indices: list[list[int]] = []
    current_group: list[int] = [selected_indices[0]]

    for current_selected_idx in selected_indices[1:]:
        previous_selected_idx = current_group[-1]

        if should_merge_sentences(
            sentences=sorted_sentences,
            prev_selected_idx=previous_selected_idx,
            next_selected_idx=current_selected_idx,
            config=config,
        ):
            for idx in range(previous_selected_idx + 1, current_selected_idx + 1):
                if idx not in current_group:
                    current_group.append(idx)
        else:
            grouped_indices.append(current_group)
            current_group = [current_selected_idx]

    grouped_indices.append(current_group)

    chunks: list[ConnectedChunk] = []
    for chunk_id, sentence_indices_in_list in enumerate(grouped_indices):
        chunk = build_connected_chunk(
            chunk_id=chunk_id,
            sentence_indices_in_list=sentence_indices_in_list,
            sentences=sorted_sentences,
        )
        chunks.append(chunk)

    return chunks


def should_merge_sentences(
    sentences: list[SentenceInfo],
    prev_selected_idx: int,
    next_selected_idx: int,
    config: ConnectionConfig,
) -> bool:
    """
    判断两个已选句子是否应该连接为同一片段。

    判断依据：
    - 中间间隔句子数是否不超过 max_gap_sentences
    - 中间句子的 token 总数是否不超过 max_gap_tokens
    """
    if next_selected_idx <= prev_selected_idx:
        return False

    gap_sentence_count = next_selected_idx - prev_selected_idx - 1
    if gap_sentence_count > config.max_gap_sentences:
        return False

    middle_sentences = sentences[prev_selected_idx + 1 : next_selected_idx]
    middle_token_count = sum(sentence.token_count for sentence in middle_sentences)

    if middle_token_count > config.max_gap_tokens:
        return False

    return True


def build_connected_chunk(
    chunk_id: int,
    sentence_indices_in_list: list[int],
    sentences: list[SentenceInfo],
) -> ConnectedChunk:
    """
    根据句子索引构建一个 ConnectedChunk。
    """
    if not sentence_indices_in_list:
        return ConnectedChunk(
            chunk_id=chunk_id,
            sentence_indices=[],
            text="",
            token_count=0,
        )

    chunk_sentences = [sentences[i] for i in sentence_indices_in_list]
    sentence_indices = [sentence.global_idx for sentence in chunk_sentences]
    chunk_text = " ".join(sentence.text for sentence in chunk_sentences).strip()
    chunk_token_count = sum(sentence.token_count for sentence in chunk_sentences)

    return ConnectedChunk(
        chunk_id=chunk_id,
        sentence_indices=sentence_indices,
        text=chunk_text,
        token_count=chunk_token_count,
    )


def merge_connected_chunks_text(chunks: list[ConnectedChunk]) -> tuple[str, int]:
    """
    将多个 ConnectedChunk 合并为一个整体文本，并返回总 token 数。

    这个函数不是必须的，但通常在第 4 步之后会直接用到。
    """
    if not chunks:
        return "", 0

    merged_text = "\n\n".join(
        chunk.text for chunk in chunks if chunk.text.strip()
    ).strip()
    merged_token_count = sum(chunk.token_count for chunk in chunks)

    return merged_text, merged_token_count
