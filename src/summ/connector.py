from __future__ import annotations

from sentence_transformers import SentenceTransformer, util

from src.summ.config import ConnectionConfig
from src.summ.schemas import ConnectedChunk, SentenceInfo

_EMBEDDING_MODEL_CACHE: dict[str, SentenceTransformer] = {}
_DEFAULT_LOCAL_EXPAND_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def connect_selected_sentences(
    sentences: list[SentenceInfo],
    config: ConnectionConfig,
) -> list[ConnectedChunk]:
    """
    将已筛选的关键句按原文顺序连接为片段。

    连接规则：
    1. 先保留所有 is_selected=True 的句子
    2. 独立执行段内前后一句补全
    3. 独立执行段内相邻锚点间隔补全
    4. 合并原始句子与两类补全结果
    5. 按原文顺序、并以段落为基本单位构建 chunk

    参数:
        sentences: 句子列表，要求已完成筛选并设置 is_selected
        config: 连接配置

    返回:
        list[ConnectedChunk]
    """
    if not sentences:
        return []

    sorted_sentences = sorted(sentences, key=lambda x: x.global_idx)
    local_expanded_global_indices, gap_expanded_global_indices = (
        collect_expanded_sentence_global_indices(
            sentences=sorted_sentences,
            config=config,
        )
    )

    final_sentence_indices_in_list = _collect_final_sentence_indices_in_list(
        sentences=sorted_sentences,
        local_expanded_global_indices=local_expanded_global_indices,
        gap_expanded_global_indices=gap_expanded_global_indices,
    )
    if not final_sentence_indices_in_list:
        return []

    grouped_indices = _group_sentence_indices_by_paragraph(
        sentences=sorted_sentences,
        sentence_indices_in_list=final_sentence_indices_in_list,
    )

    chunks: list[ConnectedChunk] = []
    for chunk_id, sentence_indices_in_list in enumerate(grouped_indices):
        chunk = build_connected_chunk(
            chunk_id=chunk_id,
            sentence_indices_in_list=sentence_indices_in_list,
            sentences=sorted_sentences,
        )
        chunks.append(chunk)

    return chunks


def collect_expanded_sentence_global_indices(
    sentences: list[SentenceInfo],
    config: ConnectionConfig,
) -> tuple[set[int], set[int]]:
    """
    统一收集两类补全结果。

    返回：
    - 段内前后一句补全得到的全局索引集合
    - 段内相邻锚点间隔补全得到的全局索引集合
    """
    local_expanded_global_indices = collect_local_expanded_sentence_global_indices(
        sentences=sentences,
        similarity_threshold=config.local_expand_similarity_threshold,
    )
    gap_expanded_global_indices = collect_gap_expanded_sentence_global_indices(
        sentences=sentences,
        config=config,
    )

    return local_expanded_global_indices, gap_expanded_global_indices


def collect_local_expanded_sentence_global_indices(
    sentences: list[SentenceInfo],
    similarity_threshold: float,
) -> set[int]:
    """
    基于原始 is_selected=True 的锚点句，做段内前后一句补全。

    只返回“补全得到”的句子全局索引，不包含原始锚点句。
    """
    if not sentences:
        return set()

    paragraph_groups = _build_paragraph_groups(sentences)
    results: set[int] = set()

    for paragraph_indices in paragraph_groups.values():
        paragraph_anchor_indices = _collect_anchor_indices_in_paragraph(
            sentences=sentences,
            paragraph_indices=paragraph_indices,
        )
        if not paragraph_anchor_indices:
            continue

        results.update(
            _expand_local_neighbors_in_paragraph(
                sentences=sentences,
                paragraph_indices=paragraph_indices,
                anchor_indices=paragraph_anchor_indices,
                similarity_threshold=similarity_threshold,
            )
        )

    return results


def collect_gap_expanded_sentence_global_indices(
    sentences: list[SentenceInfo],
    config: ConnectionConfig,
) -> set[int]:
    """
    基于原始 is_selected=True 的锚点句，做段内相邻锚点间隔补全。

    只返回“补全得到”的句子全局索引，不包含原始锚点句。
    """
    if not sentences:
        return set()

    paragraph_groups = _build_paragraph_groups(sentences)
    results: set[int] = set()

    for paragraph_indices in paragraph_groups.values():
        paragraph_anchor_indices = _collect_anchor_indices_in_paragraph(
            sentences=sentences,
            paragraph_indices=paragraph_indices,
        )
        if len(paragraph_anchor_indices) < 2:
            continue

        for prev_anchor_idx, next_anchor_idx in zip(
            paragraph_anchor_indices,
            paragraph_anchor_indices[1:],
        ):
            if not should_merge_sentences(
                sentences=sentences,
                prev_selected_idx=prev_anchor_idx,
                next_selected_idx=next_anchor_idx,
                config=config,
            ):
                continue

            for idx in range(prev_anchor_idx + 1, next_anchor_idx):
                results.add(sentences[idx].global_idx)

    return results


def _collect_final_sentence_indices_in_list(
    sentences: list[SentenceInfo],
    local_expanded_global_indices: set[int],
    gap_expanded_global_indices: set[int],
) -> list[int]:
    """
    合并原始已选句与两类补全结果，得到最终应纳入 chunk 的句子位置。
    """
    if not sentences:
        return []

    expanded_global_indices = (
        local_expanded_global_indices | gap_expanded_global_indices
    )

    results: list[int] = []
    for idx, sentence in enumerate(sentences):
        if sentence.is_selected or sentence.global_idx in expanded_global_indices:
            results.append(idx)

    return results


def _group_sentence_indices_by_paragraph(
    sentences: list[SentenceInfo],
    sentence_indices_in_list: list[int],
) -> list[list[int]]:
    """
    按原文顺序分组，并以段落为基本单位构建 chunk。
    """
    if not sentence_indices_in_list:
        return []

    grouped_indices: list[list[int]] = []
    current_group: list[int] = [sentence_indices_in_list[0]]

    for current_idx in sentence_indices_in_list[1:]:
        previous_idx = current_group[-1]
        previous_paragraph_idx = sentences[previous_idx].local_idx[0]
        current_paragraph_idx = sentences[current_idx].local_idx[0]

        if current_paragraph_idx != previous_paragraph_idx:
            grouped_indices.append(current_group)
            current_group = [current_idx]
            continue

        if current_idx == previous_idx + 1:
            current_group.append(current_idx)
        else:
            grouped_indices.append(current_group)
            current_group = [current_idx]

    grouped_indices.append(current_group)
    return grouped_indices


def _build_paragraph_groups(sentences: list[SentenceInfo]) -> dict[int, list[int]]:
    """
    按段落收集句子在当前列表中的位置。
    """
    paragraph_groups: dict[int, list[int]] = {}

    for idx, sentence in enumerate(sentences):
        paragraph_idx = sentence.local_idx[0]
        paragraph_groups.setdefault(paragraph_idx, []).append(idx)

    return paragraph_groups


def _collect_anchor_indices_in_paragraph(
    sentences: list[SentenceInfo],
    paragraph_indices: list[int],
) -> list[int]:
    """
    收集单个段落内原始 is_selected=True 的锚点句位置。
    """
    return [idx for idx in paragraph_indices if sentences[idx].is_selected]


def _expand_local_neighbors_in_paragraph(
    sentences: list[SentenceInfo],
    paragraph_indices: list[int],
    anchor_indices: list[int],
    similarity_threshold: float,
) -> set[int]:
    """
    在单个段落内，对锚点句做前后相邻句补全。
    """
    if not paragraph_indices or not anchor_indices:
        return set()

    paragraph_embeddings = _get_paragraph_sentence_embeddings(
        sentences=sentences,
        paragraph_indices=paragraph_indices,
    )
    paragraph_position_map = {
        sentence_idx: pos for pos, sentence_idx in enumerate(paragraph_indices)
    }

    results: set[int] = set()

    for anchor_idx in anchor_indices:
        anchor_pos = paragraph_position_map[anchor_idx]
        anchor_embedding = paragraph_embeddings[anchor_pos]

        if anchor_pos > 0:
            previous_idx = paragraph_indices[anchor_pos - 1]
            previous_embedding = paragraph_embeddings[anchor_pos - 1]
            if (
                _compute_cosine_similarity(previous_embedding, anchor_embedding)
                >= similarity_threshold
            ):
                results.add(sentences[previous_idx].global_idx)

        if anchor_pos < len(paragraph_indices) - 1:
            next_idx = paragraph_indices[anchor_pos + 1]
            next_embedding = paragraph_embeddings[anchor_pos + 1]
            if (
                _compute_cosine_similarity(next_embedding, anchor_embedding)
                >= similarity_threshold
            ):
                results.add(sentences[next_idx].global_idx)

    return results


def should_merge_sentences(
    sentences: list[SentenceInfo],
    prev_selected_idx: int,
    next_selected_idx: int,
    config: ConnectionConfig,
) -> bool:
    """
    判断两个已选句子是否应该连接为同一片段。

    判断依据：
    - 两句是否位于同一段落
    - 中间间隔句子数是否不超过 max_gap_sentences
    - 中间句子的 token 总数是否不超过 max_gap_tokens
    """
    if next_selected_idx <= prev_selected_idx:
        return False

    prev_paragraph_idx = sentences[prev_selected_idx].local_idx[0]
    next_paragraph_idx = sentences[next_selected_idx].local_idx[0]
    if prev_paragraph_idx != next_paragraph_idx:
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

    # 保证传入的 sentence_indices_in_list 一句是按照 global_idx 进行排序
    chunk_sentences = [sentences[i] for i in sentence_indices_in_list]
    sentence_indices = [sentence.global_idx for sentence in chunk_sentences]
    paragraph_parts = _build_chunk_paragraph_parts(chunk_sentences)
    chunk_text = "\n\n".join(paragraph_parts).strip()
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


def _build_chunk_paragraph_parts(chunk_sentences: list[SentenceInfo]) -> list[str]:
    """
    按全局顺序组织段落片段，保持段落作为基本拼接单位。
    """
    if not chunk_sentences:
        return []

    paragraph_texts: list[str] = []
    current_paragraph_idx: int | None = None
    current_sentences: list[str] = []

    for sentence in chunk_sentences:
        paragraph_idx = sentence.local_idx[0]

        if current_paragraph_idx is None:
            current_paragraph_idx = paragraph_idx

        if paragraph_idx != current_paragraph_idx:
            paragraph_text = " ".join(current_sentences).strip()
            if paragraph_text:
                paragraph_texts.append(paragraph_text)
            current_paragraph_idx = paragraph_idx
            current_sentences = []

        current_sentences.append(sentence.text)

    paragraph_text = " ".join(current_sentences).strip()
    if paragraph_text:
        paragraph_texts.append(paragraph_text)

    return paragraph_texts


def _get_paragraph_sentence_embeddings(
    sentences: list[SentenceInfo],
    paragraph_indices: list[int],
):
    """
    优先复用 selector 阶段的句向量；若缺失，则统一补编码。
    """
    if not paragraph_indices:
        return []

    existing_embeddings = [
        sentences[idx].sentence_embedding for idx in paragraph_indices
    ]
    if all(embedding is not None for embedding in existing_embeddings):
        return existing_embeddings

    paragraph_texts = [sentences[idx].text for idx in paragraph_indices]
    encoded_embeddings = _encode_sentence_embeddings(paragraph_texts)

    for pos, sentence_idx in enumerate(paragraph_indices):
        sentences[sentence_idx].sentence_embedding = encoded_embeddings[pos]

    return [sentences[idx].sentence_embedding for idx in paragraph_indices]


def _encode_sentence_embeddings(texts: list[str]):
    """
    对句子列表编码为归一化向量。
    """
    if not texts:
        return []

    model = _get_embedding_model(_DEFAULT_LOCAL_EXPAND_EMBEDDING_MODEL_NAME)
    return model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )


def _compute_cosine_similarity(left_embedding, right_embedding) -> float:
    """
    计算两个句向量的 cosine 相似度。
    """
    return float(util.cos_sim(left_embedding, right_embedding).item())


def _get_embedding_model(model_name: str) -> SentenceTransformer:
    """
    获取并缓存语义模型。
    """
    if model_name not in _EMBEDDING_MODEL_CACHE:
        _EMBEDDING_MODEL_CACHE[model_name] = SentenceTransformer(model_name)

    return _EMBEDDING_MODEL_CACHE[model_name]
