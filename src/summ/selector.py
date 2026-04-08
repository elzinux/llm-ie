from __future__ import annotations

import re

from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

from src.summ.config import MODEL_RESOURCE_CONFIG, SelectionConfig
from src.summ.schemas import ExtractionRecord, SentenceInfo
from src.utils.model_resources import get_embedding_model
from src.utils.normalize_text import normalize_text


def select_key_sentences(
    sentences: list[SentenceInfo],
    extraction_record: ExtractionRecord,
    embedding_model_name: str = SelectionConfig.embedding_model_name,
    fuzzy_threshold: int = SelectionConfig.fuzzy_threshold,
    semantic_threshold: float = SelectionConfig.semantic_threshold,
) -> list[SentenceInfo]:
    """
    根据粗提取结果筛选关键句。

    筛选逻辑保持简单：
    1. 先做关键词/关键句的直接匹配
    2. 再对未命中的句子做一次语义相似度补充
    3. 直接在 SentenceInfo 上更新 is_selected 和 semantic_score
    4. 句向量写回 SentenceInfo，供后续阶段复用

    参数:
        sentences: 句子列表
        extraction_record: 第1步得到的粗提取结果
        embedding_model_name: 语义相似度模型名称
        fuzzy_threshold: 模糊匹配阈值
        semantic_threshold: 语义相似度阈值

    返回:
        list[SentenceInfo]
    """
    if not sentences:
        return []

    if not isinstance(extraction_record, ExtractionRecord):
        return []

    clues = _build_clues(extraction_record)
    if not clues:
        return []

    # 第一步：直接匹配
    for sentence in sentences:
        match_score = _compute_direct_match_score(sentence.text, clues)
        if match_score > fuzzy_threshold:
            sentence.is_selected = True
            sentence.semantic_score = max(sentence.semantic_score, match_score)

    # 第二步：语义补充，并缓存句向量
    _apply_semantic_selection(
        sentences=sentences,
        clues=clues,
        embedding_model_name=embedding_model_name,
        semantic_threshold=semantic_threshold,
    )

    return sentences


def _build_clues(extraction_record: ExtractionRecord) -> list[str]:
    """
    将 keywords 和 key_sentences 合并为统一线索列表。
    去重时使用归一化文本，保留原始文本用于后续匹配和调试。
    """
    clues: list[str] = []
    seen: set[str] = set()

    for item in extraction_record.keywords + extraction_record.key_sentences:
        text = str(item).strip()
        if not text:
            continue

        norm = normalize_text(text)
        if not norm:
            continue

        if norm in seen:
            continue

        seen.add(norm)
        clues.append(text)

    return clues


def _compute_direct_match_score(
    sentence_text: str,
    clues: list[str],
) -> float:
    """
    计算句子与线索的直接匹配分数。

    返回值：
    - 0.0 表示未命中
    - 命中时返回 0~100 的分数
    """
    if not sentence_text.strip() or not clues:
        return 0.0

    sentence_raw = sentence_text.lower()
    sentence_norm = normalize_text(sentence_text)

    best_score = 0.0

    for clue in clues:
        clue = clue.strip()
        if not clue:
            continue

        clue_raw = clue.lower()
        clue_norm = normalize_text(clue)

        # 1. 原始小写文本直接子串命中
        if clue_raw in sentence_raw:
            return 100.0

        # 2. 归一化后子串命中
        if clue_norm and clue_norm in sentence_norm:
            return 100.0

        # 3. 模糊匹配：归一化文本上做
        fuzzy_score = fuzz.partial_ratio(clue_norm, sentence_norm)
        best_score = max(best_score, float(fuzzy_score))

    return best_score


def _normalize_for_semantic(text: str) -> str:
    """
    用于语义编码的轻量规范化：
    - 去首尾空白
    - 压缩连续空白
    - 保留正常标点和词边界
    """
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _apply_semantic_selection(
    sentences: list[SentenceInfo],
    clues: list[str],
    embedding_model_name: str,
    semantic_threshold: float,
) -> None:
    """
    对未直接命中的句子做语义相似度补充筛选。
    """
    if not sentences or not clues:
        return

    try:
        model = get_embedding_model(embedding_model_name, resource_config=MODEL_RESOURCE_CONFIG)
    except RuntimeError:
        return

    sentence_embeddings = _prepare_sentence_embeddings(
        sentences=sentences,
        model=model,
    )

    target_indices = [i for i, s in enumerate(sentences) if not s.is_selected]
    if not target_indices:
        return

    clue_texts = []
    for clue in clues:
        normalized_clue = _normalize_for_semantic(clue)
        if normalized_clue:
            clue_texts.append(normalized_clue)

    if not clue_texts:
        return

    clue_embeddings = model.encode(
        clue_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    similarity_matrix = util.cos_sim(
        sentence_embeddings[target_indices], clue_embeddings
    )

    for row_idx, sent_idx in enumerate(target_indices):
        best_similarity = float(similarity_matrix[row_idx].max().item())
        sentence = sentences[sent_idx]

        sentence.semantic_score = max(sentence.semantic_score, best_similarity)
        if best_similarity >= semantic_threshold:
            sentence.is_selected = True


def _prepare_sentence_embeddings(
    sentences: list[SentenceInfo],
    model: SentenceTransformer,
):
    """
    为全部句子统一编码，并写回 SentenceInfo 供后续阶段复用。
    """
    sentence_texts = [_normalize_for_semantic(sentence.text) for sentence in sentences]
    sentence_embeddings = model.encode(
        sentence_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    for idx, sentence in enumerate(sentences):
        sentence.sentence_embedding = sentence_embeddings[idx]

    return sentence_embeddings

