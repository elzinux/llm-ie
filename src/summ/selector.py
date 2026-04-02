from __future__ import annotations

from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

from src.summ.config import SelectionConfig
from src.summ.schemas import ExtractionRecord, SentenceInfo

_EMBEDDING_MODEL_CACHE: dict[str, SentenceTransformer] = {}


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
        return sentences

    clues = _build_clues(extraction_record)
    if not clues:
        return sentences

    # 第一步：直接匹配
    for sentence in sentences:
        match_score = _compute_direct_match_score(sentence.text, clues)
        if match_score > fuzzy_threshold:
            sentence.is_selected = True
            sentence.semantic_score = max(sentence.semantic_score, match_score)

    # 第二步：语义补充
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
    """
    clues: list[str] = []
    seen: set[str] = set()

    for item in extraction_record.keywords + extraction_record.key_sentences:
        text = str(item).strip()
        if not text:
            continue

        norm = " ".join(text.lower().split())
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

    sentence_lower = sentence_text.lower()

    best_score = 0.0

    for clue in clues:
        clue = clue.strip()
        if not clue:
            continue

        clue_lower = clue.lower()

        # 1. 子串直接命中，优先级最高
        if clue_lower in sentence_lower:
            return 100

        # 2. 模糊匹配
        fuzzy_score = fuzz.partial_ratio(clue_lower, sentence_lower)
        best_score = max(best_score, fuzzy_score)

    return best_score


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

    model = _get_embedding_model(embedding_model_name)

    sentence_texts = [s.text for s in sentences]
    clue_texts = clues

    sentence_embeddings = model.encode(
        sentence_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    clue_embeddings = model.encode(
        clue_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    similarity_matrix = util.cos_sim(sentence_embeddings, clue_embeddings)

    for i, sentence in enumerate(sentences):
        best_similarity = float(similarity_matrix[i].max().item())
        sentence.semantic_score = max(sentence.semantic_score, best_similarity)

        if best_similarity >= semantic_threshold:
            sentence.is_selected = True


def _get_embedding_model(model_name: str) -> SentenceTransformer:
    """
    获取并缓存语义相似度模型。
    """
    if model_name not in _EMBEDDING_MODEL_CACHE:
        _EMBEDDING_MODEL_CACHE[model_name] = SentenceTransformer(model_name)

    return _EMBEDDING_MODEL_CACHE[model_name]
