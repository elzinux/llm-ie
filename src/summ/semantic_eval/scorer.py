from __future__ import annotations

from functools import lru_cache
from typing import List

import networkx as nx
import numpy as np
import spacy
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"

# 参考论文中 SemanticRank 的默认设置
_SIM_THRESHOLD = 0.1
_DAMPING = 0.85


@lru_cache(maxsize=4)
def _load_spacy_model(model_name: str):
    """
    加载 spaCy 模型。
    如果模型不带句边界能力，则补一个 sentencizer。
    """
    nlp = spacy.load(model_name)
    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
    return nlp


@lru_cache(maxsize=4)
def _load_embedding_model(model_name: str):
    """加载句向量模型。"""
    return SentenceTransformer(model_name)


def _split_sentences(text: str, nlp) -> List[str]:
    """
    使用 spaCy 分句。
    若没有有效句子，则退化为整段文本一个句子。
    """
    text = (text or "").strip()
    if not text:
        return []

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    if not sentences:
        return [text]

    return sentences


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化。"""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def _self_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """计算文本内部句向量的余弦相似度矩阵。"""
    emb = _l2_normalize(embeddings)
    return emb @ emb.T


def _semantic_rank_weights(
    embeddings: np.ndarray,
    sim_threshold: float = _SIM_THRESHOLD,
    damping: float = _DAMPING,
) -> np.ndarray:
    """
    只对 reference 文本内部做 SemanticRank / TextRank 权重计算。

    步骤：
    1. 计算句子自相似矩阵
    2. 当句对相似度 > sim_threshold 时连边
    3. 用 PageRank / TextRank 得到句子重要性
    4. 归一化为和为 1 的权重
    """
    n_sent = embeddings.shape[0]

    if n_sent == 0:
        return np.array([], dtype=np.float32)

    if n_sent == 1:
        return np.array([1.0], dtype=np.float32)

    sim = _self_similarity_matrix(embeddings)

    graph = nx.Graph()
    graph.add_nodes_from(range(n_sent))

    for i in range(n_sent):
        for j in range(i + 1, n_sent):
            weight = float(sim[i, j])
            if weight > sim_threshold:
                graph.add_edge(i, j, weight=weight)

    # 如果没有有效边，则退化为均匀权重
    if graph.number_of_edges() == 0:
        return np.full(n_sent, 1.0 / n_sent, dtype=np.float32)

    scores = nx.pagerank(graph, alpha=damping, weight="weight")
    weights = np.array([scores[i] for i in range(n_sent)], dtype=np.float32)

    total = float(weights.sum())
    if total <= 0:
        return np.full(n_sent, 1.0 / n_sent, dtype=np.float32)

    return weights / total


def _cross_cosine_similarity(
    reference_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
) -> np.ndarray:
    """
    计算 reference 与 candidate 之间的原始余弦相似度矩阵。
    shape = (n_ref, n_cand)
    """
    ref = _l2_normalize(reference_embeddings)
    cand = _l2_normalize(candidate_embeddings)
    return ref @ cand.T


def score_semantic_eval(
    candidate_text: str,
    reference_text: str,
    embedding_model: str = EMBED_MODEL,
    spacy_model: str = SPACY_MODEL,
) -> float:
    """
    评估 candidate 对 reference 的语义覆盖程度。

    流程：
    1. 分句
    2. 句向量
    3. 只对 reference 内部做 TextRank 权重
    4. 计算 reference × candidate 的原始 cosine 相似度矩阵
    5. 每个 reference 句取与 candidate 中最相似的句子
    6. 用 reference 权重做加权平均

    参数:
        candidate_text: 候选文本，例如摘要
        reference_text: 参考文本，例如关键句拼接后的文本
        embedding_model: 本地句向量模型
        spacy_model: spaCy 分句模型

    返回:
        float: 语义覆盖分数，通常在 [0, 1] 左右
    """
    if not (candidate_text or "").strip():
        return 0.0
    if not (reference_text or "").strip():
        return 0.0

    nlp = _load_spacy_model(spacy_model)
    embedder = _load_embedding_model(embedding_model)

    reference_sentences = _split_sentences(reference_text, nlp)
    candidate_sentences = _split_sentences(candidate_text, nlp)

    if not reference_sentences or not candidate_sentences:
        return 0.0

    reference_embeddings = embedder.encode(
        reference_sentences,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    candidate_embeddings = embedder.encode(
        candidate_sentences,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )

    # 只对 reference 计算句子重要性权重
    reference_weights = _semantic_rank_weights(reference_embeddings)

    # 原始跨文本余弦相似度矩阵，不做行归一化
    cross_sim = _cross_cosine_similarity(reference_embeddings, candidate_embeddings)

    # 每个 reference 句找一个最相似的 candidate 句
    best_sim_per_ref = np.max(cross_sim, axis=1)

    # 保险起见，截断到 [-1, 1]
    best_sim_per_ref = np.clip(best_sim_per_ref, -1.0, 1.0)

    # 你这个任务是“覆盖度”评估，通常不希望负值把总分拉成负数
    # 所以把负相似度压到 0
    best_sim_per_ref = np.maximum(best_sim_per_ref, 0.0)

    score = float(np.sum(reference_weights * best_sim_per_ref))

    if np.isnan(score) or np.isinf(score):
        return 0.0

    return max(0.0, min(1.0, score))
