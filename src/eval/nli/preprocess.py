from __future__ import annotations

import re

from sentence_transformers import util

from src.eval.nli.schemas import (
    HypothesisSentenceMatch,
    RetrievedPremiseSentence,
    TextSentence,
)
from src.utils.model_resources import get_embedding_model, get_spacy_model

_WHITESPACE_PATTERN = re.compile(r"\s+")
_WORD_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """
    对单段文本做轻量规范化。

    处理原则：
    - 转小写
    - 统一空白符
    - 去除首尾空白
    - 保留数字、百分号、比较符号、否定词等关键信息
    - 不做会改变语义的激进清洗
    """
    if not text:
        return ""

    normalized_text = text.lower()
    normalized_text = _WHITESPACE_PATTERN.sub(" ", normalized_text).strip()
    return normalized_text


def normalize_pair_texts(
    premise_text: str,
    hypothesis_text: str,
) -> tuple[str, str]:
    """
    同时规范化 premise_text 和 hypothesis_text。
    """
    return normalize_text(premise_text), normalize_text(hypothesis_text)


def split_text_into_sentences(text: str) -> list[str]:
    """
    将文本切分为句子字符串列表。

    实现说明：
    - 优先使用 spaCy 分句
    - 自动确保 pipeline 具备 sentence boundary 能力
    - 过滤空句
    """
    normalized_text = normalize_text(text)
    if not normalized_text:
        return []

    nlp = get_spacy_model(ensure_sentence_boundaries=True)
    doc = nlp(normalized_text)

    sentences: list[str] = []
    for sentence in doc.sents:
        sentence_text = sentence.text.strip()
        if sentence_text:
            sentences.append(sentence_text)

    return sentences


def build_text_sentences(text: str) -> list[TextSentence]:
    """
    将文本切句并构造成 TextSentence 列表。

    每个句子带有其在当前文本中的顺序索引，便于后续：
    - 排序
    - 回溯
    - 匹配结果分析
    """
    sentence_texts = split_text_into_sentences(text)

    results: list[TextSentence] = []
    for index, sentence_text in enumerate(sentence_texts):
        results.append(TextSentence(text=sentence_text, index=index))

    return results


def split_premise_and_hypothesis(
    premise_text: str,
    hypothesis_text: str,
) -> tuple[list[TextSentence], list[TextSentence]]:
    """
    同时切分 premise_text 和 hypothesis_text。
    """
    premise_sentences = build_text_sentences(premise_text)
    hypothesis_sentences = build_text_sentences(hypothesis_text)
    return premise_sentences, hypothesis_sentences


def count_words(text: str) -> int:
    """
    统计文本中的词数。

    说明：
    - 使用轻量正则统计词级单元
    - 用于长度控制，不追求严格语言学意义上的 token 数
    """
    normalized_text = normalize_text(text)
    if not normalized_text:
        return 0

    return len(_WORD_PATTERN.findall(normalized_text))


def is_text_within_ideal_length(
    text: str,
    min_words: int,
    max_words: int,
) -> bool:
    """
    判断文本是否位于理想词数区间。
    """
    word_count = count_words(text)
    return min_words <= word_count <= max_words


def compute_sentence_similarity_scores(
    premise_sentences: list[TextSentence],
    hypothesis_sentence: TextSentence,
) -> list[tuple[TextSentence, float]]:
    """
    计算单个 hypothesis 句与全部 premise 句的相似度分数。

    返回：
    - [(premise_sentence, similarity), ...]

    实现说明：
    - 使用 sentence-transformers 句向量
    - 使用 cosine similarity
    - 输出按相似度从高到低排序
    """
    if not premise_sentences:
        return []

    hypothesis_text = hypothesis_sentence.text.strip()
    if not hypothesis_text:
        return []

    premise_texts = [sentence.text for sentence in premise_sentences]
    embedding_model = get_embedding_model()

    premise_embeddings = embedding_model.encode(
        premise_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    hypothesis_embedding = embedding_model.encode(
        [hypothesis_text],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    similarity_scores = util.cos_sim(hypothesis_embedding, premise_embeddings)[0]

    results: list[tuple[TextSentence, float]] = []
    for sentence, similarity in zip(premise_sentences, similarity_scores):
        results.append((sentence, float(similarity.item())))

    results.sort(key=lambda item: item[1], reverse=True)
    return results


def select_topk_premise_sentences(
    premise_sentences: list[TextSentence],
    hypothesis_sentence: TextSentence,
    top_k: int = 5,
) -> list[RetrievedPremiseSentence]:
    """
    为单个 hypothesis 句选择 top-k premise 候选句。
    """
    if top_k <= 0:
        return []

    scored_sentences = compute_sentence_similarity_scores(
        premise_sentences=premise_sentences,
        hypothesis_sentence=hypothesis_sentence,
    )

    results: list[RetrievedPremiseSentence] = []
    for sentence, similarity in scored_sentences[:top_k]:
        results.append(
            RetrievedPremiseSentence(
                sentence=sentence,
                similarity=similarity,
            )
        )

    return results


def match_hypothesis_to_premise_sentences(
    premise_sentences: list[TextSentence],
    hypothesis_sentences: list[TextSentence],
    top_k: int = 5,
) -> list[HypothesisSentenceMatch]:
    """
    为全部 hypothesis 句匹配 top-k premise 候选句。

    实现说明：
    - 先统一编码 premise 句和 hypothesis 句，避免重复编码 premise
    - 对每个 hypothesis 句取相似度最高的 top-k premise 句
    - 返回 HypothesisSentenceMatch 列表
    """
    if not premise_sentences or not hypothesis_sentences or top_k <= 0:
        return []

    premise_texts = [sentence.text for sentence in premise_sentences]
    hypothesis_texts = [sentence.text for sentence in hypothesis_sentences]

    embedding_model = get_embedding_model()

    premise_embeddings = embedding_model.encode(
        premise_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    hypothesis_embeddings = embedding_model.encode(
        hypothesis_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    similarity_matrix = util.cos_sim(hypothesis_embeddings, premise_embeddings)

    results: list[HypothesisSentenceMatch] = []

    for hypothesis_index, hypothesis_sentence in enumerate(hypothesis_sentences):
        similarity_row = similarity_matrix[hypothesis_index]

        scored_sentences: list[tuple[TextSentence, float]] = []
        for premise_index, premise_sentence in enumerate(premise_sentences):
            similarity = float(similarity_row[premise_index].item())
            scored_sentences.append((premise_sentence, similarity))

        scored_sentences.sort(key=lambda item: item[1], reverse=True)

        retrieved_sentences = [
            RetrievedPremiseSentence(sentence=sentence, similarity=similarity)
            for sentence, similarity in scored_sentences[:top_k]
        ]

        results.append(
            HypothesisSentenceMatch(
                hypothesis_sentence=hypothesis_sentence,
                candidate_premise_sentences=retrieved_sentences,
            )
        )

    return results
