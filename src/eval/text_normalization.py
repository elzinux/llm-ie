from __future__ import annotations

import re
from typing import List

import spacy
from rapidfuzz import fuzz
from sentence_transformers import util


def load_spacy_model(model_name: str = "en_core_web_sm"):
    """
    加载并返回 spaCy 英文模型，用于分句、分词和短语处理。
    """
    return spacy.load(model_name)


def normalize_text(text: str) -> str:
    """
    对文本进行基础规范化，如小写化、去除多余空白、统一标点形式，便于后续字符串匹配。
    """
    if not text:
        return ""

    text = text.lower()

    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_slot_value(value: str) -> str:
    """
    对槽位字段进行规范化处理，尽量减少大小写、空格、标点差异带来的匹配误差。
    """
    return normalize_text(value)


def split_sentences(text: str, nlp) -> List[str]:
    """
    使用 spaCy 将原文切分为句子，作为槽位验证的基本匹配单位。
    """
    if not text:
        return []

    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def tokenize_text(text: str, nlp) -> List[str]:
    """
    使用 spaCy 对文本进行分词，为精确匹配和模糊匹配提供基础。
    """
    if not text:
        return []

    doc = nlp(text)
    return [token.text for token in doc if not token.is_space]


def extract_candidate_phrases(text: str, nlp) -> List[str]:
    """
    从文本中提取可用于匹配的短语片段，辅助处理较长槽位或边界不整齐的表达。
    """
    if not text:
        return []

    doc = nlp(text)

    phrases = set()

    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        if chunk_text:
            phrases.add(chunk_text)

    for ent in doc.ents:
        ent_text = ent.text.strip()
        if ent_text:
            phrases.add(ent_text)

    return list(phrases)


def compute_fuzzy_score(a: str, b: str) -> float:
    """
    使用 RapidFuzz 计算两个字符串的模糊匹配分数。
    返回 0-100 的分值。
    """
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)

    if not a_norm or not b_norm:
        return 0.0

    return float(fuzz.partial_ratio(a_norm, b_norm))


def compute_semantic_similarity(a: str, b: str, model) -> float:
    """
    使用 SentenceTransformers 计算两个文本片段的语义相似度。
    返回 0-1 的相似度分值。
    """
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)

    if not a_norm or not b_norm:
        return 0.0

    emb_a = model.encode(a_norm, convert_to_tensor=True)
    emb_b = model.encode(b_norm, convert_to_tensor=True)

    score = util.cos_sim(emb_a, emb_b).item()
    return float(score)
