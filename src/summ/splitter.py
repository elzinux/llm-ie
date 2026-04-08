from __future__ import annotations

import re

from src.summ.config import MODEL_RESOURCE_CONFIG, SplitConfig
from src.summ.schemas import SentenceInfo
from src.utils.model_resources import get_spacy_model, get_tokenizer


def split_text_into_sentences(
    text: str,
    spacy_model_name: str = SplitConfig.spacy_model_name,
    tokenizer_name: str = SplitConfig.tokenizer_name,
    min_sentence_chars: int = SplitConfig.min_sentence_chars,
) -> list[SentenceInfo]:
    """
    将文本按段落和句子切分，并返回句子信息列表。

    处理逻辑保持简单：
    1. 先按空行切分段落
    2. 再对每个段落做句子切分
    3. 过滤过短句子
    4. 为每个句子统计 token 数
    5. 生成 SentenceInfo

    参数:
        text: 输入文本
        spacy_model_name: spaCy 英文模型名称
        tokenizer_name: token 计数器名称
        min_sentence_chars: 句子最小字符长度，小于该值则跳过

    返回:
        list[SentenceInfo]
    """
    if not isinstance(text, str) or not text.strip():
        return []

    paragraphs = split_text_into_paragraphs(text)

    results: list[SentenceInfo] = []
    global_idx = 0

    for paragraph_idx, paragraph_text in enumerate(paragraphs):
        sentences = split_paragraph_into_sentences(
            paragraph_text=paragraph_text,
            spacy_model_name=spacy_model_name,
        )

        local_sentence_idx = 0
        tmp_text = ""

        for sentence_text in sentences:
            cleaned_text = sentence_text.strip()

            if len(cleaned_text) < min_sentence_chars:
                tmp_text = cleaned_text
                continue

            if tmp_text:
                cleaned_text = tmp_text + " " + cleaned_text
                tmp_text = ""

            token_count = count_tokens(cleaned_text, tokenizer_name)

            results.append(
                SentenceInfo(
                    global_idx=global_idx,
                    local_idx=(paragraph_idx, local_sentence_idx),
                    text=cleaned_text,
                    token_count=token_count,
                    semantic_score=0.0,
                    is_selected=False,
                )
            )

            global_idx += 1
            local_sentence_idx += 1

    return results


def split_text_into_paragraphs(text: str) -> list[str]:
    """
    按空行切分段落。

    返回去除首尾空白后的非空段落列表。
    """
    if not isinstance(text, str) or not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if p]


def split_paragraph_into_sentences(
    paragraph_text: str,
    spacy_model_name: str = SplitConfig.spacy_model_name,
) -> list[str]:
    """
    将单个段落切分为句子。

    优先使用 spaCy 分句。
    如果 spaCy 分句失败，则使用正则作为保底。

    参数:
        paragraph_text: 单个段落文本
        spacy_model_name: spaCy 模型名称

    返回:
        list[str]
    """
    if not isinstance(paragraph_text, str) or not paragraph_text.strip():
        return []

    text = paragraph_text.strip()

    try:
        nlp = get_spacy_model(spacy_model_name, resource_config=MODEL_RESOURCE_CONFIG)
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        if sentences:
            return sentences
    except Exception:
        pass

    return _split_sentences_by_regex(text)


def _split_sentences_by_regex(text: str) -> list[str]:
    """
    正则保底分句。
    """
    if not isinstance(text, str) or not text.strip():
        return []

    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def count_tokens(
    text: str,
    tokenizer_name: str = SplitConfig.tokenizer_name,
) -> int:
    """
    统计文本 token 数。
    """
    if not isinstance(text, str) or not text:
        return 0

    try:
        tokenizer = get_tokenizer(tokenizer_name, resource_config=MODEL_RESOURCE_CONFIG)
        return len(tokenizer.encode(text))
    except Exception:
        return 0

