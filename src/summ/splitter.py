from __future__ import annotations

from typing import Any

import spacy
import tiktoken

from src.summ.config import SplitConfig
from src.summ.schemas import SentenceInfo

_TOKENIZER_CACHE: dict[str, Any] = {}
_NLP_CACHE: dict[str, object] = {}


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

    nlp = _get_spacy_model(spacy_model_name)
    paragraphs = split_text_into_paragraphs(text)

    results: list[SentenceInfo] = []
    global_idx = 0

    for paragraph_idx, paragraph_text in enumerate(paragraphs):
        sentences = split_paragraph_into_sentences(paragraph_text, nlp)

        local_sentence_idx = 0
        tmp_text = ""
        for sentence_text in sentences:
            cleaned_text = sentence_text.strip()
            if len(cleaned_text) < min_sentence_chars:
                tmp_text = cleaned_text
                continue
            if tmp_text != "":
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


def split_paragraph_into_sentences(paragraph_text: str, nlp) -> list[str]:
    """
    将单个段落切分为句子。

    参数:
        paragraph_text: 单个段落文本
        nlp: 已加载的 spaCy 模型

    返回:
        list[str]
    """
    if not isinstance(paragraph_text, str) or not paragraph_text.strip():
        return []

    doc = nlp(paragraph_text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def count_tokens(text: str, tokenizer_name: str = "cl100k_base") -> int:
    """
    统计文本 token 数。
    """
    if not isinstance(text, str) or not text:
        return 0

    tokenizer = _get_tokenizer(tokenizer_name)

    try:
        return len(tokenizer.encode(text))
    except Exception:
        return 0


def _get_tokenizer(tokenizer_name: str):
    """
    获取并缓存 tokenizer。
    优先按 encoding 名称获取；失败时回退到常用模型编码。
    """
    if tokenizer_name not in _TOKENIZER_CACHE:
        try:
            _TOKENIZER_CACHE[tokenizer_name] = tiktoken.get_encoding(tokenizer_name)
        except Exception:
            _TOKENIZER_CACHE[tokenizer_name] = tiktoken.encoding_for_model(
                "gpt-4o-mini"
            )

    return _TOKENIZER_CACHE[tokenizer_name]


def _get_spacy_model(model_name: str):
    """
    获取并缓存 spaCy 模型。
    """
    if model_name not in _NLP_CACHE:
        _NLP_CACHE[model_name] = spacy.load(model_name)
    return _NLP_CACHE[model_name]


if __name__ == "__main__":
    TEST_TEXT = ""
    res = split_text_into_sentences(TEST_TEXT)
    for x in res:
        print(x.global_idx)
        print(x.text)
