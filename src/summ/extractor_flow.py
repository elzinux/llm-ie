from __future__ import annotations

from typing import Any

from src.summ.config import ChunkConfig, SplitConfig
from src.summ.extractor import extract_key_elements, normalize_extraction_record
from src.summ.schemas import ExtractionRecord
from src.summ.splitter import (
    count_tokens,
    split_paragraph_into_sentences,
    split_text_into_paragraphs,
)


def extract_key_elements_with_flow(
    text: str,
    chunk_config: ChunkConfig,
    split_config: SplitConfig,
    model_config: dict[str, Any] | None = None,
    max_retries: int = 2,
) -> ExtractionRecord:
    """
    对单个章节文本执行 extractor 阶段流程。

    规则：
    1. 若章节长度安全，则整章直接抽取
    2. 若章节过长，则仅在 extractor 阶段按段落切块
    3. 单段超长时，降级为句子级切分
    4. 所有子块抽取结果合并为章节级 ExtractionRecord
    """
    if not isinstance(text, str) or not text.strip():
        return ExtractionRecord()

    if is_text_within_extractor_limit(
        text=text,
        max_tokens=chunk_config.llm_text_max_tokens,
        tokenizer_name=split_config.tokenizer_name,
    ):
        return extract_key_elements(
            text=text,
            model_config=model_config,
            max_retries=max_retries,
        )

    chunk_texts = build_extractor_chunks(
        text=text,
        max_tokens=chunk_config.llm_text_max_tokens,
        spacy_model_name=split_config.spacy_model_name,
        tokenizer_name=split_config.tokenizer_name,
    )
    if not chunk_texts:
        return extract_key_elements(
            text=text,
            model_config=model_config,
            max_retries=max_retries,
        )

    records: list[ExtractionRecord] = []
    for chunk_text in chunk_texts:
        record = extract_key_elements(
            text=chunk_text,
            model_config=model_config,
            max_retries=max_retries,
        )
        records.append(normalize_extraction_record(record))

    return merge_extraction_records(records)


def is_text_within_extractor_limit(
    text: str,
    max_tokens: int,
    tokenizer_name: str,
) -> bool:
    """
    判断章节文本是否在 extractor 安全长度内。
    """
    if not isinstance(text, str) or not text.strip():
        return True

    return count_tokens(text, tokenizer_name) <= max_tokens


def build_extractor_chunks(
    text: str,
    max_tokens: int,
    spacy_model_name: str,
    tokenizer_name: str,
) -> list[str]:
    """
    在章节内部按段落累积切块。

    规则：
    - 优先按段落切块
    - 单段超长时，降级为句子级切分
    - 不设置 overlap
    """
    if not isinstance(text, str) or not text.strip():
        return []

    paragraphs = split_text_into_paragraphs(text)
    if not paragraphs:
        return [text.strip()]

    results: list[str] = []
    current_parts: list[str] = []

    for paragraph_text in paragraphs:
        paragraph_token_count = count_tokens(paragraph_text, tokenizer_name)

        if paragraph_token_count > max_tokens:
            if current_parts:
                results.append(_merge_paragraph_parts(current_parts))
                current_parts = []

            results.extend(
                split_oversized_paragraph_into_chunks(
                    paragraph_text=paragraph_text,
                    max_tokens=max_tokens,
                    spacy_model_name=spacy_model_name,
                    tokenizer_name=tokenizer_name,
                )
            )
            continue

        candidate_parts = current_parts + [paragraph_text]
        candidate_text = _merge_paragraph_parts(candidate_parts)
        candidate_token_count = count_tokens(candidate_text, tokenizer_name)

        if not current_parts or candidate_token_count <= max_tokens:
            current_parts = candidate_parts
            continue

        results.append(_merge_paragraph_parts(current_parts))
        current_parts = [paragraph_text]

    if current_parts:
        results.append(_merge_paragraph_parts(current_parts))

    return [chunk_text for chunk_text in results if chunk_text.strip()]


def split_oversized_paragraph_into_chunks(
    paragraph_text: str,
    max_tokens: int,
    spacy_model_name: str,
    tokenizer_name: str,
) -> list[str]:
    """
    对单个超长段落降级为句子级切分。
    """
    if not isinstance(paragraph_text, str) or not paragraph_text.strip():
        return []

    sentences = split_paragraph_into_sentences(
        paragraph_text=paragraph_text,
        spacy_model_name=spacy_model_name,
    )
    if not sentences:
        return [paragraph_text.strip()]

    results: list[str] = []
    current_sentences: list[str] = []

    for sentence_text in sentences:
        cleaned_sentence = sentence_text.strip()
        if not cleaned_sentence:
            continue

        sentence_token_count = count_tokens(cleaned_sentence, tokenizer_name)
        if sentence_token_count > max_tokens:
            if current_sentences:
                results.append(_merge_sentence_parts(current_sentences))
                current_sentences = []

            results.append(cleaned_sentence)
            continue

        candidate_sentences = current_sentences + [cleaned_sentence]
        candidate_text = _merge_sentence_parts(candidate_sentences)
        candidate_token_count = count_tokens(candidate_text, tokenizer_name)

        if not current_sentences or candidate_token_count <= max_tokens:
            current_sentences = candidate_sentences
            continue

        results.append(_merge_sentence_parts(current_sentences))
        current_sentences = [cleaned_sentence]

    if current_sentences:
        results.append(_merge_sentence_parts(current_sentences))

    return [chunk_text for chunk_text in results if chunk_text.strip()]


def merge_extraction_records(records: list[ExtractionRecord]) -> ExtractionRecord:
    """
    将多个子块抽取结果合并为章节级 ExtractionRecord。
    """
    if not records:
        return ExtractionRecord()

    keywords: list[str] = []
    key_sentences: list[str] = []

    for record in records:
        normalized_record = normalize_extraction_record(record)
        keywords.extend(normalized_record.keywords)
        key_sentences.extend(normalized_record.key_sentences)

    merged_record = ExtractionRecord(
        keywords=_deduplicate_normalized_items(keywords),
        key_sentences=_deduplicate_normalized_items(key_sentences),
    )
    return normalize_extraction_record(merged_record)


def _deduplicate_normalized_items(items: list[str]) -> list[str]:
    """
    对文本项做规范化去重。
    """
    results: list[str] = []
    seen: set[str] = set()

    for item in items:
        normalized_text = _normalize_merge_text(item)
        if not normalized_text:
            continue

        dedup_key = normalized_text.lower()
        if dedup_key in seen:
            continue

        seen.add(dedup_key)
        results.append(normalized_text)

    return results


def _normalize_merge_text(text: str) -> str:
    """
    用于合并去重的轻量规范化。
    """
    if not isinstance(text, str):
        return ""

    return " ".join(text.strip().split())


def _merge_paragraph_parts(parts: list[str]) -> str:
    """
    按段落组织文本块。
    """
    valid_parts = [_normalize_merge_text(part) for part in parts if part and part.strip()]
    return "\n\n".join(valid_parts).strip()


def _merge_sentence_parts(parts: list[str]) -> str:
    """
    按句子组织文本块。
    """
    valid_parts = [_normalize_merge_text(part) for part in parts if part and part.strip()]
    return " ".join(valid_parts).strip()
