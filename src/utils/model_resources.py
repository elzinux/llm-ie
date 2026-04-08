from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import spacy
import tiktoken
from sentence_transformers import SentenceTransformer
from spacy.language import Language

DEFAULT_LOCAL_SPACY_MODEL_NAME = (
    r"D:\Code\LLM_IE\.venv\Lib\site-packages\en_core_web_sm\en_core_web_sm-3.8.0"
)
DEFAULT_SPACY_FALLBACK_NAMES = ("en_core_web_sm",)

DEFAULT_TOKENIZER_NAME = "cl100k_base"
DEFAULT_TOKENIZER_FALLBACK_MODEL_NAME = "gpt-4o-mini"

DEFAULT_LOCAL_EMBEDDING_MODEL_NAME = (
    r"C:\Users\zhulin\.cache\huggingface\hub\models--sentence-transformers--all-"
    r"MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
)
DEFAULT_EMBEDDING_FALLBACK_NAMES = ("sentence-transformers/all-MiniLM-L6-v2",)


@dataclass(frozen=True)
class ModelResourceConfig:
    """
    统一维护本地模型资源的默认路径与回退名称。

    加载顺序始终遵循：
    1. 本地路径优先
    2. 本地失败后再尝试 fallback 名称
    """

    spacy_model_name: str = DEFAULT_LOCAL_SPACY_MODEL_NAME
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME
    embedding_model_name: str = DEFAULT_LOCAL_EMBEDDING_MODEL_NAME
    spacy_fallback_names: tuple[str, ...] = DEFAULT_SPACY_FALLBACK_NAMES
    embedding_fallback_names: tuple[str, ...] = DEFAULT_EMBEDDING_FALLBACK_NAMES


DEFAULT_MODEL_RESOURCE_CONFIG = ModelResourceConfig()

_EMBEDDING_MODEL_CACHE: dict[tuple[str, tuple[str, ...]], SentenceTransformer] = {}
_SPACY_MODEL_CACHE: dict[tuple[str, bool, tuple[str, ...]], Language] = {}
_TOKENIZER_CACHE: dict[str, Any] = {}


def get_spacy_model_load_candidates(
    spacy_model_name: str | None = None,
    resource_config: ModelResourceConfig | None = None,
) -> tuple[str, ...]:
    """
    获取 spaCy 模型加载候选列表。
    """
    active_resource_config = resource_config or DEFAULT_MODEL_RESOURCE_CONFIG
    preferred_name = spacy_model_name or active_resource_config.spacy_model_name
    return _deduplicate_model_candidates(
        preferred_name,
        *active_resource_config.spacy_fallback_names,
    )


def get_embedding_model_load_candidates(
    embedding_model_name: str | None = None,
    resource_config: ModelResourceConfig | None = None,
) -> tuple[str, ...]:
    """
    获取句向量模型加载候选列表。
    """
    active_resource_config = resource_config or DEFAULT_MODEL_RESOURCE_CONFIG
    preferred_name = (
        embedding_model_name or active_resource_config.embedding_model_name
    )
    return _deduplicate_model_candidates(
        preferred_name,
        *active_resource_config.embedding_fallback_names,
    )


def get_embedding_model(
    model_name: str | None = None,
    resource_config: ModelResourceConfig | None = None,
) -> SentenceTransformer:
    """
    获取并缓存句向量模型。

    该实现保留：
    - 缓存
    - 候选顺序逐个尝试
    - 本地优先
    - fallback 其次
    - 全部失败时抛出清晰的 RuntimeError
    """
    active_resource_config = resource_config or DEFAULT_MODEL_RESOURCE_CONFIG
    resolved_model_name = model_name or active_resource_config.embedding_model_name
    candidates = get_embedding_model_load_candidates(
        embedding_model_name=resolved_model_name,
        resource_config=active_resource_config,
    )
    cache_key = (resolved_model_name, candidates)

    if cache_key in _EMBEDDING_MODEL_CACHE:
        return _EMBEDDING_MODEL_CACHE[cache_key]

    load_error: Exception | None = None

    for candidate in candidates:
        try:
            model = SentenceTransformer(candidate)
            _EMBEDDING_MODEL_CACHE[cache_key] = model
            return model
        except Exception as exc:
            load_error = exc

    raise RuntimeError(
        "Failed to load embedding model. "
        f"model_name={resolved_model_name!r}, candidates={candidates!r}"
    ) from load_error


def get_spacy_model(
    model_name: str | None = None,
    ensure_sentence_boundaries: bool = False,
    resource_config: ModelResourceConfig | None = None,
) -> Language:
    """
    获取并缓存 spaCy 模型。
    """
    active_resource_config = resource_config or DEFAULT_MODEL_RESOURCE_CONFIG
    resolved_model_name = model_name or active_resource_config.spacy_model_name
    candidates = get_spacy_model_load_candidates(
        spacy_model_name=resolved_model_name,
        resource_config=active_resource_config,
    )
    cache_key = (resolved_model_name, ensure_sentence_boundaries, candidates)

    if cache_key in _SPACY_MODEL_CACHE:
        return _SPACY_MODEL_CACHE[cache_key]

    load_error: Exception | None = None

    for candidate in candidates:
        try:
            nlp = spacy.load(candidate)
            if ensure_sentence_boundaries:
                _ensure_sentence_boundaries(nlp)
            _SPACY_MODEL_CACHE[cache_key] = nlp
            return nlp
        except Exception as exc:
            load_error = exc

    raise RuntimeError(
        "Failed to load spaCy model. "
        f"model_name={resolved_model_name!r}, "
        f"ensure_sentence_boundaries={ensure_sentence_boundaries!r}, "
        f"candidates={candidates!r}"
    ) from load_error


def get_tokenizer(
    tokenizer_name: str | None = None,
    resource_config: ModelResourceConfig | None = None,
):
    """
    获取并缓存 tokenizer。

    优先按 encoding 名称获取；失败时回退到常用模型编码。
    """
    active_resource_config = resource_config or DEFAULT_MODEL_RESOURCE_CONFIG
    resolved_tokenizer_name = tokenizer_name or active_resource_config.tokenizer_name

    if resolved_tokenizer_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[resolved_tokenizer_name]

    try:
        tokenizer = tiktoken.get_encoding(resolved_tokenizer_name)
    except Exception:
        try:
            tokenizer = tiktoken.encoding_for_model(
                DEFAULT_TOKENIZER_FALLBACK_MODEL_NAME
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load tokenizer. "
                f"tokenizer_name={resolved_tokenizer_name!r}, "
                f"fallback_model={DEFAULT_TOKENIZER_FALLBACK_MODEL_NAME!r}"
            ) from exc

    _TOKENIZER_CACHE[resolved_tokenizer_name] = tokenizer
    return tokenizer


def _deduplicate_model_candidates(*candidates: str) -> tuple[str, ...]:
    """
    保留顺序去重，并过滤空字符串。
    """
    results: list[str] = []
    seen: set[str] = set()

    for candidate in candidates:
        normalized_candidate = str(candidate or "").strip()
        if not normalized_candidate or normalized_candidate in seen:
            continue

        seen.add(normalized_candidate)
        results.append(normalized_candidate)

    return tuple(results)


def _ensure_sentence_boundaries(nlp: Language) -> None:
    """
    确保 spaCy pipeline 具备分句能力。
    """
    if "parser" in nlp.pipe_names or "senter" in nlp.pipe_names:
        return

    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
