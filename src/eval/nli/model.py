from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from torch import Tensor
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.tokenization_utils_base import BatchEncoding

from src.eval.nli.schemas import NLIProbabilities

NliLabel = Literal["contradiction", "entailment", "neutral"]
NLI_LABEL_ORDER = ("contradiction", "entailment", "neutral")

DEFAULT_LOCAL_NLI_MODEL_NAME = (
    r"C:\Users\zhulin\.cache\huggingface\hub\models--dleemiller--ModernCE-base-nli"
)
DEFAULT_NLI_FALLBACK_NAMES = ("dleemiller/ModernCE-base-nli",)


@dataclass(frozen=True)
class NLIModelResourceConfig:
    """
    NLI 模型资源配置。

    加载顺序：
    1. 本地路径优先
    2. 本地失败后再尝试 fallback 名称
    """

    nli_model_name: str = DEFAULT_LOCAL_NLI_MODEL_NAME
    nli_fallback_names: tuple[str, ...] = DEFAULT_NLI_FALLBACK_NAMES
    nli_label_order: tuple[NliLabel, NliLabel, NliLabel] = NLI_LABEL_ORDER


DEFAULT_NLI_MODEL_RESOURCE_CONFIG = NLIModelResourceConfig()

_MODEL_CACHE: dict[tuple[str, tuple[str, ...]], Any] = {}
_TOKENIZER_CACHE: dict[tuple[str, tuple[str, ...]], Any] = {}


def get_nli_model_load_candidates(
    nli_model_name: str | None = None,
    resource_config: NLIModelResourceConfig | None = None,
) -> tuple[str, ...]:
    """
    获取 NLI 模型加载候选列表。
    """
    active_resource_config = resource_config or DEFAULT_NLI_MODEL_RESOURCE_CONFIG
    preferred_name = nli_model_name or active_resource_config.nli_model_name
    return _deduplicate_candidates(
        preferred_name,
        *active_resource_config.nli_fallback_names,
    )


def get_nli_tokenizer(
    model_name: str | None = None,
    resource_config: NLIModelResourceConfig | None = None,
):
    """
    获取并缓存 NLI tokenizer。
    """
    active_resource_config = resource_config or DEFAULT_NLI_MODEL_RESOURCE_CONFIG
    resolved_model_name = model_name or active_resource_config.nli_model_name
    candidates = get_nli_model_load_candidates(
        nli_model_name=resolved_model_name,
        resource_config=active_resource_config,
    )
    cache_key = (resolved_model_name, candidates)

    if cache_key in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[cache_key]

    load_error: Exception | None = None
    for candidate in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate)
            _TOKENIZER_CACHE[cache_key] = tokenizer
            return tokenizer
        except Exception as exc:
            load_error = exc

    raise RuntimeError(
        "Failed to load NLI tokenizer. "
        f"model_name={resolved_model_name!r}, candidates={candidates!r}"
    ) from load_error


def get_nli_model(
    model_name: str | None = None,
    resource_config: NLIModelResourceConfig | None = None,
):
    """
    获取并缓存 NLI 分类模型。
    """
    active_resource_config = resource_config or DEFAULT_NLI_MODEL_RESOURCE_CONFIG
    resolved_model_name = model_name or active_resource_config.nli_model_name
    candidates = get_nli_model_load_candidates(
        nli_model_name=resolved_model_name,
        resource_config=active_resource_config,
    )
    cache_key = (resolved_model_name, candidates)

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    load_error: Exception | None = None
    for candidate in candidates:
        try:
            model = AutoModelForSequenceClassification.from_pretrained(candidate)
            model.eval()
            _MODEL_CACHE[cache_key] = model
            return model
        except Exception as exc:
            load_error = exc

    raise RuntimeError(
        "Failed to load NLI model. "
        f"model_name={resolved_model_name!r}, candidates={candidates!r}"
    ) from load_error


def load_nli_model_bundle(
    model_name: str | None = None,
    resource_config: NLIModelResourceConfig | None = None,
):
    """
    一次性加载 tokenizer 和 model。
    """
    tokenizer = get_nli_tokenizer(
        model_name=model_name,
        resource_config=resource_config,
    )
    model = get_nli_model(
        model_name=model_name,
        resource_config=resource_config,
    )
    return tokenizer, model


def predict_nli(
    premise_text: str,
    hypothesis_text: str,
    tokenizer=None,
    model=None,
    model_name: str | None = None,
    resource_config: NLIModelResourceConfig | None = None,
    max_length: int = 512,
) -> tuple[NLIProbabilities, NliLabel]:
    """
    对单个 premise / hypothesis 句对做 NLI 三分类预测。
    """
    if tokenizer is None or model is None:
        tokenizer, model = load_nli_model_bundle(
            model_name=model_name,
            resource_config=resource_config,
        )

    encoded = _tokenize_sentence_pair(
        tokenizer=tokenizer,
        premise_text=premise_text,
        hypothesis_text=hypothesis_text,
        max_length=max_length,
    )
    outputs = _run_model(model=model, encoded=encoded)
    logits = outputs.logits
    probs = _extract_probabilities(cast(Tensor, logits))
    active_resource_config = resource_config or DEFAULT_NLI_MODEL_RESOURCE_CONFIG
    label_to_score = _build_label_to_score_map(
        model=model,
        probabilities=probs,
        resource_config=active_resource_config,
    )

    nli_probabilities = NLIProbabilities(
        contradiction=label_to_score["contradiction"],
        entailment=label_to_score["entailment"],
        neutral=label_to_score["neutral"],
    )
    nli_label = _resolve_nli_label(nli_probabilities)
    return nli_probabilities, nli_label


def batch_predict_nli(
    pairs: list[tuple[str, str]],
    tokenizer=None,
    model=None,
    model_name: str | None = None,
    resource_config: NLIModelResourceConfig | None = None,
    max_length: int = 512,
) -> list[tuple[NLIProbabilities, NliLabel]]:
    """
    批量执行 NLI 预测。

    说明：
    - 当前版本为保证行为一致性，直接循环调用单条 predict_nli。
    - 这样单条与批量路径完全一致，更便于调试与维护。
    """
    if not pairs:
        return []

    if tokenizer is None or model is None:
        tokenizer, model = load_nli_model_bundle(
            model_name=model_name,
            resource_config=resource_config,
        )

    results: list[tuple[NLIProbabilities, NliLabel]] = []
    for premise_text, hypothesis_text in pairs:
        result = predict_nli(
            premise_text=premise_text,
            hypothesis_text=hypothesis_text,
            tokenizer=tokenizer,
            model=model,
            max_length=max_length,
        )
        results.append(result)

    return results


def _tokenize_sentence_pair(
    tokenizer,
    premise_text: str,
    hypothesis_text: str,
    max_length: int,
) -> BatchEncoding:
    """
    将 premise / hypothesis 编码为模型输入。
    """
    encoded = tokenizer(
        premise_text,
        hypothesis_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    return cast(BatchEncoding, encoded)


def _run_model(
    model,
    encoded: BatchEncoding,
) -> SequenceClassifierOutput:
    """
    执行模型前向推理。
    """
    with torch.no_grad():
        outputs = model(**encoded)
    return cast(SequenceClassifierOutput, outputs)


def _extract_probabilities(logits: Tensor) -> list[float]:
    """
    从 logits 提取三分类概率。
    """
    probability_tensor = torch.softmax(logits, dim=-1)[0]
    probability_list = cast(list[float], probability_tensor.tolist())
    return probability_list


def _build_label_to_score_map(
    model,
    probabilities: list[float],
    resource_config: NLIModelResourceConfig,
) -> dict[str, float]:
    """
    将概率映射为统一标签。

    说明：
    - 对当前 ModernCE NLI 模型，优先使用资源配置里声明的标签顺序。
    - 仅当未提供显式顺序时，才回退到 model.config.id2label / label2id。
    """
    label_order = tuple(resource_config.nli_label_order)
    if len(probabilities) == 3 and len(label_order) == 3:
        return {
            str(label): float(probabilities[index])
            for index, label in enumerate(label_order)
        }

    results = {"contradiction": 0.0, "entailment": 0.0, "neutral": 0.0}
    matched_labels: set[str] = set()
    id2label = _resolve_id2label(model)

    for class_index, raw_label in id2label.items():
        normalized_label = str(raw_label).strip().lower()
        if class_index >= len(probabilities):
            continue
        if "contrad" in normalized_label:
            results["contradiction"] = float(probabilities[class_index])
            matched_labels.add("contradiction")
        elif "entail" in normalized_label:
            results["entailment"] = float(probabilities[class_index])
            matched_labels.add("entailment")
        elif "neutral" in normalized_label:
            results["neutral"] = float(probabilities[class_index])
            matched_labels.add("neutral")

    if matched_labels != {"contradiction", "entailment", "neutral"}:
        raise ValueError(
            "Failed to resolve a complete NLI label mapping from model config. "
            f"id2label={id2label!r}"
        )

    return results


def _resolve_nli_label(nli_probabilities: NLIProbabilities) -> NliLabel:
    """
    根据三分类概率确定最终 nli_label。
    """
    score_map: dict[NliLabel, float] = {
        "contradiction": nli_probabilities.contradiction,
        "entailment": nli_probabilities.entailment,
        "neutral": nli_probabilities.neutral,
    }
    return max(score_map.items(), key=lambda item: item[1])[0]


def _deduplicate_candidates(*candidates: str) -> tuple[str, ...]:
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


def _resolve_id2label(model) -> dict[int, str]:
    raw_id2label = getattr(model.config, "id2label", None)
    if isinstance(raw_id2label, dict) and raw_id2label:
        return {
            int(class_index): str(raw_label)
            for class_index, raw_label in raw_id2label.items()
        }

    raw_label2id = getattr(model.config, "label2id", None)
    if isinstance(raw_label2id, dict) and raw_label2id:
        return {
            int(class_index): str(raw_label)
            for raw_label, class_index in raw_label2id.items()
        }

    raise ValueError(
        "Model config is missing id2label/label2id, cannot safely map NLI scores."
    )
