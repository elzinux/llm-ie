from __future__ import annotations

from typing import Any

from src.eval.schemas import LLMValidationResult, Quadruple
from src.llm.client import call_llm
from src.utils.json_parser import parse_json_safe

LLM_VALIDATION_SYSTEM_PROMPT = """
You are an expert evaluator for agricultural information extraction.
You will be given:
1. the original source text
2. one extracted quadruple

Your task is to evaluate the quadruple from these three aspects only:
1. Relation correctness
Check whether:
- intervention_or_policy is applied to crop
- intervention_or_policy occurs in location
- impact is truly caused by the intervention_or_policy on the crop
2. Precision
Check whether any field is too broad, too vague, too narrow, or has inaccurate boundaries.
3. Causal faithfulness
Check whether the quadruple introduces exaggeration, weakening, overgeneralization,
causal overstatement, or direction reversal.

Scoring rules:
- Use the 1-5 scoring scale.
- 1 means very poor / clearly unsupported.
- 3 means partially supported / somewhat problematic.
- 5 means fully supported / accurate.
- You may use decimals when necessary.
- Keep two decimal places.

Output requirements:
- Return only one JSON object.
- Do not output markdown.
- Do not output explanations outside JSON.
- Each reason must be one short sentence only.

Output format:
{
  "S_relation": 3,
  "relation_reason": "",
  "S_precision": 4,
  "precision_reason": "",
  "S_causal": 3,
  "causal_reason": ""
}
""".strip()


def build_llm_validation_prompt(text: str, quadruple: Quadruple) -> str:
    """
    构造第二阶段统一评估提示词，将原文和四元组组织成大模型输入。
    """
    return f"""
Original text:
{text}

Quadruple to evaluate:
{quadruple.to_dict()}

Please evaluate this quadruple and return the result in the required JSON format.
""".strip()


def call_llm_for_validation(
    prompt: str,
    model_config: dict[str, Any] | None = None,
) -> str:
    """
    调用大模型执行第二阶段评估，并返回原始文本响应。
    """
    return call_llm(
        user_input=prompt,
        system_prompt=LLM_VALIDATION_SYSTEM_PROMPT,
        model_config=model_config,
    )


def _safe_float_score(value: Any) -> float:
    """
    将分数安全转换为 float，并保留两位小数。
    """
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 1.0

    return round(score, 2)


def parse_llm_validation_response(response_text: str) -> LLMValidationResult:
    """
    解析大模型返回内容，提取 S_relation、relation_reason、S_precision、
    precision_reason、S_causal、causal_reason。
    """
    data = parse_json_safe(response_text)

    if isinstance(data, list):
        if not data:
            raise ValueError("LLM 验证结果为空列表。")
        if not isinstance(data[0], dict):
            raise ValueError("LLM 验证结果列表中的首个元素不是 JSON 对象。")
        data = data[0]

    if not isinstance(data, dict):
        raise ValueError(
            f"LLM 验证结果必须是 JSON 对象，当前类型为：{type(data).__name__}"
        )

    return LLMValidationResult(
        S_relation=_safe_float_score(data.get("S_relation", 1.0)),
        relation_reason=str(data.get("relation_reason", "") or "").strip(),
        S_precision=_safe_float_score(data.get("S_precision", 1.0)),
        precision_reason=str(data.get("precision_reason", "") or "").strip(),
        S_causal=_safe_float_score(data.get("S_causal", 1.0)),
        causal_reason=str(data.get("causal_reason", "") or "").strip(),
    )


def validate_relation_precision_causality(
    text: str,
    quadruple: Quadruple,
    model_config: dict[str, Any] | None = None,
) -> LLMValidationResult:
    """
    执行第二阶段统一评估流程，返回关系、精确性和忠实性的验证结果。
    """
    prompt = build_llm_validation_prompt(
        text=text,
        quadruple=quadruple,
    )

    response_text = call_llm_for_validation(
        prompt=prompt,
        model_config=model_config,
    )

    return parse_llm_validation_response(response_text)
