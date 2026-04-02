from __future__ import annotations

from src.eval.schemas import (
    FinalEvaluationResult,
    LLMValidationResult,
    Quadruple,
    SlotValidationResult,
)


def calculate_final_score(
    S_slot: float,
    S_relation: float,
    S_precision: float,
    S_causal: float,
) -> float:
    """
    根据固定公式计算最终总分：
    0.35 * S_slot + 0.35 * S_relation + 0.15 * S_precision + 0.15 * S_causal
    """
    score = 0.35 * S_slot + 0.35 * S_relation + 0.15 * S_precision + 0.15 * S_causal
    return round(score, 2)


def slot_validation_to_dict(slot_validation: SlotValidationResult) -> dict:
    """
    将第一阶段槽位验证结果转换为字典格式，便于结构化保存。
    """
    return slot_validation.to_dict()


def llm_validation_to_dict(llm_validation: LLMValidationResult) -> dict:
    """
    将第二阶段大模型评估结果转换为字典格式，便于结构化保存。
    """
    return llm_validation.to_dict()


def build_final_evaluation_result(
    quadruple: Quadruple,
    slot_validation: SlotValidationResult,
    llm_validation: LLMValidationResult,
) -> FinalEvaluationResult:
    """
    整合三阶段结果，生成最终统一格式的评估结果对象。
    """
    slot_scores = [
        slot_validation.intervention_or_policy.verification_score,
        slot_validation.crop.verification_score,
        slot_validation.location.verification_score,
        slot_validation.impact.verification_score,
    ]
    S_slot = round(sum(slot_scores) / len(slot_scores), 2)

    final_score = calculate_final_score(
        S_slot=S_slot,
        S_relation=llm_validation.S_relation,
        S_precision=llm_validation.S_precision,
        S_causal=llm_validation.S_causal,
    )

    return FinalEvaluationResult(
        quadruple=quadruple,
        slot_validation=slot_validation,
        S_relation=llm_validation.S_relation,
        relation_reason=llm_validation.relation_reason,
        S_precision=llm_validation.S_precision,
        precision_reason=llm_validation.precision_reason,
        S_causal=llm_validation.S_causal,
        causal_reason=llm_validation.causal_reason,
        final_score=final_score,
    )


def final_result_to_dict(result: FinalEvaluationResult) -> dict:
    """
    将最终评估结果转换为标准字典格式，便于保存为 JSON。
    """
    return result.to_dict()
