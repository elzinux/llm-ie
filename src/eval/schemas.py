from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Quadruple:
    """
    保存一条待验证四元组，包括 intervention_or_policy、crop、location、impact 四个字段。
    """

    intervention_or_policy: str
    crop: str
    location: str
    impact: str

    def to_dict(self) -> dict:
        return {
            "intervention_or_policy": self.intervention_or_policy,
            "crop": self.crop,
            "location": self.location,
            "impact": self.impact,
        }


@dataclass
class SlotMatchResult:
    """
    保存单个槽位的验证结果，包括 verification_score 和 matched_sentence。
    verification_score 为 1-5 的整数。
    """

    verification_score: float
    matched_sentence: str

    def to_dict(self) -> dict:
        return {
            "verification_score": self.verification_score,
            "matched_sentence": self.matched_sentence,
        }


@dataclass
class SlotValidationResult:
    """
    保存四个槽位整体的验证结果，分别对应四个字段的 SlotMatchResult。
    """

    intervention_or_policy: SlotMatchResult
    crop: SlotMatchResult
    location: SlotMatchResult
    impact: SlotMatchResult

    def to_dict(self) -> dict:
        return {
            "intervention_or_policy": self.intervention_or_policy.to_dict(),
            "crop": self.crop.to_dict(),
            "location": self.location.to_dict(),
            "impact": self.impact.to_dict(),
        }


@dataclass
class LLMValidationResult:
    """
    保存第二阶段大模型统一评估结果。
    三个分数均为 1-5 的整数。
    """

    S_relation: float
    relation_reason: str
    S_precision: float
    precision_reason: str
    S_causal: float
    causal_reason: str

    def to_dict(self) -> dict:
        return {
            "S_relation": self.S_relation,
            "relation_reason": self.relation_reason,
            "S_precision": self.S_precision,
            "precision_reason": self.precision_reason,
            "S_causal": self.S_causal,
            "causal_reason": self.causal_reason,
        }


@dataclass
class FinalEvaluationResult:
    """
    保存最终汇总结果，包括原始四元组、槽位验证结果、第二阶段评估结果和最终总分。
    """

    quadruple: Quadruple
    slot_validation: SlotValidationResult
    S_relation: float
    relation_reason: str
    S_precision: float
    precision_reason: str
    S_causal: float
    causal_reason: str
    final_score: float

    def to_dict(self) -> dict:
        return {
            "quadruple": self.quadruple.to_dict(),
            "slot_validation": self.slot_validation.to_dict(),
            "S_relation": self.S_relation,
            "relation_reason": self.relation_reason,
            "S_precision": self.S_precision,
            "precision_reason": self.precision_reason,
            "S_causal": self.S_causal,
            "causal_reason": self.causal_reason,
            "final_score": self.final_score,
        }
