from __future__ import annotations

from sentence_transformers import SentenceTransformer

from src.eval.llm_validation import validate_relation_precision_causality
from src.eval.schemas import (
    FinalEvaluationResult,
    Quadruple,
)
from src.eval.score_aggregation import build_final_evaluation_result
from src.eval.slot_validation import validate_all_slots
from src.eval.text_normalization import load_spacy_model

DEFAULT_SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"


def evaluate_quadruple(
    text: str,
    quadruple: Quadruple,
    nlp,
    sentence_model,
    llm_model_name: str | None = None,
) -> FinalEvaluationResult:
    """
    串联槽位验证、大模型统一评估和最终分数汇总，
    完成单条四元组的完整评估流程。
    """
    slot_validation = validate_all_slots(
        text=text,
        quadruple=quadruple,
        nlp=nlp,
        model=sentence_model,
    )

    model_config = {"model_name": llm_model_name} if llm_model_name else None

    llm_validation = validate_relation_precision_causality(
        text=text,
        quadruple=quadruple,
        model_config=model_config,
    )

    final_result = build_final_evaluation_result(
        quadruple=quadruple,
        slot_validation=slot_validation,
        llm_validation=llm_validation,
    )

    return final_result


def evaluate_quadruple_dict(
    text: str,
    quadruple_dict: dict,
    nlp,
    sentence_model,
    llm_model_name: str | None = None,
) -> dict:
    """
    接收字典形式的四元组输入，执行完整评估流程，
    并返回可直接保存的字典结果。
    """
    quadruple = Quadruple(
        intervention_or_policy=str(
            quadruple_dict.get("intervention_or_policy", "") or ""
        ),
        crop=str(quadruple_dict.get("crop", "") or ""),
        location=str(quadruple_dict.get("location", "") or ""),
        impact=str(quadruple_dict.get("impact", "") or ""),
    )

    result = evaluate_quadruple(
        text=text,
        quadruple=quadruple,
        nlp=nlp,
        sentence_model=sentence_model,
        llm_model_name=llm_model_name,
    )

    return result.to_dict()


def evaluate_quadruples(
    text: str,
    quadruples: list[dict],
    nlp,
    sentence_model,
    llm_model_name: str | None = None,
) -> list[dict]:
    """
    对同一篇原文中的多条四元组逐条进行评估，并返回结果列表。
    """
    results = []

    for quadruple_dict in quadruples:
        result = evaluate_quadruple_dict(
            text=text,
            quadruple_dict=quadruple_dict,
            nlp=nlp,
            sentence_model=sentence_model,
            llm_model_name=llm_model_name,
        )
        results.append(result)

    return results


def initialize_eval_models() -> tuple:
    """
    统一初始化评估流程所需的 spaCy 模型和 SentenceTransformers 模型，
    避免重复加载。
    """
    nlp = load_spacy_model()
    sentence_model = SentenceTransformer(DEFAULT_SENTENCE_MODEL_NAME)
    return nlp, sentence_model
