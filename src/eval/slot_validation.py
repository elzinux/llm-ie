from __future__ import annotations

from typing import List

from src.eval.schemas import Quadruple, SlotMatchResult, SlotValidationResult
from src.eval.text_normalization import (
    compute_fuzzy_score,
    compute_semantic_similarity,
    extract_candidate_phrases,
    normalize_slot_value,
    normalize_text,
    split_sentences,
)

# =========================
# 可调参数：后续只需要改这里
# =========================

FUZZY_HIGH_THRESHOLD = 80.0
SEMANTIC_HIGH_THRESHOLD = 0.70

# 说明：
# 1. RapidFuzz 分数范围是 0-100。
# 2. SentenceTransformers 常见相似度这里按 cosine 使用，通常范围近似在 0-1。
# 3. 这些映射不是“官方标准”，只是当前任务下更稳妥的经验设置。
# 4. 为了避免低分段过于激进地压到 1，这里保留了更宽松的 2 分区间。


def map_fuzzy_score_to_verification_score(score: float) -> int:
    """
    将 RapidFuzz 相似度分数映射为 1-5 的槽位验证分数。
    该函数集中管理 fuzzy 的评分映射，便于后续统一调整。
    """
    if score >= 92:
        return 5
    if score >= 82:
        return 4
    if score >= 70:
        return 3
    if score >= 55:
        return 2
    return 1


def map_semantic_score_to_verification_score(score: float) -> int:
    """
    将语义相似度分数映射为 1-5 的槽位验证分数。
    该函数集中管理 semantic 的评分映射，便于后续统一调整。
    """
    if score >= 0.84:
        return 5
    if score >= 0.74:
        return 4
    if score >= 0.64:
        return 3
    if score >= 0.52:
        return 2
    return 1


def exact_match_slot(slot_value: str, sentence: str) -> dict:
    """
    判断槽位值是否在句子中存在精确匹配，并返回匹配结果及对应分数。
    """
    slot_norm = normalize_slot_value(slot_value)
    sentence_norm = normalize_text(sentence)

    if not slot_norm or not sentence_norm:
        return {
            "is_match": False,
            "verification_score": 1,
            "matched_sentence": "",
            "similarity": 0.0,
        }

    if slot_norm in sentence_norm:
        return {
            "is_match": True,
            "verification_score": 5,
            "matched_sentence": sentence,
            "similarity": 1.0,
        }

    return {
        "is_match": False,
        "verification_score": 1,
        "matched_sentence": "",
        "similarity": 0.0,
    }


def fuzzy_match_slot(
    slot_value: str, sentence: str, threshold: float = FUZZY_HIGH_THRESHOLD
) -> dict:
    """
    判断槽位值是否与句子中的表达形成足够高的模糊匹配，并返回匹配结果及对应分数。
    """
    slot_norm = normalize_slot_value(slot_value)
    sentence_norm = normalize_text(sentence)

    if not slot_norm or not sentence_norm:
        return {
            "is_match": False,
            "verification_score": 1,
            "matched_sentence": "",
            "similarity": 0.0,
        }

    score = compute_fuzzy_score(slot_norm, sentence_norm)
    verification_score = map_fuzzy_score_to_verification_score(score)

    return {
        "is_match": score >= threshold,
        "verification_score": verification_score,
        "matched_sentence": sentence if verification_score >= 2 else "",
        "similarity": score,
    }


def semantic_match_slot(
    slot_value: str, sentence: str, model, threshold: float = SEMANTIC_HIGH_THRESHOLD
) -> dict:
    """
    判断槽位值与句子语义是否足够接近，并返回匹配结果及对应分数。
    """
    slot_norm = normalize_slot_value(slot_value)
    sentence_norm = normalize_text(sentence)

    if not slot_norm or not sentence_norm:
        return {
            "is_match": False,
            "verification_score": 1,
            "matched_sentence": "",
            "similarity": 0.0,
        }

    score = compute_semantic_similarity(slot_norm, sentence_norm, model)
    verification_score = map_semantic_score_to_verification_score(score)

    return {
        "is_match": score >= threshold,
        "verification_score": verification_score,
        "matched_sentence": sentence if verification_score >= 2 else "",
        "similarity": score,
    }


def _find_best_fuzzy_result(slot_value: str, candidates: List[str]) -> dict:
    """
    在候选文本列表中寻找 fuzzy 匹配结果最好的一个。
    """
    best_result = {
        "is_match": False,
        "verification_score": 1,
        "matched_sentence": "",
        "similarity": 0.0,
    }

    for candidate in candidates:
        result = fuzzy_match_slot(slot_value, candidate)
        if result["similarity"] > best_result["similarity"]:
            best_result = result

    return best_result


def _find_best_semantic_result(slot_value: str, candidates: List[str], model) -> dict:
    """
    在候选文本列表中寻找 semantic 匹配结果最好的一个。
    """
    best_result = {
        "is_match": False,
        "verification_score": 1,
        "matched_sentence": "",
        "similarity": 0.0,
    }

    for candidate in candidates:
        result = semantic_match_slot(slot_value, candidate, model)
        if result["similarity"] > best_result["similarity"]:
            best_result = result

    return best_result


def find_best_sentence_for_slot(
    slot_value: str, sentences: List[str], nlp, model
) -> dict:
    """
    在所有句子中为单个槽位寻找最佳匹配句，并按 exact、fuzzy、semantic 的顺序确定最终匹配方式。
    """
    if not slot_value or not sentences:
        return {
            "verification_score": 1,
            "matched_sentence": "",
        }

    # 1. exact match 优先
    for sentence in sentences:
        exact_result = exact_match_slot(slot_value, sentence)
        if exact_result["is_match"]:
            return {
                "verification_score": exact_result["verification_score"],
                "matched_sentence": exact_result["matched_sentence"],
            }

    # 2. 构造候选文本：句子本身 + 句子中的短语
    candidates = []
    for sentence in sentences:
        candidates.append(sentence)
        phrases = extract_candidate_phrases(sentence, nlp)
        candidates.extend(phrases)

    # 去重但保持顺序
    seen = set()
    unique_candidates = []
    for item in candidates:
        norm_item = normalize_text(item)
        if norm_item and norm_item not in seen:
            seen.add(norm_item)
            unique_candidates.append(item)

    # 3. fuzzy best
    best_fuzzy = _find_best_fuzzy_result(slot_value, unique_candidates)

    # 4. semantic best
    best_semantic = _find_best_semantic_result(slot_value, unique_candidates, model)

    # 5. 选择支持度更高的结果
    if best_fuzzy["verification_score"] > best_semantic["verification_score"]:
        return {
            "verification_score": best_fuzzy["verification_score"],
            "matched_sentence": best_fuzzy["matched_sentence"],
        }

    if best_semantic["verification_score"] > best_fuzzy["verification_score"]:
        return {
            "verification_score": best_semantic["verification_score"],
            "matched_sentence": best_semantic["matched_sentence"],
        }

    # 分数相同，则选相似度更高的那个
    fuzzy_similarity = best_fuzzy["similarity"]
    semantic_similarity = best_semantic["similarity"]

    # 为了可比性，把 semantic 的 0-1 放大到 0-100
    if fuzzy_similarity >= semantic_similarity * 100:
        return {
            "verification_score": best_fuzzy["verification_score"],
            "matched_sentence": best_fuzzy["matched_sentence"],
        }

    return {
        "verification_score": best_semantic["verification_score"],
        "matched_sentence": best_semantic["matched_sentence"],
    }


def validate_single_slot(
    slot_name: str, slot_value: str, sentences: List[str], nlp, model
) -> SlotMatchResult:
    """
    对单个槽位进行验证，输出标准化的槽位验证结果。
    """
    if not slot_value or not str(slot_value).strip():
        return SlotMatchResult(
            verification_score=1,
            matched_sentence="",
        )

    result = find_best_sentence_for_slot(slot_value, sentences, nlp, model)

    return SlotMatchResult(
        verification_score=int(result["verification_score"]),
        matched_sentence=result["matched_sentence"],
    )


def validate_all_slots(
    text: str, quadruple: Quadruple, nlp, model
) -> SlotValidationResult:
    """
    对四元组中的四个槽位逐一进行验证，并返回完整的第一阶段结果。
    """
    sentences = split_sentences(text, nlp)

    return SlotValidationResult(
        intervention_or_policy=validate_single_slot(
            "intervention_or_policy",
            quadruple.intervention_or_policy,
            sentences,
            nlp,
            model,
        ),
        crop=validate_single_slot(
            "crop",
            quadruple.crop,
            sentences,
            nlp,
            model,
        ),
        location=validate_single_slot(
            "location",
            quadruple.location,
            sentences,
            nlp,
            model,
        ),
        impact=validate_single_slot(
            "impact",
            quadruple.impact,
            sentences,
            nlp,
            model,
        ),
    )


def compute_slot_score(slot_validation: SlotValidationResult) -> float:
    """
    根据四个槽位的验证结果计算 S_slot，供第三阶段汇总使用。
    返回四个槽位 verification_score 的平均值，并保留两位小数。
    """
    scores = [
        slot_validation.intervention_or_policy.verification_score,
        slot_validation.crop.verification_score,
        slot_validation.location.verification_score,
        slot_validation.impact.verification_score,
    ]
    return round(sum(scores) / len(scores), 2)
