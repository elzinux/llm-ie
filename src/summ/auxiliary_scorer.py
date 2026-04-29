from src.eval.nli.pipeline import score_nli_pair
from src.eval.semantic_eval.scorer import score_semantic_eval

METHOD = "nli"


def compute_auxiliary_score(
    source_text: str,
    summary: str,
    method: str | None = METHOD,
):
    auxiliary_score = 0
    if method == "semantic_eval":
        auxiliary_score = score_semantic_eval(
            candidate_text=summary,
            reference_text=source_text,
        )
    if method == "nli":
        result = score_nli_pair(source_text, summary)
        auxiliary_score = compute_support_score(
            result.final_scores.contradiction,
            result.final_scores.entailment,
            result.final_scores.neutral,
        )
    nli_score = [
        result.final_scores.contradiction,
        result.final_scores.entailment,
        result.final_scores.neutral,
    ]
    return auxiliary_score, nli_score


def compute_support_score(c: float, e: float, n: float) -> float:
    base = (e - c + 1) / 2 + 0.20 * n * (1 - c)
    score = min(1.0, max(0.0, base))
    return round(score, 4)
