from src.eval.nli.pipeline import score_nli_pair
from src.eval.semantic_eval.scorer import score_semantic_eval

METHOD = "nli"


def compute_auxiliary_score(
    source_text: str,
    summary: str,
    method: str | None = METHOD,
) -> float:
    auxiliary_score = 0
    if method == "semantic_eval":
        auxiliary_score = score_semantic_eval(
            candidate_text=summary,
            reference_text=source_text,
        )
    if method == "nli":
        result = score_nli_pair(source_text, summary)
        auxiliary_score = result.final_scores.entailment + result.final_scores.neutral
    return auxiliary_score
