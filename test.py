from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from sentence_transformers import CrossEncoder

MODEL_NAME = "dleemiller/ModernCE-base-nli"


LABEL_ORDER = ["contradiction", "entailment", "neutral"]


@dataclass
class NLIResult:
    premise: str
    hypothesis: str
    label: str
    scores: dict[str, float]


def load_nli_model(model_name: str = MODEL_NAME) -> CrossEncoder:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_name}")
    print(f"Using device: {device}")
    return CrossEncoder(
        model_name,
        trust_remote_code=True,
        device=device,
    )


def softmax(logits: Sequence[float]) -> list[float]:
    x = torch.tensor(logits, dtype=torch.float32)
    probs = torch.softmax(x, dim=0)
    return probs.tolist()


def predict_nli(
    model: CrossEncoder,
    premise: str,
    hypothesis: str,
) -> NLIResult:
    logits = model.predict([(premise, hypothesis)])[0]
    probs = softmax(logits)

    score_map: dict[str, float] = {
        label: float(prob) for label, prob in zip(LABEL_ORDER, probs)
    }
    pred_label = max(score_map.items(), key=lambda x: x[1])[0]

    return NLIResult(
        premise=premise,
        hypothesis=hypothesis,
        label=pred_label,
        scores=score_map,
    )


def judge_support(
    model: CrossEncoder,
    premise: str,
    hypothesis: str,
    entailment_threshold: float = 0.60,
    contradiction_threshold: float = 0.60,
) -> str:
    result = predict_nli(model, premise, hypothesis)

    entailment_score = result.scores["entailment"]
    contradiction_score = result.scores["contradiction"]

    if contradiction_score >= contradiction_threshold:
        return "contradicted"
    if entailment_score >= entailment_threshold:
        return "supported"
    return "uncertain"


def main() -> None:
    model = load_nli_model()

    test_cases = [
        {
            "name": "supported_example",
            "premise": (
                "Applying nitrogen fertilizer to maize fields in Kenya "
                "increased grain yield by 12 percent."
            ),
            "hypothesis": ("Nitrogen fertilizer increased maize yield in Kenya."),
        },
        {
            "name": "contradiction_example",
            "premise": (
                "Applying nitrogen fertilizer to maize fields in Kenya "
                "increased grain yield by 12 percent."
            ),
            "hypothesis": ("Nitrogen fertilizer reduced maize yield in Kenya."),
        },
        {
            "name": "neutral_example",
            "premise": (
                "Applying nitrogen fertilizer to maize fields in Kenya "
                "increased grain yield by 12 percent."
            ),
            "hypothesis": ("Nitrogen fertilizer increased rice yield in Kenya."),
        },
        {
            "name": "summary_style_example",
            "premise": (
                "In rainfed plots, the treatment slightly increased wheat yield, "
                "but no improvement was observed in irrigated plots."
            ),
            "hypothesis": ("The treatment increased wheat yield."),
        },
    ]

    for idx, case in enumerate(test_cases, start=1):
        result = predict_nli(
            model=model,
            premise=case["premise"],
            hypothesis=case["hypothesis"],
        )

        support_label = judge_support(
            model=model,
            premise=case["premise"],
            hypothesis=case["hypothesis"],
        )

        print("=" * 10)
        print(f"Case {idx}: {case['name']}")
        print(f"Premise   : {case['premise']}")
        print(f"Hypothesis: {case['hypothesis']}")
        print(f"NLI label : {result.label}")
        print("Scores    :")
        for k, v in result.scores.items():
            print(f"  {k:>13}: {v:.4f}")
        print(f"Support   : {support_label}")


if __name__ == "__main__":
    print("-")
    main()
"""
Case 1: supported_example
Premise   : Applying nitrogen fertilizer to maize fields in Kenya increased grain yield by 12 percent.
Hypothesis: Nitrogen fertilizer increased maize yield in Kenya.
NLI label : entailment
Scores    :
  contradiction: 0.0003
     entailment: 0.9920
        neutral: 0.0077
Support   : supported
==========
Case 2: contradiction_example
Premise   : Applying nitrogen fertilizer to maize fields in Kenya increased grain yield by 12 percent.
Hypothesis: Nitrogen fertilizer reduced maize yield in Kenya.
NLI label : contradiction
Scores    :
  contradiction: 0.9990
     entailment: 0.0003
        neutral: 0.0008
Support   : contradicted
==========
Case 3: neutral_example
Premise   : Applying nitrogen fertilizer to maize fields in Kenya increased grain yield by 12 percent.
Hypothesis: Nitrogen fertilizer increased rice yield in Kenya.
NLI label : neutral
Scores    :
  contradiction: 0.0429
     entailment: 0.0023
        neutral: 0.9548
Support   : uncertain
==========
Case 4: summary_style_example
Premise   : In rainfed plots, the treatment slightly increased wheat yield, but no improvement was observed in irrigated plots.
Hypothesis: The treatment increased wheat yield.
NLI label : entailment
Scores    :
  contradiction: 0.0046
     entailment: 0.9370
        neutral: 0.0584
Support   : supported
"""
