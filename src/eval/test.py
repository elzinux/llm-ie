from __future__ import annotations

import json

from src.eval.pipeline import evaluate_quadruples, initialize_eval_models


def main() -> None:
    text = """A warming climate expands the frost-free season, plausibly allowing for increased cropping intensity in temperate regions. This paper assesses the potential of multiple cropping to offset the projected negative effects of climate change on agricultural yields in the United States. We use cross-sectional variation in observed land cover, soil characteristics, and climate to estimate farmers' propensity to double-crop winter wheat with soybeans. Our estimates imply that under current economic conditions, a 3 C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions. A fixed-effects panel model of county yields further indicates that yields of double-cropped soybeans are about 12% lower than those of single-cropped soybeans. Accounting for changes in cropping intensity and attendant effects on soybean yields, we project that at current prices, a 3 C warming would induce a shift in cropping intensity that increases revenue from soy systems by 1.3% overall, offsetting only a small fraction of the revenue impacts of predicted yield declines.""".strip()

    records = [
        {
            "intervention_or_policy": "double-cropping winter wheat with soybeans",
            "location": "United States",
            "crop": "soybeans",
            "impact": "yields are about 12% lower than those of single-cropped soybeans",
            "evidence": "yields of double-cropped soybeans are about 12% lower than those of single-cropped soybeans",
        },
        {
            "intervention_or_policy": "double-cropping winter wheat with soybeans",
            "location": "cooler regions [of the United States]",
            "crop": "soybeans",
            "impact": "increase of 2.1 percentage points in the share of current soybean area double cropped",
            "evidence": "a 3°C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions",
        },
        {
            "intervention_or_policy": "double-cropping winter wheat with soybeans",
            "location": "United States",
            "crop": "soybeans",
            "impact": "increases revenue from soy systems by 1.3% overall",
            "evidence": "a 3°C warming would induce a shift in cropping intensity that increases revenue from soy systems by 1.3% overall",
        },
    ]

    nlp, sentence_model = initialize_eval_models()

    results = evaluate_quadruples(
        text=text,
        quadruples=records,
        nlp=nlp,
        sentence_model=sentence_model,
        llm_model_name=None,
    )

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
