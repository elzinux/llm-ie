from __future__ import annotations

from typing import Any

from src.ie.pipeline import run_ie_pipeline


def run_pipeline_for_text(
    text: str,
    use_correction: bool = False,
    extraction_extra_instructions: str | None = None,
    verification_extra_instructions: str | None = None,
    correction_extra_instructions: str | None = None,
    reverify_after_correction: bool = True,
    reextract_score_threshold: int = 4,
) -> dict[str, Any]:
    """针对单条摘要文本执行 IE 流水线。"""
    return run_ie_pipeline(
        text=text,
        use_correction=use_correction,
        extraction_extra_instructions=extraction_extra_instructions,
        verification_extra_instructions=verification_extra_instructions,
        correction_extra_instructions=correction_extra_instructions,
        reverify_after_correction=reverify_after_correction,
        reextract_score_threshold=reextract_score_threshold,
    )
