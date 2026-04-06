from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.ie.extractor_runner import run_pipeline_for_text
from experiments.ie.io_reader import list_md_files, load_title_abstract_pairs
from experiments.ie.result_writer import (
    build_final_results_dir,
    build_intermediate_results_dir,
    build_output_root_dir,
    build_result_file_path,
    save_final_results,
    save_intermediate_results,
)

INPUT_DIR = Path("data/raw/data_abstract1")

USE_CORRECTION = True
EXTRACTION_EXTRA_INSTRUCTIONS: str | None = None
VERIFICATION_EXTRA_INSTRUCTIONS: str | None = None
CORRECTION_EXTRA_INSTRUCTIONS: str | None = None
REVERIFY_AFTER_CORRECTION = True
REEXTRACT_SCORE_THRESHOLD = 4


RECORD_FIELDS = [
    "intervention_or_policy",
    "location",
    "crop",
    "impact",
    "evidence",
]


def _pick_first_existing(
    data: dict[str, Any], candidate_keys: list[str], default: Any
) -> Any:
    """Internal helper."""
    if not isinstance(data, dict):
        return default

    for key in candidate_keys:
        if key in data:
            return data[key]
    return default


def _pick_first_non_none(data: dict[str, Any], candidate_keys: list[str]) -> Any:
    """Internal helper."""
    if not isinstance(data, dict):
        return None

    for key in candidate_keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _to_int_or_none(value: Any) -> int | None:
    """Internal helper."""
    if value is None:
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None

    return None


def _unwrap_result_items(items: Any, candidate_keys: list[str]) -> list[Any]:
    """Internal helper."""
    if isinstance(items, list):
        return items

    if isinstance(items, dict):
        nested_items = _pick_first_existing(items, candidate_keys, default=[])
        if isinstance(nested_items, list):
            return nested_items

    return []


def _record_defaults() -> dict[str, str]:
    """Internal helper."""
    return {field: "" for field in RECORD_FIELDS}


def _extract_record_payload(item: Any) -> dict[str, Any] | None:
    """Internal helper."""
    if not isinstance(item, dict):
        return None

    for key in ["corrected_record", "original_record", "record"]:
        nested = item.get(key)
        if isinstance(nested, dict):
            return nested

    return item


def _normalize_record(item: Any) -> dict[str, str] | None:
    """Internal helper."""
    payload = _extract_record_payload(item)
    if not isinstance(payload, dict):
        return None

    return {
        "intervention_or_policy": str(payload.get("intervention_or_policy", "") or ""),
        "location": str(payload.get("location", "") or ""),
        "crop": str(payload.get("crop", "") or ""),
        "impact": str(payload.get("impact", "") or ""),
        "evidence": str(payload.get("evidence", "") or ""),
    }


def _normalize_extraction_items(items: Any) -> list[dict[str, str]]:
    """Internal helper."""
    items = _unwrap_result_items(
        items,
        [
            "final_extraction",
            "records",
            "final_records",
            "reextract_results",
            "reextraction_results",
            "corrected_records",
            "reextracted_records",
            "failed_records",
            "results",
        ],
    )

    normalized: list[dict[str, str]] = []
    for item in items:
        normalized_item = _normalize_record(item)
        if normalized_item is not None:
            normalized.append(normalized_item)
    return normalized


def _normalize_verification_score(item: dict[str, Any]) -> int | None:
    """Internal helper."""
    raw_score = _pick_first_non_none(
        item,
        [
            "verification_score",
            "score",
            "final_score",
            "verdict_score",
        ],
    )
    return _to_int_or_none(raw_score)


def _normalize_verification_items(
    records_source: Any,
    verification_source: Any,
) -> list[dict[str, Any]]:
    """Internal helper."""
    records = _normalize_extraction_items(records_source)
    verification_items = _unwrap_result_items(
        verification_source,
        [
            "verification_records",
            "records",
            "results",
            "verification_results",
            "final_verification",
            "initial_verification",
        ],
    )

    normalized: list[dict[str, Any]] = []

    for idx, item in enumerate(verification_items):
        if not isinstance(item, dict):
            continue

        record_fields = records[idx] if idx < len(records) else _record_defaults()
        normalized.append(
            {
                **record_fields,
                "verification_score": _normalize_verification_score(item),
                "needs_reextract": bool(item.get("needs_reextract", False)),
                "reason": str(item.get("reason", "") or ""),
            }
        )

    return normalized


def build_final_result_record(
    title: str,
    text: str,
    pipeline_result: dict[str, Any],
) -> dict[str, Any]:
    """Internal helper."""
    final_extraction = _pick_first_existing(
        pipeline_result,
        candidate_keys=[
            "final_extraction",
            "records",
            "final_records",
            "final_records_after_correction",
        ],
        default=[],
    )

    return {
        "title": title,
        "text": text,
        "final_extraction": _normalize_extraction_items(final_extraction),
    }


def build_intermediate_result_record(
    text: str,
    pipeline_result: dict[str, Any],
) -> dict[str, Any]:
    """Internal helper."""
    initial_extraction = _pick_first_existing(
        pipeline_result,
        candidate_keys=["initial_extraction", "extraction"],
        default={},
    )
    initial_records = _pick_first_existing(
        initial_extraction,
        candidate_keys=["records", "final_records"],
        default=[],
    )

    initial_verification = _pick_first_existing(
        pipeline_result,
        candidate_keys=["initial_verification", "verification"],
        default={},
    )

    final_records = _pick_first_existing(
        pipeline_result,
        candidate_keys=["final_records", "records"],
        default=[],
    )
    final_verification = _pick_first_existing(
        pipeline_result,
        candidate_keys=["final_verification"],
        default={},
    )

    reextraction = _pick_first_existing(
        pipeline_result,
        candidate_keys=["reextraction"],
        default={},
    )
    if not isinstance(reextraction, dict):
        reextraction = {}

    reextraction_results = _pick_first_existing(
        reextraction,
        candidate_keys=["reextract_results", "reextraction_results", "results"],
        default=[],
    )
    failed_reextract_records = _pick_first_existing(
        reextraction,
        candidate_keys=["failed_records"],
        default=[],
    )

    return {
        "text": text,
        "initial_verification": _normalize_verification_items(
            initial_records,
            initial_verification,
        ),
        "final_verification": _normalize_verification_items(
            final_records,
            final_verification,
        ),
        "reextract_count": _to_int_or_none(reextraction.get("reextract_count")) or 0,
        "failed_reextract_count": len(failed_reextract_records)
        if isinstance(failed_reextract_records, list)
        else 0,
        "invalid_verification_count": _to_int_or_none(
            reextraction.get("invalid_verification_count")
        )
        or 0,
        "unused_verification_count": _to_int_or_none(
            reextraction.get("unused_verification_count")
        )
        or 0,
        "reextraction_results": _normalize_extraction_items(reextraction_results),
        "failed_reextract_records": _normalize_extraction_items(
            failed_reextract_records
        ),
    }


def process_single_md_file(
    md_path: str | Path,
    final_results_dir: str | Path,
    intermediate_results_dir: str | Path | None = None,
    use_correction: bool = True,
    extraction_extra_instructions: str | None = None,
    verification_extra_instructions: str | None = None,
    correction_extra_instructions: str | None = None,
    reverify_after_correction: bool = True,
    reextract_score_threshold: int = 4,
) -> None:
    """Internal helper."""
    md_path = Path(md_path)
    samples = load_title_abstract_pairs(md_path)

    if not samples:
        print(f"跳过空文件或无效文件: {md_path.name}")
        return

    final_results: list[dict[str, Any]] = []
    intermediate_results: list[dict[str, Any]] | None = [] if use_correction else None

    for sample in samples:
        title = sample["title"]
        abstract = sample["abstract"]

        pipeline_result = run_pipeline_for_text(
            text=abstract,
            use_correction=use_correction,
            extraction_extra_instructions=extraction_extra_instructions,
            verification_extra_instructions=verification_extra_instructions,
            correction_extra_instructions=correction_extra_instructions,
            reverify_after_correction=reverify_after_correction,
            reextract_score_threshold=reextract_score_threshold,
        )

        final_results.append(
            build_final_result_record(
                title=title,
                text=abstract,
                pipeline_result=pipeline_result,
            )
        )

        if intermediate_results is not None:
            intermediate_results.append(
                build_intermediate_result_record(
                    text=abstract,
                    pipeline_result=pipeline_result,
                )
            )

    final_output_path = build_result_file_path(
        output_dir=final_results_dir,
        md_stem=md_path.stem,
    )
    save_final_results(
        output_path=final_output_path,
        results=final_results,
    )

    if intermediate_results_dir is not None and intermediate_results is not None:
        intermediate_output_path = build_result_file_path(
            output_dir=intermediate_results_dir,
            md_stem=md_path.stem,
        )
        save_intermediate_results(
            output_path=intermediate_output_path,
            results=intermediate_results,
        )

    print(f"文件处理完成: {md_path.name}")


def process_md_folder(
    input_dir: str | Path,
    use_correction: bool = True,
    extraction_extra_instructions: str | None = None,
    verification_extra_instructions: str | None = None,
    correction_extra_instructions: str | None = None,
    reverify_after_correction: bool = True,
    reextract_score_threshold: int = 4,
) -> None:
    """Internal helper."""
    input_dir = Path(input_dir)

    output_root_dir = build_output_root_dir(input_dir, use_correction=use_correction)
    final_results_dir = build_final_results_dir(output_root_dir)
    intermediate_results_dir = (
        build_intermediate_results_dir(output_root_dir) if use_correction else None
    )

    md_files = list_md_files(input_dir)

    if not md_files:
        print(f"未找到 Markdown 文件: {input_dir}")
        return

    print(f"共找到 {len(md_files)} 个 Markdown 文件。")
    print(f"最终结果目录: {final_results_dir.resolve()}")
    if intermediate_results_dir is not None:
        print(f"中间结果目录: {intermediate_results_dir.resolve()}")

    for md_file in md_files:
        process_single_md_file(
            md_path=md_file,
            final_results_dir=final_results_dir,
            intermediate_results_dir=intermediate_results_dir,
            use_correction=use_correction,
            extraction_extra_instructions=extraction_extra_instructions,
            verification_extra_instructions=verification_extra_instructions,
            correction_extra_instructions=correction_extra_instructions,
            reverify_after_correction=reverify_after_correction,
            reextract_score_threshold=reextract_score_threshold,
        )

    print("全部文件处理完成。")


if __name__ == "__main__":
    # uv run python -m experiments.ie.run_ie_experiment
    process_md_folder(
        input_dir=INPUT_DIR,
        use_correction=USE_CORRECTION,
        extraction_extra_instructions=EXTRACTION_EXTRA_INSTRUCTIONS,
        verification_extra_instructions=VERIFICATION_EXTRA_INSTRUCTIONS,
        correction_extra_instructions=CORRECTION_EXTRA_INSTRUCTIONS,
        reverify_after_correction=REVERIFY_AFTER_CORRECTION,
        reextract_score_threshold=REEXTRACT_SCORE_THRESHOLD,
    )
