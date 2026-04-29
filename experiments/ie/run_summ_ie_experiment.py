from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.ie.extractor_runner import run_pipeline_for_text
from experiments.ie.result_writer import (
    build_final_results_dir,
    build_intermediate_results_dir,
    build_output_root_dir,
    build_result_file_path,
    save_final_results,
    save_intermediate_results,
)
from experiments.ie.run_ie_experiment import (
    build_final_result_record,
    build_intermediate_result_record,
)
from experiments.ie.summ_json_reader import list_json_files, load_title_summary_pairs

INPUT_DIR = Path("data/processed/e4-16-md_files_summ")

USE_CORRECTION = True
EXTRACTION_EXTRA_INSTRUCTIONS: str | None = None
VERIFICATION_EXTRA_INSTRUCTIONS: str | None = None
CORRECTION_EXTRA_INSTRUCTIONS: str | None = None
REVERIFY_AFTER_CORRECTION = True
REEXTRACT_SCORE_THRESHOLD = 4


def process_single_summary_json_file(
    json_path: str | Path,
    final_results_dir: str | Path,
    intermediate_results_dir: str | Path | None = None,
    use_correction: bool = True,
    extraction_extra_instructions: str | None = None,
    verification_extra_instructions: str | None = None,
    correction_extra_instructions: str | None = None,
    reverify_after_correction: bool = True,
    reextract_score_threshold: int = 4,
) -> None:
    """Run IE over all summaries in one summary-result JSON file."""
    json_path = Path(json_path)
    samples = load_title_summary_pairs(json_path)

    if not samples:
        print(f"跳过空文件或无有效 summary: {json_path.name}")
        return

    final_results: list[dict[str, Any]] = []
    intermediate_results: list[dict[str, Any]] | None = [] if use_correction else None

    for sample in samples:
        title = sample["title"]
        text = sample["text"]

        pipeline_result = run_pipeline_for_text(
            text=text,
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
                text=text,
                pipeline_result=pipeline_result,
            )
        )

        if intermediate_results is not None:
            intermediate_results.append(
                build_intermediate_result_record(
                    text=text,
                    pipeline_result=pipeline_result,
                )
            )

    final_output_path = build_result_file_path(
        output_dir=final_results_dir,
        md_stem=json_path.stem,
    )
    save_final_results(
        output_path=final_output_path,
        results=final_results,
    )

    if intermediate_results_dir is not None and intermediate_results is not None:
        intermediate_output_path = build_result_file_path(
            output_dir=intermediate_results_dir,
            md_stem=json_path.stem,
        )
        save_intermediate_results(
            output_path=intermediate_output_path,
            results=intermediate_results,
        )

    print(f"文件处理完成: {json_path.name}")


def process_summary_json_folder(
    input_dir: str | Path,
    use_correction: bool = True,
    extraction_extra_instructions: str | None = None,
    verification_extra_instructions: str | None = None,
    correction_extra_instructions: str | None = None,
    reverify_after_correction: bool = True,
    reextract_score_threshold: int = 4,
) -> None:
    """Run IE over all summary-result JSON files under the given folder."""
    input_dir = Path(input_dir)

    output_root_dir = build_output_root_dir(input_dir, use_correction=use_correction)
    final_results_dir = build_final_results_dir(output_root_dir)
    intermediate_results_dir = (
        build_intermediate_results_dir(output_root_dir) if use_correction else None
    )

    json_files = list_json_files(input_dir)

    if not json_files:
        print(f"未找到 JSON 文件: {input_dir}")
        return

    print(f"共找到 {len(json_files)} 个 JSON 文件。")
    print(f"最终结果目录: {final_results_dir.resolve()}")
    if intermediate_results_dir is not None:
        print(f"中间结果目录: {intermediate_results_dir.resolve()}")

    for json_file in json_files:
        process_single_summary_json_file(
            json_path=json_file,
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
    process_summary_json_folder(
        input_dir=INPUT_DIR,
        use_correction=USE_CORRECTION,
        extraction_extra_instructions=EXTRACTION_EXTRA_INSTRUCTIONS,
        verification_extra_instructions=VERIFICATION_EXTRA_INSTRUCTIONS,
        correction_extra_instructions=CORRECTION_EXTRA_INSTRUCTIONS,
        reverify_after_correction=REVERIFY_AFTER_CORRECTION,
        reextract_score_threshold=REEXTRACT_SCORE_THRESHOLD,
    )
