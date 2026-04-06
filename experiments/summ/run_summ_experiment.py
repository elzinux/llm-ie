from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.summ.readers.md_reader import read_md_file
from experiments.summ.result_writer import (
    build_summary_output_dir,
    build_summary_output_path,
    build_summary_result_records,
    save_summary_results,
)
from src.summ.config import PipelineConfig, build_default_config
from src.summ.pipeline import run_summary_pipeline


def run_summary_for_md_file(
    md_path: str | Path,
    config: PipelineConfig | None = None,
    extraction_model_config: dict[str, Any] | None = None,
    summary_model_config: dict[str, Any] | None = None,
    extraction_max_retries: int = 2,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    md_path = Path(md_path)
    input_document = read_md_file(str(md_path))
    pipeline_config = config or build_default_config()

    pipeline_result = run_summary_pipeline(
        input_document=input_document,
        config=pipeline_config,
        extraction_model_config=extraction_model_config,
        summary_model_config=summary_model_config,
        extraction_max_retries=extraction_max_retries,
    )

    summary_results = build_summary_result_records(
        input_document=input_document,
        pipeline_result=pipeline_result,
    )

    target_output_path = (
        Path(output_path)
        if output_path is not None
        else build_summary_output_path(md_path)
    )
    save_summary_results(target_output_path, summary_results)

    return summary_results


def process_md_folder(
    input_dir: str | Path,
    config: PipelineConfig | None = None,
    extraction_model_config: dict[str, Any] | None = None,
    summary_model_config: dict[str, Any] | None = None,
    extraction_max_retries: int = 2,
) -> list[dict[str, Any]]:
    input_dir = Path(input_dir)
    output_dir = build_summary_output_dir(input_dir)
    md_files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".md"
    )

    if not md_files:
        print(f"未找到 Markdown 文件: {input_dir}")
        return []

    print(f"共找到 {len(md_files)} 个 Markdown 文件。")
    print(f"输出目录: {output_dir.resolve()}")

    all_results: list[dict[str, Any]] = []
    for md_file in md_files:
        result = run_summary_for_md_file(
            md_path=md_file,
            config=config,
            extraction_model_config=extraction_model_config,
            summary_model_config=summary_model_config,
            extraction_max_retries=extraction_max_retries,
            output_path=build_summary_output_path(md_file, output_dir=output_dir),
        )
        all_results.extend(result)
        print(f"文件处理完成: {md_file.name}")

    print("全部文件处理完成。")
    return all_results


SINGLE_FILE_PATH = Path(r"data/raw/test/3.md")
BATCH_DIR_PATH = Path(r"data/raw/test")
USE_BATCH = True
EXTRACTION_MAX_RETRIES = 2


def main() -> list[dict[str, Any]]:
    if USE_BATCH:
        return process_md_folder(
            input_dir=BATCH_DIR_PATH,
            extraction_max_retries=EXTRACTION_MAX_RETRIES,
        )

    return run_summary_for_md_file(
        md_path=SINGLE_FILE_PATH,
        extraction_max_retries=EXTRACTION_MAX_RETRIES,
    )


if __name__ == "__main__":
    main()
