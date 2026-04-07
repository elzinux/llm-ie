from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.summ.schemas import (
    BlockProcessResult,
    CheckResult,
    InputDocument,
    PreparedSummaryChunk,
    SummaryChunkResult,
    SummaryPipelineResult,
)


def make_safe_filename(name: str, max_length: int = 120) -> str:
    invalid_chars = '<>:"/\\|?*'
    cleaned = "".join("_" if ch in invalid_chars else ch for ch in name.strip())
    cleaned = "_".join(cleaned.split())
    cleaned = cleaned[:max_length].strip("._")
    return cleaned or "untitled"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_summary_output_dir(input_dir: str | Path) -> Path:
    input_dir = Path(input_dir)
    return ensure_dir(Path("data") / "outputs" / f"{input_dir.name}_summ")


def build_summary_output_path(
    md_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    md_path = Path(md_path)
    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else build_summary_output_dir(md_path.parent)
    )
    safe_stem = make_safe_filename(md_path.stem)
    return ensure_dir(target_dir) / f"{safe_stem}_summ.json"


def save_summary_results(output_path: str | Path, results: list[dict[str, Any]]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_summary_result_records(
    input_document: InputDocument,
    pipeline_result: SummaryPipelineResult,
) -> list[dict[str, Any]]:
    """
    按独立 summary chunk 构造实验输出结果。
    """
    chunk_result_map = {
        chunk_result.prepared_chunk_id: chunk_result
        for chunk_result in pipeline_result.chunk_results
    }
    block_result_map = {
        block_result.block_idx: block_result
        for block_result in pipeline_result.block_results
    }

    records: list[dict[str, Any]] = []

    for prepared_chunk in pipeline_result.prepared_chunks:
        chunk_result = chunk_result_map.get(prepared_chunk.prepared_chunk_id)
        records.append(
            {
                "prepared_chunk_id": prepared_chunk.prepared_chunk_id,
                "source_block_indices": list(prepared_chunk.source_block_indices),
                "chunck": prepared_chunk.text,
                "summary": chunk_result.summary if chunk_result is not None else "",
                "accept": bool(
                    chunk_result is not None and chunk_result.check_result.passed
                ),
                "mid_res": _build_mid_res_records(
                    prepared_chunk=prepared_chunk,
                    input_document=input_document,
                    block_result_map=block_result_map,
                    chunk_result=chunk_result,
                ),
            }
        )

    return records


def _build_mid_res_records(
    prepared_chunk: PreparedSummaryChunk,
    input_document: InputDocument,
    block_result_map: dict[int, BlockProcessResult],
    chunk_result: SummaryChunkResult | None,
) -> list[dict[str, Any]]:
    mid_res_records: list[dict[str, Any]] = []

    for block_idx in prepared_chunk.source_block_indices:
        if block_idx < 0 or block_idx >= len(input_document.items):
            continue

        input_item = input_document.items[block_idx]
        block_result = block_result_map.get(block_idx)

        mid_res_records.append(
            {
                "title": input_item.title,
                "text": input_item.text,
                "keywords": _collect_keywords(block_result),
                "key_sentences": _collect_key_sentences(block_result),
                "summ_score": _build_summ_score_list(chunk_result),
                "summ_count": chunk_result.attempts if chunk_result is not None else 0,
                "reason": list(chunk_result.retry_reasons) if chunk_result is not None else [],
            }
        )

    return mid_res_records


def _collect_keywords(block_result: BlockProcessResult | None) -> list[str]:
    if block_result is None:
        return []

    keywords: list[str] = []
    seen: set[str] = set()

    for record in block_result.extraction_records:
        for keyword in getattr(record, "keywords", None) or []:
            normalized = keyword.strip()
            if not normalized:
                continue

            dedup_key = normalized.lower()
            if dedup_key in seen:
                continue

            seen.add(dedup_key)
            keywords.append(normalized)

    return keywords


def _collect_key_sentences(block_result: BlockProcessResult | None) -> list[str]:
    if block_result is None:
        return []

    key_sentences: list[str] = []
    seen: set[str] = set()

    for record in block_result.extraction_records:
        for sentence in getattr(record, "key_sentences", None) or []:
            normalized = sentence.strip()
            if not normalized:
                continue

            dedup_key = " ".join(normalized.lower().split())
            if dedup_key in seen:
                continue

            seen.add(dedup_key)
            key_sentences.append(normalized)

    return key_sentences


def _build_summ_score_list(chunk_result: SummaryChunkResult | None) -> list[float]:
    check_result = _get_chunk_check_result(chunk_result)
    return [
        float(check_result.information_coverage),
        float(check_result.semantic_eval_score),
        float(check_result.overall_score),
    ]


def _get_chunk_check_result(chunk_result: SummaryChunkResult | None) -> CheckResult:
    if chunk_result is None:
        return CheckResult(passed=False)
    return chunk_result.check_result
