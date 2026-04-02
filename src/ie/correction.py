from __future__ import annotations

import json
from typing import Any

from src.ie.parser import parse_reextraction_output
from src.ie.prompts import build_reextraction_prompt
from src.ie.schemas import build_task_schema
from src.llm.client import call_reextraction_model


def prepare_reextract_input(
    text: str,
    record: dict[str, Any],
    verification: dict[str, Any],
) -> str:
    """构造单条记录重提取时发送给模型的输入。"""
    if not text or not text.strip():
        raise ValueError("text 不能为空。")

    if not isinstance(record, dict):
        raise ValueError("record 必须是字典。")

    if not isinstance(verification, dict):
        raise ValueError("verification 必须是字典。")

    record_json = json.dumps(record, ensure_ascii=False, indent=2)
    verification_json = json.dumps(verification, ensure_ascii=False, indent=2)

    return (
        "Please correct the extracted record according to the source text and the verification feedback.\n\n"
        f"[Source Text]\n{text.strip()}\n\n"
        f"[Original Extracted Record]\n{record_json}\n\n"
        f"[Verification Feedback]\n{verification_json}"
    )


def reextract_single_record(
    text: str,
    record: dict[str, Any],
    verification: dict[str, Any],
    extra_instructions: str | None = None,
) -> dict[str, Any]:
    """对单条低分记录执行重提取。"""
    task_schema = build_task_schema()
    prompt = build_reextraction_prompt(task_schema)
    user_input = prepare_reextract_input(
        text=text,
        record=record,
        verification=verification,
    )

    llm_result = call_reextraction_model(
        user_input=user_input,
        prompt_template=prompt,
        extra_instructions=extra_instructions,
    )

    corrected_record = parse_reextraction_output(llm_result["raw_output"])

    return {
        "original_record": record,
        "verification": verification,
        "final_prompt": llm_result["final_prompt"],
        "raw_output": llm_result["raw_output"],
        "corrected_record": corrected_record,
    }


def should_reextract_record(
    verification: dict[str, Any],
    score_threshold: int,
) -> bool:
    """判断一条校验结果是否需要触发重提取。"""
    score = verification.get("verification_score")
    return isinstance(score, int) and score < score_threshold


def is_empty_record(record: dict[str, Any]) -> bool:
    """判断一条记录是否所有字段都为空。"""
    return all(not str(value).strip() for value in record.values())


def reextract_flagged_records(
    text: str,
    records: list[dict[str, Any]],
    verification_records: list[dict[str, Any]],
    extra_instructions: str | None = None,
    score_threshold: int = 4,
) -> dict[str, Any]:
    """按照分数阈值重提取低分记录，并返回最终记录列表。"""
    if not text or not text.strip():
        raise ValueError("text 不能为空。")

    if not isinstance(records, list):
        raise ValueError("records 必须是列表。")

    if not isinstance(verification_records, list):
        raise ValueError("verification_records 必须是列表。")

    final_records: list[dict[str, Any]] = []
    reextract_results: list[dict[str, Any]] = []
    failed_records: list[dict[str, Any]] = []
    invalid_verification_count = 0

    for idx, record in enumerate(records):
        if idx >= len(verification_records):
            final_records.append(record)
            continue

        verification = verification_records[idx]
        if not isinstance(verification, dict):
            invalid_verification_count += 1
            final_records.append(record)
            continue

        if not should_reextract_record(verification, score_threshold):
            final_records.append(record)
            continue

        try:
            result = reextract_single_record(
                text=text,
                record=record,
                verification=verification,
                extra_instructions=extra_instructions,
            )
            reextract_results.append(result)

            corrected_record = result["corrected_record"]
            if not is_empty_record(corrected_record):
                final_records.append(corrected_record)
        except Exception:
            failed_records.append(
                {
                    "original_record": record,
                    "verification": verification,
                }
            )
            final_records.append(record)

    return {
        "score_threshold": score_threshold,
        "reextract_count": len(reextract_results),
        "reextract_results": reextract_results,
        "failed_records": failed_records,
        "invalid_verification_count": invalid_verification_count,
        "unused_verification_count": max(0, len(verification_records) - len(records)),
        "final_records": final_records,
    }


if __name__ == "__main__":
    TEST_TEXT = ""
