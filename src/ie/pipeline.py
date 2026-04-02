from __future__ import annotations

import json
from typing import Any

from src.ie.correction import reextract_flagged_records
from src.ie.parser import parse_extraction_output, parse_verification_output
from src.ie.prompts import build_extraction_prompt, build_verification_prompt
from src.ie.schemas import build_task_schema
from src.llm.client import call_extraction_model, call_verification_model


def serialize_records(
    records: list[dict[str, Any]],
    ensure_ascii: bool = False,
    indent: int = 2,
) -> str:
    """将记录列表序列化为 JSON 字符串。"""
    return json.dumps(records, ensure_ascii=ensure_ascii, indent=indent)


def extract_records(
    text: str,
    extra_instructions: str | None = None,
) -> dict[str, Any]:
    """执行抽取阶段，并返回解析后的记录与模型元信息。"""
    if not text or not text.strip():
        raise ValueError("text 不能为空。")

    task_schema = build_task_schema()
    extraction_prompt = build_extraction_prompt(task_schema)

    llm_result = call_extraction_model(
        text=text,
        prompt_template=extraction_prompt,
        extra_instructions=extra_instructions,
    )

    records = parse_extraction_output(llm_result["raw_output"])

    return {
        "task_schema": task_schema,
        "final_prompt": llm_result["final_prompt"],
        "raw_output": llm_result["raw_output"],
        "records": records,
    }


def verify_records(
    text: str,
    records: list[dict[str, Any]],
    extra_instructions: str | None = None,
) -> dict[str, Any]:
    """执行校验阶段，并返回原始校验结果。"""
    if not text or not text.strip():
        raise ValueError("text 不能为空。")

    if not isinstance(records, list):
        raise ValueError("records 必须是列表。")

    task_schema = build_task_schema()
    verification_prompt = build_verification_prompt(task_schema)
    candidate_output = serialize_records(records, ensure_ascii=False, indent=2)

    llm_result = call_verification_model(
        text=text,
        candidate_output=candidate_output,
        verification_template=verification_prompt,
        extra_instructions=extra_instructions,
    )

    verification_records = parse_verification_output(llm_result["raw_output"])

    return {
        "task_schema": task_schema,
        "final_prompt": llm_result["final_prompt"],
        "raw_output": llm_result["raw_output"],
        "verification_records": verification_records,
    }


def run_ie_extract_only(
    text: str,
    extraction_extra_instructions: str | None = None,
) -> dict[str, Any]:
    """仅执行抽取流程，并返回未拆分的抽取结果。"""
    extraction_result = extract_records(
        text=text,
        extra_instructions=extraction_extra_instructions,
    )

    return {
        "mode": "extract_only",
        "records": extraction_result["records"],
        "raw_output": extraction_result["raw_output"],
        "final_prompt": extraction_result["final_prompt"],
    }


def run_ie_with_verification(
    text: str,
    extraction_extra_instructions: str | None = None,
    verification_extra_instructions: str | None = None,
) -> dict[str, Any]:
    """执行抽取加校验流程，但不进行重提取。"""
    extraction_result = extract_records(
        text=text,
        extra_instructions=extraction_extra_instructions,
    )

    verification_result = verify_records(
        text=text,
        records=extraction_result["records"],
        extra_instructions=verification_extra_instructions,
    )

    return {
        "mode": "with_verification",
        "extraction": extraction_result,
        "verification": verification_result,
    }


def run_ie_with_correction(
    text: str,
    extraction_extra_instructions: str | None = None,
    verification_extra_instructions: str | None = None,
    correction_extra_instructions: str | None = None,
    reverify_after_correction: bool = True,
    reextract_score_threshold: int = 4,
) -> dict[str, Any]:
    """执行完整 IE 流程：抽取、校验、重提取与可选复核。"""
    if not text or not text.strip():
        raise ValueError("text 不能为空。")

    initial_extraction = extract_records(
        text=text,
        extra_instructions=extraction_extra_instructions,
    )

    initial_verification = verify_records(
        text=text,
        records=initial_extraction["records"],
        extra_instructions=verification_extra_instructions,
    )

    reextraction_result = reextract_flagged_records(
        text=text,
        records=initial_extraction["records"],
        verification_records=initial_verification["verification_records"],
        extra_instructions=correction_extra_instructions,
        score_threshold=reextract_score_threshold,
    )

    final_records = reextraction_result["final_records"]

    final_verification: dict[str, Any] | None = None
    if reverify_after_correction:
        final_verification = verify_records(
            text=text,
            records=final_records,
            extra_instructions=verification_extra_instructions,
        )

    return {
        "mode": "with_correction",
        "initial_extraction": initial_extraction,
        "initial_verification": initial_verification,
        "reextraction": reextraction_result,
        "final_records": final_records,
        "final_verification": final_verification,
    }


def run_ie_pipeline(
    text: str,
    use_correction: bool = False,
    extraction_extra_instructions: str | None = None,
    verification_extra_instructions: str | None = None,
    correction_extra_instructions: str | None = None,
    reverify_after_correction: bool = True,
    reextract_score_threshold: int = 4,
) -> dict[str, Any]:
    """信息抽取流水线总入口。"""
    if not text or not text.strip():
        raise ValueError("text 不能为空。")

    if use_correction:
        return run_ie_with_correction(
            text=text,
            extraction_extra_instructions=extraction_extra_instructions,
            verification_extra_instructions=verification_extra_instructions,
            correction_extra_instructions=correction_extra_instructions,
            reverify_after_correction=reverify_after_correction,
            reextract_score_threshold=reextract_score_threshold,
        )

    return run_ie_extract_only(
        text=text,
        extraction_extra_instructions=extraction_extra_instructions,
    )


if __name__ == "__main__":
    print("开始执行 IE 流水线示例。")

    TEST_TEXT = ""
    result = run_ie_pipeline(TEST_TEXT, True)
    for key in result:
        print("分段结果如下：")
        print(result[key])

    print("IE 流水线示例结束。")
