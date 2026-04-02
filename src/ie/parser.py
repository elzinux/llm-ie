from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from src.utils.json_parser import ensure_list, parse_json_safe


class ExtractionRecord(BaseModel):
    """单条抽取记录。"""

    intervention_or_policy: str = Field(default="")
    location: str = Field(default="")
    crop: str = Field(default="")
    impact: str = Field(default="")
    evidence: str = Field(default="")


class VerificationRecord(BaseModel):
    """单条校验记录。"""

    verification_score: Literal[1, 2, 3, 4, 5]
    needs_reextract: bool
    reason: str = Field(default="")


def parse_extraction_output(raw_output: str) -> list[dict[str, Any]]:
    """将抽取模型输出解析成校验后的记录列表。"""
    data = parse_json_safe(raw_output)
    records = ensure_list(data)

    try:
        return [ExtractionRecord.model_validate(item).model_dump() for item in records]
    except ValidationError as exc:
        raise ValueError(f"抽取结果结构校验失败：{exc}") from exc


def parse_reextraction_output(raw_output: str) -> dict[str, Any]:
    """将重提取输出解析成单条校验后的记录。"""
    data = parse_json_safe(raw_output)
    records = ensure_list(data)

    if not records:
        return ExtractionRecord().model_dump()

    try:
        return ExtractionRecord.model_validate(records[0]).model_dump()
    except ValidationError as exc:
        raise ValueError(f"重提取结果结构校验失败：{exc}") from exc


def parse_verification_output(raw_output: str) -> list[dict[str, Any]]:
    """将校验模型输出解析成校验结果列表。"""
    data = parse_json_safe(raw_output)
    records = ensure_list(data)

    try:
        return [VerificationRecord.model_validate(item).model_dump() for item in records]
    except ValidationError as exc:
        raise ValueError(f"校验结果结构校验失败：{exc}") from exc


def pretty_print(data: Any) -> None:
    """以更易读的 JSON 形式打印数据。"""
    print(json.dumps(data, ensure_ascii=False, indent=2))
