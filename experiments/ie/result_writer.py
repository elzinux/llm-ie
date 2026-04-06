from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def make_safe_filename(name: str, max_length: int = 80) -> str:
    """将任意字符串转换为适合保存文件的安全文件名"""
    name = name.strip()
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", "_", name)
    name = name[:max_length].strip("_")
    return name or "untitled"


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，并返回目录路径对象"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_output_root_dir(
    input_dir: str | Path,
    use_correction: bool,
) -> Path:
    """根据输入目录构建实验输出根目录"""
    input_dir = Path(input_dir)
    suffix = "_ie_correction" if use_correction else "_ie_only"
    base_dir = Path("data") / "outputs" / f"{input_dir.name}{suffix}"
    candidate_dir = base_dir
    duplicate_index = 1

    while candidate_dir.exists():
        candidate_dir = Path(f"{base_dir}{duplicate_index}")
        duplicate_index += 1

    candidate_dir.mkdir(parents=True, exist_ok=False)
    return candidate_dir


def build_final_results_dir(output_root_dir: str | Path) -> Path:
    """构建最终抽取结果目录"""
    return ensure_dir(Path(output_root_dir) / "final_results")


def build_intermediate_results_dir(output_root_dir: str | Path) -> Path:
    """构建中间结果目录"""
    return ensure_dir(Path(output_root_dir) / "intermediate_results")


def build_result_file_path(
    output_dir: str | Path,
    md_stem: str,
) -> Path:
    """为单个 md 文件构造输出 json 路径"""
    output_dir = Path(output_dir)
    safe_stem = make_safe_filename(md_stem)
    return output_dir / f"{safe_stem}.json"


def save_final_results(
    output_path: str | Path,
    results: list[dict[str, Any]],
) -> None:
    """保存最终实验结果"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_intermediate_results(
    output_path: str | Path,
    results: list[dict[str, Any]],
) -> None:
    """保存中间实验结果"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
