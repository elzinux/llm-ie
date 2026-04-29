from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.json_reader import read_json


def list_json_files(input_dir: str | Path) -> list[Path]:
    """Return non-recursive JSON files under the input directory."""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    if not input_dir.is_dir():
        raise ValueError(f"输入路径不是目录: {input_dir}")

    return sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".json"
        ]
    )


def load_title_summary_pairs(json_path: str | Path) -> list[dict[str, str]]:
    """
    Load one summary result JSON file and convert it to IE samples.

    Each summary item is mapped to:
    - title: 1-based index within the file
    - text: summary
    """
    json_path = Path(json_path)
    data = read_json(str(json_path))

    if not isinstance(data, list):
        raise ValueError(f"JSON 顶层必须是列表: {json_path}")

    samples: list[dict[str, str]] = []

    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue

        summary = item.get("summary", "")
        if not isinstance(summary, str):
            summary = str(summary or "")

        summary = summary.strip()
        if not summary:
            continue

        samples.append(
            {
                "title": str(index),
                "text": summary,
            }
        )

    return samples
