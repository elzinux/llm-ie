from __future__ import annotations

from pathlib import Path


def load_title_abstract_pairs(md_path: str | Path) -> list[dict[str, str]]:
    """读取单个 Markdown 文件，并解析出标题与摘要对。"""
    md_path = Path(md_path)
    content = md_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    samples: list[dict[str, str]] = []
    current_title: str | None = None
    current_abstract_lines: list[str] = []

    def flush_current() -> None:
        """将当前缓存的标题与摘要写入结果列表。"""
        nonlocal current_title, current_abstract_lines, samples

        if current_title is None:
            return

        abstract = "\n".join(current_abstract_lines).strip()
        if not abstract:
            return

        samples.append(
            {
                "title": current_title.lstrip("#").strip(),
                "abstract": abstract,
            }
        )

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("#"):
            flush_current()
            current_title = stripped
            current_abstract_lines = []
        else:
            if current_title is not None:
                current_abstract_lines.append(line)

    flush_current()
    return samples


def list_md_files(input_dir: str | Path) -> list[Path]:
    """列出指定目录当前层级下的全部 Markdown 文件。"""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")

    return sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".md"
        ]
    )
