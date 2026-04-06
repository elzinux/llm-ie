from __future__ import annotations

import os
import re
from typing import List, Optional

from experiments.summ.readers.text_cleaner import clean_academic_paragraph
from src.summ.schemas import InputDocument, InputItem

# =========================
# 标题相关正则与配置
# =========================

HEADING_RE = re.compile(r"^\s*#+\s*(.*?)\s*$")
FIGURE_IMAGE_RE = re.compile(r"^\s*!\[\]\(images/.*?\)\s*$")
TABLE_CAPTION_RE = re.compile(r"^\s*table\s+\d+\b", re.IGNORECASE)

# 用于判断段落是否自然结束
PARAGRAPH_END_RE = re.compile(r'[.!?]["\')\]]?\s*$')

# 用于提取一行中第一个字母字符，判断其大小写
FIRST_ALPHA_RE = re.compile(r"[A-Za-z]")

# 提取标题编号：
# 3.
# 3.1.
# 2.4.1
HEADING_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*\.?)\s*(.*)$")


# =========================
# 标题规范化
# =========================


def _normalize_heading_text(raw_heading: str) -> str:
    """
    标题保存时的规范化：
    1. 去首尾空白
    2. 全部转小写
    3. 压缩连续空白为一个空格
    """
    heading = raw_heading.strip().lower()
    heading = re.sub(r"\s+", " ", heading)
    return heading


def _canonical_heading_key(raw_heading: str) -> str:
    """
    用于标题判定的强规范化：
    1. 转小写
    2. 去掉所有非字母数字字符

    例如：
    - 'A B S T R A C T' -> 'abstract'
    - 'Acknowledgements' -> 'acknowledgements'
    """
    heading = raw_heading.strip().lower()
    heading = re.sub(r"[^a-z0-9]+", "", heading)
    return heading


def _is_start_heading(raw_heading: str) -> bool:
    """
    起始标题判定：
    - abstract
    - introduction
    """
    key = _canonical_heading_key(raw_heading)
    return key in {"abstract", "introduction"}


def _is_end_heading(raw_heading: str) -> bool:
    """
    结束标题判定。
    """
    key = _canonical_heading_key(raw_heading)

    end_keys = {
        "acknowledgements",
        "acknowledgment",
        "acknowledgments",
        "acknowledgement",
        "references",
        "reference",
        "bibliography",
        "funding",
        "declarations",
        "declaration",
        "conflictofinterest",
        "conflictsofinterest",
        "competinginterests",
        "availabilityofdataandmaterials",
        "dataavailability",
        "appendix",
        "appendices",
        "supplementarymaterials",
        "supplementarymaterial",
    }

    return key in end_keys


def _split_heading_number_and_text(
    normalized_heading: str,
) -> tuple[Optional[str], str]:
    """
    从规范化标题中提取编号与正文。

    示例：
    - '3. data and methodology' -> ('3.', 'data and methodology')
    - '3.1. data' -> ('3.1.', 'data')
    - 'abstract' -> (None, 'abstract')
    """
    match = HEADING_NUMBER_RE.match(normalized_heading)
    if not match:
        return None, normalized_heading

    number = match.group(1).strip()
    text = match.group(2).strip()
    return number, text


def _is_child_heading(parent_heading: str, child_heading: str) -> bool:
    """
    判断 child_heading 是否是 parent_heading 的子标题。
    """
    parent_num, _ = _split_heading_number_and_text(parent_heading)
    child_num, _ = _split_heading_number_and_text(child_heading)

    if not parent_num or not child_num:
        return False

    parent_core = parent_num.rstrip(".")
    child_core = child_num.rstrip(".")

    if parent_core == child_core:
        return False

    return child_core.startswith(parent_core + ".")


def _update_title_parts(
    current_title_parts: List[str],
    new_heading: str,
) -> List[str]:
    """
    更新标题链。

    规则：
    1. 默认遇到新标题时，清空之前的标题，只保留当前标题
    2. 只有当新标题是当前标题的子标题时，才进行拼接
    """
    if not current_title_parts:
        return [new_heading]

    last_heading = current_title_parts[-1]

    if _is_child_heading(last_heading, new_heading):
        return [last_heading, new_heading]

    return [new_heading]


# =========================
# 行类型判断
# =========================


def _extract_heading(line: str) -> Optional[str]:
    """
    如果当前行是 Markdown 标题，返回标题文本；否则返回 None。
    """
    match = HEADING_RE.match(line)
    if not match:
        return None
    return match.group(1).strip()


def _is_blank_line(line: str) -> bool:
    return not line.strip()


def _is_figure_image_line(line: str) -> bool:
    """
    MinerU 图片行示例：
    ![](images/xxxx.jpg)
    """
    return bool(FIGURE_IMAGE_RE.match(line))


def _is_table_caption_line(line: str) -> bool:
    """
    表格说明行示例：
    Table 1 Genetic coefficients of local wheat, summer maize and early maize.
    """
    return bool(TABLE_CAPTION_RE.match(line))


def _first_alpha_is_upper(text: str) -> bool:
    """
    找到文本中第一个英文字母，判断是否为大写。
    没有英文字母则返回 False。
    """
    match = FIRST_ALPHA_RE.search(text)
    if not match:
        return False
    return match.group(0).isupper()


def _paragraph_ends(text: str) -> bool:
    """
    当前段落是否以句末标点结束。
    """
    return bool(PARAGRAPH_END_RE.search(text.strip()))


# =========================
# 文本预处理
# =========================


def _clean_line_text(line: str) -> str:
    """
    对单行文本做轻量清洗：
    - 去首尾空白
    - 合并多余空格
    """
    text = line.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _finalize_paragraph(paragraph: str) -> str:
    """
    对单个段落做最终清洗。
    返回清洗后的段落文本；若为空则返回空字符串。
    """
    paragraph = paragraph.strip()
    if not paragraph:
        return ""

    cleaned = clean_academic_paragraph(paragraph).strip()
    return cleaned


def _flush_section(
    items: List[InputItem],
    current_title_parts: List[str],
    current_section_paragraphs: List[str],
) -> None:
    """
    将当前章节缓存的所有段落写入一个 InputItem。
    """
    if not current_section_paragraphs:
        return

    current_title = "_ ".join(current_title_parts) if current_title_parts else "unknown"
    section_text = "\n\n".join(current_section_paragraphs).strip()

    if not section_text:
        return

    items.append(InputItem(title=current_title, text=section_text))


# =========================
# 主读取逻辑
# =========================


def read_md_file(
    file_path: str,
    max_drop_titles: int = 2,
) -> InputDocument:
    """
    读取 MinerU 生成的 Markdown 文件，并组织为 InputDocument。

    返回规则：
    - 一个章节标题对应一个 InputItem
    - InputItem.text 为该章节下所有段落拼接后的完整文本
    - 段落之间使用两个换行分隔

    规则概述：
    1. 从标题 abstract / introduction 开始读取；
       如果始终未命中，且已丢弃标题数达到 max_drop_titles，则从当前标题开始读取。
    2. 遇到 acknowledgements / references 及其常见变体后终止读取。
    3. 标题统一小写；默认遇到新标题时重置，
       只有在明显父子标题关系下才使用 '_ ' 拼接。
    4. 段落按 MinerU 的文本块行进行拼接：
       - 当前段落若未以 .?! 结束，且下一行不是以大写字母开头，则视为同一段落，使用一个空格拼接。
       - 否则视为新段落。
    5. 每个段落写入章节前调用 clean_academic_paragraph 清洗。
    6. 跳过图片行及其下一行图片说明；跳过表格说明行。
    """
    file_name = os.path.basename(file_path)
    items: List[InputItem] = []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    started = False
    dropped_title_count = 0

    current_title_parts: List[str] = []
    current_paragraph: Optional[str] = None
    current_section_paragraphs: List[str] = []

    i = 0
    while i < len(lines):
        raw_line = lines[i].rstrip("\n")
        line = _clean_line_text(raw_line)

        if _is_blank_line(line):
            i += 1
            continue

        # 图片：跳过当前行和下一行图片说明
        if _is_figure_image_line(line):
            i += 2
            continue

        # 表格说明：整行跳过
        if _is_table_caption_line(line):
            i += 1
            continue

        # 标题处理
        heading = _extract_heading(line)
        if heading is not None:
            normalized_heading = _normalize_heading_text(heading)

            # 终止标题：先收尾当前段落，再收尾当前章节，然后退出
            if _is_end_heading(heading):
                if current_paragraph:
                    cleaned_paragraph = _finalize_paragraph(current_paragraph)
                    if cleaned_paragraph:
                        current_section_paragraphs.append(cleaned_paragraph)
                    current_paragraph = None

                _flush_section(items, current_title_parts, current_section_paragraphs)
                current_section_paragraphs = []
                break

            # 尚未开始读取正文
            if not started:
                if _is_start_heading(heading):
                    started = True
                    current_title_parts = [normalized_heading]
                else:
                    dropped_title_count += 1
                    if dropped_title_count >= max_drop_titles:
                        started = True
                        current_title_parts = [normalized_heading]

                i += 1
                continue

            # 已开始读取正文：遇到新标题时，先收尾当前段落，再收尾当前章节
            if current_paragraph:
                cleaned_paragraph = _finalize_paragraph(current_paragraph)
                if cleaned_paragraph:
                    current_section_paragraphs.append(cleaned_paragraph)
                current_paragraph = None

            _flush_section(items, current_title_parts, current_section_paragraphs)
            current_section_paragraphs = []

            current_title_parts = _update_title_parts(
                current_title_parts, normalized_heading
            )

            i += 1
            continue

        # 未开始正文时，普通文本一律跳过
        if not started:
            i += 1
            continue

        if not current_title_parts:
            current_title_parts = ["unknown"]

        # 正文候选文本
        if current_paragraph is None:
            current_paragraph = line
            i += 1
            continue

        # 当前已有段落，判断是否拼接
        if not _paragraph_ends(current_paragraph):
            if not _first_alpha_is_upper(line):
                current_paragraph = current_paragraph + " " + line
            else:
                cleaned_paragraph = _finalize_paragraph(current_paragraph)
                if cleaned_paragraph:
                    current_section_paragraphs.append(cleaned_paragraph)
                current_paragraph = line
        else:
            cleaned_paragraph = _finalize_paragraph(current_paragraph)
            if cleaned_paragraph:
                current_section_paragraphs.append(cleaned_paragraph)
            current_paragraph = line

        i += 1

    # 文件结束时收尾
    if current_paragraph:
        cleaned_paragraph = _finalize_paragraph(current_paragraph)
        if cleaned_paragraph:
            current_section_paragraphs.append(cleaned_paragraph)

    _flush_section(items, current_title_parts, current_section_paragraphs)

    return InputDocument(
        file_name=file_name,
        items=items,
    )


if __name__ == "__main__":
    md_file = "data/raw/test/3.md"
    res = read_md_file(md_file)
    for x in res.items:
        print(x.title)
        print("-" * 5)
        print(x.text)
        print("-" * 10)
