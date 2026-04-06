import re

# =========================
# 基础正则片段
# =========================

YEAR_RE = r"(?:19|20)\d{2}[a-z]?"
# 修改：放宽作者姓名限制，允许空格和小写前缀（如 van der）
AUTHOR_TOKEN_RE = r"(?:[A-Za-z'`-]+(?:\s+[A-Za-z'`-]+)*)"
ETAL_RE = r"(?:et\s+al\.?)"


# =========================
# 1. 文献引用删除
# =========================

NUMERIC_CITATION_RE = re.compile(r"\[(?:\s*\d+\s*(?:[-–,]\s*\d+\s*)*)\]")

# 优化：采用更宽泛的匹配策略。匹配带有“典型引用特征（年份）”的括号。
# 支持前缀 (e.g., Smith, 2020) 和后缀 (Smith, 2020, p. 12)
PAREN_AUTHOR_YEAR_RE = re.compile(
    rf"""
    \(
        [^)]*?                   # 允许年份前有任意非括号字符
        \b{YEAR_RE}\b            # 必须包含年份
        [^)]*?                   # 允许年份后有任意非括号字符
    \)
    """,
    re.VERBOSE,
)

# 优化：使用捕获组 (group 1) 抓取作者部分，以便保留主语
NARRATIVE_CITATION_RE = re.compile(
    rf"""
    (                                # Group 1: 捕获作者部分
        \b
        {AUTHOR_TOKEN_RE}
        (?:,\s*{AUTHOR_TOKEN_RE})*   # 支持多个作者 (A, B, and C)
        (?:\s+(?:and|&)\s+{AUTHOR_TOKEN_RE})?
        (?:\s+{ETAL_RE})?
    )
    \s*
    \(\s*{YEAR_RE}\s*\)              # 仅匹配括号内的年份
    """,
    re.VERBOSE,
)


# =========================
# 2. 仅删除括号中的图/表/公式/章节引用
# =========================

PAREN_NON_TEXT_REF_RE = re.compile(
    r"""
    \(
        \s*
        (?:
            Fig(?:\.|ure)? |
            Table |
            Tab(?:\.)? |
            Eq(?:\.|uation)? |
            Sec(?:\.|tion)?
        )
        \s*
        [A-Za-z0-9_.\-–]+
        (?:\s*[A-Za-z0-9_.\-–]+)*
        \s*
    \)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# =========================
# 3. 工具函数
# =========================


def _cleanup_text(text: str) -> str:
    """
    清理删除后留下的空格、残余标点、空括号等。
    """
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\[\s*\]", "", text)

    # 清理标点前多余空格
    text = re.sub(r"\s+([,;:.!?])", r"\1", text)

    # 清理重复标点
    text = re.sub(r"([,;])\s*([,;])+", r"\1", text)
    text = re.sub(r"\.\s*\.", ".", text)

    # 处理括号删除后留下的多余空格
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)

    # 处理句首残留逗号/分号
    text = re.sub(r"^\s*[,;:]\s*", "", text)

    # 删除多余空格
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


# =========================
# 4. 主接口
# =========================


def clean_academic_paragraph(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    if not text.strip():
        return ""

    cleaned = text

    # 1. 删除数字型引用 [12], [3,5]
    cleaned = NUMERIC_CITATION_RE.sub("", cleaned)

    # 2. 删除括号中的图/表/公式/章节引用
    cleaned = PAREN_NON_TEXT_REF_RE.sub("", cleaned)

    # 3. 处理叙述型作者-年份引用：保留作者，删除括号和年份
    # 例如：Smith et al. (2020) -> Smith et al.
    cleaned = NARRATIVE_CITATION_RE.sub(r"\1", cleaned)

    # 4. 删除括号型引用 (Smith, 2020)
    # 因为 PAREN_AUTHOR_YEAR_RE 现在是通过判断括号内是否包含年份来操作，
    # 它会自动避开 (PRD), (WUE) 这种纯字母缩写，因此不需要额外的 _should_keep_parenthetical 函数。
    cleaned = PAREN_AUTHOR_YEAR_RE.sub("", cleaned)

    # 5. 清理残余格式
    cleaned = _cleanup_text(cleaned)

    return cleaned


if __name__ == "__main__":
    examples = [
        "Deficit irrigation improved yield stability (Smith et al., 2020) under semi-arid conditions.",
        "Partial root-zone drying (PRD) significantly improved water use efficiency (WUE).",
        "The results were consistent with previous studies [12, 15, 18].",
        "Figure 2 shows the relationship between yield and irrigation level.",
        "As shown in Table 1, PRD increased WUE.",
        "Smith et al. (2020) reported that PRD improved tomato yield.",  # 叙述型引用测试
        "The treatment was applied at flowering stage (Figure 3) and harvest was recorded weekly.",
        "The effect was minimal (e.g., van Dijk, 2019, p. 15).",  # 边缘测试：带前缀后缀和欧洲小写姓氏
        "Several authors agree (Smith, Jones, & Brown, 2021; Doe, 2018).",  # 边缘测试：多作者
    ]

    for i, example in enumerate(examples, start=1):
        print(f"[Example {i}]")
        print("Input : ", example)
        print("Output: ", clean_academic_paragraph(example))
        print("-" * 60)
