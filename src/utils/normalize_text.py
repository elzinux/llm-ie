import re


def normalize_text(text: str) -> str:
    """
    用于鲁棒匹配的轻量归一化：
    - 转小写
    - 非字母数字字符替换为空格
    - 合并多余空格
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
