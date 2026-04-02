import math

# __pycache__
# .venv
# data
# experiments
# src
# tests
# .env
# .gitignore
# .python-version
# main.py
# pyproject.toml
# README.md
# requirements.txt
# test.py
# utf8-test.txt
# uv.lock


def test_math_package() -> bool:
    """测试 `math` 标准库是否可以正常使用。"""
    number = 16
    result = math.isqrt(number)

    print(f"math.isqrt({number}) 的结果是 {result}")
    return result == 4


if __name__ == "__main__":
    if test_math_package():
        print("math 包可以正常使用。")
    else:
        print("math 包测试失败。")
