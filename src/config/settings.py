from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

API_KEY_ALI = os.getenv("API_KEY_ALI", "")

# 如果需要强制要求环境变量存在，可以取消下面这段注释。
# if not API_KEY_ALI:
#     raise ValueError("环境变量 API_KEY_ALI 未设置。")
