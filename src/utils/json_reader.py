import json
from pathlib import Path
from typing import Any


def read_json(file_path: str) -> Any:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if not path.is_file():
        raise ValueError(f"路径不是文件: {file_path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 格式错误: {e}") from e


def read_json_files_in_dir(folder_path: str) -> dict[str, Any]:
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")

    if not folder.is_dir():
        raise ValueError(f"路径不是文件夹: {folder_path}")

    result: dict[str, Any] = {}

    # 不递归，只读取当前目录下的 .json 文件
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == ".json":
            result[file_path.name] = read_json(str(file_path))

    return result


if __name__ == "__main__":
    folder_path = "./json_data"
    data = read_json_files_in_dir(folder_path)
    print(data)
