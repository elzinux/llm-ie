from pathlib import Path

from src.utils.json_reader import read_json


def print_data(items):
    cnt = [0] * 5
    cnt0 = 0
    for dic in items:
        arr = dic["initial_verification"]
        for a in arr:
            if (
                a["intervention_or_policy"] == ""
                and a["location"] == ""
                and a["crop"] == ""
                and a["impact"] == ""
            ):
                continue
                # if a["verification_score"] != 0:
                #     print("空记录分数不为 0")
            else:
                s = a["verification_score"]
                cnt[s - 1] += 1
    # print(cnt0)
    # return cnt0
    print(cnt)


def read_json_files(folder_path: str) -> None:
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")

    if not folder.is_dir():
        raise ValueError(f"路径不是文件夹: {folder_path}")

    # 不递归，只读取当前目录下的 .json 文件
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == ".json":
            # print(f"文件: {file_path.name}")
            items = read_json(str(file_path))
            print_data(items)


if __name__ == "__main__":
    json_dir1 = "data/outputs/e4-16-md_files_summ_ie_correction/intermediate_results"
    json_dir2 = "data/outputs/data_abstract1_ie_correction/intermediate_results"
    read_json_files(json_dir2)
