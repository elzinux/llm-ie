from pathlib import Path

from src.utils.json_reader import read_json
from src.utils.model_resources import (
    get_tokenizer,
)


def count_token(text: str) -> int | str:
    """
    测试 tokenizer 的分词计数功能。
    成功时返回 text 的 token 数量，失败时返回“加载模型失败”。
    """
    try:
        tokenizer = get_tokenizer()
        return len(tokenizer.encode(text))
    except Exception:
        return "加载模型失败"


def print_data(dic):
    for item in dic:
        chunck = item["chunck"]
        summary = item["summary"]
        accept = item["accept"]
        cnt_chunck = count_token(chunck)
        cnt_summary = count_token(summary)
        arr = item["mid_res"]
        cnt_w = cnt_s = 0
        for item1 in arr:
            cnt_s += len(item1["key_sentences"])
            cnt_w += len(item1["keywords"])
            score = item1["summ_score"]
        # print(f"关键词个数：{cnt_w}，关键句子个数：{cnt_s}，总数：{cnt_w + cnt_s}")
        print(
            cnt_chunck,
            cnt_summary,
            cnt_w,
            cnt_s,
            round(score[0], 2),
            round(score[1], 2),
            round(score[2], 2),
            int(accept),
        )


def read_json_files(folder_path: str) -> None:
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")

    if not folder.is_dir():
        raise ValueError(f"路径不是文件夹: {folder_path}")

    # 不递归，只读取当前目录下的 .json 文件
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == ".json":
            print(f"文件: {file_path.name}")
            items = read_json(str(file_path))
            print_data(items)


if __name__ == "__main__":
    print(
        "chunck_tokens",
        "summary_tokens",
        "keywords_cnt",
        "key_sentences_cnt",
        "information_coverage",
        "semantic_eval_score",
        "overall_score",
        "result",
    )
    dir = "data/outputs/md_files_summ"
    read_json_files(dir)
