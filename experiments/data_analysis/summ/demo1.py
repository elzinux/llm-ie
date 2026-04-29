import csv
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
            ",",
            cnt_summary,
            ",",
            cnt_w,
            ",",
            cnt_s,
            ",",
            round(score[0], 2),
            ",",
            round(score[1], 2),
            ",",
            round(score[2], 2),
            ",",
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


def read_csv(file_path):
    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        cnt1 = dict()  # information_coverage
        cnt2 = dict()  # auxiliary_score
        cnt3 = dict()  # overall_score
        length = 0
        for i, item in enumerate(reader):
            length += 1
            if i == 0:
                continue
            a = item[-4]
            b = item[-3]
            c = item[-2]
            cnt1[a] = cnt1.get(a, 0) + 1
            cnt2[b] = cnt2.get(b, 0) + 1
            cnt3[c] = cnt3.get(c, 0) + 1
        return length, cnt1, cnt2, cnt3


def print_cnt(dic: dict):
    # 0-0.1, 0.1-0.2, 0.2-0.3, 0.3-0.4, 0.4-0.5, 0.5-0.6, 0.6-0.7, 0.7-0.75, 0.75-0.8, 0.8-0.9, 0.9-1
    cnt = [0] * 11
    for k in dic:
        t = k
        k = float(k)
        if k >= 0.9:
            cnt[-1] += dic[t]
        elif k >= 0.8:
            cnt[-2] += dic[t]
        elif k >= 0.75:
            cnt[-3] += dic[t]
        elif k >= 0.7:
            cnt[-4] += dic[t]
        elif k >= 0.6:
            cnt[-5] += dic[t]
        elif k >= 0.5:
            cnt[-6] += dic[t]
        elif k >= 0.4:
            cnt[-7] += dic[t]
        elif k >= 0.3:
            cnt[-8] += dic[t]
        elif k >= 0.2:
            cnt[-9] += dic[t]
        elif k >= 0.1:
            cnt[-10] += dic[t]
        elif k >= 0.0:
            cnt[-11] += dic[t]
    print(cnt)


def count(file_path):
    length, cnt1, cnt2, cnt3 = read_csv(file_path)
    print(length - 1)
    print("-" * 5)
    print_cnt(cnt1)
    print("-" * 5)
    print_cnt(cnt2)
    print("-" * 5)
    print_cnt(cnt3)


if __name__ == "__main__":
    csv_file_path = "data/outputs/md_files_summ/log.csv"

    count(csv_file_path)

    # md_files_path = "data/outputs/md_files_summ/"
    # print(
    #     "chunck_tokens",
    #     "summary_tokens",
    #     "keywords_cnt",
    #     "key_sentences_cnt",
    #     "information_coverage",
    #     "auxiliary_score",
    #     "overall_score",
    #     "result",
    # )
    # read_json_files(md_files_path)
