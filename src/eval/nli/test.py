import os
from pathlib import Path

from src.eval.nli.pipeline import score_nli_pair
from src.utils.json_reader import read_json


def main():
    folder_path = "data/processed/e4-8-md_files_summ"
    res_file = "tests/log.txt"
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")

    if not folder.is_dir():
        raise ValueError(f"路径不是文件夹: {folder_path}")

    # 不递归，只读取当前目录下的 .json 文件
    file_cnt = 0
    res = dict()
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == ".json":
            file_cnt += 1
            print(file_path)
            items = read_json(str(file_path))

            res[file_path] = []
            for idx, item in enumerate(items):
                premise_text = item["chunck"]
                hypothesis_text = item["summary"]
                init_scores = item["mid_res"][0]["summ_score"]
                accept = item["accept"]

                result = score_nli_pair(
                    premise_text,
                    hypothesis_text,
                )

                n1 = len(premise_text)
                n2 = len(hypothesis_text)
                n3 = len(result.sentence_level_results)
                res[file_path].append(str(n1) + "_" + str(n2) + "_" + str(n3))

                os.makedirs(os.path.dirname(res_file), exist_ok=True)
                with open(res_file, "a", encoding="utf-8") as f:
                    print(
                        file_path,
                        file_cnt,
                        idx,
                        int(accept),
                        init_scores[0],
                        init_scores[1],
                        init_scores[2],
                        result.final_scores.contradiction,
                        result.final_scores.entailment,
                        result.final_scores.neutral,
                        file=f,
                    )
                # print(
                #     file_path,
                #     file_cnt,
                #     idx,
                #     int(accept),
                #     init_scores[0],
                #     init_scores[1],
                #     init_scores[2],
                #     result.final_scores.contradiction,
                #     result.final_scores.entailment,
                #     result.final_scores.neutral,
                # )
    for k in res:
        print(k, res[k])


if __name__ == "__main__":
    print("-=-" * 5)
    main()
