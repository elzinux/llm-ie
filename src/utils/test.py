# test_model_resources.py

from __future__ import annotations

from src.utils.json_reader import read_json
from src.utils.model_resources import (
    get_embedding_model,
    get_spacy_model,
    get_tokenizer,
)


def test_spacy(text: str) -> list[str] | str:
    """
    测试 spaCy 分句功能。
    成功时返回分句结果列表，失败时返回“加载模型失败”。
    """
    try:
        nlp = get_spacy_model(ensure_sentence_boundaries=True)
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except Exception:
        return "加载模型失败"


def test_tokenizer(text: str) -> int | str:
    """
    测试 tokenizer 的分词计数功能。
    成功时返回 text 的 token 数量，失败时返回“加载模型失败”。
    """
    try:
        tokenizer = get_tokenizer()
        return len(tokenizer.encode(text))
    except Exception:
        return "加载模型失败"


def test_embedding(text: str) -> tuple[int, ...] | str:
    """
    测试 embedding 模型编码功能。
    成功时返回 embedding 的 shape，失败时返回“加载模型失败”。
    """
    try:
        model = get_embedding_model()
        embeddings = model.encode([text])
        return embeddings.shape
    except Exception:
        return "加载模型失败"


def main() -> None:

    json_dir = "data/outputs/test_summ_10md/10_summ.json"
    json_content = read_json(json_dir)
    # print(
    #     "chunck_tokens", "summary_tokens", "keywords_cnt", "key_sentences_cnt", "result"
    # )
    for item in json_content:
        text1 = item["chunck"]
        text2 = item["summary"]
        accept = item["accept"]
        cnt_chunck = test_tokenizer(text1)
        cnt_summary = test_tokenizer(text2)
        # print(
        #     f"chunck_tokens:{test_tokenizer(text1)}"
        #     + "---"
        #     + f"summary_tokens:{test_tokenizer(text2)}",
        #     accept,
        # )
        arr = item["mid_res"]
        cnt_w = cnt_s = 0
        for item1 in arr:
            cnt_s += len(item1["key_sentences"])
            cnt_w += len(item1["keywords"])
        # print(f"关键词个数：{cnt_w}，关键句子个数：{cnt_s}，总数：{cnt_w + cnt_s}")
        print(cnt_chunck, cnt_summary, cnt_w, cnt_s, int(accept))


if __name__ == "__main__":
    # text = ""
    # print(test_tokenizer(text))
    main()
