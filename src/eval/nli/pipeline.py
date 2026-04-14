from __future__ import annotations

from src.eval.nli.config import NLITextLengthConfig
from src.eval.nli.model import predict_nli
from src.eval.nli.preprocess import (
    count_words,
    is_text_within_ideal_length,
    match_hypothesis_to_premise_sentences,
    normalize_pair_texts,
    split_premise_and_hypothesis,
    split_text_into_sentences,
)
from src.eval.nli.schemas import (
    HypothesisSentenceMatch,
    NliLabel,
    NLIPairResult,
    NLIProbabilities,
    SentenceLevelNLIResult,
    TextSentence,
)


def score_nli_pair(
    premise_text: str,
    hypothesis_text: str,
    length_config: NLITextLengthConfig | None = None,
) -> NLIPairResult:
    """
    执行完整的 NLI 主流程，并返回最终 NLIPairResult。

    主流程包括：
    1. 规范化 premise_text 和 hypothesis_text
    2. 判断是否适合直接整段做 NLI
    3. 若适合，直接执行单次 NLI
    4. 若不适合，执行切句、top-k 选句、局部 premise 构造、句级 NLI
    5. 聚合全部 SentenceLevelNLIResult，得到最终三分类聚合分数
    """
    active_length_config = length_config or NLITextLengthConfig()
    normalized_premise_text, normalized_hypothesis_text = normalize_pair_texts(
        premise_text=premise_text,
        hypothesis_text=hypothesis_text,
    )

    if should_use_direct_pair_nli(
        premise_text=normalized_premise_text,
        hypothesis_text=normalized_hypothesis_text,
        length_config=active_length_config,
    ):
        nli_probabilities, _ = run_direct_pair_nli(
            premise_text=normalized_premise_text,
            hypothesis_text=normalized_hypothesis_text,
        )
        return build_direct_pair_result(
            premise_text=premise_text,
            hypothesis_text=hypothesis_text,
            normalized_premise_text=normalized_premise_text,
            normalized_hypothesis_text=normalized_hypothesis_text,
            nli_probabilities=nli_probabilities,
        )

    premise_sentences, hypothesis_sentences, sentence_matches = (
        build_sentence_level_matches(
            premise_text=normalized_premise_text,
            hypothesis_text=normalized_hypothesis_text,
        )
    )

    pair_result = build_initial_long_text_pair_result(
        premise_text=premise_text,
        hypothesis_text=hypothesis_text,
        normalized_premise_text=normalized_premise_text,
        normalized_hypothesis_text=normalized_hypothesis_text,
        premise_sentences=premise_sentences,
        hypothesis_sentences=hypothesis_sentences,
        sentence_matches=sentence_matches,
    )

    sentence_level_results = run_sentence_level_nli(
        sentence_matches=sentence_matches,
        length_config=active_length_config,
    )
    final_scores = aggregate_sentence_level_results(
        sentence_level_results=sentence_level_results,
    )

    pair_result.sentence_level_results = sentence_level_results
    pair_result.final_scores = final_scores
    return pair_result


def should_use_direct_pair_nli(
    premise_text: str,
    hypothesis_text: str,
    length_config: NLITextLengthConfig,
) -> bool:
    """
    判断当前文本对是否适合直接整段做 NLI。

    判定依据：
    - premise_text 的词数是否位于理想区间
    - hypothesis_text 的词数是否位于理想区间
    - premise_text 的句子数是否不超过 max_premise_sentences
    """
    if not premise_text or not hypothesis_text:
        return False

    premise_ok = is_text_within_ideal_length(
        text=premise_text,
        min_words=length_config.premise_ideal_min_words,
        max_words=length_config.premise_ideal_max_words,
    )
    hypothesis_word_count = count_words(hypothesis_text)
    hypothesis_ok = 0 < hypothesis_word_count <= length_config.premise_ideal_max_words
    premise_sentence_count = len(split_text_into_sentences(premise_text))

    return (
        premise_ok
        and hypothesis_ok
        and premise_sentence_count <= length_config.max_premise_sentences
    )


def run_direct_pair_nli(
    premise_text: str,
    hypothesis_text: str,
) -> tuple[NLIProbabilities, NliLabel]:
    """
    对整段 premise_text / hypothesis_text 直接执行一次 NLI。

    返回：
    - NLIProbabilities
    - nli_label
    """
    return predict_nli(
        premise_text=premise_text,
        hypothesis_text=hypothesis_text,
    )


def build_sentence_level_matches(
    premise_text: str,
    hypothesis_text: str,
) -> tuple[list[TextSentence], list[TextSentence], list[HypothesisSentenceMatch]]:
    """
    为长文本路径构造句级输入。

    处理步骤：
    - 切分 premise_text
    - 切分 hypothesis_text
    - 对每个 hypothesis 句匹配 top-5 premise 候选句

    返回：
    - premise_sentences
    - hypothesis_sentences
    - sentence_matches
    """
    premise_sentences, hypothesis_sentences = split_premise_and_hypothesis(
        premise_text=premise_text,
        hypothesis_text=hypothesis_text,
    )
    sentence_matches = match_hypothesis_to_premise_sentences(
        premise_sentences=premise_sentences,
        hypothesis_sentences=hypothesis_sentences,
        top_k=5,
    )
    return premise_sentences, hypothesis_sentences, sentence_matches


def select_premise_sentences_for_match(
    match: HypothesisSentenceMatch,
    length_config: NLITextLengthConfig,
) -> list[TextSentence]:
    """
    从单个 HypothesisSentenceMatch 的 top-5 候选句中，
    选出用于构造局部 premise_text 的若干 premise 句。

    规则：
    - 先按相似度从高到低决定纳入哪些句子
    - 目标是让最终 premise 尽量落在 20–60 词之间
    - 若第一句已经超过上限，也至少保留第一句
    - 返回值只表示“选中了哪些句子”，不负责最终拼接顺序
    """
    candidates = sorted(
        match.candidate_premise_sentences,
        key=lambda item: item.similarity,
        reverse=True,
    )
    if not candidates:
        return []

    min_words = length_config.premise_ideal_min_words
    max_words = length_config.premise_ideal_max_words

    selected_sentences: list[TextSentence] = []
    total_word_count = 0

    for candidate in candidates:
        sentence = candidate.sentence
        sentence_word_count = count_words(sentence.text)

        if not selected_sentences:
            selected_sentences.append(sentence)
            total_word_count = sentence_word_count
            if total_word_count >= min_words:
                break
            continue

        next_total_word_count = total_word_count + sentence_word_count
        if next_total_word_count <= max_words:
            selected_sentences.append(sentence)
            total_word_count = next_total_word_count
            if total_word_count >= min_words:
                break
            continue

        if total_word_count < min_words and _distance_to_word_range(
            next_total_word_count,
            min_words=min_words,
            max_words=max_words,
        ) < _distance_to_word_range(
            total_word_count,
            min_words=min_words,
            max_words=max_words,
        ):
            selected_sentences.append(sentence)

        break

    return selected_sentences


def compose_ordered_premise_text(
    premise_sentences: list[TextSentence],
) -> str:
    """
    将已选中的 premise 句按其在原文中的相对顺序拼接成局部 premise_text。

    说明：
    - 不按相似度顺序拼接
    - 必须按 sentence.index 升序拼接
    """
    if not premise_sentences:
        return ""

    ordered_sentences = sorted(premise_sentences, key=lambda sentence: sentence.index)

    sentence_texts: list[str] = []
    seen_indexes: set[int] = set()

    for sentence in ordered_sentences:
        if sentence.index in seen_indexes:
            continue
        seen_indexes.add(sentence.index)

        sentence_text = sentence.text.strip()
        if sentence_text:
            sentence_texts.append(sentence_text)

    return " ".join(sentence_texts)


def build_local_premise_text_for_match(
    match: HypothesisSentenceMatch,
    length_config: NLITextLengthConfig,
) -> str:
    """
    为单个 hypothesis 句构造局部 premise_text。

    内部流程：
    1. 从 top-5 候选句中选择若干 premise 句
    2. 将这些句子按原文顺序拼接
    3. 返回最终局部 premise_text
    """
    selected_premise_sentences = select_premise_sentences_for_match(
        match=match,
        length_config=length_config,
    )
    return compose_ordered_premise_text(selected_premise_sentences)


def run_sentence_level_nli(
    sentence_matches: list[HypothesisSentenceMatch],
    length_config: NLITextLengthConfig,
) -> list[SentenceLevelNLIResult]:
    """
    对全部 hypothesis 句执行句级 NLI。

    处理步骤：
    - 遍历每个 hypothesis 句对应的匹配结果
    - 构造该句对应的局部 premise_text
    - 使用局部 premise_text 与当前 hypothesis 句执行一次 NLI
    - 生成 SentenceLevelNLIResult 列表
    """
    results: list[SentenceLevelNLIResult] = []

    for match in sentence_matches:
        local_premise_text = build_local_premise_text_for_match(
            match=match,
            length_config=length_config,
        )

        if local_premise_text:
            nli_probs, nli_label = predict_nli(
                premise_text=local_premise_text,
                hypothesis_text=match.hypothesis_sentence.text,
            )
        else:
            nli_probs = NLIProbabilities()
            nli_label = "neutral"

        results.append(
            SentenceLevelNLIResult(
                premise_sentence=TextSentence(
                    text=local_premise_text,
                    index=_resolve_local_premise_index(match),
                ),
                hypothesis_sentence=match.hypothesis_sentence,
                nli_probs=nli_probs,
                nli_label=nli_label,
            )
        )

    return results


def aggregate_sentence_level_results(
    sentence_level_results: list[SentenceLevelNLIResult],
) -> NLIProbabilities:
    """
    根据全部 SentenceLevelNLIResult 聚合最终结果。

    返回：
    - NLIProbabilities

    说明：
    - 这是最终统计函数
    - 后续若增加 numeric / polarity 等模块，应直接扩展 SentenceLevelNLIResult，
      再由本函数统一读取这些字段并完成最终打分
    """
    if not sentence_level_results:
        return NLIProbabilities()

    result_count = len(sentence_level_results)
    contradiction = sum(
        result.nli_probs.contradiction for result in sentence_level_results
    ) / result_count
    entailment = sum(
        result.nli_probs.entailment for result in sentence_level_results
    ) / result_count
    neutral = sum(result.nli_probs.neutral for result in sentence_level_results) / (
        result_count
    )

    return NLIProbabilities(
        contradiction=float(contradiction),
        entailment=float(entailment),
        neutral=float(neutral),
    )


def build_direct_pair_result(
    premise_text: str,
    hypothesis_text: str,
    normalized_premise_text: str,
    normalized_hypothesis_text: str,
    nli_probabilities: NLIProbabilities,
) -> NLIPairResult:
    """
    为直接整段 NLI 路径构造最终 NLIPairResult。
    """
    return NLIPairResult(
        premise_text=premise_text,
        hypothesis_text=hypothesis_text,
        normalized_premise_text=normalized_premise_text,
        normalized_hypothesis_text=normalized_hypothesis_text,
        final_scores=nli_probabilities,
    )


def build_initial_long_text_pair_result(
    premise_text: str,
    hypothesis_text: str,
    normalized_premise_text: str,
    normalized_hypothesis_text: str,
    premise_sentences: list[TextSentence],
    hypothesis_sentences: list[TextSentence],
    sentence_matches: list[HypothesisSentenceMatch],
) -> NLIPairResult:
    """
    为长文本路径构造初始 NLIPairResult，
    先写入切句和匹配结果，最终分数与标签由后续聚合函数补齐。
    """
    return NLIPairResult(
        premise_text=premise_text,
        hypothesis_text=hypothesis_text,
        normalized_premise_text=normalized_premise_text,
        normalized_hypothesis_text=normalized_hypothesis_text,
        premise_sentences=premise_sentences,
        hypothesis_sentences=hypothesis_sentences,
        sentence_matches=sentence_matches,
    )


def _resolve_local_premise_index(match: HypothesisSentenceMatch) -> int:
    if match.candidate_premise_sentences:
        return min(
            candidate.sentence.index for candidate in match.candidate_premise_sentences
        )
    return match.hypothesis_sentence.index


def _distance_to_word_range(
    word_count: int,
    min_words: int,
    max_words: int,
) -> int:
    if min_words <= word_count <= max_words:
        return 0
    if word_count < min_words:
        return min_words - word_count
    return word_count - max_words
