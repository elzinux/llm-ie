from dataclasses import dataclass, field
from typing import Literal

# ---------------------------
# 类型别名
# ---------------------------
# 这些 Literal 用于约束各阶段输出的合法取值，
# 便于后续流程保持统一。

NliLabel = Literal["entailment", "contradiction", "neutral"]

NumericStatus = Literal["match", "conflict", "none", "unknown"]


@dataclass
class NLIProbabilities:
    """
    NLI 模型对三分类标签的概率输出。

    字段说明：
    - contradiction: 矛盾概率
    - entailment: 蕴含概率
    - neutral: 中性概率
    """

    contradiction: float = 0.0
    entailment: float = 0.0
    neutral: float = 0.0


@dataclass
class NumericMatchResult:
    """
    数字一致性检查结果。

    字段说明：
    - premise_numbers:
        从 premise_text 中提取到的显式数字表达。
    - hypothesis_numbers:
        从 hypothesis_text 中提取到的显式数字表达。
    - numeric_status:
        数字检查结论：
        - match: 数字一致
        - conflict: 数字冲突
        - none: hypothesis 中无显式数字
        - unknown: 当前规则无法稳定判断
    """

    premise_numbers: list[str] = field(default_factory=list)
    hypothesis_numbers: list[str] = field(default_factory=list)
    numeric_status: NumericStatus = "none"


@dataclass
class TextSentence:
    """
    文本切句后的单个句子对象。

    字段说明：
    - text:
        当前句子的原始文本内容。
    - index:
        该句子在原文本中的顺序索引，从 0 开始。
    """

    text: str
    index: int


@dataclass
class RetrievedPremiseSentence:
    """
    为某个 hypothesis 句检索到的候选 premise 句。

    字段说明：
    - sentence: 检索到的 premise 句子。
    - similarity: 当前 premise 句与目标 hypothesis 句之间的相关度分数。
    """

    sentence: TextSentence
    similarity: float = 0.0


@dataclass
class HypothesisSentenceMatch:
    """
    单个 hypothesis 句及其对应的 premise 候选句集合。

    字段说明：
    - hypothesis_sentence:
        当前待验证的 hypothesis 句。
    - candidate_premise_sentences:
        为该 hypothesis 句检索得到的 top-k premise 候选句。
    """

    hypothesis_sentence: TextSentence
    candidate_premise_sentences: list[RetrievedPremiseSentence] = field(
        default_factory=list
    )


@dataclass
class SentenceLevelNLIResult:
    """
    单个句子级 NLI 判定结果。

    字段说明：
    - premise_sentence:
        当前参与 NLI 判定的 premise 句。
    - hypothesis_sentence:
        当前参与 NLI 判定的 hypothesis 句。
    - nli_probs:
        NLI 模型三分类概率输出。
    - nli_label:
        当前句对的 NLI 主判定标签。
    """

    premise_sentence: TextSentence
    hypothesis_sentence: TextSentence
    nli_probs: NLIProbabilities = field(default_factory=NLIProbabilities)
    nli_label: NliLabel = "neutral"


@dataclass
class NLIPairResult:
    """
    单个 premise_text / hypothesis_text 的完整 NLI 支持判定结果。

    字段说明：
    - premise_text:
        原始 premise 文本。
    - hypothesis_text:
        原始 hypothesis 文本。
    - normalized_premise_text:
        规范化后的 premise 文本。
    - normalized_hypothesis_text:
        规范化后的 hypothesis 文本。
    - premise_sentences:
        premise_text 切句后的句子列表。
    - hypothesis_sentences:
        hypothesis_text 切句后的句子列表。
    - sentence_matches:
        对每个 hypothesis 句检索得到的 top-k premise 候选句。
    - sentence_level_results:
        句子级 premise / hypothesis 配对后的 NLI 结果列表。
    - final_scores:
        整体文本对最终聚合后的 NLI 三分类分数。
    """

    premise_text: str
    hypothesis_text: str

    normalized_premise_text: str = ""
    normalized_hypothesis_text: str = ""

    premise_sentences: list[TextSentence] = field(default_factory=list)
    hypothesis_sentences: list[TextSentence] = field(default_factory=list)

    sentence_matches: list[HypothesisSentenceMatch] = field(default_factory=list)
    sentence_level_results: list[SentenceLevelNLIResult] = field(default_factory=list)

    final_scores: NLIProbabilities = field(default_factory=NLIProbabilities)
