from dataclasses import dataclass


@dataclass
class NLITextLengthConfig:
    """
    NLI 输入文本长度控制配置。

    设计原则：
    - hypothesis_text 应尽量短，最好只表达一个原子事实或单一判断。
    - premise_text 保留 hypothesis 最相关的局部证据，不是整段或整章文本。

    字段说明：
    - premise_ideal_min_words: premise_text 的理想最小词数。
    - premise_ideal_max_words: premise_text 的理想最大词数。
    - max_premise_sentences: premise_text 建议保留的最大句子数。

    """

    premise_ideal_min_words: int = 20
    premise_ideal_max_words: int = 60

    max_premise_sentences: int = 5
