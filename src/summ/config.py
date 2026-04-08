from dataclasses import dataclass, field

from src.utils.model_resources import DEFAULT_MODEL_RESOURCE_CONFIG, ModelResourceConfig

# summ 模块当前选择使用的模型资源配置。
# 默认直接复用公共层提供的全局默认资源配置；若后续需要覆盖，
# 只应在这里替换为新的 ModelResourceConfig，而不是把底层定义复制回来。
MODEL_RESOURCE_CONFIG: ModelResourceConfig = DEFAULT_MODEL_RESOURCE_CONFIG


@dataclass
class SplitConfig:
    """
    第 2 步：句子切分配置。

    参数说明：
    - spacy_model_name: 英文句子切分使用的 spaCy 模型名称
    - tokenizer_name: token 计数器名称
    - min_sentence_chars: 过滤过短句子的最小字符数
    """

    spacy_model_name: str = MODEL_RESOURCE_CONFIG.spacy_model_name
    tokenizer_name: str = MODEL_RESOURCE_CONFIG.tokenizer_name
    min_sentence_chars: int = 5


@dataclass
class SelectionConfig:
    """
    第 3 步：句子筛选配置。

    设计原则：
    这一阶段应适当宽松，优先提高召回，减少后续逻辑断裂风险。

    参数说明：
    - embedding_model_name: 语义相似度模型名称
    - fuzzy_threshold: 关键词或关键句的模糊匹配阈值
    - semantic_threshold: 语义相似度阈值
    """

    embedding_model_name: str = MODEL_RESOURCE_CONFIG.embedding_model_name
    fuzzy_threshold: int = 70
    semantic_threshold: float = 0.60


@dataclass
class ConnectionConfig:
    """
    第 4 步：关键句连接配置。

    参数说明：
    - max_gap_sentences: 两个关键句之间最多允许间隔多少句仍可补全连接
    - max_gap_tokens: 两个关键句之间中间句的最大 token 数
    - local_expand_similarity_threshold: 段内局部补全时的相邻句相似度阈值
    """

    max_gap_sentences: int = 3
    max_gap_tokens: int = 120
    local_expand_similarity_threshold: float = 0.55


@dataclass
class ChunkConfig:
    """
    第 5 步：摘要输入块配置。

    说明：
    这里不区分 API 模型 / 本地模型的运行实现，
    只保留两类推荐区间，供上层在实际调用时选择。

    大模型推荐：
    - 总 token 最佳区间：2000 ~ 4000
    - prompt 建议：500 ~ 1500
    - 正文文本块建议：1500 ~ 2500

    小模型推荐：
    - 总 token 最佳区间：1200 ~ 2500
    - prompt 建议：300 ~ 500
    - 正文文本块建议：900 ~ 2000
    """

    llm_text_min_tokens: int = 1000
    llm_text_max_tokens: int = 2000
    small_model_text_min_tokens: int = 800
    small_model_text_max_tokens: int = 1500
    overlap_tokens: int = 150


@dataclass
class RetryConfig:
    """
    第 6 步：摘要配置。

    参数说明：
    - max_retry_times: 摘要不通过检查时的最大重试次数
    """

    max_retry_times: int = 2


@dataclass
class CheckConfig:
    """
    参数说明：
    - coverage_threshold: 信息覆盖得分阈值
    - keyword_fuzzy_threshold: 关键词覆盖检查时的模糊匹配阈值
    - sentence_semantic_threshold: 关键句覆盖检查时的语义相似阈值
    - overall_threshold: 总分的阈值
    - coverage_weight: 信息覆盖的权重
    - semantic_eval_weight: 全局覆盖（事实一致性）的权重
    """

    coverage_threshold: float = 0.80

    keyword_fuzzy_threshold: int = 70
    sentence_semantic_threshold: float = 0.60

    overall_threshold: float = 0.70

    coverage_weight: float = 0.5
    semantic_eval_weight: float = 0.5


@dataclass
class PipelineConfig:
    """
    全流程配置。

    只保留流程真正需要的配置，不包含运行环境配置、
    不包含冗余模型参数配置。
    """

    split: SplitConfig = field(default_factory=SplitConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    summary: RetryConfig = field(default_factory=RetryConfig)
    check: CheckConfig = field(default_factory=CheckConfig)


def build_default_config() -> PipelineConfig:
    """
    构造默认配置。

    默认策略：
    - 筛选阶段尽量宽松，优先保留更多可能相关的句子
    - 连接阶段适度放宽，减少文本块内部断裂
    - 摘要输入块优先按 API 大模型推荐区间控制
    - 检查阶段阈值不设得过严，避免过度回退
    """
    return PipelineConfig(
        split=SplitConfig(
            spacy_model_name=MODEL_RESOURCE_CONFIG.spacy_model_name,
            tokenizer_name=MODEL_RESOURCE_CONFIG.tokenizer_name,
            min_sentence_chars=5,
        ),
        selection=SelectionConfig(
            embedding_model_name=MODEL_RESOURCE_CONFIG.embedding_model_name,
            fuzzy_threshold=70,
            semantic_threshold=0.60,
        ),
        connection=ConnectionConfig(
            max_gap_sentences=3,
            max_gap_tokens=120,
            local_expand_similarity_threshold=0.55,
        ),
        chunk=ChunkConfig(
            llm_text_min_tokens=1500,
            llm_text_max_tokens=2500,
            small_model_text_min_tokens=900,
            small_model_text_max_tokens=2000,
            overlap_tokens=150,
        ),
        summary=RetryConfig(
            max_retry_times=2,
        ),
        check=CheckConfig(
            coverage_threshold=0.80,
            keyword_fuzzy_threshold=70,
            sentence_semantic_threshold=0.60,
            overall_threshold=0.70,
            coverage_weight=0.5,
            semantic_eval_weight=0.5,
        ),
    )
