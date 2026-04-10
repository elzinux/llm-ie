from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class InputItem:
    """
    需要按章节组织，且章节的不同段落之间是两个换行.
    """

    title: str = ""
    text: str = ""


@dataclass
class InputDocument:
    file_name: str = ""
    items: list[InputItem] = field(default_factory=list)


@dataclass
class ExtractionRecord:
    """
    目标信息粗提取结果。

    这一阶段不对目标信息做严格分类，
    只保留两类用于后续筛选的线索：
    - keywords: 目标相关关键词
    - key_sentences: 目标相关关键句
    """

    keywords: list[str] = field(default_factory=list)
    key_sentences: list[str] = field(default_factory=list)


@dataclass
class ExtractionCandidate:
    """
    extractor / reviewer / checker 阶段使用的候选信息结构。

    字段说明：
    - text:
        候选文本本身。可以是 field_candidate，也可以是 sentence 或更短的证据片段。
    - candidate_type:
        候选类型，只允许:"field_candidate", "evidence_span"
    - target_fields:
        该候选涉及到的目标字段列表。
        可包含的字段通常为:"intervention_or_policy", "location", "crop", "impact"
    - relevance:
        与目标信息的相关程度，只允许:"high", "medium", "low", "none
    - reason:
        简短说明该候选为什么与目标信息相关。
    """

    text: str = ""
    candidate_type: Literal["field_candidate", "evidence_span"] = "field_candidate"
    target_fields: list[
        Literal["intervention_or_policy", "location", "crop", "impact"]
    ] = field(default_factory=list)
    relevance: Literal["high", "medium", "low", "none"] = "none"
    reason: str = ""


@dataclass
class SentenceInfo:
    """
    句子级信息。

    说明：
    - global_idx: 全局句子索引，用于排序、连接、回溯
    - local_idx: (paragraph_idx, sentence_idx_in_paragraph)
      用于保留段落内位置信息，同时避免分散多个 idx 字段
    - token_count: 句子 token 数
    - semantic_score: 句子在筛选阶段的语义相关分数
    - is_selected: 是否被筛选保留
    - sentence_embedding: 句子向量，可供后续连接阶段复用
    """

    global_idx: int
    local_idx: tuple[int, int]
    text: str
    token_count: int
    semantic_score: float = 0.0
    is_selected: bool = False
    sentence_embedding: Any = None


@dataclass
class ConnectedChunk:
    """
    关键句连接后的片段。

    只保留最必要的信息：
    - chunk_id
    - sentence_indices
    - text
    - token_count
    """

    chunk_id: int
    sentence_indices: list[int]
    text: str
    token_count: int


@dataclass
class PreparedSummaryChunk:
    """
    整理后的摘要输入块。

    说明：
    - prepared_chunk_id: 摘要输入块编号
    - text: 最终送入摘要模型的文本
    - token_count: 当前块 token 数
    - extraction_records: 当前块对应的粗提取结果
    - source_block_indices: 当前块来源的原始 block 编号
    """

    prepared_chunk_id: int
    text: str
    token_count: int
    extraction_records: list[ExtractionRecord] = field(default_factory=list)
    source_block_indices: list[int] = field(default_factory=list)


@dataclass
class CheckResult:
    passed: bool
    information_coverage: float = 0.0
    semantic_eval_score: float = 0.0
    overall_score: float = 0.0
    status: str = ""


@dataclass
class BlockProcessResult:
    """
    单个原始文本块处理结果。

    用于保存第 1~4 步的核心中间结果，保持简洁：
    - block_idx: 原始块编号
    - original_text: 原始文本块
    - extraction_records: 第 1 步粗提取得到的线索
    - sentences: 句子切分结果（其中包含筛选状态）
    - connected_chunks: 连接后的关键片段
    - merged_text: 当前块经连接后的可摘要文本
    - merged_token_count: merged_text 的 token 数
    """

    block_idx: int
    original_text: str
    extraction_records: list[ExtractionRecord] = field(default_factory=list)
    sentences: list[SentenceInfo] = field(default_factory=list)
    connected_chunks: list[ConnectedChunk] = field(default_factory=list)
    merged_text: str = ""
    merged_token_count: int = 0


@dataclass
class SummaryChunkResult:
    """
    单个摘要输入块的最终结果。

    只保留最核心的输出：
    - prepared_chunk_id: 对应的摘要输入块编号
    - source_text: 输入给摘要模型的文本
    - summary: 模型生成的摘要
    - attempts: 实际摘要尝试次数
    - check_result: 检查结果
    - retry_reasons: 每次重摘要对应的失败分数项
    """

    prepared_chunk_id: int
    source_text: str
    summary: str
    attempts: int
    check_result: CheckResult
    retry_reasons: list[list[str]] = field(default_factory=list)


@dataclass
class SummaryPipelineResult:
    """
    总流程最终输出。

    尽量简洁，只保留：
    - final_summary: 最终摘要结果
    - block_results: 每个原始文本块的处理结果
    - prepared_chunks: 第 5 步得到的摘要输入块
    - chunk_results: 每个摘要输入块的摘要结果
    """

    final_summary: str
    block_results: list[BlockProcessResult] = field(default_factory=list)
    prepared_chunks: list[PreparedSummaryChunk] = field(default_factory=list)
    chunk_results: list[SummaryChunkResult] = field(default_factory=list)
