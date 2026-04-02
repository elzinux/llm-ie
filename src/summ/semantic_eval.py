from __future__ import annotations

import os
import re

import networkx as nx
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

EMBED_MODEL = "text-embedding-3-large"


class TextSegmenter:
    """
    与原仓库保持一致的轻量句子切分器。
    """

    def __init__(self, split_pattern: str | None = None, threshold: int = 3) -> None:
        if split_pattern is None:
            self.split_pattern = r'(?<!\d)(?<=[.!?])\s+(?=["“‘\(A-Z])'
        else:
            self.split_pattern = split_pattern
        self.threshold = threshold

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def refine(self, raw_text: str) -> list[dict[str, str]]:
        if not isinstance(raw_text, str) or not raw_text.strip():
            return []

        refined = re.sub(r"(\d+\.)\s+", r"\1 ", raw_text)
        segments = re.split(self.split_pattern, refined)

        final_segments: list[dict[str, str]] = []
        for s in segments:
            s = s.strip()
            if not s:
                continue
            if re.match(r"^\d+\.$", s):
                continue
            if self.count_tokens(s) >= self.threshold:
                final_segments.append({"sentence": s})

        return final_segments


class EmbeddingClient:
    """
    仅把原仓库的 embedding 调用改成新版 OpenAI SDK。
    其余行为尽量保持原思路。
    """

    def __init__(
        self, api_key: str | None = None, model_name: str = EMBED_MODEL
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OpenAI API key.")
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name

    def embed_one(self, input_text: str) -> list[float] | None:
        if not isinstance(input_text, str) or not input_text.strip():
            return None

        try:
            response = self.client.embeddings.create(
                input=input_text,
                model=self.model_name,
            )
            return response.data[0].embedding
        except Exception:
            return None

    def gather(self, block: list[dict[str, str]], prefix: str) -> list[str]:
        found: list[str] = []
        for item in block:
            for k, v in item.items():
                if k.startswith(prefix):
                    found.append(v)
                    break
        return found

    def package(
        self, records: list[dict[str, str]], name: str
    ) -> dict[str, np.ndarray | list[str] | None]:
        phrases = self.gather(records, "sentence")
        storage: list[list[float]] = []

        for element in phrases:
            emb = self.embed_one(element)
            if emb is not None:
                storage.append(emb)

        matrix = np.array(storage, dtype=float) if storage else None
        return {"phrases": phrases, "matrix": matrix}


class GraphRanker:
    """
    与原仓库 GraphRanker 的计算流程保持一致。
    """

    def __init__(self, threshold: float = 0.1, alpha: float = 0.85) -> None:
        self.threshold = threshold
        self.alpha = alpha

    def compute_smat(self, emb: np.ndarray) -> np.ndarray:
        return cosine_similarity(emb)

    def create_graph(self, sm: np.ndarray) -> nx.Graph:
        g = nx.Graph()
        n = sm.shape[0]
        g.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                if sm[i][j] >= self.threshold:
                    g.add_edge(i, j, weight=sm[i][j])

        return g

    def rank(self, g: nx.Graph) -> dict[int, float]:
        return nx.pagerank(g, alpha=self.alpha, weight="weight")

    def scale(self, w: dict[int, float]) -> np.ndarray:
        arr = np.array(list(w.values()), dtype=float)
        tot = arr.sum()
        if tot == 0:
            raise ValueError("Zero sum.")
        return arr / tot

    def process(self, emb: np.ndarray) -> np.ndarray:
        sm = self.compute_smat(emb)
        g = self.create_graph(sm)
        r = self.rank(g)
        return self.scale(r)


class SemanticEvalScorer:
    """
    这里刻意按原仓库的 finalize/merge_weights/compute_normalized_reweights 行为实现。
    也就是说：保留原仓库中的若干可疑点，以便最大程度复现原始结果。
    """

    def compute_normalized_reweights(
        self,
        max_vals: np.ndarray,
        ma: float,
        se: float,
    ) -> list[float]:
        selected_weights_np = np.array(max_vals)

        if len(selected_weights_np) == 0:
            return []
        if len(selected_weights_np) == 1:
            return [1.0]

        ma_index = np.argmax(selected_weights_np)
        temp_weights = selected_weights_np.copy()
        temp_weights[ma_index] = -np.inf
        se_index = np.argmax(temp_weights)

        re_weight = 1 - ma - se
        remaining_values = np.delete(selected_weights_np, [ma_index, se_index])
        num_remaining = len(remaining_values)

        # 保留原仓库行为：n == 2 时直接返回 [ma, se]
        if num_remaining == 0:
            return [ma, se]

        if np.sum(remaining_values) == 0:
            remaining = [re_weight / num_remaining] * num_remaining
        else:
            total = np.sum(remaining_values)
            relative = remaining_values / total
            remaining = relative * re_weight

        # 这里不能再写 remaining.tolist()，因为 remaining 有时是 list
        norm_w = list(remaining)
        norm_w.insert(se_index, se)
        norm_w.insert(ma_index, ma)

        return norm_w

    def merge_weights(self, cw: np.ndarray, lw: np.ndarray) -> np.ndarray:
        return (cw[:, np.newaxis] + lw) / 2

    def finalize(
        self,
        c_data,
        l_data,
        w: np.ndarray,
    ) -> float | None:
        c_mat = c_data.get("matrix")
        l_mat = l_data.get("matrix")

        if c_mat is None or l_mat is None:
            return None

        c_mat = np.asarray(c_mat, dtype=float)
        l_mat = np.asarray(l_mat, dtype=float)

        try:
            smat = cosine_similarity(c_mat, l_mat)
            smat = np.nan_to_num(smat)
        except Exception:
            return None

        max_vals = smat.max(axis=1)
        index_vals = smat.argmax(axis=1)

        # 保留原仓库行为：sel_w 计算了，但最终并未参与打分
        row_ids = np.arange(w.shape[0])
        _sel_w = w[row_ids, index_vals]

        norm_w = self.compute_normalized_reweights(max_vals, ma=0.5, se=0.3)
        final_val = np.sum(np.array(norm_w, dtype=float) * max_vals)
        final_val = round(float(final_val), 2)
        return final_val


def score_semantic_eval(
    candidate_text: str,
    reference_text: str,
    api_key: str | None = None,
    embedding_model: str = EMBED_MODEL,
) -> float:
    """
    计算候选文本与参考文本之间的语义评估分数。

    注意：为了完全对齐原仓库的计算逻辑，此处的内部变量流向保留了原仓库中
    略显“反直觉”的命名和映射关系。

    Args:
        candidate_text: 待评估的候选文本 (对应原仓库的 row['candidate'])
        reference_text: 参考标准文本 (对应原仓库的 row['reference'])
        api_key: OpenAI API 密钥
        embedding_model: 调用的 Embedding 模型名称

    Returns:
        基于图排序 (PageRank) 和余弦相似度计算的最终语义得分。
        如果计算失败或输入无效，则返回 0.0。
    """
    # 1. 前置检查：验证输入文本是否为有效字符串且不为空
    if not isinstance(candidate_text, str) or not candidate_text.strip():
        return 0.0
    if not isinstance(reference_text, str) or not reference_text.strip():
        return 0.0

    segmenter = TextSegmenter()

    # 2. 文本分句：保留原仓库 run() 的实际流向
    # 【重点映射】
    # lab_sents 处理的是候选文本 (candidate_text)
    # cand_sents 处理的是参考文本 (reference_text)
    lab_sents = segmenter.refine(candidate_text)
    cand_sents = segmenter.refine(reference_text)

    # 如果任一文本无法被有效切分为句子，则直接返回 0分
    if not lab_sents or not cand_sents:
        return 0.0

    # 3. 初始化核心组件
    embedder = EmbeddingClient(api_key=api_key, model_name=embedding_model)
    scorer = SemanticEvalScorer()
    ranker = GraphRanker()

    # 4. 向量化打包：将切分好的句子转化为 Embedding 向量矩阵
    # lab_vector_data 实际承载的是 candidate_text 的特征
    # candidate_vector_data 实际承载的是 reference_text 的特征
    lab_vector_data = embedder.package(lab_sents, "Lab")
    candidate_vector_data = embedder.package(cand_sents, "candidate")

    lab_matrix = lab_vector_data.get("matrix")
    candidate_matrix = candidate_vector_data.get("matrix")

    # 确保两边都成功提取到了向量矩阵，否则无法进行后续计算
    if lab_matrix is None or candidate_matrix is None:
        return 0.0

    # 统一转换为 numpy 数组，规范化数据类型
    lab_matrix = np.asarray(lab_matrix, dtype=float)
    candidate_matrix = np.asarray(candidate_matrix, dtype=float)

    # 5. 图排序处理：基于句子间的相似度网络，计算各个句子的重要性权重
    lw = ranker.process(lab_matrix)
    cw = ranker.process(candidate_matrix)

    # 合并候选与参考文本的句子权重矩阵
    wmat = scorer.merge_weights(cw, lw)

    # 6. 最终得分计算
    # 先传 candidate_vector_data (实际是 reference)，再传 lab_vector_data (实际是 candidate)
    result = scorer.finalize(
        {"phrases": candidate_vector_data.get("phrases"), "matrix": candidate_matrix},
        {"phrases": lab_vector_data.get("phrases"), "matrix": lab_matrix},
        wmat,
    )

    # 7. 返回结果：如果 finalize 因异常返回了 None，则退回 0.0
    return 0.0 if result is None else float(result)
