# import json
# import logging
# import re
# import sys

# import networkx as nx
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# EMBED_MODEL = r"sentence-transformers/all-MiniLM-L6-v2"


# class TextSegmenter:
#     def __init__(self, split_pattern=None, threshold=3):
#         if split_pattern is None:
#             self.split_pattern = r'(?<!\d)(?<=[.!?])\s+(?=["“‘\(A-Z])'
#         else:
#             self.split_pattern = split_pattern
#         self.threshold = threshold

#     def count_tokens(self, sample: str) -> int:
#         return len(sample.split())

#     def refine(self, raw_text: str) -> list[dict[str, str]]:
#         refined = re.sub(r"(\d+\.)\s+", r"\1 ", raw_text)
#         segments = re.split(self.split_pattern, refined)

#         final_segments = []
#         for s in segments:
#             s = s.strip()
#             if re.match(r"^\d+\.$", s):
#                 continue
#             if self.count_tokens(s) >= self.threshold:
#                 final_segments.append({"sentence": s})
#         return final_segments

#     def encode_json(self, data, label="sentences"):
#         return json.dumps({label: data}, ensure_ascii=False, indent=4)

#     def save(self, data, path, label="sentences"):
#         with open(path, "w", encoding="utf-8") as f:
#             json.dump({label: data}, f, ensure_ascii=False, indent=4)


# class LocalVectorSimilarity:
#     """
#     仅替换向量模型与编码方式：
#     - 原脚本使用 OpenAI Embedding API
#     - 这里改为本地 SentenceTransformer
#     其余加权与最终得分逻辑保持不变
#     """

#     _model_cache: dict[str, SentenceTransformer] = {}

#     def __init__(self, embedding_model: str = EMBED_MODEL):
#         logging.basicConfig(
#             level=logging.INFO,
#             format="%(asctime)s [%(levelname)s] %(message)s",
#             handlers=[logging.StreamHandler(sys.stdout)],
#         )
#         self.embedding_model = embedding_model
#         self.model = self._load_model(embedding_model)

#     @classmethod
#     def _load_model(cls, model_name_or_path: str) -> SentenceTransformer:
#         if model_name_or_path not in cls._model_cache:
#             cls._model_cache[model_name_or_path] = SentenceTransformer(
#                 model_name_or_path
#             )
#         return cls._model_cache[model_name_or_path]

#     def embed(self, input_text: str, model: str | None = None) -> list[float] | None:
#         """
#         保持与原先 embed 接口含义一致，但改为本地编码。
#         """
#         try:
#             encoder = (
#                 self.model
#                 if model is None or model == self.embedding_model
#                 else self._load_model(model)
#             )
#             vec = encoder.encode(
#                 input_text,
#                 convert_to_numpy=True,
#                 normalize_embeddings=False,
#             )
#             return vec.tolist()
#         except Exception as e:
#             logging.error(f"Local embedding error: {e}")
#             return None

#     def gather(self, block, prefix):
#         found = []
#         for i in block:
#             for k, v in i.items():
#                 if k.startswith(prefix):
#                     found.append(v)
#                     break
#         return found

#     def normalize(self, vals):
#         total = sum(vals)
#         if total == 0:
#             raise ValueError("Zero sum.")
#         return np.array([v / total for v in vals])

#     def package(self, records, name):
#         phrases = self.gather(records, "sentence")
#         storage = []

#         for element in phrases:
#             e = self.embed(element)
#             if e is not None:
#                 storage.append(e)
#             else:
#                 logging.warning(f"Failed on: {element}")

#         if storage:
#             mat = np.array(storage, dtype=np.float32)
#         else:
#             logging.warning(f"No embeddings found: {name}")
#             mat = None

#         return {"phrases": phrases, "matrix": mat}

#     def compute_normalized_reweights(self, max_vals, ma, se):
#         selected_weights_np = np.array(max_vals)
#         if len(selected_weights_np) == 0:
#             return []
#         if len(selected_weights_np) == 1:
#             return [1.0]

#         ma_index = np.argmax(selected_weights_np)
#         temp_weights = selected_weights_np.copy()
#         temp_weights[ma_index] = -np.inf
#         se_index = np.argmax(temp_weights)

#         re = 1 - ma - se
#         remaining_values = np.delete(selected_weights_np, [ma_index, se_index])
#         num_remaining = len(remaining_values)

#         if num_remaining == 0:
#             return [ma, se]

#         if np.sum(remaining_values) == 0:
#             remaining = [re / num_remaining] * num_remaining
#         else:
#             total = np.sum(remaining_values)
#             relative = remaining_values / total
#             remaining = relative * re

#         norm_w = list(remaining)
#         norm_w.insert(se_index, se)
#         norm_w.insert(ma_index, ma)
#         return norm_w

#     def finalize(self, c_data, l_data, w):
#         c_mat = c_data.get("matrix")
#         l_mat = l_data.get("matrix")
#         if c_mat is None or l_mat is None:
#             logging.error("Missing embeddings.")
#             return None

#         try:
#             smat = cosine_similarity(c_mat, l_mat)
#             smat = np.nan_to_num(smat)
#         except Exception as e:
#             logging.error(f"Similarity error: {e}")
#             return None

#         max_vals = smat.max(axis=1)
#         index_vals = smat.argmax(axis=1)

#         row_ids = np.arange(w.shape[0])
#         _sel_w = w[row_ids, index_vals]  # 保留原脚本结构，虽然最终未参与计算

#         norm_w = self.compute_normalized_reweights(max_vals, ma=0.5, se=0.3)
#         final_val = np.sum(np.array(norm_w) * max_vals)
#         final_val = round(float(final_val), 2)
#         return final_val

#     def merge_weights(self, cw, lw):
#         return (cw[:, np.newaxis] + lw) / 2


# class GraphRanker:
#     def __init__(self, threshold=0.1, alpha=0.85):
#         self.threshold = threshold
#         self.alpha = alpha

#     def compute_smat(self, emb):
#         return cosine_similarity(emb)

#     def create_graph(self, sm):
#         g = nx.Graph()
#         n = sm.shape[0]
#         g.add_nodes_from(range(n))
#         for i in range(n):
#             for j in range(i + 1, n):
#                 if sm[i][j] >= self.threshold:
#                     g.add_edge(i, j, weight=float(sm[i][j]))
#         return g

#     def rank(self, g):
#         return nx.pagerank(g, alpha=self.alpha, weight="weight")

#     def scale(self, w):
#         arr = np.array(list(w.values()), dtype=np.float32)
#         tot = arr.sum()
#         if tot == 0:
#             raise ValueError("Zero sum.")
#         return arr / tot

#     def process(self, emb):
#         sm = self.compute_smat(emb)
#         g = self.create_graph(sm)
#         r = self.rank(g)
#         return self.scale(r)


# def score_semantic_eval(
#     candidate_text: str,
#     reference_text: str,
#     embedding_model: str = EMBED_MODEL,
# ) -> float:
#     """
#     按照原脚本的计算分数逻辑计算语义匹配分数。
#     仅替换 embedding 模型为本地 sentence-transformers 模型，
#     不修改原有匹配分数计算规则。

#     参数:
#         candidate_text: 候选文本
#         reference_text: 参考文本
#         embedding_model: 本地 sentence-transformers 模型名或路径

#     返回:
#         float: 最终语义匹配分数，保留两位小数
#     """
#     seg = TextSegmenter()
#     vsim = LocalVectorSimilarity(embedding_model=embedding_model)
#     gproc = GraphRanker()

#     ref_sents = seg.refine(reference_text)
#     cand_sents = seg.refine(candidate_text)

#     if not ref_sents or not cand_sents:
#         raise ValueError("Text segmentation produced no valid sentences.")

#     ref_vector_data = vsim.package(ref_sents, "reference")
#     cand_vector_data = vsim.package(cand_sents, "candidate")

#     if ref_vector_data["matrix"] is None or cand_vector_data["matrix"] is None:
#         raise ValueError("Failed to build embedding matrices.")

#     lw = gproc.process(ref_vector_data["matrix"])
#     cw = gproc.process(cand_vector_data["matrix"])
#     wmat = vsim.merge_weights(cw, lw)

#     result = vsim.finalize(cand_vector_data, ref_vector_data, wmat)
#     if result is None:
#         raise ValueError("Semantic evaluation failed.")

#     return result


# if __name__ == "__main__":
#     text1 = "Amoxicillin is a type of antibiotic. It is mainly used to treat a variety of bacterial infections. For example, it can be used to treat infections of the middle ear, sinuses, throat, lungs, urinary tract and skin. It works by stopping the growth of bacteria. However, it is important to note that it is only effective against bacterial infections and not viral infections."
#     text2 = "Amoxicillin is a penicillin - like antibiotic. It works by stopping the growth of bacteria. It is used to treat a variety of bacterial infections, such as infections of the middle ear, tonsils, throat, urinary tract, skin, and respiratory tract (like pneumonia, bronchitis). However, it is ineffective against viral infections."

#     text3 = "Shanghai captivates me with its seamless blend of history and modernity. Strolling along the Bund at dusk, watching the neon-lit skyline of Pudong reflect on the Huangpu River, fills me with awe. The city’s energy is palpable—a dynamic rhythm that pulses through its bustling streets, art deco buildings, and hidden lanes filled with local snacks like xiaolongbao. Whether exploring the timeless charm of Yuyuan Garden or marveling at the futuristic towers of Lujiazui, Shanghai never fails to inspire. It’s a place where tradition and innovation coexist, creating a magnetic allure that feels like home."
#     text4 = "I’m deeply enamored with Shanghai’s ability to bridge past and present. The Bund’s colonial architecture stands in striking contrast to the glowing skyscrapers of Pudong, a visual symphony of eras. There’s an indescribable vitality here—a buzz in the air from the crowds, the clatter of woks in street-side kitchens, and the hum of ambition in its tech-driven districts. From sipping tea in a century-old teahouse to losing myself in the neon-drenched chaos of Nanjing Road, Shanghai offers a mosaic of experiences. It’s a city that embraces change while honoring its roots, and this duality is what makes my heart swell with admiration every time I wander its streets."

#     text5 = "Living in the city is an exhilarating adventure filled with endless possibilities. The vibrant energy of bustling streets, diverse cultures, and innovative opportunities invigorate the spirit. Every corner holds a new experience—a cozy café tucked between skyscrapers, a hidden gallery showcasing local art, or a lively festival uniting strangers. Public transport connects you to the world, while careers in cutting-edge fields thrive here. Though chaos sometimes lingers, it’s the chaos of progress, of humanity collaborating and dreaming aloud. The city is a living, breathing testament to what we can achieve when we embrace curiosity and connection."
#     text6 = "Urban existence feels like a relentless marathon, where peace is a luxury few can afford. The constant roar of traffic, the suffocating crowds, and the pressure to 'keep up’ drain the soul. Skyscrapers cast long shadows over parks once meant for tranquility, replacing greenery with concrete. Relationships grow superficial, forged in hurried elevator rides or digital screens rather than genuine dialogue. Opportunities? They exist, but at the cost of sanity—long commutes, skyrocketing costs, and a sense of being just another cog in a machine. The city promises progress but delivers isolation, a cacophony of noise drowning out the quiet moments that make life meaningful."

#     score = score_semantic_eval(text1, text2)
#     print(score)
