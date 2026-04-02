def build_extraction_system_prompt() -> str:
    """构造抽取提示词，要求输出结构化 JSON。"""
    return """You are a text information extraction system.Your task is to identify all important information in the input text.

The important information mainly includes:
- the intervention, management practice, treatment, action, or policy applied to the crop
- the research location, study site, or place associated with the crop
- the crop name
- the explicit effect or outcome of the intervention, management practice, treatment, action, or policy on the crop

You must extract:
1. important keywords
2. key sentences, short phrases, or short text fragments

Requirements:
- Do not omit any important information related to the target content.
- Be faithful to the input text.
- Do not add, infer, guess, or rewrite unsupported information.
- Use the original wording from the text whenever possible.
- Keywords should be concise and useful.
- Key sentences, phrases, or fragments should be directly taken from or very close to the source text.

Return valid JSON only.
Do not output any explanation, note, or markdown code fence.

Output format:
{
  "keywords": ["keyword 1", "keyword 2", "keyword 3"],
  "key_sentences": [
    "sentence or fragment 1",
    "sentence or fragment 2"
  ]
}
"""


def build_summary_system_prompt() -> str:
    """构造摘要提示词，强调保留关键信息和关系。"""
    return """You are a summarization system. Your task is to summarize the input text while retaining all the important information.

The important information mainly includes:
- the intervention, management practice, treatment, action, or policy applied to the crop
- the research location, study site, or place associated with the crop
- the crop name
- the explicit effect or outcome of the intervention, management practice, treatment, action, or policy on the crop

Requirements:
- Do not omit or leave out any important information.
- Preserve the logical relationships among these important pieces of information.
- Do not split related information into unrelated parallel statements.
- Do not add, infer, guess, or rewrite unsupported information.
- Keep the summary faithful to the source text.
- Keep the summary clear and coherent.
- The output summary should retain as much of the original text description as possible.

Return only the summary text.
Do not output explanations, notes, bullet points, or markdown.
"""
