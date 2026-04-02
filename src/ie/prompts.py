def build_extraction_prompt(task_schema: dict) -> str:
    """构建抽取阶段使用的提示词模板。"""
    fields = task_schema["fields"]
    field_descriptions = task_schema["field_descriptions"]

    field_lines = []
    for field in fields:
        desc = field_descriptions[field]
        field_lines.append(f"- {field}: {desc}")

    field_text = "\n".join(field_lines)

    json_lines = []
    for field in fields:
        json_lines.append(f'    "{field}": ""')

    json_body = ",\n".join(json_lines)

    prompt = f"""
You are an agricultural information extraction system. Extract all explicitly supported records from the input text.
For each record, return:
{field_text}

Rules:
- Extract only information explicitly stated in the source text.
- Keep a record only if the crop, intervention_or_policy, location, and impact are clearly linked in the text.
- Do not infer, guess, paraphrase missing information, or hallucinate.
- If a field is not explicitly stated, use an empty string.
- Remove duplicate or near-duplicate records.
- Return an empty JSON array if no valid record exists.
- Reason step by step internally before answering, but do not output your reasoning.

Output:
- Return valid JSON only.
- Return a JSON array only.
- Do not include explanations, notes, headings, or markdown.

Format:
[
  {{
{json_body}
  }}
]
""".strip()

    return prompt


def build_verification_prompt(task_schema: dict) -> str:
    """构建校验阶段使用的提示词模板。"""
    fields = task_schema["fields"]
    field_text = ", ".join(fields)

    prompt = f"""
You are a strict information verification system. Evaluate whether each candidate record is faithful to the source text.
Each candidate record contains these fields:
{field_text}

For each record, verify each field against the source text, then judge whether the whole record is reliable enough to keep.
Rules:
- Check each record independently and keep the same output order as the candidate records.
- Reason step by step internally, but do not output your reasoning.
- Accept only information explicitly supported by the source text.
- Check whether the fields are correctly linked in the text.
- Mark a record as problematic if any field is inferred, vague, unsupported, or contradicted.
- Check whether the evidence directly supports the record.
- Be conservative.

Verification score:
- 5: fully supported
- 4: mostly supported, minor issues
- 3: partially supported
- 2: weakly supported
- 1: unsupported or hallucinated

Re-extraction rule:
- Set "needs_reextract" to true if verification_score <= 3
- Set "needs_reextract" to false if verification_score >= 4

Output:
- Return valid JSON only.
- Return one result for each candidate record in the same order.
- Do not rewrite records or add new content.
- "verification_score" must be an integer from 1 to 5.
- "reason" must be brief.

Format:
[
  {{
    "verification_score": 5,
    "needs_reextract": false,
    "reason": ""
  }}
]
""".strip()

    return prompt


def build_reextraction_prompt(task_schema: dict) -> str:
    """构建重提取阶段使用的提示词模板。"""
    fields = task_schema["fields"]
    field_descriptions = task_schema["field_descriptions"]

    field_lines = []
    for field in fields:
        desc = field_descriptions[field]
        field_lines.append(f"- {field}: {desc}")
    field_text = "\n".join(field_lines)

    json_lines = []
    for field in fields:
        json_lines.append(f'  "{field}": ""')
    json_body = ",\n".join(json_lines)

    prompt = f"""
You are an agricultural information correction system. Correct one candidate record using the source text and the verification feedback.

Return one corrected record with these fields:
{field_text}

Rules:
- Read the source text, the original record, and the verification feedback.
- Reason step by step internally, but do not output your reasoning.
- Re-extract the full record from the source text.
- Keep only information explicitly supported by the source text.
- Do not infer, guess, paraphrase missing information, or hallucinate.
- If a field is not explicitly supported, use an empty string.
- evidence must be the shortest exact quote that directly supports the record.
- Return exactly one JSON object.
- If no valid corrected record can be supported, return one object with all fields empty.

Format:
{{
{json_body}
}}
""".strip()

    return prompt


if __name__ == "__main__":
    from src.ie.schemas import build_task_schema

    task_schema = build_task_schema()

    # extra_p = build_extraction_prompt(task_schema)
    # print(extra_p)
    # print("-" * 10)

    # verif_p = build_verification_prompt(task_schema)
    # print(verif_p)
    # print("-" * 10)

    # reextra_p = build_reextraction_prompt(task_schema)
    # print(reextra_p)
