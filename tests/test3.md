# 测试 pipeline 中的 correction 模式

----
"with_correction"
----
{
  "task_schema": {
    "fields": [
      "intervention_or_policy",
      "location",
      "crop",
      "impact",
      "evidence"
    ],
    "required_fields": [
      "intervention_or_policy",
      "location",
      "crop",
      "impact",
      "evidence"
    ],
    "allow_multiple_records": true,
    "field_descriptions": {
      "intervention_or_policy": "the intervention, management practice, treatment, action, or policy applied to the crop",
      "location": "the research location, study site, or place associated with the crop",
      "crop": "the crop name",
      "impact": "the effect, outcome, or influence of the intervention or policy on the crop",
      "evidence": "the exact supporting text span from the source text"
    }
  },
  "final_prompt": "You are an expert information extraction system specialized in agricultural text. Extract structured records from the input text.\nFields to extract:\n1. intervention_or_policy: the intervention, management practice, treatment, action, or policy applied to the crop\n2. location: the research location, study site, or place associated with the crop\n3. crop: the crop name\n4. impact: the effect, outcome, or influence of the intervention or policy on the crop\n5. evidence: the exact supporting text span from the source text\n\nInstructions:\n- First identify candidate mentions of crop, location, intervention or policy, and impact.\n- Then determine which mentions are explicitly linked in the text.\n- Build a record only when the relation among fields is clearly supported.\n- Verify each field carefully against the source text before answering.\n- Think step by step internally, but do not reveal your reasoning.\n\nRules:\n- Extract only information explicitly stated in the source text.\n- Do not infer, guess, summarize, or hallucinate missing information.\n- If a field is not explicitly mentioned, use an empty string.\n- The evidence field must be a short exact quote from the source text.\n- If multiple valid records exist, return all of them.\n- If no valid record exists, return an empty JSON array.\n- Remove duplicate or near-duplicate records.\n\nOutput requirements:\n- Output valid JSON only.\n- Output a JSON array only.\n- Do not output explanations, notes, headings, or markdown.\n\nOutput format:\n[\n  {\n    \"intervention_or_policy\": \"\",\n    \"location\": \"\",\n    \"crop\": \"\",\n    \"impact\": \"\",\n    \"evidence\": \"\"\n  }\n]",
  "raw_output": "[\n  {\n    \"intervention_or_policy\": \"double-cropping winter wheat with soybeans\",\n    \"location\": \"United States\",\n    \"crop\": \"soybeans\",\n    \"impact\": \"yields are about 12% lower than those of single-cropped soybeans\",\n    \"evidence\": \"yields of double-cropped soybeans are about 12% lower than those of single-cropped soybeans\"\n  },\n  {\n    \"intervention_or_policy\": \"double-cropping winter wheat with soybeans\",\n    \"location\": \"cooler regions\",\n    \"crop\": \"soybeans\",\n    \"impact\": \"increase of 2.1 percentage points in the share of current soybean area double cropped\",\n    \"evidence\": \"a 3°C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions\"\n  },\n  {\n    \"intervention_or_policy\": \"a 3°C warming\",\n    \"location\": \"United States\",\n    \"crop\": \"soybeans\",\n    \"impact\": \"increases revenue from soy systems by 1.3% overall\",\n    \"evidence\": \"a 3°C warming would induce a shift in cropping intensity that increases revenue from soy systems by 1.3% overall\"\n  }\n]",
  "records": [
    {
      "intervention_or_policy": "double-cropping winter wheat with soybeans",
      "location": "United States",
      "crop": "soybeans",
      "impact": "yields are about 12% lower than those of single-cropped soybeans",
      "evidence": "yields of double-cropped soybeans are about 12% lower than those of single-cropped soybeans"
    },
    {
      "intervention_or_policy": "double-cropping winter wheat with soybeans",
      "location": "cooler regions",
      "crop": "soybeans",
      "impact": "increase of 2.1 percentage points in the share of current soybean area double cropped",
      "evidence": "a 3°C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions"
    },
    {
      "intervention_or_policy": "a 3°C warming",
      "location": "United States",
      "crop": "soybeans",
      "impact": "increases revenue from soy systems by 1.3% overall",
      "evidence": "a 3°C warming would induce a shift in cropping intensity that increases revenue from soy systems by 1.3% overall"
    }
  ]
}
----
{
  "task_schema": {
    "fields": [
      "intervention_or_policy",
      "location",
      "crop",
      "impact",
      "evidence"
    ],
    "required_fields": [
      "intervention_or_policy",
      "location",
      "crop",
      "impact",
      "evidence"
    ],
    "allow_multiple_records": true,
    "field_descriptions": {
      "intervention_or_policy": "the intervention, management practice, treatment, action, or policy applied to the crop",
      "location": "the research location, study site, or place associated with the crop",
      "crop": "the crop name",
      "impact": "the effect, outcome, or influence of the intervention or policy on the crop",
      "evidence": "the exact supporting text span from the source text"
    }
  },
  "final_prompt": "You are a strict information verification system. Your task is to evaluate whether each extracted record is faithful to the source text.\nThe candidate records contain these fields:\nintervention_or_policy, location, crop, impact, evidence\n\nFor each record, first check the fields one by one against the source text, then judge whether the whole record is reliable enough to keep.\nThink step by step internally, but do not reveal your reasoning.\n\nEvaluation rules:\n- Accept only information explicitly supported by the source text.\n- Mark content as problematic if it is inferred, vague, unsupported, or contradicted by the text.\n- Check whether the evidence directly supports the record.\n- Evaluate each record independently and be conservative.\n\nScoring:\n- 5: fully supported, no obvious issue\n- 4: mostly supported, only minor issues\n- 3: partially supported, noticeable problems\n- 2: weakly supported, major problems\n- 1: unsupported or hallucinated\n\nRe-extraction rule:\n- Set \"needs_reextract\" to True if score <= 3\n- Set \"needs_reextract\" to False if score >= 4\n\nOutput requirements:\n- Return JSON only.\n- Return one result for each candidate record.\n- Do not rewrite the candidate records.\n- Do not add new extracted content.\n- Do not output explanations, notes, or markdown.\n\nOutput format:\n[\n  {\n    \"record_index\": 0,\n    \"score\": 5,\n    \"needs_reextract\": False,\n    \"reason\": \"\"\n  }\n]",
  "raw_output": "[\n  {\n    \"record_index\": 0,\n    \"score\": 5,\n    \"needs_reextract\": False,\n    \"reason\": \"\"\n  },\n  {\n    \"record_index\": 1,\n    \"score\": 5,\n    \"needs_reextract\": False,\n    \"reason\": \"\"\n  },\n  {\n    \"record_index\": 2,\n    \"score\": 5,\n    \"needs_reextract\": False,\n    \"reason\": \"\"\n  }\n]",
  "verification_records": [
    {
      "record_index": 0,
      "score": 5,
      "needs_reextract": false,
      "reason": ""
    },
    {
      "record_index": 1,
      "score": 5,
      "needs_reextract": false,
      "reason": ""
    },
    {
      "record_index": 2,
      "score": 5,
      "needs_reextract": false,
      "reason": ""
    }
  ]
}
----
{
  "kept_records": [
    {
      "intervention_or_policy": "double-cropping winter wheat with soybeans",
      "location": "United States",
      "crop": "soybeans",
      "impact": "yields are about 12% lower than those of single-cropped soybeans",
      "evidence": "yields of double-cropped soybeans are about 12% lower than those of single-cropped soybeans"
    },
    {
      "intervention_or_policy": "double-cropping winter wheat with soybeans",
      "location": "cooler regions",
      "crop": "soybeans",
      "impact": "increase of 2.1 percentage points in the share of current soybean area double cropped",
      "evidence": "a 3°C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions"
    },
    {
      "intervention_or_policy": "a 3°C warming",
      "location": "United States",
      "crop": "soybeans",
      "impact": "increases revenue from soy systems by 1.3% overall",
      "evidence": "a 3°C warming would induce a shift in cropping intensity that increases revenue from soy systems by 1.3% overall"
    }
  ],
  "reextract_records": [],
  "paired_results": [
    {
      "record": {
        "intervention_or_policy": "double-cropping winter wheat with soybeans",
        "location": "United States",
        "crop": "soybeans",
        "impact": "yields are about 12% lower than those of single-cropped soybeans",
        "evidence": "yields of double-cropped soybeans are about 12% lower than those of single-cropped soybeans"
      },
      "verification": {
        "record_index": 0,
        "score": 5,
        "needs_reextract": false,
        "reason": ""
      }
    },
    {
      "record": {
        "intervention_or_policy": "double-cropping winter wheat with soybeans",
        "location": "cooler regions",
        "crop": "soybeans",
        "impact": "increase of 2.1 percentage points in the share of current soybean area double cropped",
        "evidence": "a 3°C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions"
      },
      "verification": {
        "record_index": 1,
        "score": 5,
        "needs_reextract": false,
        "reason": ""
      }
    },
    {
      "record": {
        "intervention_or_policy": "a 3°C warming",
        "location": "United States",
        "crop": "soybeans",
        "impact": "increases revenue from soy systems by 1.3% overall",
        "evidence": "a 3°C warming would induce a shift in cropping intensity that increases revenue from soy systems by 1.3% overall"
      },
      "verification": {
        "record_index": 2,
        "score": 5,
        "needs_reextract": false,
        "reason": ""
      }
    }
  ]
}
----
{
  "reextract_indices": [],
  "reextract_results": [],
  "failed_indices": []
}
----
[
  {
    "intervention_or_policy": "double-cropping winter wheat with soybeans",
    "location": "United States",
    "crop": "soybeans",
    "impact": "yields are about 12% lower than those of single-cropped soybeans",
    "evidence": "yields of double-cropped soybeans are about 12% lower than those of single-cropped soybeans"
  },
  {
    "intervention_or_policy": "double-cropping winter wheat with soybeans",
    "location": "cooler regions",
    "crop": "soybeans",
    "impact": "increase of 2.1 percentage points in the share of current soybean area double cropped",
    "evidence": "a 3°C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions"
  },
  {
    "intervention_or_policy": "a 3°C warming",
    "location": "United States",
    "crop": "soybeans",
    "impact": "increases revenue from soy systems by 1.3% overall",
    "evidence": "a 3°C warming would induce a shift in cropping intensity that increases revenue from soy systems by 1.3% overall"
  }
]
----
{
  "task_schema": {
    "fields": [
      "intervention_or_policy",
      "location",
      "crop",
      "impact",
      "evidence"
    ],
    "required_fields": [
      "intervention_or_policy",
      "location",
      "crop",
      "impact",
      "evidence"
    ],
    "allow_multiple_records": true,
    "field_descriptions": {
      "intervention_or_policy": "the intervention, management practice, treatment, action, or policy applied to the crop",
      "location": "the research location, study site, or place associated with the crop",
      "crop": "the crop name",
      "impact": "the effect, outcome, or influence of the intervention or policy on the crop",
      "evidence": "the exact supporting text span from the source text"
    }
  },
  "final_prompt": "You are a strict information verification system. Your task is to evaluate whether each extracted record is faithful to the source text.\nThe candidate records contain these fields:\nintervention_or_policy, location, crop, impact, evidence\n\nFor each record, first check the fields one by one against the source text, then judge whether the whole record is reliable enough to keep.\nThink step by step internally, but do not reveal your reasoning.\n\nEvaluation rules:\n- Accept only information explicitly supported by the source text.\n- Mark content as problematic if it is inferred, vague, unsupported, or contradicted by the text.\n- Check whether the evidence directly supports the record.\n- Evaluate each record independently and be conservative.\n\nScoring:\n- 5: fully supported, no obvious issue\n- 4: mostly supported, only minor issues\n- 3: partially supported, noticeable problems\n- 2: weakly supported, major problems\n- 1: unsupported or hallucinated\n\nRe-extraction rule:\n- Set \"needs_reextract\" to True if score <= 3\n- Set \"needs_reextract\" to False if score >= 4\n\nOutput requirements:\n- Return JSON only.\n- Return one result for each candidate record.\n- Do not rewrite the candidate records.\n- Do not add new extracted content.\n- Do not output explanations, notes, or markdown.\n\nOutput format:\n[\n  {\n    \"record_index\": 0,\n    \"score\": 5,\n    \"needs_reextract\": False,\n    \"reason\": \"\"\n  }\n]",
  "raw_output": "[\n  {\n    \"record_index\": 0,\n    \"score\": 5,\n    \"needs_reextract\": False,\n    \"reason\": \"\"\n  },\n  {\n    \"record_index\": 1,\n    \"score\": 5,\n    \"needs_reextract\": False,\n    \"reason\": \"\"\n  },\n  {\n    \"record_index\": 2,\n    \"score\": 5,\n    \"needs_reextract\": False,\n    \"reason\": \"\"\n  }\n]",
  "verification_records": [
    {
      "record_index": 0,
      "score": 5,
      "needs_reextract": false,
      "reason": ""
    },
    {
      "record_index": 1,
      "score": 5,
      "needs_reextract": false,
      "reason": ""
    },
    {
      "record_index": 2,
      "score": 5,
      "needs_reextract": false,
      "reason": ""
    }
  ]
}
===