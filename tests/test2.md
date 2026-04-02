测试脚本 correction
---

====================
TEST: prepare_reextract_input
====================
Please correct the extracted record according to the source text and the verification feedback.

[Source Text]
A warming climate expands the frost-free season, plausibly allowing for increased cropping intensity in temperate regions. This paper assesses the potential of multiple cropping to offset the projected negative effects of climate change on agricultural yields in the United States. We use cross-sectional variation in observed land cover, soil characteristics, and climate to estimate farmers’ propensity to double-crop winter wheat with soybeans. Our estimates imply that under current economic conditions, a 3C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions. A fixed-effects panel model of county yields further indicates that yields of double-cropped soybeans are about 12% lower than those of single-cropped soybeans. Accounting for changes in cropping intensity and attendant effects on soybean yields, we project that at current prices, a 3C warming would induce a shift in cropping intensity that increases revenue from soy systems by 1.3% overall, offsetting only a small fraction of the revenue impacts of predicted yield declines.

[Original Extracted Record]
{
  "intervention_or_policy": "double-cropping winter wheat with soybeans",
  "location": "United States",
  "crop": "soybeans",
  "impact": "yields are about 14% lower than those of single-cropped soybeans",
  "evidence": "yields of double-cropped soybeans are about 12% lower than those of single-cropped soybeans"
}

[Verification Feedback]
{
  "record_index": 0,
  "score": 2,
  "needs_reextract": true,
  "reason": ""
}

====================
TEST: reextract_single_record
====================
{
  "record_index": 0,
  "original_record": {
    "intervention_or_policy": "double-cropping winter wheat with soybeans",
    "location": "United States",
    "crop": "soybeans",
    "impact": "yields are about 14% lower than those of single-cropped soybeans",
    "evidence": "yields of double-cropped soybeans are about 12% lower than those of single-cropped soybeans"
  },
  "verification": {
    "record_index": 0,
    "score": 2,
    "needs_reextract": true,
    "reason": ""
  },
  "final_prompt": "You are an expert information correction system specialized in agricultural text. Your task is to correct one extracted record using the source text and the verification feedback.\nThe corrected record must contain these fields:\n1. intervention_or_policy: the intervention, management practice, treatment, action, or policy applied to the crop\n2. location: the research location, study site, or place associated with the crop\n3. crop: the crop name\n4. impact: the effect, outcome, or influence of the intervention or policy on the crop\n5. evidence: the exact supporting text span from the source text\n\nInstructions:\n- Read the source text carefully.\n- Review the original extracted record and the verification feedback.\n- Identify which parts of the original record are unsupported or incorrect.\n- Re-extract the full corrected record from the source text.\n- Think step by step internally, but do not reveal your reasoning.\n\nRules:\n- Keep only information explicitly supported by the source text.\n- Do not infer, guess, summarize, or hallucinate missing information.\n- If a field is not explicitly supported, use an empty string.\n- The evidence field must be a short exact quote from the source text.\n- Return exactly one corrected JSON object.\n- If the original record cannot be supported as a valid record, return one object with all fields set to empty strings.\n- Do not output explanations, notes, headings, or markdown.\n- Output valid JSON only.\n\nOutput format:\n{\n  \"intervention_or_policy\": \"\",\n  \"location\": \"\",\n  \"crop\": \"\",\n  \"impact\": \"\",\n  \"evidence\": \"\"\n}",
  "raw_output": "{\n  \"intervention_or_policy\": \"double-cropping winter wheat with soybeans\",\n  \"location\": \"United States\",\n  \"crop\": \"soybeans\",\n  \"impact\": \"yields are about 12% lower than those of single-cropped soybeans\",\n  \"evidence\": \"yields of double-cropped soybeans are about 12% lower than those of single-cropped soybeans\"\n}",
  "corrected_record": {
    "intervention_or_policy": "double-cropping winter wheat with soybeans",
    "location": "United States",
    "crop": "soybeans",
    "impact": "yields are about 12% lower than those of single-cropped soybeans",
    "evidence": "yields of double-cropped soybeans are about 12% lower than those of single-cropped soybeans"
  }
}

====================
TEST: reextract_flagged_records
====================
{
  "reextract_indices": [
    0,
    2
  ],
  "reextract_results": [
    {
      "record_index": 0,
      "original_record": {
        "intervention_or_policy": "double-cropping winter wheat with soybeans",
        "location": "United States",
        "crop": "soybeans",
        "impact": "yields are about 14% lower than those of single-cropped soybeans",
        "evidence": "yields of double-cropped soybeans are about 12% lower than those of single-cropped soybeans"
      },
      "verification": {
        "record_index": 0,
        "score": 2,
        "needs_reextract": true,
        "reason": ""
      },
      "final_prompt": "You are an expert information correction system specialized in agricultural text. Your task is to correct one extracted record using the source text and the verification feedback.\nThe corrected record must contain these fields:\n1. intervention_or_policy: the intervention, management practice, treatment, action, or policy applied to the crop\n2. location: the research location, study site, or place associated with the crop\n3. crop: the crop name\n4. impact: the effect, outcome, or influence of the intervention or policy on the crop\n5. evidence: the exact supporting text span from the source text\n\nInstructions:\n- Read the source text carefully.\n- Review the original extracted record and the verification feedback.\n- Identify which parts of the original record are unsupported or incorrect.\n- Re-extract the full corrected record from the source text.\n- Think step by step internally, but do not reveal your reasoning.\n\nRules:\n- Keep only information explicitly supported by the source text.\n- Do not infer, guess, summarize, or hallucinate missing information.\n- If a field is not explicitly supported, use an empty string.\n- The evidence field must be a short exact quote from the source text.\n- Return exactly one corrected JSON object.\n- If the original record cannot be supported as a valid record, return one object with all fields set to empty strings.\n- Do not output explanations, notes, headings, or markdown.\n- Output valid JSON only.\n\nOutput format:\n{\n  \"intervention_or_policy\": \"\",\n  \"location\": \"\",\n  \"crop\": \"\",\n  \"impact\": \"\",\n  \"evidence\": \"\"\n}",
      "raw_output": "{\n  \"intervention_or_policy\": \"double-cropping winter wheat with soybeans\",\n  \"location\": \"United States\",\n  \"crop\": \"soybeans\",\n  \"impact\": \"yields are about 12% lower than those of single-cropped soybeans\",\n  \"evidence\": \"yields of double-cropped soybeans are about 12% lower than those of single-cropped soybeans\"\n}",
      "corrected_record": {
        "intervention_or_policy": "double-cropping winter wheat with soybeans",
        "location": "United States",
        "crop": "soybeans",
        "impact": "yields are about 12% lower than those of single-cropped soybeans",
        "evidence": "yields of double-cropped soybeans are about 12% lower than those of single-cropped soybeans"
      }
    },
    {
      "record_index": 2,
      "original_record": {
        "intervention_or_policy": "double-cropping winter wheat with soybeans",
        "location": "United States",
        "crop": "soybeans",
        "impact": "a 3°C warming would induce a shift in cropping intensity that increases revenue from soy systems by 13% overall",
        "evidence": "a 3°C warming would induce a shift in cropping intensity that increases revenue from soy systems by 13% overall"
      },
      "verification": {
        "record_index": 2,
        "score": 2,
        "needs_reextract": true,
        "reason": ""
      },
      "final_prompt": "You are an expert information correction system specialized in agricultural text. Your task is to correct one extracted record using the source text and the verification feedback.\nThe corrected record must contain these fields:\n1. intervention_or_policy: the intervention, management practice, treatment, action, or policy applied to the crop\n2. location: the research location, study site, or place associated with the crop\n3. crop: the crop name\n4. impact: the effect, outcome, or influence of the intervention or policy on the crop\n5. evidence: the exact supporting text span from the source text\n\nInstructions:\n- Read the source text carefully.\n- Review the original extracted record and the verification feedback.\n- Identify which parts of the original record are unsupported or incorrect.\n- Re-extract the full corrected record from the source text.\n- Think step by step internally, but do not reveal your reasoning.\n\nRules:\n- Keep only information explicitly supported by the source text.\n- Do not infer, guess, summarize, or hallucinate missing information.\n- If a field is not explicitly supported, use an empty string.\n- The evidence field must be a short exact quote from the source text.\n- Return exactly one corrected JSON object.\n- If the original record cannot be supported as a valid record, return one object with all fields set to empty strings.\n- Do not output explanations, notes, headings, or markdown.\n- Output valid JSON only.\n\nOutput format:\n{\n  \"intervention_or_policy\": \"\",\n  \"location\": \"\",\n  \"crop\": \"\",\n  \"impact\": \"\",\n  \"evidence\": \"\"\n}",
      "raw_output": "{\n  \"intervention_or_policy\": \"double-cropping winter wheat with soybeans\",\n  \"location\": \"United States\",\n  \"crop\": \"soybeans\",\n  \"impact\": \"a 3°C warming would induce a shift in cropping intensity that increases revenue from soy systems by 1.3% overall, offsetting only a small fraction of the revenue impacts of predicted yield declines\",\n  \"evidence\": \"a 3°C warming would induce a shift in cropping intensity that increases revenue from soy systems by 1.3% overall, offsetting only a small fraction of the revenue impacts of predicted yield declines\"\n}",
      "corrected_record": {
        "intervention_or_policy": "double-cropping winter wheat with soybeans",
        "location": "United States",
        "crop": "soybeans",
        "impact": "a 3°C warming would induce a shift in cropping intensity that increases revenue from soy systems by 1.3% overall, offsetting only a small fraction of the revenue impacts of predicted yield declines",
        "evidence": "a 3°C warming would induce a shift in cropping intensity that increases revenue from soy systems by 1.3% overall, offsetting only a small fraction of the revenue impacts of predicted yield declines"
      }
    }
  ],
  "failed_indices": []
}

====================
TEST: merge_corrected_records
====================
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
    "location": "United States",
    "crop": "soybeans",
    "impact": "a 3°C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions",
    "evidence": "a 3°C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions"
  },
ion of the revenue impacts of predicted yield declines",
        "evidence": "a 3°C warming would induce a shift in cropping intensity that increases revenue from soy systems by 1.3% overall, offsetting only a small fraction of the revenue impacts of predicted yield declines"
      }
    }
  ],
  "failed_indices": []
}

====================
TEST: merge_corrected_records
====================
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
    "location": "United States",
    "crop": "soybeans",
    "impact": "a 3°C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions",
    "evidence": "a 3°C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions"
  },
      }
    }
  ],
  "failed_indices": []
}

====================
TEST: merge_corrected_records
====================
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
    "location": "United States",
    "crop": "soybeans",
    "impact": "a 3°C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions",
    "evidence": "a 3°C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions"
  },

====================
TEST: merge_corrected_records
====================
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
    "location": "United States",
    "crop": "soybeans",
    "impact": "a 3°C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions",
    "evidence": "a 3°C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions"
  },
====================
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
    "location": "United States",
    "crop": "soybeans",
    "impact": "a 3°C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions",
    "evidence": "a 3°C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions"
  },
    "intervention_or_policy": "double-cropping winter wheat with soybeans",
    "location": "United States",
    "crop": "soybeans",
    "impact": "yields are about 12% lower than those of single-cropped soybeans",
    "evidence": "yields of double-cropped soybeans are about 12% lower than those of single-cropped soybeans"
  },
  {
    "intervention_or_policy": "double-cropping winter wheat with soybeans",
    "location": "United States",
    "crop": "soybeans",
    "impact": "a 3°C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions",
    "evidence": "a 3°C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions"
  },
    "evidence": "yields of double-cropped soybeans are about 12% lower than those of single-cropped soybeans"
  },
  {
    "intervention_or_policy": "double-cropping winter wheat with soybeans",
    "location": "United States",
    "crop": "soybeans",
    "impact": "a 3°C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions",
    "evidence": "a 3°C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions"
  },
    "location": "United States",
    "crop": "soybeans",
    "impact": "a 3°C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions",
    "evidence": "a 3°C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions"
  },
ons in cooler regions",
    "evidence": "a 3°C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions"
  },
  {
    "intervention_or_policy": "double-cropping winter wheat with soybeans",
sions in cooler regions"
  },
  {
    "intervention_or_policy": "double-cropping winter wheat with soybeans",
    "location": "United States",
  {
    "intervention_or_policy": "double-cropping winter wheat with soybeans",
    "location": "United States",
    "crop": "soybeans",
    "impact": "a 3°C warming would induce a shift in cropping intensity that increases revenue from soy systems by 1.3% overall, offsetting only a small fraction     "location": "United States",
    "crop": "soybeans",
    "impact": "a 3°C warming would induce a shift in cropping intensity that increases revenue from soy systems by 1.3% overall, offsetting only a small fraction     "crop": "soybeans",
    "impact": "a 3°C warming would induce a shift in cropping intensity that increases revenue from soy systems by 1.3% overall, offsetting only a small fraction of the revenue impacts of predicted yield declines",
    "evidence": "a 3°C warming would induce a shift in cropping intensity that increases revenue from soy systems by 1.3% overall, offsetting only a small fraction of the revenue impacts of predicted yield declines"
  }
]