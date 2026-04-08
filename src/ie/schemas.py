def build_task_schema() -> dict:
    """构建当前信息抽取任务的字段定义。"""
    return {
        "fields": [
            "intervention_or_policy",
            "location",
            "crop",
            "impact",
            "evidence",
        ],
        "required_fields": [
            "intervention_or_policy",
            "location",
            "crop",
            "impact",
            "evidence",
        ],
        "allow_multiple_records": True,
        "field_descriptions": {
            "intervention_or_policy": (
                "The intervention, management practice, treatment, action, "
                "or policy applied to the crop."
            ),
            "location": (
                "The location where the crop is grown or related to the crop."
            ),
            "crop": "The crop name.",
            "impact": (
                "The explicit effect or outcome of the intervention_or_policy "
                "on the crop."
            ),
            "evidence": "The shortest exact quote that directly supports the record.",
        },
    }
