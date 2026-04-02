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
                "the intervention, management practice, treatment, action, "
                "or policy applied to the crop"
            ),
            "location": (
                "the research location, study site, or place associated with the crop"
            ),
            "crop": "the crop name",
            "impact": (
                "the explicit effect or outcome of the intervention_or_policy "
                "on the crop"
            ),
            "evidence": "the shortest exact quote that directly supports the record",
        },
    }
