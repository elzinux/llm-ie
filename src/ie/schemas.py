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
                """Specific management practices, treatments, actions, or policies (e.g., "drip irrigation", "nitrogen application", "subsidies")."""
            ),
            "location": (
                """Geographical sites, administrative regions, or specific environments where the crop is grown (e.g., "North China Plain", "greenhouse")."""
            ),
            "crop": """Common or scientific names of the crop (e.g., "Triticum aestivum", "maize").""",
            "impact": (
                """The explicit, measurable outcome or effect on the crop(e.g., "yield increase", "biomass reduction")."""
            ),
            "evidence": "The shortest exact quote that directly supports the record.",
        },
    }
