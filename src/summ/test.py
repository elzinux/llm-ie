from sentence_transformers import util

from src.summ.config import MODEL_RESOURCE_CONFIG
from src.utils.model_resources import get_embedding_model
from src.utils.normalize_text import normalize_text

_DEFAULT_EMBEDDING_MODEL_NAME = MODEL_RESOURCE_CONFIG.embedding_model_name


def is_key_sentence_covered(
    key_sentence: str,
    summary: str,
) -> float:
    normalized_key_sentence = normalize_text(key_sentence)
    normalized_summary = normalize_text(summary)
    if normalized_key_sentence in normalized_summary:
        return 1

    if not normalized_key_sentence or not normalized_summary:
        return 0

    if normalized_key_sentence in normalized_summary:
        return 1

    model = get_embedding_model(_DEFAULT_EMBEDDING_MODEL_NAME)

    embeddings = model.encode(
        [normalized_key_sentence, normalized_summary],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    similarity = float(util.cos_sim(embeddings[0], embeddings[1]).item())
    return round(similarity, 4)


if __name__ == "__main__":
    print("-=-" * 5)
    s1 = "increase the revenue of soy agriculture by 1.3% on average"
    s2 = "Climate change is projected to have large negative effects on US soybean and wheat yields—17% and 24% decreases, respectively, under a 3°C uniform warming scenario—leading to a 17.3% aggregate decline in revenue from winter wheat–soy double-cropping (DC) systems due to yield effects alone; in contrast, adoption of DC practices is estimated to increase soy agriculture revenue by only 1.3% on average, meaning DC expansion would offset just 1.3 percentage points of the yield-driven revenue decline, resulting in a net 16% reduction in soy revenues."
    s3 = "adoption of DC practices is estimated to increase soy agriculture revenue by only 1.3% on average, meaning DC expansion would offset just 1.3 percentage points of the yield-driven revenue decline"

    # res1 = is_key_sentence_covered(
    #     s1,
    #     s2,
    # )
    # res2 = is_key_sentence_covered(
    #     s1,
    #     s3,
    # )
    # print(res1, res2)

    ss = """
    Based on the crop trial data during 1981–2009 at 57 agricultural experimental stations across the North Eastern China Plain (NECP) and the middle and lower reaches of Yangtze River (MLRYR), we investigated how major climate variables had changed and how the climate change had affected crop growth and yield in a setting in which agronomic management practices were taken based on actual weather. We found a significant warming trend during rice growing season, and a general decreasing trend in solar radiation (SRD) in the MLRYR during 1981–2009. Rice transplanting, heading, and maturity dates were generally advanced, but the heading and maturity dates of single rice in the MLRYR (YZ_SR) and NECP (NE_SR) were delayed. Climate warming had a negative impact on growth period lengths at about 80% of the investigated stations. Nevertheless, the actual growth period lengths of YZ_SR and NE_SR, as well as the actual length of reproductive growth period (RGP) of early rice in the MLRYR (YZ_ER), were generally prolonged due to adoption of cultivars with longer growth period to obtain higher yield. In contrast, the actual growth period length of late rice in the MLRYR (YZ_LR) was shortened by both climate warming and adoption of early mature cultivars to prevent cold damage and obtain higher yield. During 1981–2009, climate warming and decrease in SRD changed the yield of YZ_ER by 0.59 to 2.4%; climate warming during RGP increased the yield of YZ_LR by 8.38–9.56%; climate warming and decrease in SRD jointly reduced yield of YZ_SR by 7.14–9.68%; climate warming and increase in SRD jointly increased the yield of NE_SR by 1.01–3.29%. Our study suggests that rice production in China has been affected by climate change, yet at the same time changes in varieties continue to be the major factor driving yield and growing period trends.
    """
    print(ss)
