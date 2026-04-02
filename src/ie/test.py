from src.ie.pipeline import run_ie_pipeline

import json


def pretty_print(data) -> None:
    """以更易读的 JSON 格式打印结果。"""
    print(json.dumps(data, ensure_ascii=False, indent=2))


print("开始执行 IE 测试脚本。")
TEST_TEXT = """A warming climate expands the frost-free season, plausibly allowing for increased cropping intensity in temperate regions. This paper assesses the potential of multiple cropping to offset the projected negative effects of climate change on agricultural yields in the United States. We use cross-sectional variation in observed land cover, soil characteristics, and climate to estimate farmers' propensity to double-crop winter wheat with soybeans. Our estimates imply that under current economic conditions, a 3 C warming would result in an increase of 2.1 percentage points in the share of current soybean area double cropped, primarily driven by expansions in cooler regions. A fixed-effects panel model of county yields further indicates that yields of double-cropped soybeans are about 12% lower than those of single-cropped soybeans. Accounting for changes in cropping intensity and attendant effects on soybean yields, we project that at current prices, a 3 C warming would induce a shift in cropping intensity that increases revenue from soy systems by 1.3% overall, offsetting only a small fraction of the revenue impacts of predicted yield declines.""".strip()
result = run_ie_pipeline(TEST_TEXT, True)
for key in result:
    print("输出分段：")
    pretty_print(result[key])
print("IE 测试脚本执行结束。")
