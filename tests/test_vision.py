from vayusetu import config
from vayusetu.models import vision


def _a(name):
    return vision.analyze_image(config.SAMPLES_DIR / name)


def test_sample_categories():
    assert _a("clear_skyline.jpg")["category"] == "Clear sky"
    assert _a("smoke_plume.jpg")["category"] in ("Smoke / open burning", "Fire / open burning")
    assert _a("dust_site.jpg")["category"] == "Dust / construction"
    assert _a("industrial_stack.jpg")["category"] == "Industrial smoke / stack"


def test_haze_ordering_and_credibility():
    clear, hazy = _a("clear_skyline.jpg"), _a("hazy_skyline.jpg")
    assert clear["haze_index"] < hazy["haze_index"]
    hi, _ = vision.report_credibility(hazy, 0.8, sensor_excess=80, minutes_old=30, has_gps=True)
    lo, _ = vision.report_credibility(clear, 0.3, sensor_excess=0, minutes_old=3000, has_gps=False)
    assert hi > 0.8 > lo
