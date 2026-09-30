import math

from vayusetu import aqi as A


def test_pm25_breakpoints():
    assert A.subindex("pm25", 30) == 50
    assert A.subindex("pm25", 60) == 100
    assert A.subindex("pm25", 90) == 200
    assert A.subindex("pm25", 120) == 300
    assert A.subindex("pm25", 250) == 400
    assert A.subindex("pm25", 2000) == 500  # capped


def test_overall_is_max_and_nan_safe():
    assert A.aqi(pm25=100, pm10=50) == A.subindex("pm25", 100)
    assert A.aqi(pm25=float("nan"), pm10=300) == A.subindex("pm10", 300)
    assert math.isnan(A.aqi(pm25=float("nan")))


def test_categories_and_grap():
    assert A.category(50) == "Good" and A.category(51) == "Satisfactory"
    assert A.category(301) == "Very Poor" and A.category(450) == "Severe"
    assert [A.grap_stage(v) for v in (150, 250, 350, 420, 480)] == [0, 1, 2, 3, 4]
    assert A.color_hex(420) == "#8B1A1A"
