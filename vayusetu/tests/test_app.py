"""Every page of the Streamlit app renders without exceptions (replay mode)."""
import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["views/command_center.py", "views/hotspot_radar.py", "views/forecast.py", "views/citizen_reports.py",
         "views/alerts.py", "views/federated.py", "views/open_data.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file("../app.py", default_timeout=300)
    at.run()
    if page != PAGES[0]:
        at.switch_page(page)
        at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_pages_after_citizen_report_submission():
    """Regression: a submitted citizen report must flow into hotspots and alerts without errors."""
    at = AppTest.from_file("../app.py", default_timeout=300)
    at.run()
    at.session_state["user_reports"] = [{
        "report_id": "U-TEST01", "time": "2019-10-30 22:00:00", "lat": 28.624, "lon": 77.32,
        "category": "Smoke / open burning", "kind": "burning", "haze_score": 0.6, "reporter_trust": 0.6,
        "image": "smoke_plume.jpg", "note": "test report", "source": "citizen (you)", "event_id": None, "city": "Delhi"}]
    for page in ["views/hotspot_radar.py", "views/alerts.py", "views/citizen_reports.py", "views/command_center.py"]:
        at.switch_page(page)
        at.run()
        assert not at.exception, (page, [e.value for e in at.exception])
