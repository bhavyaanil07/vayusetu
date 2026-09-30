import xml.etree.ElementTree as ET

from vayusetu import alerts as AL
from vayusetu import engine

NS = {"cap": "urn:oasis:names:tc:emergency:cap:1.2"}


def test_alerts_route_and_export_cap():
    st = engine.build_replay_state("2019-10-30 22:00")
    fcs = engine.forecast_all(st, ["Delhi", "Gurugram", "Mumbai"])
    alerts = engine.build_alerts(st, fcs, {"Delhi": engine.city_hotspots(st, "Delhi")})
    assert alerts and alerts[0].level == "Emergency"
    delhi_fc = next(a for a in alerts if a.kind == "spike_forecast" and a.city == "Delhi")
    assert any("CAQM" in x["name"] for x in delhi_fc.authorities)
    assert any("GRAP" in x for x in delhi_fc.actions)
    assert not any(a.city == "Mumbai" and a.level != "Advisory" for a in alerts)
    for a in alerts[:5]:
        root = ET.fromstring(AL.to_cap_xml(a).encode())
        assert root.find("cap:status", NS).text == "Exercise"  # replay alerts are exercises
        info = root.find("cap:info", NS)
        for tag in ("event", "urgency", "severity", "certainty", "headline", "area"):
            assert info.find(f"cap:{tag}", NS) is not None
    hs = [a for a in alerts if a.kind == "hotspot"]
    assert hs and all(not a.public for a in hs)  # responder-only
    assert AL.dispatch(alerts[0])["mode"] == "simulated"
