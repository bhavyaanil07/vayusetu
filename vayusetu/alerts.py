"""
Alert engine: turns hotspots, forecasts and regional smoke transport into
actionable alerts for the right authority.

Alerts are exported in the Common Alerting Protocol (CAP 1.2, OASIS), the
standard used by India's NDMA SACHET integrated alert system, so VayuSetu
alerts can flow into existing state and national dissemination channels.
"""
from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from xml.dom import minidom

import pandas as pd

from . import aqi as A
from . import config

CAP_NS = "urn:oasis:names:tc:emergency:cap:1.2"
SENDER = "alerts.vayusetu@example.org"

LEVEL_ORDER = {"Emergency": 0, "Warning": 1, "Advisory": 2}
LEVEL_COLOR = {"Emergency": "#8B1A1A", "Warning": "#E53935", "Advisory": "#F7941D"}


@dataclass
class Alert:
    alert_id: str
    created: pd.Timestamp
    city: str
    kind: str                      # hotspot | spike_forecast | regional_transport
    level: str                     # Advisory | Warning | Emergency
    title: str
    summary: str
    actions: list = field(default_factory=list)
    authorities: list = field(default_factory=list)   # [{"name", "role"}]
    evidence: list = field(default_factory=list)
    lat: float | None = None
    lon: float | None = None
    radius_km: float = 3.0
    wards: list = field(default_factory=list)
    valid_from: pd.Timestamp | None = None
    valid_to: pd.Timestamp | None = None
    confidence: float = 0.0
    cap_severity: str = "Moderate"   # Extreme | Severe | Moderate | Minor
    cap_urgency: str = "Expected"    # Immediate | Expected | Future
    cap_certainty: str = "Likely"    # Observed | Likely | Possible
    public: bool = False
    exercise: bool = False
    source_ref: str = ""
    status: str = "New"
    parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("created", "valid_from", "valid_to"):
            d[k] = None if d[k] is None else pd.Timestamp(d[k]).isoformat()
        return d


# ----------------------------------------------------------------------------
# Routing
# ----------------------------------------------------------------------------
def authorities_for(city: str, kind: str, source: str | None = None, level: str = "Warning") -> list[dict]:
    c = config.CITIES[city]
    spcb = config.spcb_for(city)
    muni = c.municipal_body
    sdma = "Delhi Disaster Management Authority (DDMA)" if c.state == "Delhi" else f"{c.state} State Disaster Management Authority"
    health = "Delhi Health & Family Welfare Department" if c.state == "Delhi" else f"{c.state} Health Department"
    out: list[dict] = []
    if kind == "hotspot":
        if source == "industrial":
            out += [{"name": spcb, "role": "Inspect unit, verify consent conditions, issue closure/show-cause"},
                    {"name": f"District Magistrate ({city})", "role": "Enforcement support"}]
        elif source == "burning":
            out += [{"name": muni, "role": "Stop open/landfill burning; deploy sanitation & water tenders"},
                    {"name": f"{city} Fire & Emergency Services", "role": "Douse smouldering fire"},
                    {"name": spcb, "role": "Penalty under Solid Waste Management Rules"}]
        elif source == "dust":
            out += [{"name": muni, "role": "C&D site inspection; enforce dust-control (covering, sprinkling)"},
                    {"name": spcb, "role": "Environmental compensation for dust-norm violations"}]
        elif source == "traffic":
            out += [{"name": f"{city} Traffic Police", "role": "Decongest junction; stop idling; divert heavy vehicles"},
                    {"name": f"{c.state} Transport Department", "role": "PUC drive at hotspot"}]
    elif kind == "spike_forecast":
        out += [{"name": spcb, "role": "Activate city air-quality action plan"},
                {"name": muni, "role": "Intensify sweeping, sprinkling and anti-burning patrols"},
                {"name": sdma, "role": "Public advisory via SACHET (CAP)"},
                {"name": health, "role": "Alert hospitals; advisories for vulnerable groups"}]
        if c.ncr:
            out.insert(0, {"name": config.CAQM, "role": "Invoke GRAP stage for NCR"})
        if level == "Emergency":
            out.append({"name": f"{c.state} Education Department", "role": "Consider hybrid/online classes"})
    elif kind == "regional_transport":
        out += [{"name": "Agriculture Departments of Punjab, Haryana & Uttar Pradesh", "role": "Crop-residue management teams to fire clusters"},
                {"name": "District Magistrates of top fire districts", "role": "Field enforcement against stubble burning"},
                {"name": spcb, "role": "Prepare for incoming smoke; pre-emptive local controls"}]
        if c.ncr:
            out.insert(0, {"name": config.CAQM, "role": "Coordinate NCR & neighbouring states"})
    return out


# ----------------------------------------------------------------------------
# Alert generation
# ----------------------------------------------------------------------------
def _tz(t) -> pd.Timestamp:
    return pd.Timestamp(t)


def alerts_from_hotspots(city: str, hotspots: list[dict], now, exercise: bool = False,
                         min_confidence: float = 0.6) -> list[Alert]:
    out = []
    for h in hotspots:
        if h["confidence"] < min_confidence:
            continue
        src = h["source"]
        label = config.SOURCE_TYPES[src]["label"]
        level = "Warning" if (h["severity"] == "Critical" or h["peak_excess"] >= 120) else "Advisory"
        where = ", ".join(h["wards"][:3]) + (" ..." if len(h["wards"]) > 3 else "")
        poi_txt = ""
        if h.get("nearest_poi"):
            near = sorted(h["nearest_poi"].values(), key=lambda x: x["km"])[0]
            if near["km"] < 2.5:
                poi_txt = f" near {near['name']}"
        actions = {
            "industrial": ["Dispatch SPCB flying squad for night inspection", "Check stack monitoring (OCEMS) data and consent limits",
                           "Seal/stop unit if violation confirmed"],
            "burning": ["Send sanitation team and water tender", "Identify and penalise burning as per SWM Rules",
                        "Deploy patrol for the next 3 nights"],
            "dust": ["Inspect construction/demolition site", "Enforce covering, wind-breakers and water sprinkling",
                     "Issue environmental compensation notice if non-compliant"],
            "traffic": ["Deploy traffic marshals to clear congestion", "Stop idling of buses/trucks",
                        "Run PUC check drive at the junction"],
        }[src]
        out.append(Alert(
            alert_id=f"VS-{h['hotspot_id']}", created=_tz(now), city=city, kind="hotspot", level=level,
            title=f"{config.SOURCE_TYPES[src]['icon']} {label} hotspot{poi_txt}",
            summary=(f"Local PM2.5 is {h['peak_excess']:.0f} µg/m³ above the neighbourhood background "
                     f"({h['peak_pm25']:.0f} µg/m³ peak) across {h['n_sensors']} sensor(s) in {where}. "
                     f"Likely source: {label.lower()} ({h['source_probs'][src]:.0%})."),
            actions=actions, authorities=authorities_for(city, "hotspot", src, level),
            evidence=h["evidence"], lat=h["lat"], lon=h["lon"], radius_km=max(1.0, h["radius_km"]),
            wards=h["wards"], valid_from=_tz(now), valid_to=_tz(now) + pd.Timedelta(hours=6),
            confidence=h["confidence"], cap_severity="Severe" if level == "Warning" else "Moderate",
            cap_urgency="Immediate", cap_certainty="Observed", public=False, exercise=exercise,
            source_ref=h["hotspot_id"],
            parameters={"PM25_EXCESS_UGM3": round(h["peak_excess"], 1), "SOURCE": src,
                        "SOURCE_PROBABILITY": round(h["source_probs"][src], 2),
                        "CONFIDENCE": round(h["confidence"], 2)},
        ))
    return out


def alerts_from_forecast(city: str, fc: dict, now, exercise: bool = False) -> list[Alert]:
    th = config.THRESHOLDS
    wins = fc["windows"]
    worst = max(wins, key=lambda w: w["prob_very_poor"])
    severe = max(wins, key=lambda w: w["prob_severe"])
    c = config.CITIES[city]
    if severe["prob_severe"] >= 0.5:
        level, win, prob, cat_txt = "Emergency", severe, severe["prob_severe"], "Severe"
    elif worst["prob_very_poor"] >= th.spike_prob_alert:
        level, win, prob, cat_txt = "Warning", worst, worst["prob_very_poor"], "Very Poor or worse"
    else:
        poor = max(wins, key=lambda w: w["prob_poor"])
        if poor["prob_poor"] >= 0.6:
            level, win, prob, cat_txt = "Advisory", poor, poor["prob_poor"], "Poor or worse"
        else:
            return []
    aqi_val = win["aqi"]
    stage = A.grap_stage(aqi_val)
    actions = []
    if c.ncr and stage:
        actions.append(f"GRAP Stage {'I' * stage if stage < 4 else 'IV'} expected - pre-position these measures:")
        actions += A.GRAP_ACTIONS[stage]
    else:
        actions += ["Activate the city clean-air action plan for the forecast window",
                    "Intensify inspections at VayuSetu hotspots and known source clusters",
                    "Issue a health advisory for children, elderly and people with heart/lung disease"]
    start, end = pd.Timestamp(win["start"]), pd.Timestamp(win["end"])
    return [Alert(
        alert_id=f"VS-FC-{city[:3].upper()}-{pd.Timestamp(now):%Y%m%d%H}", created=_tz(now), city=city,
        kind="spike_forecast", level=level,
        title=f"📈 {cat_txt} air forecast for {city} ({win['window']})",
        summary=(f"VayuCast forecasts a 24-h average PM2.5 of {win['pm25_p50']:.0f} µg/m³ "
                 f"(80% range {win['pm25_p10']:.0f}-{win['pm25_p90']:.0f}), AQI about {aqi_val:.0f} "
                 f"({win['category']}), between {start:%d %b %H:%M} and {end:%d %b %H:%M}. "
                 f"Probability of {cat_txt}: {prob:.0%}."),
        actions=actions, authorities=authorities_for(city, "spike_forecast", level=level),
        evidence=[f"Last 24-h mean PM2.5: {fc['persistence_24h']:.0f} µg/m³",
                  f"Peak hour forecast: {win['peak_hour_pm25']:.0f} µg/m³ at {pd.Timestamp(win['peak_time']):%d %b %H:%M}",
                  "Model: VayuCast quantile gradient boosting (backtested on CPCB data)"],
        lat=c.lat, lon=c.lon, radius_km=25.0, wards=[], valid_from=start, valid_to=end,
        confidence=prob, cap_severity="Extreme" if level == "Emergency" else ("Severe" if level == "Warning" else "Moderate"),
        cap_urgency="Expected", cap_certainty="Likely" if prob >= 0.5 else "Possible", public=True,
        exercise=exercise, source_ref=f"forecast:{city}:{pd.Timestamp(now):%Y%m%d%H}",
        parameters={"FORECAST_PM25_UGM3": round(win["pm25_p50"], 1), "FORECAST_AQI": round(aqi_val),
                    "PROBABILITY": round(prob, 2), "GRAP_STAGE": stage if c.ncr else "n/a"},
    )]


def alert_from_regional(city: str, upwind: dict, wind_speed: float, wind_dir: float, now,
                        exercise: bool = False, source_label: str = "NASA VIIRS") -> Alert | None:
    th = config.THRESHOLDS
    if upwind["count"] < th.fire_count_alert:
        return None
    c = config.CITIES[city]
    level = "Warning" if upwind["count"] >= 2 * th.fire_count_alert else "Advisory"
    return Alert(
        alert_id=f"VS-RT-{city[:3].upper()}-{pd.Timestamp(now):%Y%m%d%H}", created=_tz(now), city=city,
        kind="regional_transport", level=level,
        title=f"🌾 Crop-residue smoke moving towards {city}",
        summary=(f"{upwind['count']:,} satellite fire detections ({source_label}, last 24 h) lie upwind of {city} "
                 f"within {config.THRESHOLDS.fire_upwind_km:.0f} km (wind from {wind_dir:.0f} deg at {wind_speed:.1f} m/s). "
                 f"Total fire radiative power {upwind['frp_mw']:,.0f} MW."),
        actions=["Target crop-residue management teams to the densest fire clusters",
                 "Pre-emptive local controls in the receiving city for the next 48 h",
                 "Public advisory if forecast AQI crosses Very Poor"],
        authorities=authorities_for(city, "regional_transport"),
        evidence=[f"Upwind sector {upwind['sector'][0] % 360:.0f}-{upwind['sector'][1] % 360:.0f} deg",
                  f"Nearest upwind fire {upwind['nearest_km']:.0f} km" if upwind.get("nearest_km") else ""],
        lat=c.lat, lon=c.lon, radius_km=50.0, valid_from=_tz(now), valid_to=_tz(now) + pd.Timedelta(hours=48),
        confidence=min(0.95, 0.5 + upwind["count"] / 2000), cap_severity="Severe" if level == "Warning" else "Moderate",
        cap_urgency="Expected", cap_certainty="Likely", public=False, exercise=exercise,
        source_ref=f"fires:{city}", parameters={"UPWIND_FIRES_24H": upwind["count"], "FRP_MW": round(upwind["frp_mw"])},
    )


def sort_alerts(alerts: list[Alert]) -> list[Alert]:
    return sorted(alerts, key=lambda a: (LEVEL_ORDER.get(a.level, 9), -a.confidence))


# ----------------------------------------------------------------------------
# CAP 1.2 export
# ----------------------------------------------------------------------------
def _iso(t) -> str:
    t = pd.Timestamp(t)
    if t.tzinfo is None:
        t = t.tz_localize("Asia/Kolkata")
    return t.isoformat(timespec="seconds")


def to_cap_xml(a: Alert) -> str:
    ET.register_namespace("", CAP_NS)
    q = lambda tag: f"{{{CAP_NS}}}{tag}"  # noqa: E731
    root = ET.Element(q("alert"))

    def sub(parent, tag, text):
        e = ET.SubElement(parent, q(tag))
        e.text = str(text)
        return e

    sub(root, "identifier", a.alert_id)
    sub(root, "sender", SENDER)
    sub(root, "sent", _iso(a.created))
    sub(root, "status", "Exercise" if a.exercise else "Actual")
    sub(root, "msgType", "Alert")
    sub(root, "scope", "Public" if a.public else "Restricted")
    if not a.public:
        sub(root, "restriction", "For authorised government responders")
    info = ET.SubElement(root, q("info"))
    sub(info, "language", "en-IN")
    sub(info, "category", "Env")
    sub(info, "category", "Health")
    sub(info, "event", {"hotspot": "Localised air pollution hotspot",
                        "spike_forecast": "Air quality deterioration forecast",
                        "regional_transport": "Regional smoke transport"}[a.kind])
    sub(info, "responseType", "Monitor" if a.kind != "spike_forecast" else "Prepare")
    sub(info, "urgency", a.cap_urgency)
    sub(info, "severity", a.cap_severity)
    sub(info, "certainty", a.cap_certainty)
    if a.valid_from is not None:
        sub(info, "effective", _iso(a.valid_from))
    if a.valid_to is not None:
        sub(info, "expires", _iso(a.valid_to))
    sub(info, "senderName", f"VayuSetu - {a.city} node")
    sub(info, "headline", a.title.encode("ascii", "ignore").decode().strip())
    sub(info, "description", a.summary)
    if a.actions:
        sub(info, "instruction", " | ".join(a.actions))
    for k, v in a.parameters.items():
        p = ET.SubElement(info, q("parameter"))
        sub(p, "valueName", k)
        sub(p, "value", v)
    area = ET.SubElement(info, q("area"))
    desc = f"{a.city}" + (f" - wards: {', '.join(a.wards[:6])}" if a.wards else "")
    sub(area, "areaDesc", desc)
    if a.lat is not None:
        sub(area, "circle", f"{a.lat:.5f},{a.lon:.5f} {a.radius_km:.1f}")
    xml = ET.tostring(root, encoding="unicode")
    return minidom.parseString(xml).toprettyxml(indent="  ")


# ----------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------
def dispatch(a: Alert, webhook_url: str | None = None, timeout: float = 8.0) -> dict:
    """Send an alert to a webhook (Slack/Teams/Discord-compatible JSON) if configured.

    Without a webhook the dispatch is simulated and only logged.
    """
    url = webhook_url or os.environ.get("VAYUSETU_WEBHOOK_URL")
    payload = {"text": f"[{a.level}] {a.title}\n{a.summary}", "alert": a.to_dict()}
    if not url:
        return {"ok": True, "mode": "simulated", "recipients": [x["name"] for x in a.authorities]}
    try:
        import requests
        r = requests.post(url, data=json.dumps(payload, default=str),
                          headers={"Content-Type": "application/json"}, timeout=timeout)
        return {"ok": r.ok, "mode": "webhook", "status_code": r.status_code}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "mode": "webhook", "error": str(exc)}
