"""Alerts & Response: triage, route and dispatch alerts; export CAP 1.2 for SACHET."""
import json

import pandas as pd
import streamlit as st

from vayusetu import alerts as AL
from vayusetu import ui

mode, now_str = ui.current_key()
ss = st.session_state
state = ui.get_state(mode, now_str)
with st.spinner("Collecting alerts from forecasts, hotspot scans and satellite fires..."):
    alerts = ui.get_alerts(mode, now_str)

ui.page_header("Alerts & Response",
               f"{len(alerts)} alerts at {state.now:%d %b %Y, %H:%M} IST · routed to the responsible authority · "
               "exportable as CAP 1.2 (the format of India's SACHET alert system)", state)
if state.exercise:
    st.caption("ℹ️ Replay mode: alerts are marked as an Exercise in CAP and dispatch is simulated unless a webhook is configured.")

status = ss["alert_status"]
f1, f2, f3 = st.columns([1.2, 1.4, 1])
levels = f1.pills("Level", ["Emergency", "Warning", "Advisory"], selection_mode="multi",
                  default=["Emergency", "Warning", "Advisory"])
kinds_map = {"Forecast spikes": "spike_forecast", "Local hotspots": "hotspot", "Regional smoke": "regional_transport"}
kinds = f2.pills("Type", list(kinds_map), selection_mode="multi", default=list(kinds_map))
cities = ["All cities"] + sorted({a.city for a in alerts})
city = f3.selectbox("City", cities)
sel = [a for a in alerts if a.level in (levels or []) and a.kind in [kinds_map[k] for k in (kinds or [])]
       and (city == "All cities" or a.city == city)]

lv = pd.Series([a.level for a in alerts]).value_counts() if alerts else pd.Series(dtype=int)
st_counts = pd.Series([status.get(a.alert_id, "New") for a in alerts]).value_counts() if alerts else pd.Series(dtype=int)
ui.kpi_row([
    ("Emergency", f"{lv.get('Emergency', 0)}", "Severe air expected"),
    ("Warning", f"{lv.get('Warning', 0)}", "Very Poor air / strong hotspots"),
    ("Advisory", f"{lv.get('Advisory', 0)}", "early notice"),
    ("Acknowledged", f"{st_counts.get('Acknowledged', 0) + st_counts.get('Dispatched', 0) + st_counts.get('Resolved', 0)}", "by responders"),
    ("Resolved", f"{st_counts.get('Resolved', 0)}", "closed with action taken"),
])
st.write("")

webhook = ui.get_secrets().get("VAYUSETU_WEBHOOK_URL")


def _set(aid: str, new: str, alert: AL.Alert) -> None:
    status[aid] = new
    info = ""
    if new == "Dispatched":
        r = AL.dispatch(alert, webhook)
        info = f"{r['mode']}: {'ok' if r['ok'] else r.get('error', r.get('status_code'))}"
    ss["alert_log"].append({"time": pd.Timestamp.now().strftime("%H:%M:%S"), "alert": aid, "city": alert.city,
                            "action": new, "detail": info or "-"})


if not sel:
    st.info("No alerts match these filters.")
for a in sel[:40]:
    cur = status.get(a.alert_id, "New")
    with st.container(border=True):
        h1, h2 = st.columns([4, 1.3])
        h1.markdown(f"{ui.level_chip(a.level)} &nbsp;<b style='font-size:1.02rem;color:#0b0b0b'>{ui.esc(a.title)}</b>"
                    f"<div class='vs-note' style='margin-top:.2rem'>{ui.esc(a.city)} · issued {pd.Timestamp(a.created):%d %b %H:%M}"
                    f" · valid to {pd.Timestamp(a.valid_to):%d %b %H:%M} · confidence {a.confidence:.0%} · "
                    f"status <b>{ui.esc(cur)}</b></div>", unsafe_allow_html=True)
        b1, b2, b3 = h2.columns(3)
        b1.button("Ack", key=f"ack_{a.alert_id}", disabled=cur != "New", on_click=_set, args=(a.alert_id, "Acknowledged", a),
                  help="Acknowledge")
        b2.button("Send", key=f"dis_{a.alert_id}", disabled=cur in ("Dispatched", "Resolved"), type="primary",
                  on_click=_set, args=(a.alert_id, "Dispatched", a), help="Dispatch to the routed authorities")
        b3.button("Done", key=f"res_{a.alert_id}", disabled=cur == "Resolved", on_click=_set, args=(a.alert_id, "Resolved", a),
                  help="Mark resolved")
        st.markdown(f"<div style='color:#52514e;font-size:.9rem;margin:.2rem 0 .4rem 0'>{ui.esc(a.summary)}</div>",
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown("**Recommended actions**")
            st.markdown("\n".join(f"- {ui.esc(x)}" for x in a.actions))
        with c2:
            st.markdown("**Routed to**")
            st.markdown("\n".join(f"- **{ui.esc(x['name'])}** - {ui.esc(x['role'])}" for x in a.authorities))
        with st.expander("Evidence and CAP 1.2 export"):
            ev = [e for e in a.evidence if e]
            if ev:
                st.markdown("\n".join(f"- {ui.esc(e)}" for e in ev))
            xml = AL.to_cap_xml(a)
            st.code(xml, language="xml")
            d1, d2 = st.columns(2)
            d1.download_button("Download CAP XML", xml, file_name=f"{a.alert_id}.xml", mime="application/xml",
                               key=f"x_{a.alert_id}", on_click="ignore")
            d2.download_button("Download JSON", json.dumps(a.to_dict(), indent=2, default=str), file_name=f"{a.alert_id}.json",
                               mime="application/json", key=f"j_{a.alert_id}", on_click="ignore")

if ss["alert_log"]:
    st.markdown('<div class="vs-section">Response log</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(ss["alert_log"])[::-1], hide_index=True)
