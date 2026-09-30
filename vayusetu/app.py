"""
VayuSetu - federated clean-air intelligence for Indian cities.

Run locally:   streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="VayuSetu · Clean-air intelligence", page_icon="🌬️", layout="wide",
                   initial_sidebar_state="expanded")

from vayusetu import ui  # noqa: E402  (import after page config)

ui.inject_css()
ui.init_session()

pages = {
    "Monitor": [
        st.Page("views/command_center.py", title="Command Center", icon="🛰️", default=True),
        st.Page("views/hotspot_radar.py", title="Hotspot Radar", icon="📍"),
        st.Page("views/forecast.py", title="Spike Forecast", icon="📈"),
    ],
    "Act": [
        st.Page("views/citizen_reports.py", title="Citizen Reports", icon="📸"),
        st.Page("views/alerts.py", title="Alerts & Response", icon="🚨"),
    ],
    "Collaborate": [
        st.Page("views/federated.py", title="Federated Network", icon="🤝"),
        st.Page("views/open_data.py", title="Open Data & API", icon="🔌"),
    ],
}
nav = st.navigation(pages)
ui.sidebar()
nav.run()
