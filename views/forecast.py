"""Spike Forecast: 72-hour probabilistic forecasts, GRAP guidance and model evidence."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from vayusetu import aqi as A
from vayusetu import config, ui

mode, now_str = ui.current_key()
ss = st.session_state
state = ui.get_state(mode, now_str)
with st.spinner("Running VayuCast for every city..."):
    fcs = ui.get_forecasts(mode, now_str)

cities = sorted(fcs, key=lambda c: (c != "Delhi", c))
if ss.get("city") not in cities:
    ss["city"] = "Delhi" if "Delhi" in cities else cities[0]
sel_l, sel_r = st.columns([1, 3])
city = sel_l.selectbox("City", cities, index=cities.index(ss["city"]), key="fc_city")
ss["city"] = city
fc = fcs[city]
meta = config.CITIES[city]

ui.page_header(f"Spike Forecast · {city}",
               f"VayuCast issued {state.now:%d %b %Y, %H:%M} IST · 72-hour PM2.5 with calibrated uncertainty"
               + (" · blended with Copernicus CAMS" if fc.get("hybrid") else ""), state)

cols = st.columns(3)
for col, w in zip(cols, fc["windows"]):
    chip = ui.aqi_chip(w["aqi"], "AQI %.0f · %s" % (w["aqi"], w["category"]))
    col.markdown(
        f'<div class="vs-card"><div class="vs-row"><b style="color:#0b0b0b">{ui.esc(w["window"])}</b>'
        f'<span>{pd.Timestamp(w["start"]):%d %b %H:%M} - {pd.Timestamp(w["end"]):%d %b %H:%M}</span></div>'
        f'<div style="margin:.45rem 0">{chip}</div>'
        f'<div class="vs-row"><span>24-h mean PM2.5</span><b>{w["pm25_p50"]:.0f} µg/m³</b></div>'
        f'<div class="vs-row"><span>80% range</span><span>{w["pm25_p10"]:.0f} - {w["pm25_p90"]:.0f}</span></div>'
        f'<div class="vs-row" style="margin-top:.35rem"><span>P(Very Poor or worse)</span><b>{w["prob_very_poor"]:.0%}</b></div>'
        f'{ui.prob_bar(w["prob_very_poor"])}'
        f'<div class="vs-row"><span>P(Severe)</span><b>{w["prob_severe"]:.0%}</b></div>{ui.prob_bar(w["prob_severe"])}'
        f'<div class="vs-note">Peak hour ≈ {w["peak_hour_pm25"]:.0f} µg/m³ around {pd.Timestamp(w["peak_time"]):%d %b %H:%M}</div></div>',
        unsafe_allow_html=True)

show_actual = False
if state.mode == "replay":
    show_actual = st.toggle("Reveal what actually happened (replay only)", value=True,
                            help="Overlay the real CPCB measurements for the forecast period - the forecast itself only "
                                 "used data up to 'now'.")
ui.show_fig(ui.forecast_figure(state, city, fc, show_actual), key="fc_main")
st.caption("Shaded band: 80% forecast range (conformally calibrated). Background bands: NAQI categories for PM2.5.")

g_l, g_r = st.columns([1.1, 1], gap="medium")
with g_l:
    worst = max(fc["windows"], key=lambda w: w["aqi"])
    stage = A.grap_stage(worst["aqi"])
    if meta.ncr:
        st.markdown(f'<div class="vs-section">GRAP guidance (Delhi-NCR) · expected Stage '
                    f'{["-", "I", "II", "III", "IV"][stage]}</div>', unsafe_allow_html=True)
        if stage:
            st.markdown(f"Forecast AQI {worst['aqi']:.0f} ({worst['category']}) in the {worst['window'].lower()} window. "
                        "Measures to pre-position now:")
            for a in A.GRAP_ACTIONS[stage]:
                st.markdown(f"- {a}")
            st.caption("Aligned with the CAQM Graded Response Action Plan stages; verify against the latest CAQM schedule.")
        else:
            st.success("No GRAP stage expected in the next 72 hours.")
    else:
        st.markdown('<div class="vs-section">City action guidance</div>', unsafe_allow_html=True)
        st.markdown(f"Worst forecast window: **{worst['window']}**, AQI {worst['aqi']:.0f} ({worst['category']}).")
        st.markdown(f"- Health advice: {A.health_advice(worst['aqi'])}")
        if worst["aqi"] > 200:
            st.markdown("- Activate the city clean-air action plan (dust control, anti-burning patrols, traffic management).")
            st.markdown("- Target inspections at the hotspots flagged on the Hotspot Radar.")
with g_r:
    st.markdown('<div class="vs-section">Spike probability across cities</div>', unsafe_allow_html=True)
    rows = sorted(fcs.items(), key=lambda kv: -max(w["prob_very_poor"] for w in kv[1]["windows"]))[:12]
    Z = np.array([[w["prob_very_poor"] for w in f["windows"]] for _, f in rows])
    fig = go.Figure(go.Heatmap(
        z=Z, x=[w["window"] for w in fc["windows"]], y=[c for c, _ in rows], zmin=0, zmax=1,
        colorscale=[[0, "#f0efec"], [0.5, "#86b6ef"], [1, "#184f95"]], xgap=2, ygap=2,
        text=[[f"{v:.0%}" for v in r] for r in Z], texttemplate="%{text}", textfont=dict(size=11),
        colorbar=dict(title=dict(text="P(Very Poor+)", side="top", font=dict(size=11)), tickformat=".0%", thickness=10, len=0.75),
        hovertemplate="%{y} · %{x}<br>P(Very Poor or worse) %{z:.0%}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    ui.style_fig(fig, height=390, legend=False, margin=dict(l=10, r=10, t=10, b=10))
    ui.show_fig(fig, key="fc_matrix")

st.markdown('<div class="vs-section">Does it work? Evidence from out-of-sample testing on real CPCB data</div>',
            unsafe_allow_html=True)
metrics = ui.load_json("reports/forecast_metrics.json")
card = ui.load_json("data/models/forecaster_card.json")
wm = metrics["windows"]["excl_covid_lockdown"]
k = st.columns(4)
k[0].markdown(ui.kpi("Error vs. persistence · next 24 h", f"-{wm['0']['skill_vs_persistence']:.0%}",
                     f"MAE {wm['0']['mae_model']:.1f} vs {wm['0']['mae_persistence']:.1f} µg/m³"), unsafe_allow_html=True)
k[1].markdown(ui.kpi("Error vs. persistence · 48-72 h", f"-{wm['2']['skill_vs_persistence']:.0%}",
                     f"MAE {wm['2']['mae_model']:.1f} vs {wm['2']['mae_persistence']:.1f} µg/m³"), unsafe_allow_html=True)
k[2].markdown(ui.kpi("Very Poor+ days caught (F1)", f"{wm['0']['very_poor_f1']:.2f}",
                     f"precision {wm['0']['very_poor_precision']:.2f} · recall {wm['0']['very_poor_recall']:.2f}"), unsafe_allow_html=True)
k[3].markdown(ui.kpi("80% range really covers", f"{wm['0']['p10_p90_coverage']:.0%}", "of outcomes (calibrated)"),
              unsafe_allow_html=True)
st.caption("Test year Jul 2019 - Jun 2020 (never seen in training), 26 cities, one forecast per day at 18:00 IST. "
           "Persistence = 'the next 24 h will look like the last 24 h', a strong baseline for smog. COVID-19 lockdown "
           "weeks excluded here; full-period numbers are in reports/forecast_metrics.json.")

t1, t2, t3, t4 = st.tabs(["Backtest: Delhi, Oct-Nov 2019", "Skill by horizon", "Are the probabilities honest?", "What drives it"])
with t1:
    bt = ui.load_csv("reports/delhi_nov2019_backtest.csv", ("issue", "start"))
    b = bt[bt["window"] == "Next 24 h"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(b["start"]) + list(b["start"][::-1]), y=list(b["pred_p90"]) + list(b["pred_p10"][::-1]),
                             fill="toself", fillcolor="rgba(235,104,52,0.13)", line=dict(width=0), name="80% range",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=b["start"], y=b["pred_p50"], name="VayuCast (issued the evening before)",
                             line=dict(color=ui.SERIES[1], width=2), mode="lines+markers", marker=dict(size=8, line=dict(color="white", width=2)),
                             hovertemplate="%{x|%d %b}<br>Forecast %{y:.0f} µg/m³<extra></extra>"))
    fig.add_trace(go.Scatter(x=b["start"], y=b["actual"], name="Actual 24-h mean (CPCB)", mode="lines+markers",
                             line=dict(color=ui.INK, width=2), marker=dict(size=8, line=dict(color="white", width=2)),
                             hovertemplate="%{x|%d %b}<br>Actual %{y:.0f} µg/m³<extra></extra>"))
    ui.aqi_bands(fig, float(np.nanmax([b["actual"].max(), b["pred_p90"].max()])) * 1.05, b["start"].min(), b["start"].max())
    fig.add_vline(x=pd.Timestamp("2019-11-01 19:00"), line_width=1, line_color=ui.BASE)
    fig.add_annotation(x=pd.Timestamp("2019-11-01 19:00"), y=1, yref="paper", text="Health emergency declared (1 Nov)",
                       showarrow=False, xanchor="left", yanchor="top", font=dict(size=11, color=ui.INK2))
    fig.update_yaxes(title_text="Next-day mean PM2.5 (µg/m³)")
    ui.style_fig(fig, height=380, margin=dict(l=10, r=80, t=40, b=10))
    ui.show_fig(fig, key="bt")
    row = bt[(bt["window"] == "24-48 h") & (bt["issue"] == pd.Timestamp("2019-10-30 18:00"))]
    row2 = bt[(bt["window"] == "24-48 h") & (bt["issue"] == pd.Timestamp("2019-11-01 18:00"))]
    if len(row) and len(row2):
        st.markdown(f"On **30 Oct, 6 pm**, VayuCast gave a **{row['prob_severe'].iloc[0]:.0%} chance of Severe air** "
                    f"24-48 h ahead (actual: {row['actual'].iloc[0]:.0f} µg/m³, Severe). On **1 Nov**, the day the "
                    f"emergency was declared, it gave **{row2['prob_severe'].iloc[0]:.0%}** for the following day "
                    f"(actual {row2['actual'].iloc[0]:.0f} µg/m³). It could not foresee the wind change that cleared the "
                    "smog on 4-5 Nov - that is why live mode blends in meteorology-driven CAMS forecasts.")
with t2:
    rows = []
    for lab, key in (("Next 24 h", "0"), ("24-48 h", "1"), ("48-72 h", "2")):
        rows.append({"Window": lab, "VayuCast MAE": wm[key]["mae_model"], "Persistence MAE": wm[key]["mae_persistence"],
                     "Skill": wm[key]["skill_vs_persistence"], "Category accuracy": wm[key]["category_accuracy_model"],
                     "Very Poor+ F1": wm[key]["very_poor_f1"], "Persistence F1": wm[key]["very_poor_f1_persistence"]})
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Window"], y=df["Persistence MAE"], name="Persistence", marker=dict(color=ui.MUTED, cornerradius=4), width=0.3,
                         offset=-0.31, hovertemplate="%{x}: %{y:.1f} µg/m³<extra>Persistence</extra>"))
    fig.add_trace(go.Bar(x=df["Window"], y=df["VayuCast MAE"], name="VayuCast", marker=dict(color=ui.SERIES[0], cornerradius=4), width=0.3,
                         offset=0.01, text=[f"-{s:.0%}" for s in df["Skill"]], textposition="outside", textfont=dict(color=ui.INK2),
                         hovertemplate="%{x}: %{y:.1f} µg/m³<extra>VayuCast</extra>"))
    fig.update_yaxes(title_text="Mean absolute error, 24-h PM2.5 (µg/m³)")
    ui.style_fig(fig, height=330)
    ui.show_fig(fig, key="skill")
    st.dataframe(df.round(3), hide_index=True)
with t3:
    rel = pd.DataFrame(metrics["reliability_next24h"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=ui.BASE, width=1), name="Perfectly calibrated",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=rel["mean_prob"], y=rel["observed_freq"], mode="lines+markers", name="VayuCast",
                             line=dict(color=ui.SERIES[0], width=2), marker=dict(size=9, line=dict(color="white", width=2)),
                             text=rel["n"], hovertemplate="Forecast %{x:.0%} → happened %{y:.0%}<br>n = %{text}<extra></extra>"))
    fig.update_xaxes(title_text="Forecast probability of a Very Poor+ day", tickformat=".0%", range=[0, 1])
    fig.update_yaxes(title_text="How often it happened", tickformat=".0%", range=[0, 1])
    ui.style_fig(fig, height=360)
    ui.show_fig(fig, key="rel")
    st.caption(f"Brier score {wm['0']['brier_very_poor']:.3f} (lower is better). When VayuCast says 90%, it happens about 9 times in 10.")
with t4:
    imp = pd.DataFrame(metrics["importance_window_model"]).head(12)[::-1]
    names = {"lp_mean24": "Last 24 h mean PM2.5", "anom_mean24": "How unusual the last 24 h were", "doy_cos": "Season (day of year)",
             "lp_mean168": "Last 7 days mean", "clim_win": "Typical level for this season/hour", "h": "Hours ahead",
             "clim_shift": "Seasonal trend ahead", "anom_0": "How unusual this hour is", "lat": "Latitude",
             "lp_d24": "Change vs. yesterday", "diwali_d": "Days to/from Diwali", "lp_lag168": "Same hour last week",
             "lp_std24": "Volatility (24 h)", "nb_n": "Neighbouring stations reporting", "lp10_mean24": "PM10 (24 h)",
             "nb_anom24": "Pollution anomaly in neighbouring cities", "doy_sin": "Season (sin)", "lon": "Longitude",
             "lp_mean72": "Last 3 days mean", "win": "Forecast window", "nb_d24": "Neighbours' 24-h change"}
    fig = go.Figure(go.Bar(x=imp["importance"], y=[names.get(f, f) for f in imp["feature"]], orientation="h",
                           marker=dict(color=ui.SERIES[0], cornerradius=4), width=0.55,
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="Permutation importance (increase in error when shuffled)")
    ui.style_fig(fig, height=420, legend=False)
    ui.show_fig(fig, key="imp")
with st.expander("Model card"):
    st.json(card)
