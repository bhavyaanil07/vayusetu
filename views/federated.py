"""Federated Network: cities share models, not data."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from vayusetu import config, ui
from vayusetu.models import federated as FL

mode, now_str = ui.current_key()
state = ui.get_state(mode, now_str)
res = ui.load_json("data/models/federated_results.json")
card = ui.load_json("data/models/federated_card.json")
fc_card = ui.load_json("data/models/forecaster_card.json")
summ = res["summary"]
pc = pd.DataFrame(res["per_city"])
cold = pd.DataFrame(res["cold_start"])

ui.page_header("Federated Network",
               "Cities and states train shared forecasting models together - raw measurements never leave their servers",
               state)

s1, s2, s3 = st.columns(3)
for col, (t, d) in zip((s1, s2, s3), [
        ("1 · Data stays home", "Each city node keeps its station, citizen-sensor and photo data on its own server "
                                "(data sovereignty, DPDP-friendly)."),
        ("2 · Only model weights travel", "The hub sends a model; each node trains locally and returns weights. The hub "
                                          "averages them (FedAvg). Optional differential-privacy noise."),
        ("3 · Everyone gets a better model", "Small or new cities borrow strength from the whole network on day one and "
                                             "fine-tune locally.")]):
    col.markdown(f'<div class="vs-card"><h4>{t}</h4><div class="meta">{d}</div></div>', unsafe_allow_html=True)

imp_local = 1 - summ["mean_mae_federated_seen"] / summ["mean_mae_local_only_seen"]
imp_pers = 1 - summ["mean_mae_federated_seen"] / summ["mean_mae_persistence_seen"]
gap_c = summ["mean_mae_federated_seen"] / summ["mean_mae_centralised_seen"] - 1
lk = cold[cold["city"] == "Lucknow"].iloc[0] if (cold["city"] == "Lucknow").any() else cold.iloc[0]
ui.kpi_row([
    ("Cities in the federation", f"{summ['federated_cities']}", "real CPCB data, 2015-2019"),
    ("Error vs. training alone", f"-{imp_local:.0%}", f"federated beats local-only in {summ['cities_where_federated_beats_local']} of {summ['federated_cities']}"),
    ("Error vs. persistence", f"-{imp_pers:.0%}", "next-day 24-h mean PM2.5"),
    ("Gap to pooling all raw data", f"+{gap_c:.0%}", "privacy costs little accuracy"),
    ("New city, 14 days of data", f"{lk['mae_local_14d']:.0f} → {lk['mae_federated_finetuned']:.0f}", f"MAE µg/m³ ({lk['city']}): alone → federated"),
])
st.write("")

left, right = st.columns([1.35, 1], gap="medium")
with left:
    hub = (21.15, 79.09)  # geographic centre of India (Nagpur) as a neutral hub location
    nodes = []
    for _, r in pc.iterrows():
        c = config.CITIES[r["city"]]
        nodes.append({"lon": c.lon, "lat": c.lat, "city": r["city"], "member": bool(r["in_federation"]),
                      "radius": 9000 + 9 * r["train_samples"],
                      "color": ui.hex_rgb(ui.SERIES[0], 235) if r["in_federation"] else [255, 255, 255, 255],
                      "line": ui.hex_rgb(ui.SERIES[0], 255),
                      **ui.tipf(r["city"], f"Member node · {r['train_samples']:,} training samples"
                                if r["in_federation"] else "Joins later (uses the shared model zero-shot)")})
    nd = pd.DataFrame(nodes)
    arcs = [{"source": [n["lon"], n["lat"]], "target": [hub[1], hub[0]], **ui.tipf(f"{n['city']} ⇄ hub", "Model weights only - no raw data")}
            for n in nodes if n["member"]]
    layers = [ui._states_layer(12),
              pdk.Layer("ArcLayer", arcs, get_source_position="source", get_target_position="target",
                        get_source_color=ui.hex_rgb(ui.SERIES[0], 160), get_target_color=ui.hex_rgb(ui.SERIES[2], 200),
                        get_width=2, pickable=True),
              *[ui.dots(ui.records(nd[(nd["radius"] >= lo) & (nd["radius"] < hi)], ["lon", "lat", "color", "line", *ui.TIP_COLS]),
                        px, "color", "line", 2) for lo, hi, px in ((0, 25000, 6), (25000, 50000, 9), (50000, 1e12, 12))
                if ((nd["radius"] >= lo) & (nd["radius"] < hi)).any()],
              ui.dots([{"lon": hub[1], "lat": hub[0], **ui.tipf("Federation hub", "FedAvg aggregation of city models")}], 10,
                      ui.hex_rgb(ui.SERIES[2], 255), [255, 255, 255, 255], 2)]
    st.pydeck_chart(ui._deck(layers, pdk.ViewState(latitude=21.3, longitude=81.0, zoom=3.5)), height=470)
    st.markdown(ui.legend_html([("Member node (size = data volume)", ui.SERIES[0]),
                                ("Hub (averages weights)", ui.SERIES[2])]), unsafe_allow_html=True)
with right:
    h = pd.DataFrame(res["history"])
    fig = go.Figure(go.Scatter(x=h["round"], y=h["eval_loss"], mode="lines+markers", line=dict(color=ui.SERIES[0], width=2),
                               marker=dict(size=7, line=dict(color="white", width=2)),
                               hovertemplate="Round %{x}<br>Loss %{y:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="Federated round")
    fig.update_yaxes(title_text="Held-out loss (Huber)")
    ui.style_fig(fig, height=235, legend=False)
    fig.update_layout(title=dict(text="Global model improves every round", x=0, font=dict(size=13)))
    ui.show_fig(fig, key="fl_loss")
    d = pd.DataFrame(res["dp"])
    fig = go.Figure(go.Bar(x=[("No noise" if v == 0 else f"σ = {v}") for v in d["noise_multiplier"]], y=d["mae"],
                           marker=dict(color=ui.SERIES[0], cornerradius=4), width=0.45,
                           text=[f"{v:.1f}" for v in d["mae"]], textposition="outside", textfont=dict(color=ui.INK2),
                           hovertemplate="%{x}: MAE %{y:.1f} µg/m³<extra></extra>"))
    fig.update_yaxes(title_text="MAE (µg/m³)")
    ui.style_fig(fig, height=220, legend=False)
    fig.update_layout(title=dict(text="Differential-privacy noise vs. accuracy", x=0, font=dict(size=13)))
    ui.show_fig(fig, key="fl_dp")

st.markdown('<div class="vs-section">City by city: training alone vs. together (next-day PM2.5 error, test year)</div>',
            unsafe_allow_html=True)
mem = pc[pc["in_federation"]].sort_values("mae_local_only")
fig = go.Figure()
series = [("Persistence", "mae_persistence", ui.MUTED, "circle"), ("Local-only", "mae_local_only", ui.SERIES[1], "diamond"),
          ("Federated", "mae_federated", ui.SERIES[0], "circle"), ("Centralised (pooled raw data)", "mae_centralised", ui.SERIES[2], "square")]
for name, col, color, sym in series:
    fig.add_trace(go.Scatter(x=mem[col], y=mem["city"], mode="markers", name=name,
                             marker=dict(color=color, size=10, symbol=sym, line=dict(color="white", width=2)),
                             hovertemplate="%{y}<br>" + name + ": %{x:.1f} µg/m³<extra></extra>"))
fig.update_xaxes(title_text="Mean absolute error (µg/m³) - lower is better", showgrid=True, gridcolor=ui.GRID)
fig.update_yaxes(showgrid=False)
ui.style_fig(fig, height=560)
ui.show_fig(fig, key="fl_cities")
st.caption("Cities with little history (e.g. Brajrajnagar, Jorapokhar, Patna, Ahmedabad) overfit when training alone; "
           "the federated model fixes that without anyone sharing raw data.")

a, b = st.columns([1.2, 1], gap="medium")
with a:
    st.markdown('<div class="vs-section">Cold start: a city joins with only 14 days of data</div>', unsafe_allow_html=True)
    fig = go.Figure()
    for i, (name, col) in enumerate([("Persistence", "mae_persistence"), ("Train alone on 14 days", "mae_local_14d"),
                                     ("Federated model, zero-shot", "mae_federated_zero_shot"),
                                     ("Federated + 14-day fine-tune", "mae_federated_finetuned")]):
        color = [ui.MUTED, ui.SERIES[1], ui.SERIES[0], ui.SERIES[2]][i]
        fig.add_trace(go.Bar(x=cold["city"], y=cold[col], name=name, marker=dict(color=color, cornerradius=4),
                             hovertemplate="%{x}: %{y:.1f} µg/m³<extra>" + name + "</extra>"))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.12)
    fig.update_yaxes(title_text="MAE (µg/m³)")
    ui.style_fig(fig, height=330)
    ui.show_fig(fig, key="fl_cold")
with b:
    st.markdown('<div class="vs-section">Try it: run federated training live</div>', unsafe_allow_html=True)
    demo = np.load(config.MODELS_DIR / "fl_demo_clients.npz")
    avail = sorted({k.split("__")[0] for k in demo.files if not k.startswith("__")})
    chosen = st.multiselect("Participating cities", avail, default=avail[:8])
    rounds = st.slider("Rounds", 5, 30, 12)
    noise = st.select_slider("Privacy noise (σ)", [0.0, 0.25, 0.5], value=0.0)
    if st.button("Start federated training", type="primary", icon="🤝", disabled=len(chosen) < 2):
        clients = {c: (demo[f"{c}__X"].astype(np.float32), demo[f"{c}__y"].astype(np.float32)) for c in chosen}
        Xe, ye, be = demo["__eval_X"].astype(np.float32), demo["__eval_y"].astype(np.float32), demo["__eval_b"]
        obs = np.expm1(ye + be)
        prog = st.progress(0.0, text="Round 0")
        chart = st.empty()
        maes = []

        def cb(rec, w):
            pred = np.expm1(FL.predict(w, Xe) + be)
            maes.append(float(np.mean(np.abs(pred - obs))))
            prog.progress(rec["round"] / rounds, text=f"Round {rec['round']}/{rounds} · {rec['clients']} cities averaged")
            f = go.Figure(go.Scatter(x=list(range(1, len(maes) + 1)), y=maes, mode="lines+markers",
                                     line=dict(color=ui.SERIES[0], width=2), marker=dict(size=7)))
            f.update_xaxes(title_text="Round")
            f.update_yaxes(title_text="Held-out MAE (µg/m³)")
            ui.style_fig(f, height=230, legend=False)
            chart.plotly_chart(f, theme=None, key=f"live_fl_{len(maes)}")

        FL.run_fedavg(clients, rounds=rounds, local_epochs=2, clip=1.0 if noise > 0 else None,
                      noise_multiplier=noise, seed=7, callback=cb)
        st.success(f"Done: held-out MAE {maes[-1]:.1f} µg/m³ after {rounds} rounds with {len(chosen)} cities. "
                   "No raw readings left any city's node.")

st.markdown('<div class="vs-section">Model registry · interoperable model exchange</div>', unsafe_allow_html=True)
reg = pd.DataFrame([
    {"Model": fc_card["name"], "ID": fc_card["model_id"], "Version": fc_card["version"], "Owner": fc_card["owner_node"],
     "Type": "Gradient boosting, quantile", "Size (MB)": fc_card["size_mb"],
     "Headline": f"-{fc_card['headline_metrics']['skill_vs_persistence_next24h']:.0%} error vs persistence (24 h)"},
    {"Model": card["name"], "ID": card["model_id"], "Version": card["version"], "Owner": "Federation (19 city nodes)",
     "Type": "MLP via FedAvg", "Size (MB)": round((config.MODELS_DIR / "federated_global.npz").stat().st_size / 1e6, 3),
     "Headline": f"-{imp_local:.0%} error vs training alone"},
])
st.dataframe(reg, hide_index=True)
d1, d2, d3, d4 = st.columns(4)
d1.download_button("VayuCast model card (JSON)", json.dumps(fc_card, indent=2), "vayucast_card.json", "application/json", on_click="ignore")
d2.download_button("Federated model card (JSON)", json.dumps(card, indent=2), "federated_card.json", "application/json", on_click="ignore")
d3.download_button("Federated weights (.npz)", (config.MODELS_DIR / "federated_global.npz").read_bytes(), "federated_global.npz",
                   "application/octet-stream", on_click="ignore")
d4.download_button("Experiment results (JSON)", json.dumps(res, indent=2), "federated_results.json", "application/json", on_click="ignore")
st.caption("Models travel with a machine-readable card (training data, metrics, limits, licence) so a receiving city "
           "knows exactly what it is adopting. Weights are plain NumPy arrays - no vendor lock-in.")
