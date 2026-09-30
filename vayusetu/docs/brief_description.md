# VayuSetu — brief description

## Short version (about 100 words)

VayuSetu is a federated AI platform that finds hidden, street-level pollution hotspots in Indian cities. It forecasts
air-quality spikes 72 hours ahead and alerts the right authority. It fuses a ward-level network of low-cost citizen
sensors and geotagged photos with NASA satellite fire data, Copernicus air-quality forecasts and weather.

In a Delhi replay it detected 97% of hidden pollution events, against 39% for a regulatory-density network. On a
year of unseen CPCB data from 26 cities, it cut forecast error by 9–11% compared with persistence. Cities share
models, not raw data: federated learning cut error by 9% compared with training alone.

## Full version (about 230 words)

India monitors air quality at city scale, so industrial emissions, open burning and construction dust at street
level go unseen. **VayuSetu** is a federated, AI-powered clean-air platform that closes this gap.

It combines citizen-sourced data (about one low-cost PM2.5 sensor per ward, plus geotagged photo reports analysed for
haze, smoke and fire) with satellite data (NASA FIRMS fire detections, NASA GIBS imagery, Copernicus CAMS) and
meteorology. With these it does three things:

1. **Finds hidden hotspots.** Local-background and Getis-Ord Gi\* statistics pinpoint street-level events. An
   explainable engine attributes each one to industry, burning, dust or traffic, using chemical fingerprints,
   time-of-day patterns, photos and fire data. In a Delhi replay it found 97% of hidden events (against 39% for a
   regulatory-density network), with zero false alerts at the alert threshold.
2. **Forecasts spikes 72 hours ahead across five economic corridors,** with calibrated probabilities. On a year of
   unseen CPCB data, error fell 9–11% compared with persistence. It gave a 61% warning of Severe air two days before
   Delhi's November 2019 health emergency.
3. **Alerts the right authority:** pollution control boards, municipal bodies, traffic police, and CAQM with GRAP
   stage guidance. Alerts export as CAP 1.2, the format of NDMA's SACHET system.

Cities and states **share models, not data.** Federated learning cut error by 9% compared with training alone, and
lets a new city start with just two weeks of data. Open schemas, model cards, GeoJSON and a REST API make it
interoperable.

Built with Python, scikit-learn, Streamlit and FastAPI.
