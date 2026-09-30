"""Live connectors are tested against recorded response formats (no network needed)."""
import pandas as pd

from vayusetu.data import live

OM_SINGLE = {"latitude": 28.625, "longitude": 77.25, "utc_offset_seconds": 19800, "timezone": "Asia/Kolkata",
             "hourly_units": {"time": "iso8601", "pm2_5": "μg/m³", "carbon_monoxide": "μg/m³"},
             "hourly": {"time": ["2026-10-01T00:00", "2026-10-01T01:00"], "pm2_5": [120.5, None],
                        "pm10": [180.0, 175.0], "carbon_monoxide": [1500.0, 1400.0]}}


def test_open_meteo_single_and_multi():
    df = live.parse_open_meteo(OM_SINGLE, ["Delhi"], live.AQ_VARS)
    assert list(df["city"].unique()) == ["Delhi"] and len(df) == 2
    assert df["co"].iloc[0] == 1.5  # ug/m3 -> mg/m3
    assert pd.isna(df["pm25"].iloc[1])
    multi = live.parse_open_meteo([OM_SINGLE, OM_SINGLE], ["Delhi", "Mumbai"], live.AQ_VARS)
    assert set(multi["city"]) == {"Delhi", "Mumbai"} and len(multi) == 4


FIRMS_CSV = """latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight
30.25,75.84,330.1,0.4,0.37,2026-10-20,0812,N,VIIRS,n,2.0NRT,290.2,6.3,D
30.90,75.85,340.5,0.5,0.4,2026-10-20,812,N,VIIRS,h,2.0NRT,295.0,12.1,D
31.40,70.10,335.0,0.4,0.4,2026-10-20,0815,N,VIIRS,n,2.0NRT,292.0,4.0,D
"""


def test_firms_parse_time_and_bbox():
    df = live.parse_firms_csv(FIRMS_CSV, "VIIRS S-NPP")
    assert len(df) == 3  # bbox keeps the Pakistan point; state assignment later drops it
    assert df["time"].iloc[0] == pd.Timestamp("2026-10-20 13:42")  # 08:12 UTC -> 13:42 IST
    from vayusetu.geo import assign_states
    states = assign_states(df["latitude"].values, df["longitude"].values)
    assert states[0] == "Punjab" and states[2] == ""


def test_cpcb_records_both_field_variants():
    old = [{"state": "Delhi", "city": "Delhi", "station": "ITO, Delhi - CPCB", "last_update": "01-10-2026 10:00:00",
            "latitude": "28.628", "longitude": "77.241", "pollutant_id": "PM2.5", "pollutant_avg": "210"}]
    new = [dict(old[0], **{"avg_value": "205"})]
    del new[0]["pollutant_avg"]
    for recs, val in ((old, 210), (new, 205)):
        df = live.parse_cpcb_records(recs)
        assert df["pm25"].iloc[0] == val
    means = live.cpcb_city_means(live.parse_cpcb_records(old))
    assert round(means["Delhi"]) == 210


def test_thingspeak_and_gibs():
    js = {"channel": {"name": "Kottayam node", "latitude": "9.59", "longitude": "76.52"},
          "feeds": [{"created_at": "2026-10-01T04:30:00Z", "entry_id": 1, "field1": "35.2", "field2": "60.1"}]}
    df = live.parse_thingspeak(js)
    assert df["pm25"].iloc[0] == 35.2 and df["time"].iloc[0] == pd.Timestamp("2026-10-01 10:00")
    url = live.gibs_wms_url("MODIS_Terra_CorrectedReflectance_TrueColor", "2026-10-01", (76.9, 28.3, 77.5, 28.9))
    assert "BBOX=76.9,28.3,77.5,28.9" in url and "TIME=2026-10-01" in url


def test_live_state_falls_back_to_replay(monkeypatch):
    from vayusetu import engine
    monkeypatch.setattr(live, "fetch_open_meteo_aq", lambda **k: live.FetchResult(pd.DataFrame(), False, "x", "offline"))
    st = engine.build_live_state({})
    assert st.mode == "replay" and "unavailable" in st.notices[0]
