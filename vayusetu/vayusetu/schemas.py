"""
Open data schemas for interoperability between city nodes, states and the hub.

These pydantic models validate API payloads and are exported as JSON Schema to
docs/schemas/ (run `python scripts/export_schemas.py`). Field names follow
CPCB/NAQI conventions; alerts additionally use OASIS CAP 1.2 (XML).
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """One sensor reading (hourly mean). Compatible with OGC SensorThings 'Observation' concepts."""
    sensor_id: str = Field(..., examples=["DEL-0123"])
    time: datetime = Field(..., description="End of the averaging hour, ISO 8601 with offset (IST = +05:30)")
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    pm25: float = Field(..., ge=0, le=2000, description="PM2.5, µg/m³")
    pm10: float | None = Field(None, ge=0, le=3000, description="PM10, µg/m³")
    co: float | None = Field(None, ge=0, description="Carbon monoxide, mg/m³")
    no2: float | None = Field(None, ge=0, description="Nitrogen dioxide, µg/m³")
    rh: float | None = Field(None, ge=0, le=100, description="Relative humidity, %")
    temp: float | None = Field(None, ge=-30, le=60, description="Air temperature, °C")
    sensor_model: str | None = Field(None, examples=["SDS011"])
    calibrated: bool = Field(False, description="True if humidity/bias correction was already applied")
    quality_flag: Literal["ok", "offline", "stuck", "suspect-drift"] = "ok"


class CitizenReport(BaseModel):
    """A geotagged citizen photo/observation report."""
    report_id: str | None = None
    city: str = Field(..., examples=["Delhi"])
    time: datetime
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    category: Literal["Fire / open burning", "Smoke / open burning", "Industrial smoke / stack",
                      "Dust / construction", "Haze / smog", "Vehicle exhaust / congestion"]
    note: str | None = Field(None, max_length=500)
    haze_index: float | None = Field(None, ge=0, le=1, description="From on-device dark-channel analysis")
    image_url: str | None = None


class SourceProbabilities(BaseModel):
    industrial: float
    burning: float
    dust: float
    traffic: float


class Hotspot(BaseModel):
    """A detected localised pollution hotspot (served as a GeoJSON Feature's properties)."""
    hotspot_id: str
    city: str
    time: datetime
    lat: float
    lon: float
    radius_km: float
    peak_pm25: float
    peak_excess: float = Field(..., description="µg/m³ above the neighbourhood background")
    source: Literal["industrial", "burning", "dust", "traffic"]
    source_probs: SourceProbabilities
    confidence: float = Field(..., ge=0, le=1)
    severity: Literal["Moderate", "High", "Critical"]
    wards: list[str]
    evidence: list[str]


class ForecastWindow(BaseModel):
    window: Literal["Next 24 h", "24-48 h", "48-72 h"]
    start: datetime
    end: datetime
    pm25_p10: float
    pm25_p50: float
    pm25_p90: float
    aqi: float
    category: str
    prob_very_poor: float = Field(..., ge=0, le=1)
    prob_severe: float = Field(..., ge=0, le=1)


class Forecast(BaseModel):
    city: str
    issue_time: datetime
    model_id: str = "vayucast-pm25-v1.1"
    windows: list[ForecastWindow]
    hourly: list[dict] = Field(..., description="[{time, h, p10, p50, p90}] hourly PM2.5 in µg/m³")


class ModelCard(BaseModel):
    model_id: str
    name: str
    version: str
    license: str
    headline_metrics: dict
