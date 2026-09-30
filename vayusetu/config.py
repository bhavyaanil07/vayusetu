"""
Central configuration for VayuSetu.

Everything that describes the "world" the platform works in lives here:
cities, economic corridors, the authorities that receive alerts, and the
points of interest used by the source-attribution engine.

Coordinates of points of interest (POIs) are approximate (about 1 km) and are
used only to explain hotspots and to place simulated citizen-sensor events.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CPCB_PARQUET = DATA_DIR / "cpcb" / "city_hour.parquet"
GEO_DIR = DATA_DIR / "geo"
WARDS_DIR = GEO_DIR / "wards"
SENSORS_FILE = GEO_DIR / "sensors.json"
MODELS_DIR = DATA_DIR / "models"
SAMPLES_DIR = DATA_DIR / "samples"
REPORTS_DIR = ROOT / "reports"

APP_NAME = "VayuSetu"
APP_TAGLINE = "Federated clean-air intelligence for Indian cities"
APP_VERSION = "1.0.0"

# Replay scenario: the Delhi-NCR smog episode of Oct-Nov 2019 (real CPCB data).
# On 1 Nov 2019 a public health emergency was declared in Delhi-NCR.
REPLAY_START = "2019-10-24 00:00"
REPLAY_END = "2019-11-17 23:00"
REPLAY_DEFAULT_TIME = "2019-10-30 22:00"


@dataclass(frozen=True)
class City:
    name: str
    lat: float
    lon: float
    state: str
    corridors: tuple[str, ...]
    municipal_body: str
    has_wards: bool = False
    ncr: bool = False
    population_m: float = 1.0  # millions (approximate, urban agglomeration)


# The 26 cities with CPCB hourly records in the bundled dataset.
CITIES: dict[str, City] = {
    c.name: c
    for c in [
        City("Delhi", 28.6139, 77.2090, "Delhi", ("AKIC", "DMIC"), "Municipal Corporation of Delhi (MCD)", True, True, 32.9),
        City("Gurugram", 28.4595, 77.0266, "Haryana", ("DMIC",), "Municipal Corporation of Gurugram", False, True, 1.5),
        City("Chandigarh", 30.7333, 76.7794, "Chandigarh", ("AKIC",), "Municipal Corporation Chandigarh", False, False, 1.2),
        City("Amritsar", 31.6340, 74.8723, "Punjab", ("AKIC",), "Municipal Corporation Amritsar", False, False, 1.3),
        City("Lucknow", 26.8467, 80.9462, "Uttar Pradesh", ("AKIC",), "Lucknow Municipal Corporation", True, False, 3.9),
        City("Patna", 25.5941, 85.1376, "Bihar", ("AKIC",), "Patna Municipal Corporation", False, False, 2.5),
        City("Jorapokhar", 23.7070, 86.4130, "Jharkhand", ("AKIC",), "Dhanbad Municipal Corporation", False, False, 1.3),
        City("Kolkata", 22.5726, 88.3639, "West Bengal", ("AKIC", "ECEC"), "Kolkata Municipal Corporation", True, False, 15.3),
        City("Jaipur", 26.9124, 75.7873, "Rajasthan", ("DMIC",), "Jaipur Municipal Corporation", True, False, 4.2),
        City("Ahmedabad", 23.0225, 72.5714, "Gujarat", ("DMIC",), "Ahmedabad Municipal Corporation", True, False, 8.9),
        City("Mumbai", 19.0760, 72.8777, "Maharashtra", ("DMIC",), "Brihanmumbai Municipal Corporation", False, False, 21.7),
        City("Bhopal", 23.2599, 77.4126, "Madhya Pradesh", ("Central",), "Bhopal Municipal Corporation", False, False, 2.5),
        City("Hyderabad", 17.3850, 78.4867, "Telangana", ("HBIC",), "Greater Hyderabad Municipal Corporation", True, False, 11.0),
        City("Bengaluru", 12.9716, 77.5946, "Karnataka", ("CBIC", "HBIC"), "Greater Bengaluru Authority", True, False, 13.6),
        City("Chennai", 13.0827, 80.2707, "Tamil Nadu", ("CBIC", "ECEC"), "Greater Chennai Corporation", True, False, 11.8),
        City("Coimbatore", 11.0168, 76.9558, "Tamil Nadu", ("CBIC",), "Coimbatore City Municipal Corporation", False, False, 3.0),
        City("Kochi", 9.9674, 76.3180, "Kerala", ("CBIC",), "Kochi Municipal Corporation", True, False, 3.3),
        City("Ernakulam", 10.0746, 76.3013, "Kerala", ("CBIC",), "Kochi Municipal Corporation", False, False, 3.3),
        City("Thiruvananthapuram", 8.5241, 76.9366, "Kerala", ("Kerala",), "Thiruvananthapuram Corporation", False, False, 1.8),
        City("Visakhapatnam", 17.6868, 83.2185, "Andhra Pradesh", ("ECEC",), "Greater Visakhapatnam Municipal Corporation", False, False, 2.4),
        City("Amaravati", 16.5150, 80.5180, "Andhra Pradesh", ("ECEC",), "Amaravati Capital Region authorities", False, False, 0.5),
        City("Talcher", 20.9500, 85.2333, "Odisha", ("ECEC",), "Talcher Municipality", False, False, 0.2),
        City("Brajrajnagar", 21.8160, 83.9180, "Odisha", ("ECEC",), "Brajrajnagar Municipality", False, False, 0.1),
        City("Guwahati", 26.1445, 91.7362, "Assam", ("Northeast",), "Guwahati Municipal Corporation", False, False, 1.2),
        City("Shillong", 25.5788, 91.8933, "Meghalaya", ("Northeast",), "Shillong Municipal Board", False, False, 0.4),
        City("Aizawl", 23.7271, 92.7176, "Mizoram", ("Northeast",), "Aizawl Municipal Corporation", False, False, 0.4),
    ]
}

# Economic corridors (National Industrial Corridor Development programme and
# regional groupings used for coordination).
CORRIDORS: dict[str, dict] = {
    "AKIC": {"name": "Amritsar-Kolkata Industrial Corridor", "color": [230, 126, 34],
             "path": ["Amritsar", "Chandigarh", "Delhi", "Lucknow", "Patna", "Jorapokhar", "Kolkata"]},
    "DMIC": {"name": "Delhi-Mumbai Industrial Corridor", "color": [52, 152, 219],
             "path": ["Delhi", "Gurugram", "Jaipur", "Ahmedabad", "Mumbai"]},
    "CBIC": {"name": "Chennai-Bengaluru Industrial Corridor (Kochi extension)", "color": [26, 188, 156],
             "path": ["Chennai", "Bengaluru", "Coimbatore", "Kochi", "Ernakulam"]},
    "HBIC": {"name": "Hyderabad-Bengaluru Industrial Corridor", "color": [155, 89, 182],
             "path": ["Hyderabad", "Bengaluru"]},
    "ECEC": {"name": "East Coast Economic Corridor", "color": [241, 196, 15],
             "path": ["Kolkata", "Brajrajnagar", "Talcher", "Visakhapatnam", "Amaravati", "Chennai"]},
    "Central": {"name": "Central India", "color": [149, 165, 166], "path": ["Bhopal"]},
    "Kerala": {"name": "Kerala coast", "color": [39, 174, 96], "path": ["Thiruvananthapuram", "Kochi"]},
    "Northeast": {"name": "North-East cities", "color": [127, 140, 141], "path": ["Guwahati", "Shillong", "Aizawl"]},
}

MAJOR_CORRIDORS = ["AKIC", "DMIC", "CBIC", "HBIC", "ECEC"]

# State Pollution Control Boards / Committees.
SPCB: dict[str, str] = {
    "Delhi": "Delhi Pollution Control Committee (DPCC)",
    "Haryana": "Haryana State Pollution Control Board",
    "Punjab": "Punjab Pollution Control Board",
    "Chandigarh": "Chandigarh Pollution Control Committee",
    "Uttar Pradesh": "Uttar Pradesh Pollution Control Board",
    "Bihar": "Bihar State Pollution Control Board",
    "Jharkhand": "Jharkhand State Pollution Control Board",
    "West Bengal": "West Bengal Pollution Control Board",
    "Rajasthan": "Rajasthan State Pollution Control Board",
    "Gujarat": "Gujarat Pollution Control Board",
    "Maharashtra": "Maharashtra Pollution Control Board",
    "Madhya Pradesh": "Madhya Pradesh Pollution Control Board",
    "Telangana": "Telangana State Pollution Control Board",
    "Karnataka": "Karnataka State Pollution Control Board",
    "Tamil Nadu": "Tamil Nadu Pollution Control Board",
    "Kerala": "Kerala State Pollution Control Board",
    "Andhra Pradesh": "Andhra Pradesh Pollution Control Board",
    "Odisha": "State Pollution Control Board, Odisha",
    "Assam": "Pollution Control Board, Assam",
    "Meghalaya": "Meghalaya State Pollution Control Board",
    "Mizoram": "Mizoram Pollution Control Board",
}

CAQM = "Commission for Air Quality Management in NCR (CAQM)"

# Upwind agricultural-burning source region used for regional-transport checks.
CROP_BURNING_STATES = ["Punjab", "Haryana", "Uttar Pradesh"]


@dataclass(frozen=True)
class POI:
    name: str
    lat: float
    lon: float
    kind: str  # industrial | landfill | traffic | port


POIS: dict[str, list[POI]] = {
    "Delhi": [
        POI("Bawana Industrial Area", 28.7978, 77.0390, "industrial"),
        POI("Narela Industrial Area", 28.8467, 77.1031, "industrial"),
        POI("Okhla Industrial Area", 28.5328, 77.2744, "industrial"),
        POI("Wazirpur Industrial Area", 28.6996, 77.1653, "industrial"),
        POI("Mundka Industrial Area", 28.6834, 77.0296, "industrial"),
        POI("Mayapuri Industrial Area", 28.6376, 77.1269, "industrial"),
        POI("Patparganj Industrial Area", 28.6380, 77.3050, "industrial"),
        POI("Ghazipur Landfill", 28.6240, 77.3265, "landfill"),
        POI("Bhalswa Landfill", 28.7406, 77.1582, "landfill"),
        POI("Okhla Landfill", 28.5105, 77.2860, "landfill"),
        POI("ITO Junction", 28.6289, 77.2410, "traffic"),
        POI("Anand Vihar ISBT", 28.6469, 77.3159, "traffic"),
        POI("Ashram Chowk", 28.5708, 77.2585, "traffic"),
        POI("Dhaula Kuan", 28.5918, 77.1612, "traffic"),
        POI("Mukarba Chowk", 28.7330, 77.1600, "traffic"),
        POI("Kashmere Gate ISBT", 28.6675, 77.2285, "traffic"),
    ],
    "Kochi": [
        POI("Eloor-Edayar Industrial Belt", 10.0746, 76.3013, "industrial"),
        POI("Ambalamugal Refinery Area", 9.9600, 76.3700, "industrial"),
        POI("Brahmapuram Waste Plant", 9.9947, 76.3698, "landfill"),
        POI("Vyttila Mobility Hub", 9.9674, 76.3180, "traffic"),
        POI("Edappally Junction", 10.0244, 76.3084, "traffic"),
        POI("Willingdon Island Port", 9.9667, 76.2667, "port"),
    ],
    "Kolkata": [
        POI("Dhapa Dumping Ground", 22.5470, 88.4210, "landfill"),
        POI("Taratala Industrial Area", 22.5010, 88.3170, "industrial"),
        POI("Howrah Industrial Belt", 22.5900, 88.3100, "industrial"),
        POI("Esplanade", 22.5646, 88.3510, "traffic"),
        POI("Ultadanga", 22.5950, 88.3930, "traffic"),
    ],
    "Lucknow": [
        POI("Talkatora Industrial Area", 26.8330, 80.8930, "industrial"),
        POI("Shivri Waste Plant", 26.7700, 80.9900, "landfill"),
        POI("Charbagh", 26.8320, 80.9230, "traffic"),
        POI("Hazratganj", 26.8510, 80.9460, "traffic"),
    ],
    "Bengaluru": [
        POI("Peenya Industrial Area", 13.0285, 77.5197, "industrial"),
        POI("Bommasandra Industrial Area", 12.8160, 77.6960, "industrial"),
        POI("Silk Board Junction", 12.9172, 77.6230, "traffic"),
        POI("Hebbal Flyover", 13.0358, 77.5970, "traffic"),
        POI("Mandur Landfill", 13.0800, 77.7300, "landfill"),
    ],
    "Chennai": [
        POI("Manali Industrial Area", 13.1667, 80.2667, "industrial"),
        POI("Guindy Industrial Estate", 13.0100, 80.2100, "industrial"),
        POI("Kodungaiyur Dump Yard", 13.1350, 80.2550, "landfill"),
        POI("Perungudi Dump Yard", 12.9560, 80.2330, "landfill"),
        POI("Kathipara Junction", 13.0070, 80.2050, "traffic"),
        POI("Koyambedu", 13.0694, 80.1948, "traffic"),
    ],
    "Hyderabad": [
        POI("Jeedimetla Industrial Area", 17.5100, 78.4550, "industrial"),
        POI("Jawaharnagar Dump Yard", 17.5130, 78.5970, "landfill"),
        POI("Punjagutta", 17.4260, 78.4500, "traffic"),
        POI("Uppal", 17.4010, 78.5590, "traffic"),
    ],
    "Ahmedabad": [
        POI("Vatva GIDC", 22.9660, 72.6280, "industrial"),
        POI("Naroda GIDC", 23.0870, 72.6660, "industrial"),
        POI("Pirana Landfill", 22.9700, 72.5660, "landfill"),
        POI("Kalupur", 23.0270, 72.6010, "traffic"),
        POI("ISKCON Cross Road", 23.0270, 72.5070, "traffic"),
    ],
    "Jaipur": [
        POI("Vishwakarma Industrial Area", 26.9750, 75.7730, "industrial"),
        POI("Sitapura Industrial Area", 26.7780, 75.8430, "industrial"),
        POI("Sindhi Camp", 26.9230, 75.8000, "traffic"),
        POI("Gopalpura Tonk Road", 26.8600, 75.7900, "traffic"),
    ],
}

# Source categories used across the platform.
SOURCE_TYPES = {
    "industrial": {"label": "Industrial emission", "icon": "🏭", "color": [142, 68, 173]},
    "burning": {"label": "Biomass / waste burning", "icon": "🔥", "color": [211, 84, 0]},
    "dust": {"label": "Construction / road dust", "icon": "🏗️", "color": [185, 140, 80]},
    "traffic": {"label": "Vehicular traffic", "icon": "🚗", "color": [52, 73, 94]},
}


@dataclass
class Thresholds:
    hotspot_gi_z: float = 1.96          # Getis-Ord Gi* z-score (95%)
    hotspot_resid_z: float = 2.0        # sensor residual z-score
    hotspot_min_excess: float = 25.0    # ug/m3 above local background
    cluster_eps_km: float = 2.2
    spike_prob_alert: float = 0.55      # probability that triggers a spike warning
    severe_aqi: int = 401
    very_poor_aqi: int = 301
    fire_upwind_km: float = 350.0
    fire_upwind_half_angle: float = 35.0
    fire_count_alert: int = 150


THRESHOLDS = Thresholds()

# Diwali dates (festival fireworks drive short, sharp PM2.5 spikes).
DIWALI_DATES = [
    "2015-11-11", "2016-10-30", "2017-10-19", "2018-11-07", "2019-10-27",
    "2020-11-14", "2021-11-04", "2022-10-24", "2023-11-12", "2024-10-31",
    "2025-10-20", "2026-11-08",
]


def city_names() -> list[str]:
    return list(CITIES.keys())


def ward_cities() -> list[str]:
    return [c.name for c in CITIES.values() if c.has_wards]


def spcb_for(city: str) -> str:
    state = CITIES[city].state
    return SPCB.get(state, f"{state} State Pollution Control Board")
