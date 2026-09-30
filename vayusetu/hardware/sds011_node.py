"""
VayuSetu citizen sensor node - Raspberry Pi + Nova SDS011 (USB).

Reads PM2.5/PM10 from the SDS011 over its USB-serial adapter, averages over
the reporting interval, and posts to a ThingSpeak channel (field1=PM2.5,
field2=PM10, field3=temperature, field4=humidity). The VayuSetu app reads the
channel when THINGSPEAK_CHANNEL_ID is set in Streamlit secrets.

Usage:
    pip install pyserial requests
    python sds011_node.py --port /dev/ttyUSB0 --write-key YOUR_THINGSPEAK_WRITE_KEY
    python sds011_node.py --simulate --write-key KEY      # no hardware: sends plausible values

Optional DHT22 humidity sensor: pip install adafruit-circuitpython-dht and pass --dht-pin 4.
"""
from __future__ import annotations

import argparse
import random
import statistics
import time

import requests

THINGSPEAK_URL = "https://api.thingspeak.com/update"


def read_sds011(ser) -> tuple[float, float] | None:
    """Read one 10-byte SDS011 frame: AA C0 PM25L PM25H PM10L PM10H ID1 ID2 CHK AB."""
    while True:
        b = ser.read(1)
        if not b:
            return None
        if b == b"\xaa":
            rest = ser.read(9)
            if len(rest) == 9 and rest[0] == 0xC0 and rest[8] == 0xAB:
                data = rest[1:7]
                if sum(data) % 256 != rest[7]:
                    continue  # checksum mismatch
                pm25 = (data[0] + data[1] * 256) / 10.0
                pm10 = (data[2] + data[3] * 256) / 10.0
                return pm25, pm10


def read_dht(pin: int):
    try:
        import adafruit_dht
        import board
        dev = adafruit_dht.DHT22(getattr(board, f"D{pin}"))
        return dev.temperature, dev.humidity
    except Exception:  # noqa: BLE001 - sensor optional
        return None, None


def post(write_key: str, pm25: float, pm10: float, temp=None, rh=None) -> bool:
    params = {"api_key": write_key, "field1": round(pm25, 1), "field2": round(pm10, 1)}
    if temp is not None:
        params["field3"] = round(temp, 1)
    if rh is not None:
        params["field4"] = round(rh, 1)
    try:
        r = requests.get(THINGSPEAK_URL, params=params, timeout=15)
        return r.ok and r.text.strip() != "0"
    except requests.RequestException as exc:
        print("upload failed:", exc)
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--write-key", required=True)
    ap.add_argument("--interval", type=int, default=300, help="seconds between uploads (ThingSpeak free: >=15)")
    ap.add_argument("--dht-pin", type=int, default=None)
    ap.add_argument("--simulate", action="store_true", help="send plausible values without hardware")
    args = ap.parse_args()

    ser = None
    if not args.simulate:
        import serial  # pyserial
        ser = serial.Serial(args.port, baudrate=9600, timeout=2)
    print("VayuSetu node started - posting every", args.interval, "s")
    while True:
        t0, pm25s, pm10s = time.time(), [], []
        while time.time() - t0 < args.interval:
            if args.simulate:
                base = 60 + 40 * random.random()
                pm25s.append(base)
                pm10s.append(base / random.uniform(0.45, 0.7))
                time.sleep(5)
            else:
                v = read_sds011(ser)
                if v:
                    pm25s.append(v[0])
                    pm10s.append(v[1])
        if pm25s:
            temp, rh = read_dht(args.dht_pin) if args.dht_pin is not None else (None, None)
            ok = post(args.write_key, statistics.median(pm25s), statistics.median(pm10s), temp, rh)
            print(time.strftime("%H:%M:%S"), f"PM2.5={statistics.median(pm25s):.1f} PM10={statistics.median(pm10s):.1f}",
                  "uploaded" if ok else "upload failed")


if __name__ == "__main__":
    main()
