"""Export the pydantic data models as JSON Schema files into docs/schemas/."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vayusetu import schemas  # noqa: E402

OUT = ROOT / "docs" / "schemas"
OUT.mkdir(parents=True, exist_ok=True)
for name, model in {"observation": schemas.Observation, "citizen_report": schemas.CitizenReport,
                    "hotspot": schemas.Hotspot, "forecast": schemas.Forecast}.items():
    sch = model.model_json_schema()
    sch["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    sch["$id"] = f"urn:vayusetu:schema:{name}:v1"
    (OUT / f"{name}.schema.json").write_text(json.dumps(sch, indent=2))
    print("wrote", name)
