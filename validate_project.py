"""Static/runtime validation for the MechSense project."""
from pathlib import Path
import json
import py_compile

BASE = Path(__file__).resolve().parent
for name in ("app.py", "train_lstm.py"):
    py_compile.compile(str(BASE / name), doraise=True)

meta = json.loads((BASE / "model" / "metadata.json").read_text())
scaler = json.loads((BASE / "model" / "scaler.json").read_text())
assert len(meta["features"]) == len(scaler["features"])
assert meta["features"] == scaler["features"]
assert len(scaler["mean"]) == len(meta["features"])
assert len(scaler["scale"]) == len(meta["features"])
assert (BASE / "model" / "lstm.pt").exists()
assert (BASE / "data" / "cmapss" / "train_FD001.csv").exists()
assert (BASE / "data" / "cmapss" / "test_FD001.csv").exists()
assert (BASE / "data" / "cmapss" / "RUL_FD001.csv").exists()
print("Static validation passed.")
