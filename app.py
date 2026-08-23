from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from pathlib import Path
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "cmapss"
MODEL_DIR = BASE / "model"
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)
SEQ_LEN = 30

meta = json.loads((MODEL_DIR / "metadata.json").read_text())
scaler = json.loads((MODEL_DIR / "scaler.json").read_text())
FEATURES = scaler["features"]
MEAN = np.array(scaler["mean"], dtype=float)
SCALE = np.array(scaler["scale"], dtype=float)
THRESHOLD = float(meta["threshold"])

class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(.25)
        self.lstm2 = nn.LSTM(hidden, 32, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Dropout(.2), nn.Linear(16, 1))

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.drop(x)
        x, _ = self.lstm2(x)
        return self.fc(x[:, -1, :]).squeeze(-1)

model = LSTMClassifier(len(FEATURES))
model.load_state_dict(torch.load(MODEL_DIR / "lstm.pt", map_location="cpu", weights_only=True))
model.eval()
RUL_MODEL_PATH = MODEL_DIR / "rul_estimator.joblib"
rul_model = None
if RUL_MODEL_PATH.exists():
    try:
        rul_model = joblib.load(RUL_MODEL_PATH)
    except Exception as exc:
        # Optional artifact: if it was serialized with an incompatible sklearn version,
        # keep the main prediction app fully usable without RUL estimation.
        rul_model = None
rul_meta_path = MODEL_DIR / "rul_metadata.json"
rul_meta = json.loads(rul_meta_path.read_text()) if rul_meta_path.exists() else {}

# Separate model for common AI4I / industrial predictive-maintenance CSVs.
INDUSTRIAL_MODEL_PATH = MODEL_DIR / "industrial_lstm.pt"
INDUSTRIAL_SCALER_PATH = MODEL_DIR / "industrial_lstm_scaler.json"
INDUSTRIAL_META_PATH = MODEL_DIR / "industrial_lstm_metadata.json"
industrial_model = None
industrial_scaler = {}
industrial_meta = {}
INDUSTRIAL_FEATURES = []
INDUSTRIAL_SEQ_LEN = 30
INDUSTRIAL_THRESHOLD = 0.60

if INDUSTRIAL_MODEL_PATH.exists() and INDUSTRIAL_SCALER_PATH.exists() and INDUSTRIAL_META_PATH.exists():
    industrial_meta = json.loads(INDUSTRIAL_META_PATH.read_text())
    industrial_scaler = json.loads(INDUSTRIAL_SCALER_PATH.read_text())
    INDUSTRIAL_FEATURES = industrial_scaler.get("features", industrial_meta.get("features", []))
    INDUSTRIAL_SEQ_LEN = int(industrial_meta.get("sequence_length", 30))
    INDUSTRIAL_THRESHOLD = float(industrial_meta.get("threshold", 0.60))

    class IndustrialLSTM(nn.Module):
        def __init__(self, n_features, hidden=64):
            super().__init__()
            self.lstm1 = nn.LSTM(n_features, hidden, batch_first=True)
            self.dropout1 = nn.Dropout(0.25)
            self.lstm2 = nn.LSTM(hidden, 32, batch_first=True)
            self.classifier = nn.Sequential(
                nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.20), nn.Linear(16, 1)
            )
        def forward(self, x):
            x, _ = self.lstm1(x)
            x = self.dropout1(x)
            x, _ = self.lstm2(x)
            return self.classifier(x[:, -1, :]).squeeze(-1)

    industrial_model = IndustrialLSTM(len(INDUSTRIAL_FEATURES))
    industrial_model.load_state_dict(torch.load(INDUSTRIAL_MODEL_PATH, map_location="cpu", weights_only=True))
    industrial_model.eval()

app = Flask(__name__)
APP_VERSION = "CSV-RAW-AI4I-FINAL-FIX-2026-08-17"
app.secret_key = os.environ.get("MECHSENSE_SECRET", "mechsense-dev")

TRAIN = pd.read_csv(DATA / "train_FD001.csv")
TEST = pd.read_csv(DATA / "test_FD001.csv")
RUL = pd.read_csv(DATA / "RUL_FD001.csv", header=None).iloc[:, 0].to_numpy()

# ---------------------------------------------------------------------------
# Dataset state
# ---------------------------------------------------------------------------
# The original FD001 test set is NEVER modified.  A CSV uploaded by the user
# becomes a separate active dataset in memory and is persisted under uploads/
# only for the current Flask process.  /reset-default switches back instantly.
ACTIVE_DF = TEST.copy()
ACTIVE_SOURCE = "default"
ACTIVE_FILENAME = "test_FD001.csv"
ACTIVE_RESULT = None
ACTIVE_MODE = "cmapss"
INDUSTRIAL_PREDICTIONS = []
ACTIVE_STATE_PATH = UPLOADS / "active_dataset.json"


def _clean_column_names(df):
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def _looks_like_numeric_header(df):
    if not len(df.columns):
        return False
    try:
        [float(str(c).strip()) for c in df.columns]
        return True
    except (TypeError, ValueError):
        return False


def _normalise_cmapss_columns(df):
    """Accept common C-MAPSS exports, including headerless 26-column files."""
    df = _clean_column_names(df)

    # NASA C-MAPSS raw files contain 26 values per row and usually have no header.
    # pandas treats the first data row as headers in that case, so detect numeric
    # headers and restore the canonical 26-column schema.
    if len(df.columns) == 26 and _looks_like_numeric_header(df):
        raw_columns = [
            "engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3",
            *[f"sensor_{i}" for i in range(1, 22)],
        ]
        df.columns = raw_columns

    aliases = {
        "unit_number": "engine_id", "unit number": "engine_id", "unit": "engine_id",
        "engine_id": "engine_id", "engineid": "engine_id", "engine_id": "engine_id",
        "time_cycles": "cycle", "time_cycle": "cycle", "cycle": "cycle",
        "setting1": "op_setting_1", "setting_1": "op_setting_1",
        "setting2": "op_setting_2", "setting_2": "op_setting_2",
        "setting3": "op_setting_3", "setting_3": "op_setting_3",
    }
    for i in range(1, 22):
        aliases.update({
            f"s{i}": f"sensor_{i}",
            f"s_{i}": f"sensor_{i}",
            f"sensor{i}": f"sensor_{i}",
        })

    rename = {}
    for col in df.columns:
        key = str(col).strip().lower().replace("-", "_")
        if key in aliases and aliases[key] not in df.columns:
            rename[col] = aliases[key]
    if rename:
        df = df.rename(columns=rename)
    return df


def _has_industrial_numeric_columns(df):
    """Recognize the raw 7-column AI4I/industrial CSV used by this app.

    IMPORTANT: raw uploads do NOT need TWF/HDF/PWF/OSF/RNF. Those are derived
    internally from the five physical measurements before the industrial LSTM
    runs. This detector deliberately does not depend on the trained feature list.
    """
    df = _clean_column_names(df)
    numeric_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    if not all(c in df.columns for c in numeric_features):
        return False
    try:
        return all(pd.to_numeric(df[c], errors="coerce").notna().all() for c in numeric_features)
    except Exception:
        return False


def _add_industrial_derived_features(df):
    """Build the AI4I failure-mode feature columns when a raw CSV omits them."""
    df = _clean_column_names(df).copy()
    if not _has_industrial_numeric_columns(df):
        return df

    air = pd.to_numeric(df["Air temperature [K]"], errors="coerce")
    process = pd.to_numeric(df["Process temperature [K]"], errors="coerce")
    speed = pd.to_numeric(df["Rotational speed [rpm]"], errors="coerce")
    torque = pd.to_numeric(df["Torque [Nm]"], errors="coerce")
    wear = pd.to_numeric(df["Tool wear [min]"], errors="coerce")
    power = torque * speed * (2 * np.pi / 60.0)

    # These are the standard deterministic AI4I failure-mode rules. RNF is a
    # random failure label and cannot be inferred from a raw measurement row,
    # so it is kept at zero when the source CSV does not provide it.
    df["TWF"] = (wear >= 200).astype(int) if "TWF" not in df.columns else pd.to_numeric(df["TWF"], errors="coerce")
    df["HDF"] = (((process - air) < 8.6) & (speed < 1380)).astype(int) if "HDF" not in df.columns else pd.to_numeric(df["HDF"], errors="coerce")
    df["PWF"] = ((power < 3500) | (power > 9000)).astype(int) if "PWF" not in df.columns else pd.to_numeric(df["PWF"], errors="coerce")

    if "OSF" not in df.columns:
        type_values = df["Type"].astype(str).str.upper() if "Type" in df.columns else pd.Series("M", index=df.index)
        limit = type_values.map({"L": 11000, "M": 12000, "H": 13000}).fillna(12000)
        df["OSF"] = ((wear * torque) > limit).astype(int)
    else:
        df["OSF"] = pd.to_numeric(df["OSF"], errors="coerce")

    if "RNF" not in df.columns:
        df["RNF"] = 0
    else:
        df["RNF"] = pd.to_numeric(df["RNF"], errors="coerce")

    return df


def is_industrial_dataframe(df):
    df = _clean_column_names(df)
    return _has_industrial_numeric_columns(df)

def _validate_industrial_model_domain(df):
    """Reject measurements that are far outside the distribution seen by the industrial LSTM.

    This prevents an uploaded CSV with incompatible units (for example Celsius instead of
    Kelvin, or torque/speed in different units) from producing confident but meaningless
    probabilities and forecasts. A generous 4-sigma envelope is used so legitimate variation
    is accepted while obvious unit/scale errors are rejected.
    """
    numeric = [
        "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
    ]
    mean = np.asarray(industrial_scaler.get("mean", []), dtype=float)
    scale = np.asarray(industrial_scaler.get("scale", []), dtype=float)
    features = industrial_scaler.get("features", INDUSTRIAL_FEATURES)
    if len(mean) != len(features) or len(scale) != len(features):
        raise ValueError("Industrial LSTM scaler metadata is invalid.")
    limits = dict(zip(features, zip(mean, np.where(scale == 0, 1.0, scale))))
    problems = []
    for col in numeric:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        mu, sd = limits[col]
        lo, hi = mu - 4.0 * sd, mu + 4.0 * sd
        bad = (values < lo) | (values > hi)
        if bad.any():
            vmin, vmax = float(values.min()), float(values.max())
            problems.append(f"{col}: observed {vmin:.2f}–{vmax:.2f}, expected roughly {lo:.2f}–{hi:.2f}")
    for col in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.isna().any() or not values.isin([0, 1]).all():
                problems.append(f"{col}: must contain only 0 or 1")
    if problems:
        raise ValueError(
            "CSV rejected: sensor values are outside the trained Industrial LSTM range. "
            "Check units/column mapping. " + " | ".join(problems[:3])
        )


def industrial_scaled(df):
    x = df[INDUSTRIAL_FEATURES].apply(pd.to_numeric, errors="coerce")
    if x.isna().any().any():
        bad = x.columns[x.isna().any()].tolist()
        raise ValueError("Non-numeric or missing values in: " + ", ".join(bad))
    mean = np.asarray(industrial_scaler.get("mean", []), dtype=np.float32)
    scale = np.asarray(industrial_scaler.get("scale", []), dtype=np.float32)
    scale = np.where(scale == 0, 1.0, scale)
    if len(mean) != len(INDUSTRIAL_FEATURES) or len(scale) != len(INDUSTRIAL_FEATURES):
        raise ValueError("Industrial LSTM scaler does not match the trained feature list")
    return ((x.to_numpy(dtype=np.float32) - mean) / scale).astype(np.float32)

def industrial_sequence(values, end_index):
    start = max(0, end_index - INDUSTRIAL_SEQ_LEN + 1)
    seq = values[start:end_index + 1]
    if len(seq) < INDUSTRIAL_SEQ_LEN:
        seq = np.concatenate([np.repeat(values[[0]], INDUSTRIAL_SEQ_LEN-len(seq), axis=0), seq], axis=0)
    return seq[-INDUSTRIAL_SEQ_LEN:]

def industrial_predict_dataframe(df):
    """Score one complete 30-step window per machine, using its latest readings."""
    if industrial_model is None:
        raise ValueError("Industrial LSTM model files are missing")

    work = df.copy()
    machine_col = industrial_machine_column(work)
    if not machine_col:
        raise ValueError("Industrial CSV has no machine identifier.")

    out = []
    for machine_id, g in work.groupby(machine_col, sort=False):
        g = g.sort_values("cycle").reset_index(drop=True)
        if len(g) < INDUSTRIAL_SEQ_LEN:
            raise ValueError(
                f"Machine {normalize_engine_id(machine_id)} has only {len(g)} readings; "
                f"the trained Industrial LSTM requires {INDUSTRIAL_SEQ_LEN}."
            )

        scaled = industrial_scaled(g)
        seq = scaled[-INDUSTRIAL_SEQ_LEN:]
        with torch.no_grad():
            tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            p = float(torch.sigmoid(industrial_model(tensor)).item())

        status = "critical" if p >= INDUSTRIAL_THRESHOLD else (
            "warning" if p >= INDUSTRIAL_THRESHOLD * 0.65 else "healthy"
        )
        last = g.iloc[-1]
        item = {
            "row": int(g.index[-1]) + 1,
            "engine_id": normalize_engine_id(machine_id),
            "failure_probability": round(p * 100, 2),
            "failureProbabilityPct": round(p * 100, 2),
            "predicted_failure": p >= INDUSTRIAL_THRESHOLD,
            "predictedFailure": p >= INDUSTRIAL_THRESHOLD,
            "status": status,
            "health_score": max(0, min(100, round((1 - p) * 100))),
            "healthScore": max(0, min(100, round((1 - p) * 100))),
            "maintenance_priority": "URGENT" if status == "critical" else ("HIGH" if status == "warning" else "NORMAL"),
            "maintenance_action": "Inspect and service immediately" if status == "critical" else (
                "Schedule inspection and monitor closely" if status == "warning" else
                "Continue routine monitoring"
            ),
            "readings": int(len(g)),
            "cycle": int(last["cycle"]),
        }
        if "Product ID" in g.columns:
            item["product_id"] = str(last["Product ID"])
        if "Machine failure" in g.columns:
            actual = pd.to_numeric(pd.Series([last["Machine failure"]]), errors="coerce").iloc[0]
            item["actual_failure"] = None if pd.isna(actual) else bool(float(actual) > 0)
        out.append(item)

    return out

def industrial_summary_rows(predictions, limit=100):
    return sorted(predictions, key=lambda r: r.get("failure_probability",0), reverse=True)[:limit]

def normalize_engine_id(value):
    """Keep user-provided machine IDs safe for both numeric and string CSVs."""
    if pd.isna(value):
        return "UNKNOWN"
    text = str(value).strip()
    if not text:
        return "UNKNOWN"
    try:
        number = float(text)
        if number.is_integer():
            return int(number)
    except (ValueError, TypeError):
        pass
    return text


def sort_engine_ids(values):
    return sorted(values, key=lambda x: (str(x).isdigit() is False, str(x)))


def engine_label(value):
    """Human-readable machine label without assuming a numeric ID."""
    return f"MS-{normalize_engine_id(value)}"



def predict_sequence(df):
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    if len(df) < SEQ_LEN:
        raise ValueError(f"Need at least {SEQ_LEN} readings")
    x = ((df[FEATURES].astype(float).values[-SEQ_LEN:] - MEAN) / SCALE).astype(np.float32)
    with torch.no_grad():
        p = float(torch.sigmoid(model(torch.tensor(x).unsqueeze(0))).item())
    return p


def active_is_uploaded():
    return ACTIVE_SOURCE == "uploaded"


def active_dataset_label():
    if active_is_uploaded():
        return ACTIVE_FILENAME
    return "C-MAPSS FD001 default dataset"


def _validate_numeric_columns(df, columns, dataset_name):
    """Reject malformed model inputs instead of silently coercing bad values."""
    bad = []
    for col in columns:
        if col not in df.columns:
            bad.append(f"missing '{col}'")
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            bad.append(f"'{col}' contains non-numeric/blank values")
        elif not np.isfinite(values.to_numpy(dtype=float)).all():
            bad.append(f"'{col}' contains infinite values")
    if bad:
        raise ValueError(
            f"Invalid {dataset_name} CSV: " + "; ".join(bad) +
            ". Fix the CSV and upload it again."
        )


def _validate_industrial_timeseries(df):
    """Validate the minimum contract required by the trained Industrial LSTM."""
    machine_col = industrial_machine_column(df)
    if not machine_col:
        raise ValueError(
            "Invalid industrial CSV: a machine identifier is required "
            "(for example 'Machine ID', 'Machine_ID', 'machine_id' or 'asset_id')."
        )

    has_cycle = any(c in df.columns for c in ("cycle", "Cycle", "time_cycle", "time_cycles"))
    has_timestamp = any(c in df.columns for c in ("Timestamp", "timestamp", "datetime", "Date", "date", "time"))
    if not (has_cycle or has_timestamp):
        raise ValueError(
            "Invalid industrial CSV: a time-order column is required "
            "(for example 'cycle' or 'Timestamp'). A one-row-per-machine CSV "
            "cannot be analyzed by the 30-step LSTM or forecast as a time series."
        )

    _validate_numeric_columns(df, [
        "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
    ], "industrial")

    work = df.copy()
    if not has_cycle:
        time_col = next(c for c in ("Timestamp", "timestamp", "datetime", "Date", "date", "time") if c in work.columns)
        parsed = pd.to_datetime(work[time_col], errors="coerce")
        if parsed.isna().any():
            raise ValueError(f"Invalid industrial CSV: '{time_col}' contains invalid timestamps.")
        work["_time_order"] = parsed
        work["cycle"] = work.groupby(machine_col, sort=False).cumcount() + 1
    else:
        cycle_col = next(c for c in ("cycle", "Cycle", "time_cycle", "time_cycles") if c in work.columns)
        work["cycle"] = pd.to_numeric(work[cycle_col], errors="coerce")
        if work["cycle"].isna().any() or not np.isfinite(work["cycle"].to_numpy(dtype=float)).all():
            raise ValueError(f"Invalid industrial CSV: '{cycle_col}' must contain numeric cycle values.")

    ids = work[machine_col].map(normalize_engine_id)
    if (ids == "UNKNOWN").any():
        raise ValueError(f"Invalid industrial CSV: '{machine_col}' contains blank machine IDs.")

    counts = ids.value_counts()
    too_short = counts[counts < INDUSTRIAL_SEQ_LEN]
    if not too_short.empty:
        examples = ", ".join(f"{idx} ({int(v)} rows)" for idx, v in too_short.head(5).items())
        raise ValueError(
            f"Invalid industrial time series: every machine needs at least "
            f"{INDUSTRIAL_SEQ_LEN} readings for the trained LSTM. "
            f"Too-short machines: {examples}."
        )

    _validate_industrial_model_domain(work)

    work["_machine_key"] = ids
    sort_cols = ["_machine_key", "cycle"]
    if "_time_order" in work.columns:
        sort_cols = ["_machine_key", "_time_order", "cycle"]
    work = work.sort_values(sort_cols).drop(columns=["_machine_key", "_time_order"], errors="ignore")
    return work.reset_index(drop=True)


def _canonicalize_uploaded_df(df):
    """Validate uploaded data strictly before it becomes the active dataset."""
    df = _clean_column_names(df)
    if df.empty:
        raise ValueError("The uploaded CSV is empty.")

    if is_industrial_dataframe(df):
        return _add_industrial_derived_features(_validate_industrial_timeseries(df)).reset_index(drop=True)

    df = _normalise_cmapss_columns(df)
    required = ["engine_id", "cycle"] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Invalid C-MAPSS CSV. Required columns are engine/unit ID, cycle, "
            "the C-MAPSS operating settings, and the trained sensor columns. "
            "Missing: " + ", ".join(missing)
        )
    _validate_numeric_columns(df, FEATURES, "C-MAPSS")
    df["engine_id"] = df["engine_id"].map(normalize_engine_id)
    if (df["engine_id"] == "UNKNOWN").any():
        raise ValueError("Invalid C-MAPSS CSV: engine/unit IDs cannot be blank.")
    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
    if df["cycle"].isna().any() or not np.isfinite(df["cycle"].to_numpy(dtype=float)).all():
        raise ValueError("Invalid C-MAPSS CSV: cycle must contain numeric values.")
    counts = df["engine_id"].value_counts()
    too_short = counts[counts < SEQ_LEN]
    if not too_short.empty:
        examples = ", ".join(f"{idx} ({int(v)} rows)" for idx, v in too_short.head(5).items())
        raise ValueError(
            f"Invalid C-MAPSS time series: every engine needs at least {SEQ_LEN} readings "
            f"for the trained LSTM. Too-short engines: {examples}."
        )
    return df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)


def activate_uploaded_dataset(df, filename="uploaded.csv", mode="cmapss", predictions=None):
    global ACTIVE_DF, ACTIVE_SOURCE, ACTIVE_FILENAME, ACTIVE_RESULT, ACTIVE_MODE, INDUSTRIAL_PREDICTIONS
    ACTIVE_DF = _canonicalize_uploaded_df(df)
    ACTIVE_SOURCE = "uploaded"
    ACTIVE_FILENAME = filename or "uploaded.csv"
    ACTIVE_MODE = mode
    INDUSTRIAL_PREDICTIONS = predictions or []
    ACTIVE_RESULT = None
    safe_name = Path(ACTIVE_FILENAME).name or "uploaded.csv"
    ACTIVE_DF.to_csv(UPLOADS / safe_name, index=False)
    ACTIVE_STATE_PATH.write_text(json.dumps({
        "source": "uploaded",
        "filename": safe_name,
        "mode": ACTIVE_MODE,
    }, indent=2), encoding="utf-8")


def reset_to_default():
    global ACTIVE_DF, ACTIVE_SOURCE, ACTIVE_FILENAME, ACTIVE_RESULT, ACTIVE_MODE, INDUSTRIAL_PREDICTIONS
    ACTIVE_DF = TEST.copy()
    ACTIVE_SOURCE = "default"
    ACTIVE_FILENAME = "test_FD001.csv"
    ACTIVE_RESULT = None
    ACTIVE_MODE = "cmapss"
    INDUSTRIAL_PREDICTIONS = []
    try:
        if ACTIVE_STATE_PATH.exists():
            ACTIVE_STATE_PATH.unlink()
    except OSError:
        pass


def _load_persisted_dataset_state():
    global ACTIVE_DF, ACTIVE_SOURCE, ACTIVE_FILENAME, ACTIVE_MODE, INDUSTRIAL_PREDICTIONS
    if not ACTIVE_STATE_PATH.exists():
        return
    try:
        state = json.loads(ACTIVE_STATE_PATH.read_text(encoding="utf-8"))
        csv_name = Path(state.get("filename", "")).name
        csv_path = UPLOADS / csv_name
        if state.get("source") == "uploaded" and csv_name and csv_path.exists():
            loaded = pd.read_csv(csv_path)
            mode = state.get("mode", "industrial" if is_industrial_dataframe(loaded) else "cmapss")
            ACTIVE_DF = _canonicalize_uploaded_df(loaded)
            ACTIVE_SOURCE = "uploaded"
            ACTIVE_FILENAME = csv_name
            ACTIVE_MODE = mode
            INDUSTRIAL_PREDICTIONS = industrial_predict_dataframe(ACTIVE_DF) if mode == "industrial" else []
    except Exception:
        ACTIVE_DF = TEST.copy()
        ACTIVE_SOURCE = "default"
        ACTIVE_FILENAME = "test_FD001.csv"
        ACTIVE_MODE = "cmapss"
        INDUSTRIAL_PREDICTIONS = []


_load_persisted_dataset_state()


def industrial_machine_column(df=None):
    frame = ACTIVE_DF if df is None else df
    # Prefer true asset/machine identifiers over UDI. In AI4I-style exports,
    # UDI is a row identifier and is often unique for every observation; using
    # it as the machine ID would incorrectly turn a 20-machine time series
    # into 2,000 one-row machines.
    return next((c for c in [
        "machine_id", "Machine ID", "Machine_ID", "machineId",
        "asset_id", "Asset ID", "assetId", "engine_id", "Engine ID", "UDI"
    ] if c in frame.columns), None)

def active_engine_frame(engine_id):
    target = normalize_engine_id(engine_id)
    if ACTIVE_MODE == "industrial":
        machine_col = industrial_machine_column()
        if not machine_col:
            raise KeyError(engine_id)
        keys = ACTIVE_DF[machine_col].map(normalize_engine_id)
        g = ACTIVE_DF.loc[keys == target].copy()
        if g.empty:
            raise KeyError(engine_id)
        if "cycle" not in g.columns:
            g["cycle"] = np.arange(1, len(g) + 1)
        return g.sort_values("cycle").reset_index(drop=True)
    keys = ACTIVE_DF["engine_id"].map(normalize_engine_id)
    g = ACTIVE_DF.loc[keys == target].sort_values("cycle").copy()
    if g.empty:
        raise KeyError(engine_id)
    return g

def active_engine_ids():
    if ACTIVE_MODE == "industrial":
        col = industrial_machine_column()
        if not col:
            return []
        return sort_engine_ids(ACTIVE_DF[col].map(normalize_engine_id).drop_duplicates().tolist())
    return sort_engine_ids(ACTIVE_DF["engine_id"].map(normalize_engine_id).drop_duplicates().tolist())

def display_readings(engine_id, limit=60):
    g = active_engine_frame(engine_id)
    if ACTIVE_MODE == "industrial":
        rows = []
        for i, (_, r) in enumerate(g.tail(limit).iterrows(), start=max(1, len(g)-min(limit,len(g))+1)):
            rows.append({
                "cycle": i,
                "sensor_4": round(float(r.get("Process temperature [K]", 0)), 4),
                "sensor_2": round(float(r.get("Air temperature [K]", 0)), 4),
                "sensor_11": round(float(r.get("Rotational speed [rpm]", 0)), 4),
                "sensor_15": round(float(r.get("Tool wear [min]", 0)), 4),
            })
        return rows
    return g.tail(limit)[["cycle"] + FEATURES].to_dict("records")

def temperature_series(engine_id, limit=60):
    g = active_engine_frame(engine_id)
    if ACTIVE_MODE == "industrial":
        rows = []
        for i, (_, r) in enumerate(g.tail(limit).iterrows(), start=max(1, len(g)-min(limit,len(g))+1)):
            rows.append({
                "cycle": i,
                "sensor_4": round(float(r.get("Process temperature [K]", 0)), 4),
                "sensor_2": round(float(r.get("Air temperature [K]", 0)), 4),
                "sensor_11": round(float(r.get("Rotational speed [rpm]", 0)), 4),
            })
        return rows
    return g.tail(limit)[["cycle", "sensor_4", "sensor_2", "sensor_11"]].to_dict("records")


def engine_frame(engine_id):
    return active_engine_frame(engine_id)


def _last_number(g, column, default=0.0):
    if column not in g.columns or g.empty:
        return default
    try:
        return float(g.iloc[-1][column])
    except (TypeError, ValueError):
        return default


def rul_features(df):
    if len(df) < SEQ_LEN:
        raise ValueError(f"Need at least {SEQ_LEN} readings for RUL estimation")
    g = df.sort_values("cycle") if "cycle" in df.columns else df
    w = g.tail(SEQ_LEN)[FEATURES].astype(float)
    last = w.iloc[-1].to_numpy()
    mean = w.mean().to_numpy()
    std = w.std().fillna(0).to_numpy()
    slope = (w.iloc[-1].to_numpy() - w.iloc[0].to_numpy()) / max(1, len(w) - 1)
    return np.r_[last, mean, std, slope].reshape(1, -1)


def estimate_rul(df):
    if rul_model is None or len(df) < SEQ_LEN:
        return None
    return float(np.clip(rul_model.predict(rul_features(df))[0], 0, rul_meta.get("rul_cap", 125)))


def maintenance_advice(status, probability, rul):
    if status == "critical":
        return {"priority": "URGENT", "action": "Inspect and service immediately", "reason": f"Failure risk is {probability*100:.1f}%"}
    if status == "warning":
        return {"priority": "HIGH", "action": "Schedule inspection and monitor closely", "reason": f"Risk is elevated at {probability*100:.1f}%"}
    if rul is not None and rul <= 30:
        return {"priority": "PLANNED", "action": "Plan preventive maintenance", "reason": f"Estimated remaining life is about {rul:.0f} cycles"}
    return {"priority": "NORMAL", "action": "Continue routine monitoring", "reason": "No immediate maintenance trigger detected"}




def maintenance_profile(machine, status=None):
    """Create deterministic, machine-specific maintenance guidance for the UI.
    These are planning estimates, not quoted vendor/service costs.
    """
    eid = str(machine.get("engineId", "0"))
    try:
        n = int(''.join(ch for ch in eid if ch.isdigit()) or 0)
    except ValueError:
        n = 0
    risk = float(machine.get("failureProbabilityPct", 0) or 0)
    status = str(status or machine.get("status", "")).lower()
    rul = machine.get("estimatedRUL")
    if rul is None:
        rul = max(6, round((100 - risk) * 0.45, 1))

    profiles = [
        ("Cooling & thermal inspection", "Inspect cooling circuit, radiator/fan, thermal contacts and airflow.", 95, 4.5, 42000, 185000),
        ("Bearing & rotating assembly service", "Inspect bearings, shaft alignment, coupling and abnormal vibration sources.", 72, 5.5, 36500, 160000),
        ("Lubrication system service", "Check lubricant level/quality, seals, pump delivery and contamination.", 120, 3.0, 28500, 125000),
        ("Drive-train preventive maintenance", "Inspect belts/couplings, motor load, alignment and transmission wear.", 80, 6.0, 51000, 210000),
        ("Sensor & control-system inspection", "Verify sensor wiring, calibration, signal stability and controller diagnostics.", 140, 2.5, 19500, 98000),
        ("Full preventive service", "Perform a complete mechanical inspection with lubrication and sensor checks.", 60, 8.0, 68000, 275000),
    ]
    service, reason, base_window, duration, base_cost, failure_cost = profiles[n % len(profiles)]
    # Higher risk shortens the estimated operating window and increases urgency/cost.
    urgency_factor = max(0.35, 1.0 - risk / 130.0)
    operating_hours = max(2, round(base_window * urgency_factor * (0.7 + min(float(rul), 125) / 250.0)))
    duration_hours = round(duration * (1.0 + min(risk, 100) / 250.0), 1)
    estimated_cost = int(round(base_cost * (0.85 + risk / 180.0), -2))
    exposure = int(round(failure_cost * (0.75 + risk / 125.0), -2))
    savings = max(0, exposure - estimated_cost)

    # Healthy machines are monitored, not scheduled for paid intervention.
    # Cost/exposure estimates are intentionally unavailable for healthy status.
    if status == "healthy":
        estimated_cost = 0
        exposure = 0
        savings = 0

    precaution_sets = [
        ["Reduce load before the next high-demand cycle.", "Inspect coolant level and airflow path.", "Check temperature rise at idle and under load.", "Do not return to full load until thermal readings stabilize."],
        ["Avoid sudden speed changes.", "Inspect bearing housing for noise or vibration.", "Check shaft/coupling alignment and mounting bolts.", "Confirm vibration trend is falling after service."],
        ["Check lubricant level and contamination.", "Inspect seals for leakage.", "Verify pump/flow delivery before restarting.", "Use the approved lubricant grade for refill."],
        ["Keep the machine below peak load until inspection.", "Inspect belts, couplings and alignment.", "Check motor current and drive temperature.", "Run a controlled test cycle after maintenance."],
        ["Verify sensor connectors and cable condition.", "Check calibration against a known reference.", "Review sudden sensor spikes before clearing the alert.", "Confirm stable readings across two consecutive cycles."],
        ["Schedule a controlled maintenance window.", "Inspect mechanical, thermal and electrical subsystems.", "Record replaced parts and abnormal findings.", "Perform a post-service validation cycle before production."],
    ]
    precautions = precaution_sets[n % len(precaution_sets)]
    return {
        "type": service,
        "reason": reason,
        "estimatedCostINR": estimated_cost,
        "potentialFailureCostINR": exposure,
        "estimatedSavingsINR": savings,
        "maintenanceTimeHours": duration_hours,
        "estimatedOperatingHours": operating_hours,
        "operatingWindow": f"~{operating_hours} hours before planned intervention",
        "precautions": precautions,
        "planningNote": "Estimated from model risk and machine readings; use a maintenance engineer's inspection for final decisions."
    }

def status_for_probability(p):
    if p >= THRESHOLD:
        return "critical"
    if p >= THRESHOLD * 0.65:
        return "warning"
    return "healthy"


def engine_summary(engine_id):
    if ACTIVE_MODE == "industrial":
        matches = [
            r for r in INDUSTRIAL_PREDICTIONS
            if str(r.get("engine_id")) == str(normalize_engine_id(engine_id))
        ]
        if not matches:
            raise KeyError(engine_id)
        r = matches[-1]
        machine_col = industrial_machine_column(ACTIVE_DF)
        g = ACTIVE_DF.loc[
            ACTIVE_DF[machine_col].map(normalize_engine_id) == normalize_engine_id(engine_id)
        ].sort_values("cycle")
        if g.empty:
            raise KeyError(engine_id)
        last = g.iloc[-1]
        sensor_values={
            "Air temperature [K]": float(last.get("Air temperature [K]",0)),
            "Process temperature [K]": float(last.get("Process temperature [K]",0)),
            "Rotational speed [rpm]": float(last.get("Rotational speed [rpm]",0)),
            "Torque [Nm]": float(last.get("Torque [Nm]",0)),
            "Tool wear [min]": float(last.get("Tool wear [min]",0)),
        }
        profile=[]
        for label,col in [("Air temperature","Air temperature [K]"),("Process temperature","Process temperature [K]"),("Rotational speed","Rotational speed [rpm]"),("Torque","Torque [Nm]"),("Tool wear","Tool wear [min]")]:
            vals=pd.to_numeric(ACTIVE_DF[col],errors="coerce").dropna() if col in ACTIVE_DF.columns else pd.Series(dtype=float)
            if len(vals):
                lo=float(vals.quantile(.05)); hi=float(vals.quantile(.95)); v=sensor_values[col]
                score=50 if hi<=lo else float(np.clip((v-lo)/(hi-lo)*100,0,100))
                profile.append({"label":label,"value":round(v,2),"score":round(score,1)})
        rul_est=max(1, round((1-float(r["failure_probability"])/100)*180,1))
        return {"engineId":r["engine_id"],"cycle":int(last.get("cycle", r.get("cycle", len(g)))),"failureProbability":round(r["failure_probability"]/100,4),"failureProbabilityPct":r["failure_probability"],"predictedFailure":r["predicted_failure"],"status":r["status"],"healthScore":r["health_score"],"estimatedRUL":rul_est,"rulUnit":"rows","maintenance":{"priority":r["maintenance_priority"],"action":r["maintenance_action"]},"maintenanceProfile":maintenance_profile({"engineId":r["engine_id"],"failureProbabilityPct":r["failure_probability"],"estimatedRUL":rul_est}, r["status"]),"threshold":INDUSTRIAL_THRESHOLD,"readings":int(len(g)),"sensor4":round(sensor_values["Process temperature [K]"],3),"sensor2":round(sensor_values["Air temperature [K]"],3),"sensor11":round(sensor_values["Rotational speed [rpm]"],3),"sensor15":round(sensor_values["Tool wear [min]"],3),"sensorValues":sensor_values,"sensorProfile":profile}
    g = engine_frame(engine_id)
    p = predict_sequence(g)
    last = g.iloc[-1]
    normalized_id = normalize_engine_id(engine_id)
    status = status_for_probability(p)
    rul = estimate_rul(g)
    advice = maintenance_advice(status, p, rul)
    try: cycle = int(float(last.get("cycle", len(g))))
    except (TypeError, ValueError): cycle = len(g)
    return {"engineId":normalized_id,"cycle":cycle,"failureProbability":round(p,4),"failureProbabilityPct":round(p*100,2),"predictedFailure":bool(p>=THRESHOLD),"status":status,"healthScore":max(0,min(100,round((1-p)*100))),"estimatedRUL":round(rul,1) if rul is not None else None,"rulUnit":"cycles","maintenance":advice,"maintenanceProfile":maintenance_profile({"engineId":normalized_id,"failureProbabilityPct":round(p*100,2),"estimatedRUL":rul}, status),"threshold":THRESHOLD,"readings":int(len(g)),"sensor4":round(_last_number(g,"sensor_4"),3),"sensor2":round(_last_number(g,"sensor_2"),3),"sensor11":round(_last_number(g,"sensor_11"),3),"sensor15":round(_last_number(g,"sensor_15"),4)}

def all_engine_summaries():
    if ACTIVE_MODE == "industrial":
        return [engine_summary(r["engine_id"]) for r in industrial_summary_rows(INDUSTRIAL_PREDICTIONS, 100)]
    rows=[]
    ids=ACTIVE_DF["engine_id"].map(normalize_engine_id).drop_duplicates().tolist()
    for eid in sort_engine_ids(ids):
        try: rows.append(engine_summary(eid))
        except (ValueError,TypeError): continue
    return rows


def _scaled_sensor_profile(df):
    """Return a visual 0-100 sensor condition profile for industrial uploads."""
    names = [
        ("Air temperature", "Air temperature [K]"),
        ("Process temperature", "Process temperature [K]"),
        ("Rotational speed", "Rotational speed [rpm]"),
        ("Torque", "Torque [Nm]"),
        ("Tool wear", "Tool wear [min]"),
    ]
    out=[]
    for label,col in names:
        if col not in df.columns:
            continue
        vals=pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        lo=float(vals.quantile(.05)); hi=float(vals.quantile(.95))
        latest=float(vals.iloc[-1])
        score=50.0 if hi<=lo else float(np.clip((latest-lo)/(hi-lo)*100,0,100))
        out.append({"label":label,"value":round(latest,2),"score":round(score,1)})
    return out

def dashboard_data():
    rows = all_engine_summaries()
    counts = {k: sum(1 for r in rows if r["status"] == k) for k in ("healthy", "warning", "critical")}

    # Always initialize dashboard visual data before branching.
    # This prevents UnboundLocalError when the default FD001 dataset is active.
    trend = []
    temperature_trend = []
    sensor_profile = []

    if ACTIVE_MODE == "industrial":
        # Uploaded industrial files are commonly one row per machine. Build
        # risk and temperature visuals from every uploaded record without
        # inventing machine cycles.
        trend = [{
            "index": int(r["row"]),
            "risk": float(r["failure_probability"]),
        } for r in INDUSTRIAL_PREDICTIONS]

        temp_series = pd.to_numeric(
            ACTIVE_DF.get("Process temperature [K]", pd.Series(dtype=float)),
            errors="coerce"
        ).dropna().tail(80)

        values = temp_series.to_numpy(dtype=float)
        temperature_trend = []
        for i, value in enumerate(values):
            start = max(0, i - 9)
            window = values[start:i + 1]
            baseline = float(np.mean(window))
            spread = float(np.std(window))
            z = abs(float(value) - baseline) / max(spread, 0.05)
            recent = values[max(0, i - 4):i + 1]
            slope = float(np.polyfit(np.arange(len(recent)), recent, 1)[0]) if len(recent) >= 3 else 0.0
            direction = "increasing" if slope > 0.08 else ("decreasing" if slope < -0.08 else "stable")
            temperature_trend.append({
                "index": int(i + 1),
                "value": round(float(value), 2),
                "anomaly": bool(z >= 2.5),
                "direction": direction,
                "zScore": round(float(z), 2),
            })

        sensor_profile = _scaled_sensor_profile(ACTIVE_DF)

    else:
        # Default C-MAPSS FD001 dataset.
        if "sensor_4" in ACTIVE_DF.columns:
            series = (
                ACTIVE_DF.groupby("cycle")["sensor_4"]
                .mean()
                .tail(80)
                .reset_index()
            )

            temperature_trend = [
                {"index": int(x.cycle), "value": round(float(x.sensor_4), 2)}
                for x in series.itertuples()
            ]

        # Build a useful multi-sensor condition profile for the default dataset.
        sensor_names = [
            ("T50 / Sensor 4", "sensor_4"),
            ("Sensor 2", "sensor_2"),
            ("Sensor 11", "sensor_11"),
            ("Sensor 15", "sensor_15"),
        ]

        for label, col in sensor_names:
            if col not in ACTIVE_DF.columns:
                continue

            vals = pd.to_numeric(ACTIVE_DF[col], errors="coerce").dropna()
            if vals.empty:
                continue

            lo = float(vals.quantile(0.05))
            hi = float(vals.quantile(0.95))
            latest = float(vals.iloc[-1])

            score = (
                50.0
                if hi <= lo
                else float(np.clip((latest - lo) / (hi - lo) * 100, 0, 100))
            )

            sensor_profile.append({
                "label": label,
                "value": round(latest, 2),
                "score": round(score, 1),
            })

    costs = [
        int(m.get("maintenanceProfile", {}).get("estimatedCostINR", 0) or 0)
        for m in rows
    ]

    exposure = [
        int(m.get("maintenanceProfile", {}).get("potentialFailureCostINR", 0) or 0)
        for m in rows
    ]

    model_artifact = industrial_meta if ACTIVE_MODE == "industrial" else meta

    # Report-ready model view.  The Industrial metadata stores metrics under
    # validation/test because those are two distinct evaluation stages.
    # Reports must show the final held-out TEST metrics, not validation metrics.
    report_model = model_artifact
    if ACTIVE_MODE == "industrial" and industrial_meta:
        test_metrics = dict(industrial_meta.get("metrics", {}).get("test", {}))
        validation_metrics = dict(industrial_meta.get("metrics", {}).get("validation", {}))
        report_model = dict(industrial_meta)
        report_model["metrics"] = test_metrics
        report_model["validation_metrics"] = validation_metrics
        report_model["train_engines"] = len(industrial_meta.get("train_machines", []))
        report_model["validation_engines"] = len(industrial_meta.get("validation_machines", []))
        report_model["test_engines"] = len(industrial_meta.get("test_machines", []))
        report_model["total_machines"] = int(industrial_meta.get("machines", 0))
        report_model["total_readings"] = int(industrial_meta.get("rows", len(ACTIVE_DF)))
        report_model["test_rows"] = int(test_metrics.get("rows", 0))
        report_model["validation_rows"] = int(validation_metrics.get("rows", 0))
        report_model["split_strategy"] = industrial_meta.get("split_strategy", "")

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "machines": len(rows),
        "counts": counts,
        "trainReadings": len(TRAIN),
        "testReadings": len(ACTIVE_DF),
        "model": model_artifact,
        "reportModel": report_model,
        "activeMode": ACTIVE_MODE,
        "trend": trend,
        "temperatureTrend": temperature_trend,
        "sensorProfile": sensor_profile,
        "engines": rows,
        "datasetSource": ACTIVE_SOURCE,
        "datasetName": active_dataset_label(),
        "datasetRows": int(len(ACTIVE_DF)),
        "uploaded": active_is_uploaded(),
        "estimatedMaintenanceCost": int(sum(costs)),
        "failureExposure": int(sum(exposure)),
    }


def page_context(active):
    return {
        "active": active,
        "meta": meta,
        "threshold_pct": round((INDUSTRIAL_THRESHOLD if ACTIVE_MODE == "industrial" else THRESHOLD) * 100),
        "active_mode": ACTIVE_MODE,
        "dataset_source": ACTIVE_SOURCE,
        "dataset_name": active_dataset_label(),
        "dataset_rows": len(ACTIVE_DF),
        "dataset_uploaded": active_is_uploaded(),
    }


# ---------------- Web UI ----------------
@app.route("/")
def index():
    return render_template("dashboard.html", **page_context("dashboard"), data=dashboard_data())

@app.route("/machines")
def machines_page():
    return render_template("machines.html", **page_context("machines"), machines=all_engine_summaries())

@app.route("/analyze")
def analyze_page():
    engine_id = request.args.get("engine", default="1")
    ids = active_engine_ids()
    try:
        machine = engine_summary(engine_id)
    except (KeyError, ValueError):
        machine = engine_summary(ids[0]) if ids else None
    readings = []
    if machine:
        try:
            readings = display_readings(machine["engineId"], INDUSTRIAL_SEQ_LEN if ACTIVE_MODE == "industrial" else SEQ_LEN)
        except (KeyError, ValueError):
            readings = []
    return render_template("analyze.html", **page_context("analyze"), machine=machine, engine_ids=ids, readings=readings)

@app.route("/machine/<path:engine_id>")
def machine_page(engine_id):
    try:
        m = engine_summary(engine_id)
        readings = display_readings(engine_id, INDUSTRIAL_SEQ_LEN if ACTIVE_MODE == "industrial" else SEQ_LEN)
        p = float(m["failureProbability"])
        threshold = INDUSTRIAL_THRESHOLD if ACTIVE_MODE == "industrial" else THRESHOLD
    except KeyError:
        return render_template("machine.html", **page_context("machines"), error=f"Engine {engine_id} not found"), 404
    except ValueError as exc:
        return render_template("machine.html", **page_context("machines"), error=str(exc), engine_id=engine_id), 400
    return render_template("machine.html", **page_context("machines"), engine_id=engine_id, probability=p, threshold=threshold, readings=readings, error=None)


@app.route("/temperature")
def temperature_page():
    engine_id = request.args.get("engine", default="1")
    ids = active_engine_ids()
    if ids and normalize_engine_id(engine_id) not in ids:
        engine_id = ids[0]
    try:
        trend = temperature_series(engine_id, 60)
    except KeyError:
        if not ids:
            trend = []
        else:
            engine_id = ids[0]
            trend = temperature_series(engine_id, 60)
    temperature_label = "Process Temperature [K]" if ACTIVE_MODE == "industrial" else "T50 / Sensor 4"
    return render_template("temperature.html", **page_context("temperature"), engine_id=engine_id, engine_ids=ids, trend=trend, temperature_label=temperature_label)

@app.route("/forecast")
def forecast_page():
    """Render a machine-specific time-series forecast page.

    The selected machine is passed as /forecast?engine=<machine_id>.
    The available machine IDs are supplied to the template so the user can
    switch machines directly from the forecast page.
    """
    ids = active_engine_ids()

    if not ids:
        empty_result = {
            "engineId": "—",
            "sensor": "sensor_4",
            "label": "No forecast data",
            "method": "linear trend baseline",
            "historical": [],
            "forecast": [],
            "slopePerCycle": 0,
        }
        return render_template(
            "forecast.html",
            **page_context("forecast"),
            result=empty_result,
            engine_ids=[],
            selected_engine=None,
        )

    requested_engine = request.args.get("engine")
    selected_engine = normalize_engine_id(requested_engine) if requested_engine else ids[0]

    if selected_engine not in ids:
        selected_engine = ids[0]

    try:
        result = forecast_engine(selected_engine)
    except Exception as exc:
        flash(str(exc))
        selected_engine = ids[0]
        try:
            result = forecast_engine(selected_engine)
        except Exception:
            result = {
                "engineId": selected_engine,
                "sensor": "sensor_4",
                "label": "No forecast data",
                "method": "linear trend baseline",
                "historical": [],
                "forecast": [],
                "slopePerCycle": 0,
            }

    return render_template(
        "forecast.html",
        **page_context("forecast"),
        result=result,
        engine_ids=ids,
        selected_engine=selected_engine,
    )

@app.route("/alerts")
def alerts_page():
    machines = all_engine_summaries()
    alerts = []
    for m in machines:
        if m["status"] != "healthy":
            alerts.append({
                "id": f"ALT-{normalize_engine_id(m['engineId'])}",
                "engineId": m["engineId"],
                "severity": "Critical" if m["status"] == "critical" else "Warning",
                "type": "Failure risk",
                "probability": m["failureProbabilityPct"],
                "cycle": m["cycle"],
                "status": "Open",
                "healthScore": m.get("healthScore"),
                "rul": m.get("estimatedRUL"),
                "sensor4": m.get("sensor4", 0),
                "sensor2": m.get("sensor2", 0),
                "sensor11": m.get("sensor11", 0),
                "sensor15": m.get("sensor15", 0),
                "maintenance": m.get("maintenanceProfile", {}),
            })
    alerts.sort(key=lambda x: (-float(x["probability"]), str(x["engineId"])))
    return render_template("alerts.html", **page_context("alerts"), alerts=alerts)

@app.route("/maintenance")
def maintenance_page():
    machines = [m for m in all_engine_summaries() if m["status"] != "healthy"]
    actions = [{
        "engineId": m["engineId"],
        "priority": "Urgent" if m["status"] == "critical" else "Planned",
        "reason": f"Failure probability {m['failureProbabilityPct']}%",
        "cycle": m["cycle"],
        "action": "Inspect / service machine",
    } for m in machines]
    return render_template("maintenance.html", **page_context("maintenance"), actions=actions)

@app.route("/reports")
def reports_page():
    data = dashboard_data()
    return render_template("reports.html", **page_context("reports"), data=data)

@app.route("/settings")
def settings_page():
    return render_template("settings.html", **page_context("settings"))

@app.route("/api-center")
def api_center_page():
    return render_template("api.html", **page_context("api-center"))

@app.route("/reset-default")
def reset_default():
    reset_to_default()
    flash("Returned to the default C-MAPSS FD001 dataset. Uploaded CSV data was kept separate.")
    return redirect(url_for("index"))


@app.route("/lstm-testing", methods=["GET", "POST"])
def lstm_testing():
    global ACTIVE_RESULT
    result = ACTIVE_RESULT if active_is_uploaded() else None
    columns = ACTIVE_DF.columns.tolist() if active_is_uploaded() else None
    if request.method == "POST":
        # A failed upload must never leave the previous file looking like
        # the newly submitted file was analyzed.
        result = None
        f = request.files.get("file")
        if not f or not f.filename or not f.filename.lower().endswith(".csv"):
            flash("Upload rejected: please select a CSV file.")
        else:
            try:
                df = pd.read_csv(f)
                if df.empty: raise ValueError("The uploaded CSV is empty.")
                df = _clean_column_names(df)
                if is_industrial_dataframe(df):
                    prepared = _canonicalize_uploaded_df(df)
                    predictions = industrial_predict_dataframe(prepared)
                    result = {"mode":"industrial","model":industrial_meta.get("model","PyTorch Industrial LSTM"),"filename":Path(f.filename).name,"rows":len(prepared),"machines":len(predictions),"predictions":predictions,"critical":sum(r["status"]=="critical" for r in predictions),"warning":sum(r["status"]=="warning" for r in predictions),"healthy":sum(r["status"]=="healthy" for r in predictions),"threshold":INDUSTRIAL_THRESHOLD}
                    activate_uploaded_dataset(prepared, Path(f.filename).name, mode="industrial", predictions=predictions)
                else:
                    prepared=_canonicalize_uploaded_df(df)
                    predictions=[]
                    for eid,g in prepared.groupby("engine_id",dropna=False):
                        try:
                            g=g.sort_values("cycle"); p=predict_sequence(g); status=status_for_probability(p); rul=estimate_rul(g); advice=maintenance_advice(status,p,rul)
                            predictions.append({"engine_id":normalize_engine_id(eid),"failure_probability":round(p*100,2),"predicted_failure":bool(p>=THRESHOLD),"status":status,"health_score":max(0,min(100,round((1-p)*100))),"estimated_rul":round(rul,1) if rul is not None else None,"maintenance_priority":advice["priority"],"maintenance_action":advice["action"]})
                        except (ValueError,TypeError) as exc: predictions.append({"engine_id":normalize_engine_id(eid),"error":str(exc)})
                    result={"mode":"cmapss","model":meta.get("model"),"filename":Path(f.filename).name,"rows":len(prepared),"machines":len(predictions),"predictions":predictions,"critical":sum(r.get("status")=="critical" for r in predictions),"warning":sum(r.get("status")=="warning" for r in predictions),"healthy":sum(r.get("status")=="healthy" for r in predictions),"threshold":THRESHOLD}
                    activate_uploaded_dataset(prepared,Path(f.filename).name,mode="cmapss",predictions=None)
                ACTIVE_RESULT=result
                columns=ACTIVE_DF.columns.tolist()
                flash(f"CSV analyzed successfully: {len(df):,} rows. The uploaded dataset is now active. The default FD001 dataset remains unchanged.")
            except Exception as exc:
                # Keep the currently active dataset untouched, but make the
                # rejection explicit in the upload screen.
                result = None
                columns = ACTIVE_DF.columns.tolist() if active_is_uploaded() else None
                flash(f"CSV rejected: {exc}")
    context = page_context("testing")
    context.update({"result": result, "columns": columns, "seq": INDUSTRIAL_SEQ_LEN if ACTIVE_MODE == "industrial" else SEQ_LEN, "features": INDUSTRIAL_FEATURES if ACTIVE_MODE == "industrial" else FEATURES})
    return render_template("lstm_testing.html", **context)


# ---------------- REST API ----------------
def forecast_engine(engine_id, horizon=12):
    """Build a machine-specific forecast from that machine's own ordered history.

    IMPORTANT: never substitute fleet-wide data for a machine. Doing so makes a
    one-machine forecast look valid while actually forecasting another series.
    """
    g = active_engine_frame(engine_id)
    if ACTIVE_MODE == "industrial":
        if len(g) < INDUSTRIAL_SEQ_LEN:
            raise ValueError(
                f"Machine {normalize_engine_id(engine_id)} has only {len(g)} readings; "
                f"at least {INDUSTRIAL_SEQ_LEN} readings are required for a time-series forecast."
            )
        series = pd.DataFrame({
            "cycle": pd.to_numeric(g["cycle"], errors="coerce"),
            "value": pd.to_numeric(g["Process temperature [K]"], errors="coerce")
        }).dropna().tail(60)
        sensor = "Process temperature [K]"
        label = "Process temperature trend forecast"
    else:
        series = g[["cycle", "sensor_4"]].copy().rename(columns={"sensor_4": "value"}).tail(60)
        sensor = "sensor_4"
        label = "T50 / Sensor 4 trend forecast"

    y = series["value"].to_numpy(dtype=float)
    x = series["cycle"].to_numpy(dtype=float)
    if len(y) < 10:
        raise ValueError("Not enough readings for forecast")

    # Fit against elapsed steps rather than large raw cycle numbers. This is
    # numerically stable and makes the slope directly interpretable.
    t = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(t, y, 1)
    future_t = np.arange(len(y), len(y) + horizon, dtype=float)
    raw_forecast = intercept + slope * future_t

    # Prevent a noisy linear fit from exploding to physically implausible
    # values. The trend is retained, but extrapolation is bounded relative
    # to this machine's own recent operating range.
    anomalies = []
    if ACTIVE_MODE == "industrial":
        lo = float(np.min(y))
        hi = float(np.max(y))
        span = max(hi - lo, 0.5)
        recent_mean = float(np.mean(y[-min(12, len(y)):]))
        recent_std = float(np.std(y[-min(12, len(y)):]))
        # Keep the forecast close to the machine's own recent operating range,
        # and also inside the distribution used to train the industrial model.
        lower_bound = lo - 0.15 * span
        upper_bound = hi + 0.15 * span
        if sensor in industrial_scaler.get("features", []):
            idx = industrial_scaler["features"].index(sensor)
            mu = float(industrial_scaler["mean"][idx])
            sd = max(float(industrial_scaler["scale"][idx]), 1e-6)
            lower_bound = max(lower_bound, mu - 4.0 * sd)
            upper_bound = min(upper_bound, mu + 4.0 * sd)
        if lower_bound >= upper_bound:
            lower_bound, upper_bound = lo, hi
        forecast = np.clip(raw_forecast, lower_bound, upper_bound)

        # Mark points that are unusually far from the machine's recent baseline.
        baseline = recent_mean
        threshold = max(3.0 * recent_std, 0.75)
        for c, v in zip(x, y):
            if abs(float(v) - baseline) > threshold:
                anomalies.append({"cycle": int(c), "value": round(float(v), 4), "type": "deviation"})
    else:
        forecast = raw_forecast

    direction_threshold = max(1e-6, 0.005 * max(float(np.mean(np.abs(y))), 1.0))
    direction = "increasing" if slope > direction_threshold else ("decreasing" if slope < -direction_threshold else "stable")
    return {
        "engineId": normalize_engine_id(engine_id),
        "sensor": sensor,
        "label": label,
        "method": "bounded linear trend baseline",
        "historical": [{"cycle": int(c), "value": round(float(v), 4)} for c, v in zip(x, y)],
        "forecast": [{"cycle": int(c), "value": round(float(v), 4)} for c, v in zip(x[-1] + np.arange(1, horizon + 1), forecast)],
        "slopePerCycle": round(float(slope), 6),
        "trendDirection": direction,
        "forecastMin": round(float(np.min(forecast)), 4),
        "forecastMax": round(float(np.max(forecast)), 4),
        "anomalies": anomalies,
    }


def all_forecasts(horizon=12):
    """Return one forecast object per unique active machine/engine."""
    results = []
    errors = []
    for engine_id in active_engine_ids():
        try:
            results.append(forecast_engine(engine_id, horizon))
        except (KeyError, ValueError) as exc:
            errors.append({"engineId": normalize_engine_id(engine_id), "error": str(exc)})
    return {"count": len(results), "forecasts": results, "errors": errors}


@app.get("/api/health")
def api_health():
    return jsonify({"status": "ok", "service": "MechSense API", "model": meta["model"]})

@app.get("/api/dashboard")
def api_dashboard():
    return jsonify(dashboard_data())

@app.get("/api/machines")
def api_machines():
    status = request.args.get("status")
    rows = all_engine_summaries()
    if status:
        rows = [r for r in rows if r["status"] == status.lower()]
    return jsonify({"count": len(rows), "machines": rows})

@app.get("/api/machines/<path:engine_id>")
def api_machine(engine_id):
    try:
        g = engine_frame(engine_id)
        m = engine_summary(engine_id)
    except KeyError:
        return jsonify({"error": "Engine not found", "engineId": engine_id}), 404
    readings = display_readings(engine_id, INDUSTRIAL_SEQ_LEN if ACTIVE_MODE == "industrial" else SEQ_LEN)
    return jsonify({"machine": m, "readings": readings})

@app.get("/api/predict/<path:engine_id>")
def api_predict(engine_id):
    try:
        return jsonify(engine_summary(engine_id) | {"sequenceLength": SEQ_LEN, "model": meta["model"]})
    except KeyError:
        return jsonify(error="Engine not found"), 404
    except ValueError as e:
        return jsonify(error=str(e)), 400

@app.get("/api/trends/<path:engine_id>")
def api_trends(engine_id):
    try:
        engine_frame(engine_id)
    except KeyError:
        return jsonify(error="Engine not found"), 404
    limit = min(max(request.args.get("limit", 60, type=int), 10), 300)
    return jsonify({"engineId": normalize_engine_id(engine_id), "points": temperature_series(engine_id, limit)})

@app.get("/api/forecast/<path:engine_id>")
def api_forecast(engine_id):
    horizon = min(max(request.args.get("horizon", 12, type=int), 1), 48)
    try:
        return jsonify(forecast_engine(engine_id, horizon))
    except KeyError:
        return jsonify(error="Engine not found"), 404
    except ValueError as e:
        return jsonify(error=str(e)), 400


@app.get("/api/forecasts")
def api_forecasts():
    """Batch endpoint used by the forecast page: one request for all machines."""
    horizon = min(max(request.args.get("horizon", 12, type=int), 1), 48)
    return jsonify(all_forecasts(horizon))

@app.post("/api/predict")
def api_predict_upload():
    f = request.files.get("file")
    if f:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            return jsonify(error=f"Invalid CSV: {e}"), 400
    else:
        payload = request.get_json(silent=True) or {}
        rows = payload.get("readings")
        if not isinstance(rows, list):
            return jsonify(error="Send multipart file=CSV or JSON {readings:[...]}."), 400
        df = pd.DataFrame(rows)
    try:
        if "engine_id" in df.columns:
            predictions = []
            for eid, g in df.groupby("engine_id"):
                try:
                    p = predict_sequence(g.sort_values("cycle"))
                    predictions.append({"engineId": normalize_engine_id(eid), "failureProbability": round(p, 4), "predictedFailure": bool(p >= THRESHOLD), "status": status_for_probability(p)})
                except ValueError as e:
                    predictions.append({"engineId": normalize_engine_id(eid), "error": str(e)})
            return jsonify({"mode": "machine", "predictions": predictions})
        p = predict_sequence(df)
        return jsonify({"mode": "single", "failureProbability": round(p, 4), "failureProbabilityPct": round(p * 100, 2), "predictedFailure": bool(p >= THRESHOLD), "status": status_for_probability(p), "threshold": THRESHOLD})
    except ValueError as e:
        return jsonify(error=str(e)), 400

@app.get("/api/metrics")
def metrics():
    return jsonify(meta)


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=os.environ.get("FLASK_DEBUG", "0") == "1")