from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Reuse the exact model/data/inference logic used by the Flask dashboard.
from app import (
    FEATURES,
    SEQ_LEN,
    THRESHOLD,
    all_engine_summaries,
    dashboard_data,
    engine_frame,
    engine_summary,
    forecast_engine,
    meta,
    predict_sequence,
    status_for_probability,
    estimate_rul,
    maintenance_advice,
)

app = FastAPI(
    title="MechSense Predictive Maintenance API",
    description=(
        "Production-style REST API for the MechSense C-MAPSS FD001 "
        "predictive-maintenance model. Includes fleet analytics, machine "
        "risk, sensor trends, forecasting, and CSV/JSON inference."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Reading(BaseModel):
    """One sensor/operating-setting reading used for prediction."""

    model_config = {"extra": "allow"}


class PredictionRequest(BaseModel):
    readings: List[dict[str, Any]] = Field(
        ..., min_length=1, description="At least 30 rows are required by the model."
    )


def _predict_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        raise ValueError("The supplied dataset is empty.")

    if "engine_id" in df.columns:
        predictions = []
        for eid, group in df.groupby("engine_id"):
            try:
                ordered = group.sort_values("cycle") if "cycle" in group.columns else group
                p = predict_sequence(ordered)
                status = status_for_probability(p)
                rul = estimate_rul(ordered)
                predictions.append(
                    {
                        "engineId": int(eid),
                        "failureProbability": round(p, 4),
                        "failureProbabilityPct": round(p * 100, 2),
                        "predictedFailure": bool(p >= THRESHOLD),
                        "status": status,
                        "healthScore": max(0, min(100, round((1.0 - p) * 100))),
                        "estimatedRUL": round(rul, 1) if rul is not None else None,
                        "maintenance": maintenance_advice(status, p, rul),
                    }
                )
            except (ValueError, TypeError) as exc:
                predictions.append({"engineId": int(eid), "error": str(exc)})
        return {"mode": "machine", "predictions": predictions}

    p = predict_sequence(df)
    status = status_for_probability(p)
    rul = estimate_rul(df)
    return {
        "mode": "single",
        "failureProbability": round(p, 4),
        "failureProbabilityPct": round(p * 100, 2),
        "predictedFailure": bool(p >= THRESHOLD),
        "status": status,
        "healthScore": max(0, min(100, round((1.0 - p) * 100))),
        "estimatedRUL": round(rul, 1) if rul is not None else None,
        "maintenance": maintenance_advice(status, p, rul),
        "threshold": THRESHOLD,
        "sequenceLength": SEQ_LEN,
    }



def _health_score(probability: float) -> int:
    """Display score derived from model failure probability; not a RUL estimate."""
    return max(0, min(100, round((1.0 - probability) * 100)))


def _alerts() -> list[dict[str, Any]]:
    alerts = []
    for m in all_engine_summaries():
        if m["status"] != "healthy":
            alerts.append(
                {
                    "id": f"ALT-{m['engineId']:03d}",
                    "engineId": m["engineId"],
                    "severity": "Critical" if m["status"] == "critical" else "Warning",
                    "type": "Failure risk",
                    "probability": m["failureProbabilityPct"],
                    "cycle": m["cycle"],
                    "status": "Open",
                }
            )
    return sorted(alerts, key=lambda x: (-x["probability"], x["engineId"]))


def _maintenance() -> list[dict[str, Any]]:
    return [
        {
            "engineId": m["engineId"],
            "priority": "Urgent" if m["status"] == "critical" else "Planned",
            "reason": f"Failure probability {m['failureProbabilityPct']}%",
            "cycle": m["cycle"],
            "action": "Inspect / service machine",
        }
        for m in all_engine_summaries()
        if m["status"] != "healthy"
    ]


@app.get("/", tags=["System"])
def api_root():
    return {
        "name": "MechSense Predictive Maintenance API",
        "version": app.version,
        "docs": "/docs",
        "redoc": "/redoc",
        "model": meta["model"],
    }


@app.get("/api/v1/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "service": "MechSense API",
        "model": meta["model"],
        "sequenceLength": SEQ_LEN,
        "threshold": THRESHOLD,
    }


@app.get("/api/v1/dashboard", tags=["Dashboard"])
def dashboard():
    return dashboard_data()


@app.get("/api/v1/metrics", tags=["Model"])
def metrics():
    return meta


@app.get("/api/v1/machines", tags=["Machines"])
def machines(
    status: Optional[str] = Query(None, description="healthy, warning, or critical"),
    limit: int = Query(100, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    rows = all_engine_summaries()
    if status:
        status = status.lower()
        if status not in {"healthy", "warning", "critical"}:
            raise HTTPException(400, "status must be healthy, warning, or critical")
        rows = [r for r in rows if r["status"] == status]

    total = len(rows)
    rows = rows[offset : offset + limit]
    return {"total": total, "limit": limit, "offset": offset, "machines": rows}


@app.get("/api/v1/machines/{engine_id}", tags=["Machines"])
def machine(engine_id: int):
    try:
        g = engine_frame(engine_id)
        summary = engine_summary(engine_id)
    except KeyError:
        raise HTTPException(404, f"Engine {engine_id} not found")

    readings = g.tail(SEQ_LEN)[["cycle"] + FEATURES].to_dict("records")
    return {"machine": summary, "readings": readings}


@app.get("/api/v1/predict/{engine_id}", tags=["Prediction"])
def predict(engine_id: int):
    try:
        result = engine_summary(engine_id)
        result["healthScore"] = _health_score(result["failureProbability"])
        result["recommendation"] = (
            "Immediate inspection and maintenance" if result["status"] == "critical"
            else "Schedule inspection and continue monitoring" if result["status"] == "warning"
            else "Continue normal monitoring"
        )
        return result | {
            "sequenceLength": SEQ_LEN,
            "model": meta["model"],
        }
    except KeyError:
        raise HTTPException(404, f"Engine {engine_id} not found")
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/v1/machines/{engine_id}/intelligence", tags=["Machine Intelligence"])
def machine_intelligence(engine_id: int):
    try:
        summary = engine_summary(engine_id)
        g = engine_frame(engine_id)
        trend = g.tail(60)[["cycle", "sensor_2", "sensor_4", "sensor_11", "sensor_15", "sensor_21"]].to_dict("records")
        return {
            "machine": summary,
            "trend": trend,
            "model": meta,
            "rulModel": rul_meta,
        }
    except KeyError:
        raise HTTPException(404, f"Engine {engine_id} not found")


@app.post("/api/v1/predict", tags=["Prediction"])
def predict_json(payload: PredictionRequest):
    try:
        return _predict_dataframe(pd.DataFrame(payload.readings))
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/v1/predict/csv", tags=["Prediction"])
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a .csv file")

    try:
        content = await file.read()
        from io import BytesIO
        df = pd.read_csv(BytesIO(content))
        return _predict_dataframe(df)
    except Exception as exc:
        raise HTTPException(400, f"Could not analyze CSV: {exc}")


@app.post("/api/v1/predict/csv/report", tags=["Prediction"])
async def predict_csv_report(file: UploadFile = File(...)):
    """Run batch prediction and return a CSV report suitable for download."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a .csv file")
    try:
        from io import BytesIO
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
        result = _predict_dataframe(df)
        rows = result.get("predictions") if result.get("mode") == "machine" else [result]
        report = pd.DataFrame(rows).to_csv(index=False)
        return Response(content=report, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=mechsense_predictions.csv"})
    except Exception as exc:
        raise HTTPException(400, f"Could not create CSV report: {exc}")


@app.get("/api/v1/trends/{engine_id}", tags=["Sensors"])
def trends(
    engine_id: int,
    limit: int = Query(60, ge=10, le=300),
):
    try:
        g = engine_frame(engine_id)
    except KeyError:
        raise HTTPException(404, f"Engine {engine_id} not found")

    cols = ["cycle", "sensor_2", "sensor_4", "sensor_11", "sensor_15", "sensor_21"]
    return {
        "engineId": engine_id,
        "points": g.tail(limit)[cols].to_dict("records"),
    }


@app.get("/api/v1/forecast/{engine_id}", tags=["Forecast"])
def forecast(
    engine_id: int,
    horizon: int = Query(12, ge=1, le=48),
):
    try:
        return forecast_engine(engine_id, horizon)
    except KeyError:
        raise HTTPException(404, f"Engine {engine_id} not found")
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/v1/analytics/overview", tags=["Analytics"])
def analytics_overview():
    rows = all_engine_summaries()
    ranked = sorted(rows, key=lambda r: r["failureProbability"], reverse=True)
    avg_risk = sum(r["failureProbability"] for r in rows) / len(rows) if rows else 0
    return {
        "machines": len(rows),
        "averageRiskPct": round(avg_risk * 100, 2),
        "averageHealthScore": _health_score(avg_risk),
        "critical": sum(r["status"] == "critical" for r in rows),
        "warning": sum(r["status"] == "warning" for r in rows),
        "healthy": sum(r["status"] == "healthy" for r in rows),
        "topRiskMachines": [
            {"engineId": r["engineId"], "riskPct": r["failureProbabilityPct"], "status": r["status"]}
            for r in ranked[:5]
        ],
    }


@app.get("/api/v1/alerts", tags=["Operations"])
def alerts(
    severity: Optional[str] = Query(None, description="Warning or Critical"),
):
    rows = _alerts()
    if severity:
        rows = [r for r in rows if r["severity"].lower() == severity.lower()]
    return {"count": len(rows), "alerts": rows}


@app.get("/api/v1/maintenance", tags=["Operations"])
def maintenance():
    rows = _maintenance()
    return {"count": len(rows), "actions": rows}


@app.get("/api/v1/model", tags=["Model"])
def model_info():
    return {
        "model": meta["model"],
        "sequenceLength": meta["sequence_length"],
        "rulCutoffCycles": meta["rul_cutoff_cycles"],
        "threshold": THRESHOLD,
        "features": FEATURES,
        "metrics": meta["metrics"],
    }
