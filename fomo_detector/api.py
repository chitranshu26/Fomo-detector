"""HTTP API that exposes the existing FOMO Detector training and simulation."""

import argparse
import threading
import uuid
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fomo_detector.config import DATA_PATH, LOG_PATH, MODEL_PATH
from fomo_detector.simulate import run_simulation
from fomo_detector.train import train_model


app = FastAPI(
    title="FOMO Detector API",
    version="1.0.0",
    description="Local API for the synthetic FOMO model and simulation.",
)

_sessions = {}
_sessions_lock = threading.Lock()
_simulation_log_path = LOG_PATH


class SimulationRequest(BaseModel):
    duration_seconds: float = Field(default=30, gt=0, le=3600)


@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "model_available": MODEL_PATH.is_file(),
        "dataset_available": DATA_PATH.is_file(),
    }


@app.post("/api/v1/train")
def train():
    if not DATA_PATH.is_file():
        raise HTTPException(status_code=404, detail="Default training dataset was not found.")
    try:
        model = train_model()
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "status": "trained",
        "model_path": str(MODEL_PATH),
        "classes": [int(label) for label in model.classes_],
        "features": list(model.feature_names_in_),
    }


def _run_session(session_id, duration):
    with _sessions_lock:
        session = _sessions[session_id]
        session["status"] = "running"
        stop_event = session["stop_event"]

    def record_sample(sample):
        sample["recorded_at"] = datetime.now(timezone.utc).isoformat()
        with _sessions_lock:
            session = _sessions[session_id]
            session["samples"].append(sample)
            if len(session["samples"]) > 100:
                del session["samples"][:-100]

    try:
        run_simulation(
            duration,
            log_path=_simulation_log_path,
            on_sample=record_sample,
            stop_event=stop_event,
        )
        with _sessions_lock:
            _sessions[session_id]["status"] = (
                "stopped" if stop_event.is_set() else "completed"
            )
    except Exception as error:
        with _sessions_lock:
            _sessions[session_id]["status"] = "failed"
            _sessions[session_id]["error"] = str(error)


@app.post("/api/v1/simulations", status_code=202)
def start_simulation(request: SimulationRequest):
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=409, detail="Train a model before starting a simulation.")

    with _sessions_lock:
        active = any(
            session["status"] in {"starting", "running", "stopping"}
            for session in _sessions.values()
        )
        if active:
            raise HTTPException(status_code=409, detail="A simulation is already running.")

        session_id = uuid.uuid4().hex
        _sessions[session_id] = {
            "id": session_id,
            "status": "starting",
            "duration_seconds": request.duration_seconds,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "samples": [],
            "error": None,
            "stop_event": threading.Event(),
        }
        worker = threading.Thread(
            target=_run_session,
            args=(session_id, request.duration_seconds),
            name=f"fomo-simulation-{session_id[:8]}",
            daemon=True,
        )
        worker.start()

    return {"id": session_id, "status": "starting"}


@app.get("/api/v1/simulations/{session_id}")
def simulation_status(session_id: str):
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Simulation session was not found.")
        return {
            "id": session["id"],
            "status": session["status"],
            "duration_seconds": session["duration_seconds"],
            "started_at": session["started_at"],
            "samples": list(session["samples"]),
            "error": session["error"],
        }


@app.post("/api/v1/simulations/{session_id}/stop")
def stop_simulation(session_id: str):
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Simulation session was not found.")
        if session["status"] not in {"starting", "running"}:
            return {"id": session_id, "status": session["status"]}
        session["status"] = "stopping"
        session["stop_event"].set()
        return {"id": session_id, "status": "stopping"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the local FOMO Detector API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log-path", default=str(LOG_PATH))
    arguments = parser.parse_args(argv)
    global _simulation_log_path
    _simulation_log_path = arguments.log_path
    uvicorn.run(app, host=arguments.host, port=arguments.port, log_level="info")


if __name__ == "__main__":
    main()