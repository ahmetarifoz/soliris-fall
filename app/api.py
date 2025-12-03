"""REST API server for fall detection control."""

from __future__ import annotations

import threading
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Will be set by run.py when starting the API
_detector = None
_config = None
_lock = threading.Lock()


def set_detector(detector, config):
    """Set the detector instance for API control."""
    global _detector, _config
    with _lock:
        _detector = detector
        _config = config


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class DetectionSettings(BaseModel):
    """Detection threshold settings."""
    confidence_threshold: Optional[float] = Field(
        None,
        description="Pose confidence threshold (0.0-1.0)",
        example=0.16
    )
    angle_threshold: Optional[float] = Field(
        None,
        description="Fall angle threshold in degrees",
        example=46
    )
    temporal_window: Optional[int] = Field(
        None,
        description="Rolling window size for temporal filtering (frames)",
        example=60
    )
    temporal_min_hits: Optional[int] = Field(
        None,
        description="Minimum frames needed to confirm detection",
        example=3
    )


class AlertSettings(BaseModel):
    """Alert configuration settings."""
    cooldown_seconds: Optional[float] = Field(
        None,
        description="Minimum seconds between alerts per track",
        example=0.4
    )
    candidate_alert_sec: Optional[float] = Field(
        None,
        description="Seconds before candidate state triggers alert",
        example=10.0
    )
    fallen_alert_sec: Optional[float] = Field(
        None,
        description="Seconds before fallen state triggers alert",
        example=5.0
    )
    op500_alert_type: Optional[str] = Field(
        None,
        description="OP500 alert type (e.g., FALL)",
        example="FALL"
    )
    rtp_port: Optional[int] = Field(
        None,
        description="RTP port for OP500 integration",
        example=10160
    )


class ConfigUpdate(BaseModel):
    """Configuration update request."""
    detection: Optional[DetectionSettings] = None
    alert: Optional[AlertSettings] = None


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fall Detection API",
    version="1.0",
    description="REST API for controlling fall detection system"
)

# CORS middleware - allow all origins including localhost:5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm origin'lere izin ver
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST, PUT, DELETE, OPTIONS vs.
    allow_headers=["*"],  # Tüm header'lara izin ver
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "fall-detection"}


@app.get("/status")
async def get_status():
    """Get detection status."""
    with _lock:
        if _detector is None:
            raise HTTPException(status_code=503, detail="Detector not initialized")
        
        return {
            "enabled": _detector.detection_enabled,
            "camera_id": _detector.camera_id,
            "active_tracks": len(_detector.track_state),
            "rtp_loopback_active": _detector.rtp_loopback_active,
            "op500_enabled": _detector.op500_enabled,
            "op500_rtp_port": _detector.op500_rtp_port,
        }


@app.post("/enable")
async def enable_detection():
    """Enable detection."""
    with _lock:
        if _detector is None:
            raise HTTPException(status_code=503, detail="Detector not initialized")
        
        _detector.detection_enabled = True
        return {"status": "ok", "detection_enabled": True}


@app.post("/disable")
async def disable_detection():
    """Disable detection."""
    with _lock:
        if _detector is None:
            raise HTTPException(status_code=503, detail="Detector not initialized")
        
        _detector.detection_enabled = False
        return {"status": "ok", "detection_enabled": False}


@app.post("/reset-cooldown")
async def reset_cooldown():
    """Reset cooldown timers."""
    with _lock:
        if _detector is None:
            raise HTTPException(status_code=503, detail="Detector not initialized")
        
        # Clear all cooldown timers
        _detector.last_alert.clear()
        
        # Reset track states
        for tid in _detector.track_state:
            _detector.track_state[tid]["candidate_notified"] = False
            _detector.track_state[tid]["fallen_notified"] = False
        
        return {"status": "ok", "message": "Cooldown timers reset"}


@app.get("/config")
async def get_config():
    """Get detection configuration (thresholds, cooldowns, alerts)."""
    with _lock:
        if _config is None:
            raise HTTPException(status_code=503, detail="Config not initialized")
        
        detect = _config.detect
        return {
            "detection": {
                "confidence_threshold": detect.conf,
                "angle_threshold": detect.angle_th,
                "temporal_window": detect.temporal_window,
                "temporal_min_hits": detect.temporal_min_hits,
                "confirm_sec": detect.confirm_sec,
                "recover_sec": detect.recover_sec,
                "recover_angle": detect.recover_angle,
            },
            "alert": {
                "cooldown_seconds": detect.cooldown_sec,
                "candidate_alert_sec": detect.candidate_alert_sec,
                "fallen_alert_sec": detect.fallen_alert_sec,
                "alert_hold": detect.alert_hold,
                "op500_enabled": _config.op500.enabled if _config.op500 else False,
                "op500_base_url": _config.op500.base_url if _config.op500 else None,
                "rtp_auto_close_seconds": _config.op500.rtp_auto_close_seconds if _config.op500 else None,
            },
            "webhook": {
                "url": _config.webhook.url,
                "cooldown_seconds": _config.webhook.cooldown_seconds,
            }
        }


@app.put("/config")
async def update_config(update: ConfigUpdate):
    """Update detection configuration (thresholds, cooldowns, alerts)."""
    with _lock:
        if _detector is None or _config is None:
            raise HTTPException(status_code=503, detail="Detector not initialized")
        
        updated = {}
        
        # Note: Since config dataclasses are frozen, we update detector's runtime state
        # For persistent changes, config.yaml should be modified
        
        if update.detection:
            det = update.detection
            if det.confidence_threshold is not None:
                # Update model conf threshold
                _detector.model.conf = det.confidence_threshold
                updated["confidence_threshold"] = det.confidence_threshold
            
            if det.angle_threshold is not None:
                # Store in detector for runtime use
                _detector._runtime_angle_th = det.angle_threshold
                updated["angle_threshold"] = det.angle_threshold
            
            if det.temporal_window is not None:
                _detector._runtime_temporal_window = det.temporal_window
                updated["temporal_window"] = det.temporal_window
            
            if det.temporal_min_hits is not None:
                _detector._runtime_temporal_min_hits = det.temporal_min_hits
                updated["temporal_min_hits"] = det.temporal_min_hits
        
        if update.alert:
            alert = update.alert
            if alert.cooldown_seconds is not None:
                _detector._runtime_cooldown_sec = alert.cooldown_seconds
                updated["cooldown_seconds"] = alert.cooldown_seconds
            
            if alert.candidate_alert_sec is not None:
                _detector._runtime_candidate_alert_sec = alert.candidate_alert_sec
                updated["candidate_alert_sec"] = alert.candidate_alert_sec
            
            if alert.fallen_alert_sec is not None:
                _detector._runtime_fallen_alert_sec = alert.fallen_alert_sec
                updated["fallen_alert_sec"] = alert.fallen_alert_sec
            
            if alert.rtp_port is not None:
                _detector.op500_rtp_port = alert.rtp_port
                updated["rtp_port"] = alert.rtp_port
        
        return {"status": "ok", "updated": updated}


@app.get("/tracks")
async def get_tracks():
    """Get current track information."""
    with _lock:
        if _detector is None:
            raise HTTPException(status_code=503, detail="Detector not initialized")
        
        tracks = []
        for tid, state in _detector.track_state.items():
            tracks.append({
                "track_id": tid,
                "state": state.get("state", "idle"),
                "angle": state.get("angle", 0.0),
                "last_seen": _detector.last_seen.get(tid, 0.0),
                "alert_until": _detector.alert_until.get(tid, 0.0),
                "in_cooldown": tid in _detector.last_alert,
            })
        
        return {"tracks": tracks, "count": len(tracks)}


@app.post("/rtp/start")
async def start_rtp():
    """Start RTP streaming."""
    with _lock:
        if _detector is None:
            raise HTTPException(status_code=503, detail="Detector not initialized")
        
        if _detector.stream_reader is None:
            raise HTTPException(status_code=400, detail="Stream reader not available")
        
        _detector.stream_reader.trigger_rtp_stream()
        return {"status": "ok", "rtp_active": True}


@app.post("/rtp/stop")
async def stop_rtp():
    """Stop RTP streaming."""
    with _lock:
        if _detector is None:
            raise HTTPException(status_code=503, detail="Detector not initialized")
        
        if _detector.stream_reader is None:
            raise HTTPException(status_code=400, detail="Stream reader not available")
        
        _detector.stream_reader.stop_rtp_sender()
        return {"status": "ok", "rtp_active": False}


def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server in a separate thread."""
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")


def start_api_thread(host: str = "0.0.0.0", port: int = 8000) -> threading.Thread:
    """Start API server in a background thread."""
    thread = threading.Thread(
        target=run_api_server,
        args=(host, port),
        daemon=True,
        name="api-server"
    )
    thread.start()
    return thread


__all__ = ["app", "set_detector", "start_api_thread", "run_api_server"]
