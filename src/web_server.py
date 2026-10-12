"""
Web Server - FastAPI application for mobile audio upload and transcription
"""
import asyncio
import json
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from settings_manager import SettingsManager
from diarization_engine import DiarizationEngine

# ---------------------------------------------------------------------------
# App and global state
# ---------------------------------------------------------------------------

app = FastAPI(title="Teams-to-Transcript Web")
settings = SettingsManager()
executor = ThreadPoolExecutor(max_workers=2)

# Job storage: job_id -> job dict
jobs: Dict[str, dict] = {}

# WebSocket connections: client_id -> WebSocket
ws_connections: Dict[str, WebSocket] = {}

# Cached model instances (by model size string)
_whisper_models: Dict[str, object] = {}
_diarization_engine: Optional[DiarizationEngine] = None

# Directories
output_dir = Path(settings.get("output_directory"))
UPLOAD_DIR = output_dir / "web_uploads"
TRANSCRIPT_DIR = output_dir / "web_transcripts"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

VALID_MODELS = [
    "tiny", "base", "small", "medium",
    "large", "large-v2", "large-v3", "large-v3-turbo", "distil-large-v3",
]

# /api/transcribe-local only accepts media files, so the endpoint (and the job's
# download-audio link) can't be used to read arbitrary files off this machine.
MEDIA_EXTENSIONS = {
    ".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".wma", ".aac", ".webm",
    ".mp4", ".mkv", ".mov",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_whisper_model(model_size: str):
    """Return a cached WhisperModel, loading it if necessary."""
    if model_size in _whisper_models:
        return _whisper_models[model_size]

    from faster_whisper import WhisperModel

    device = _detect_device()
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Loading whisper model {model_size} on {device} ({compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    _whisper_models[model_size] = model
    print(f"Whisper model {model_size} loaded.")
    return model


def _get_diarization_engine() -> Optional[DiarizationEngine]:
    """Return the shared DiarizationEngine, or None if no HF token configured."""
    global _diarization_engine
    hf_token = settings.get("hf_token", "")
    if not hf_token:
        return None
    if _diarization_engine is None:
        _diarization_engine = DiarizationEngine(hf_token=hf_token)
    return _diarization_engine


def _format_seconds(total: float) -> str:
    h = int(total // 3600)
    m = int((total % 3600) // 60)
    s = int(total % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


async def _ws_send(client_id: str, message: dict):
    """Send a JSON message to a specific WebSocket client."""
    ws = ws_connections.get(client_id)
    if ws:
        try:
            await ws.send_json(message)
        except Exception:
            ws_connections.pop(client_id, None)


def _ws_send_sync(client_id: str, message: dict):
    """Non-async helper so worker threads can send WS messages."""
    ws = ws_connections.get(client_id)
    if ws:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return
        asyncio.run_coroutine_threadsafe(_ws_send(client_id, message), loop)

# ---------------------------------------------------------------------------
# Processing pipeline (runs in thread pool)
# ---------------------------------------------------------------------------

def _process_job(job_id: str, loop: asyncio.AbstractEventLoop):
    """Full transcription + diarization pipeline for a single job."""
    job = jobs[job_id]
    client_id = job.get("client_id", "")
    audio_path = job["file_path"]
    model_size = job.get("model", settings.get("web_model", "large-v3"))
    num_speakers = job.get("num_speakers")
    language = job.get("language")

    def send(msg: dict):
        ws = ws_connections.get(client_id)
        if ws:
            asyncio.run_coroutine_threadsafe(_ws_send(client_id, msg), loop)

    try:
        job["status"] = "loading_model"
        send({"type": "progress", "job_id": job_id, "stage": "loading_model",
              "message": f"Loading whisper model ({model_size})..."})

        model = _get_whisper_model(model_size)

        # --- Transcription ---
        job["status"] = "transcribing"
        send({"type": "progress", "job_id": job_id, "stage": "transcribing",
              "message": "Transcribing audio..."})

        transcribe_kwargs = {"beam_size": 5, "word_timestamps": True}
        if language:
            transcribe_kwargs["language"] = language

        segments_gen, info = model.transcribe(audio_path, **transcribe_kwargs)

        whisper_segments = []
        full_text_parts = []
        for seg in segments_gen:
            text = seg.text.strip()
            if not text:
                continue
            entry = {"start": seg.start, "end": seg.end, "text": text}
            whisper_segments.append(entry)
            full_text_parts.append(text)

            send({"type": "segment", "job_id": job_id,
                  "start": seg.start, "end": seg.end, "text": text})

        audio_duration = info.duration if hasattr(info, "duration") else (
            whisper_segments[-1]["end"] if whisper_segments else 0
        )
        job["duration"] = audio_duration

        send({"type": "progress", "job_id": job_id, "stage": "transcription_done",
              "message": f"Transcription complete: {len(whisper_segments)} segments"})

        # --- Diarization (optional) ---
        final_segments = whisper_segments
        num_speakers_found = 0
        run_diarization = job.get("diarize", False)

        if not run_diarization:
            send({"type": "progress", "job_id": job_id, "stage": "diarization_skipped",
                  "message": "Diarization skipped (not requested)"})
        else:
            diarization_engine = _get_diarization_engine()
            if not diarization_engine:
                send({"type": "progress", "job_id": job_id, "stage": "diarization_skipped",
                      "message": "Diarization skipped (no HF token configured)"})
            else:
                try:
                    job["status"] = "diarizing"
                    send({"type": "progress", "job_id": job_id, "stage": "diarizing",
                          "message": "Running speaker diarization..."})

                    def diar_progress(msg):
                        send({"type": "progress", "job_id": job_id, "stage": "diarizing",
                              "message": msg})

                    turns = diarization_engine.diarize(
                        audio_path, num_speakers=num_speakers, progress_callback=diar_progress,
                    )
                    labeled = DiarizationEngine.align_with_transcript(whisper_segments, turns)
                    final_segments = DiarizationEngine.merge_consecutive(labeled)
                    num_speakers_found = len(set(s.get("speaker", "") for s in final_segments))

                    send({"type": "progress", "job_id": job_id, "stage": "diarization_done",
                          "message": f"Diarization complete: {num_speakers_found} speakers"})
                except Exception as diar_exc:
                    import traceback
                    traceback.print_exc()
                    send({"type": "progress", "job_id": job_id, "stage": "diarization_skipped",
                          "message": f"Diarization failed, continuing without speakers: {diar_exc}"})

        # --- Save transcript ---
        job["status"] = "saving"
        transcript_name = f"{job_id}.txt"
        transcript_path = TRANSCRIPT_DIR / transcript_name

        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write("# Audio Transcript\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write(f"Duration: {_format_seconds(audio_duration)}\n")
            f.write(f"Model: faster-whisper ({model_size})\n")
            if num_speakers_found:
                f.write(f"Speakers: {num_speakers_found}\n")
            f.write("\n---\n\n")

            for seg in final_segments:
                start_str = _format_seconds(seg["start"])
                end_str = _format_seconds(seg["end"])
                speaker = seg.get("speaker", "")
                if speaker:
                    f.write(f"[{start_str} - {end_str}] {speaker}: {seg['text']}\n")
                else:
                    f.write(f"[{start_str} - {end_str}] {seg['text']}\n")

            f.write("\n---\n\nEnd of transcript.\n")

        job["status"] = "complete"
        job["transcript_path"] = str(transcript_path)
        job["segments"] = final_segments
        job["num_speakers"] = num_speakers_found

        send({"type": "complete", "job_id": job_id,
              "message": "Processing complete!",
              "segments": final_segments,
              "num_speakers": num_speakers_found,
              "duration": audio_duration})

    except Exception as exc:
        import traceback
        traceback.print_exc()
        job["status"] = "error"
        job["error"] = str(exc)
        send({"type": "error", "job_id": job_id, "message": str(exc)})

# ---------------------------------------------------------------------------
# Static files and SPA
# ---------------------------------------------------------------------------

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = static_dir / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@app.get("/api/models")
async def get_models():
    return {"models": VALID_MODELS, "default": settings.get("web_model", "large-v3")}


@app.post("/api/ping")
async def ping():
    return {"pong": True}


@app.get("/api/status")
async def get_status():
    device = _detect_device()
    gpu_available = device == "cuda"
    gpu_name = ""
    if gpu_available:
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return {
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "loaded_models": list(_whisper_models.keys()),
        "diarization_available": bool(settings.get("hf_token", "")),
    }


@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:12]
    ext = Path(file.filename or "audio.wav").suffix or ".wav"
    dest = UPLOAD_DIR / f"{job_id}{ext}"

    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    jobs[job_id] = {
        "id": job_id,
        "status": "uploaded",
        "filename": file.filename,
        "file_path": str(dest),
        "created": datetime.now().isoformat(),
        "transcript_path": None,
        "segments": None,
        "error": None,
    }

    return {"job_id": job_id, "filename": file.filename}


@app.post("/api/jobs/{job_id}/start")
async def start_job(job_id: str, model: str = "", num_speakers: int = 0, language: str = "", client_id: str = "", diarize: bool = False):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] not in ("uploaded", "error"):
        raise HTTPException(status_code=400, detail=f"Job already in state: {job['status']}")

    effective_model = model if model and model in VALID_MODELS else settings.get("web_model", "large-v3")
    job["model"] = effective_model
    job["num_speakers"] = num_speakers if num_speakers > 0 else None
    job["language"] = language or None
    job["client_id"] = client_id
    job["diarize"] = diarize
    job["status"] = "queued"
    job["error"] = None

    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _process_job, job_id, loop)

    return {"job_id": job_id, "status": "queued", "model": effective_model}


class LocalFileRequest(BaseModel):
    file_path: str
    model: str = ""
    diarize: bool = False
    num_speakers: int = 0
    language: str = ""
    client_id: str = ""


@app.post("/api/transcribe-local")
async def transcribe_local(req: LocalFileRequest):
    """Start transcription of a file already on this machine (no upload needed)."""
    path = Path(req.file_path).expanduser().resolve()
    if path.suffix.lower() not in MEDIA_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Not an audio/video file: {path.name}")
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    job_id = uuid.uuid4().hex[:12]
    effective_model = req.model if req.model and req.model in VALID_MODELS else settings.get("web_model", "large-v3")

    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "filename": path.name,
        "file_path": str(path),
        "created": datetime.now().isoformat(),
        "transcript_path": None,
        "segments": None,
        "error": None,
        "model": effective_model,
        "num_speakers": req.num_speakers if req.num_speakers > 0 else None,
        "language": req.language or None,
        "client_id": req.client_id,
        "diarize": req.diarize,
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _process_job, job_id, loop)

    return {"job_id": job_id, "filename": path.name, "status": "queued", "model": effective_model}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return {
        "id": job["id"],
        "status": job["status"],
        "filename": job.get("filename"),
        "model": job.get("model"),
        "created": job.get("created"),
        "error": job.get("error"),
        "num_speakers": job.get("num_speakers", 0),
        "duration": job.get("duration"),
        "has_transcript": job.get("transcript_path") is not None,
    }


@app.get("/api/jobs/{job_id}/download")
async def download_transcript(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    path = job.get("transcript_path")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Transcript not ready")
    filename = job.get("filename", "audio")
    dl_name = Path(filename).stem + "_transcript.txt"
    return FileResponse(path, filename=dl_name, media_type="text/plain")


@app.get("/api/jobs/{job_id}/download-audio")
async def download_audio(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    path = job.get("file_path")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path, filename=job.get("filename", "audio.wav"))

# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    ws_connections[client_id] = websocket
    try:
        while True:
            # Keep connection alive; client can send pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        ws_connections.pop(client_id, None)
    except Exception:
        ws_connections.pop(client_id, None)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = settings.get("web_host", "127.0.0.1")
    port = int(settings.get("web_port", 8000))
    print(f"Starting web server on {host}:{port}")
    print(f"Upload dir:     {UPLOAD_DIR}")
    print(f"Transcript dir: {TRANSCRIPT_DIR}")
    print(f"GPU available:  {_detect_device() == 'cuda'}")
    print(f"HF token set:   {bool(settings.get('hf_token', ''))}")

    uvicorn.run(app, host=host, port=port)
