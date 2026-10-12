"""
Live Transcript Server - read-only web API for the live meeting transcript.

Each recording is a session folder (recordings/<title>/) holding the audio,
transcript, screenshots and a session.json manifest. CaptureManager writes the
manifest the moment recording starts and TranscriptionEngine rewrites the
transcript after every audio chunk, so this server can announce a new session
immediately and stream segments and screenshots as they appear - LLMs (or
anything else on the tailnet) can follow the meeting live with text and
images as context.

The read side never writes to any file, never touches audio, and does not
import torch/whisper - safe to run alongside an in-progress recording. Older
flat recordings (recordings/*_teams-call.txt, no folder) are still served.

It also hosts the remote ingest API (POST /ingest/*) plus the /record web
page, which let a phone or another computer stream mic audio to this machine
for GPU transcription - those sessions are written as normal session folders
and served right back through the same read endpoints. Ingest is lazy: the
inference backend only loads when a remote session actually starts.

Run standalone:
    python src/live_server.py [--host HOST] [--port PORT] [--dir RECORDINGS_DIR]

Or it is started automatically as a background thread by the GUI app
(main.py) when live_server_enabled is true in settings.
"""
import argparse
import asyncio
import atexit
import json
import re
import socket
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse

from settings_manager import SettingsManager

# Transcript lines look like: [00:01:30 - 00:01:58]  some text
SEGMENT_RE = re.compile(r"^\[(\d{2,}):(\d{2}):(\d{2}) - (\d{2,}):(\d{2}):(\d{2})\]\s+(.+)$")
STARTED_RE = re.compile(r"^(?:Recording started|Start time):\s*(.+)$")

REFRESH_INTERVAL = 0.5  # min seconds between disk checks
ACTIVE_WINDOW = 120     # minimum staleness window before "active" turns false

IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0


def _parse_transcript(text: str) -> Tuple[List[dict], str]:
    segments = []
    started = ""
    for line in text.splitlines():
        m = SEGMENT_RE.match(line)
        if m:
            h1, m1, s1, h2, m2, s2, seg_text = m.groups()
            segments.append({
                "index": len(segments),
                "start": int(h1) * 3600 + int(m1) * 60 + int(s1),
                "end": int(h2) * 3600 + int(m2) * 60 + int(s2),
                "start_hms": f"{h1}:{m1}:{s1}",
                "end_hms": f"{h2}:{m2}:{s2}",
                "text": seg_text.strip(),
            })
            continue
        sm = STARTED_RE.match(line)
        if sm and not started:
            started = sm.group(1).strip()
    return segments, started


class SessionWatcher:
    """Polls the recordings directory and tracks the freshest recording session."""

    def __init__(self, recordings_dir: Path):
        self.recordings_dir = Path(recordings_dir)
        self._lock = threading.Lock()
        self._last_check = 0.0
        self._sig = None       # change-detection signature of last load
        self.generation = 0    # bumps whenever the active session changes
        self.identity: Optional[str] = None
        self.name = ""
        self.status = "none"   # starting|recording|stopped|error|legacy|none
        self.manifest: Optional[dict] = None
        self.session_dir: Optional[Path] = None
        self.transcript_path: Optional[Path] = None
        self.raw_text = ""
        self.segments: List[dict] = []
        self.screenshots: List[dict] = []
        self.recording_started = ""
        self.last_activity: Optional[float] = None

    # -- discovery -----------------------------------------------------

    def _find_session(self):
        """Return ('manifest'|'legacy', path) for the freshest session, or None."""
        best = None  # (freshness, kind, path)
        for pattern in ("*/session.json", "*/*/session.json"):
            for manifest_path in self.recordings_dir.glob(pattern):
                freshness = _mtime(manifest_path)
                for txt in manifest_path.parent.glob("*.txt"):
                    freshness = max(freshness, _mtime(txt))
                if freshness >= 0 and (best is None or freshness > best[0]):
                    best = (freshness, "manifest", manifest_path)
        # Legacy flat layout (pre session-folder recordings)
        for txt in self.recordings_dir.glob("*_teams-call.txt"):
            freshness = _mtime(txt)
            if freshness >= 0 and (best is None or freshness > best[0]):
                best = (freshness, "legacy", txt)
        return best

    # -- loading -------------------------------------------------------

    def _reset(self):
        self.identity = None
        self.name = ""
        self.status = "none"
        self.manifest = None
        self.session_dir = None
        self.transcript_path = None
        self.raw_text = ""
        self.segments = []
        self.screenshots = []
        self.recording_started = ""
        self.last_activity = None
        self._sig = None

    def _bump_if_new(self, identity: str):
        if self.identity != identity:
            self.generation += 1
            self.identity = identity

    def _load_manifest_session(self, manifest_path: Path):
        session_dir = manifest_path.parent
        try:
            m_stat = manifest_path.stat()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        transcript_path = None
        t_stat = None
        transcript_name = manifest.get("transcript_file") or ""
        if transcript_name:
            candidate = session_dir / transcript_name
            try:
                t_stat = candidate.stat()
                transcript_path = candidate
            except OSError:
                pass

        sig = ("manifest", str(session_dir), m_stat.st_mtime_ns,
               (t_stat.st_mtime_ns, t_stat.st_size) if t_stat else None)
        if sig == self._sig:
            return

        raw, segments = "", []
        if transcript_path:
            try:
                raw = transcript_path.read_text(encoding="utf-8", errors="replace")
                segments, _ = _parse_transcript(raw)
            except OSError:
                transcript_path = None

        self._bump_if_new(str(session_dir))
        self._sig = sig
        self.session_dir = session_dir
        self.manifest = manifest
        self.name = manifest.get("name") or session_dir.name
        self.status = manifest.get("status", "unknown")
        self.recording_started = manifest.get("started") or ""
        self.transcript_path = transcript_path
        self.raw_text = raw
        self.segments = segments
        self.screenshots = [
            {**shot, "url": f"/screenshots/{shot['file']}"}
            for shot in manifest.get("screenshots", [])
            if isinstance(shot, dict) and shot.get("file")
        ]
        self.last_activity = max(m_stat.st_mtime,
                                 t_stat.st_mtime if t_stat else 0.0)

    def _load_legacy_session(self, txt_path: Path):
        try:
            t_stat = txt_path.stat()
        except OSError:
            return
        sig = ("legacy", str(txt_path), t_stat.st_mtime_ns, t_stat.st_size)
        if sig == self._sig:
            return
        try:
            raw = txt_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return

        segments, started = _parse_transcript(raw)
        self._bump_if_new(str(txt_path))
        self._sig = sig
        self.session_dir = txt_path.parent
        self.manifest = None
        self.name = txt_path.stem
        self.status = "legacy"
        self.recording_started = started
        self.transcript_path = txt_path
        self.raw_text = raw
        self.segments = segments
        self.screenshots = []
        self.last_activity = t_stat.st_mtime

    # -- public API ----------------------------------------------------

    def refresh(self) -> None:
        """Re-scan the recordings directory if enough time has passed."""
        with self._lock:
            now = time.monotonic()
            if now - self._last_check < REFRESH_INTERVAL:
                return
            self._last_check = now

            found = self._find_session()
            if found is None:
                if self.identity is not None:
                    self.generation += 1
                    self._reset()
                return
            _, kind, path = found
            if kind == "manifest":
                self._load_manifest_session(path)
            else:
                self._load_legacy_session(path)

    def get_state(self):
        """(generation, name, status, raw_text, segments, screenshots) snapshot.

        Lists are replaced wholesale on refresh, never mutated, so handing
        out the references is safe.
        """
        self.refresh()
        with self._lock:
            return (self.generation, self.name, self.status,
                    self.raw_text, self.segments, self.screenshots)

    def _active_window(self) -> float:
        chunk = 30
        if self.manifest:
            try:
                chunk = int(self.manifest.get("chunk_duration") or 30)
            except (TypeError, ValueError):
                chunk = 30
        return max(ACTIVE_WINDOW, 3 * chunk)

    def snapshot(self) -> dict:
        self.refresh()
        with self._lock:
            age = (time.time() - self.last_activity) if self.last_activity else None
            if self.status in ("recording", "starting", "legacy"):
                active = age is not None and age < self._active_window()
            else:
                active = False
            return {
                "session": self.name or None,
                "status": self.status,
                "recording_active": active,
                "generation": self.generation,
                "recording_started": self.recording_started,
                "session_dir": str(self.session_dir) if self.session_dir else None,
                "transcript_file": str(self.transcript_path) if self.transcript_path else None,
                "segment_count": len(self.segments),
                "screenshot_count": len(self.screenshots),
                "last_update": datetime.fromtimestamp(self.last_activity).isoformat()
                               if self.last_activity else None,
                "seconds_since_update": round(age, 1) if age is not None else None,
            }

    def screenshot_path(self, filename: str) -> Optional[Path]:
        """Resolve a screenshot filename inside the current session dir, safely."""
        self.refresh()
        with self._lock:
            if not self.session_dir or not filename or "/" in filename or ".." in filename:
                return None
            if Path(filename).suffix.lower() not in IMAGE_TYPES:
                return None
            path = self.session_dir / filename
            return path if path.is_file() else None


settings = SettingsManager()
watcher = SessionWatcher(settings.get_recordings_dir())

app = FastAPI(title="Live Transcript Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/")
async def index():
    return {
        "service": "teams-to-transcript live transcript API",
        "description": "Read-only view of the live meeting: transcript segments "
                       "plus screenshots, both timestamped on the recording clock. "
                       "A new chunk is transcribed roughly every "
                       f"{settings.get('chunk_duration', 30)}s while recording.",
        "endpoints": {
            "GET /record": "web page: record on a phone/other device, chunks "
                           "stream here for transcription",
            "POST /ingest/start": "start a remote session "
                                  "(JSON: {name, chunk_seconds})",
            "POST /ingest/chunk?session=ID": "body = one audio chunk "
                                             "(WAV, or anything ffmpeg decodes)",
            "POST /ingest/photo?session=ID": "attach a JPEG/PNG photo, stamped "
                                             "on the recording clock",
            "POST /ingest/stop?session=ID": "finalize the remote session",
            "GET /ingest/status": "remote ingest state",
            "GET /status": "session name, recording state, segment/screenshot counts",
            "GET /transcript": "full transcript as plain text (?tail=N for last N segments)",
            "GET /transcript.json": "all segments and screenshots as JSON",
            "GET /segments?since=N&generation=G":
                "segments with index >= N; returns reset=true (and all segments) "
                "if the session changed since generation G",
            "GET /screenshots": "screenshot list with elapsed-clock timestamps and URLs",
            "GET /screenshots/{filename}": "the screenshot image itself",
            "GET /timeline": "segments and screenshots interleaved by recording clock - "
                             "ready to use as multimodal LLM context",
            "GET /stream":
                "Server-Sent Events: 'session' on new recording (fires the moment "
                "recording starts), 'segments' per transcribed chunk, 'screenshot' "
                "when one is captured, 'status' on recording state changes",
        },
        "status": watcher.snapshot(),
    }


@app.get("/status")
async def status():
    return watcher.snapshot()


@app.get("/transcript", response_class=PlainTextResponse)
async def transcript(tail: int = Query(0, ge=0)):
    _, name, _, raw_text, segments, _ = watcher.get_state()
    if not name:
        return PlainTextResponse("No transcript available.\n", status_code=404)
    if tail > 0:
        lines = [f"[{s['start_hms']} - {s['end_hms']}]  {s['text']}"
                 for s in segments[-tail:]]
        return "\n".join(lines) + "\n"
    return raw_text


@app.get("/transcript.json")
async def transcript_json():
    generation, name, session_status, _, segments, screenshots = watcher.get_state()
    return {
        "generation": generation,
        "session": name or None,
        "recording_status": session_status,
        "segment_count": len(segments),
        "segments": segments,
        "screenshot_count": len(screenshots),
        "screenshots": screenshots,
        "status": watcher.snapshot(),
    }


@app.get("/segments")
async def segments(since: int = Query(0, ge=0), generation: int = -1):
    gen, name, session_status, _, segs, shots = watcher.get_state()
    reset = generation not in (-1, gen)
    start = 0 if reset else min(since, len(segs))
    return {
        "generation": gen,
        "session": name or None,
        "recording_status": session_status,
        "reset": reset,
        "segments": segs[start:],
        "next_since": len(segs),
        "screenshots_total": len(shots),
    }


@app.get("/screenshots")
async def screenshots():
    gen, name, _, _, _, shots = watcher.get_state()
    return {"generation": gen, "session": name or None,
            "count": len(shots), "screenshots": shots}


@app.get("/screenshots/{filename}")
async def screenshot_file(filename: str):
    path = watcher.screenshot_path(filename)
    if path is None:
        return PlainTextResponse("Screenshot not found.\n", status_code=404)
    return FileResponse(path, media_type=IMAGE_TYPES[path.suffix.lower()])


@app.get("/timeline")
async def timeline():
    """Segments and screenshots merged into one recording-clock-ordered list."""
    gen, name, session_status, _, segs, shots = watcher.get_state()
    items = [{"type": "speech", **seg} for seg in segs]
    items += [{
        "type": "screenshot",
        "time": shot.get("elapsed_seconds", 0),
        "time_hms": shot.get("elapsed_hms", ""),
        "file": shot.get("file"),
        "url": shot.get("url"),
        "taken_at": shot.get("taken_at"),
    } for shot in shots]
    items.sort(key=lambda i: i.get("start", i.get("time", 0)) or 0)
    return {
        "generation": gen,
        "session": name or None,
        "recording_status": session_status,
        "items": items,
    }


@app.get("/stream")
async def stream(since: int = Query(0, ge=0), generation: int = -1):
    async def event_gen():
        gen, name, session_status, _, segs, shots = watcher.get_state()
        seg_cursor = min(since, len(segs)) if generation in (-1, gen) else 0
        yield _sse("session", {"generation": gen, "session": name or None,
                               "status": session_status,
                               "segment_count": len(segs),
                               "screenshot_count": len(shots)})
        if segs[seg_cursor:]:
            yield _sse("segments", {"generation": gen,
                                    "segments": segs[seg_cursor:],
                                    "next_since": len(segs)})
        if shots:
            yield _sse("screenshot", {"generation": gen, "screenshots": shots})
        seg_cursor, shot_cursor = len(segs), len(shots)
        last_gen, last_status = gen, session_status
        last_sent = time.monotonic()

        while True:
            await asyncio.sleep(1.0)
            gen, name, session_status, _, segs, shots = watcher.get_state()
            if gen != last_gen:
                last_gen = gen
                seg_cursor = shot_cursor = 0
                last_status = session_status
                yield _sse("session", {"generation": gen, "session": name or None,
                                       "status": session_status,
                                       "segment_count": len(segs),
                                       "screenshot_count": len(shots)})
                last_sent = time.monotonic()
            elif session_status != last_status:
                last_status = session_status
                yield _sse("status", {"generation": gen, "session": name or None,
                                      "status": session_status})
                last_sent = time.monotonic()
            if len(segs) > seg_cursor:
                yield _sse("segments", {"generation": gen,
                                        "segments": segs[seg_cursor:],
                                        "next_since": len(segs)})
                seg_cursor = len(segs)
                last_sent = time.monotonic()
            if len(shots) > shot_cursor:
                yield _sse("screenshot", {"generation": gen,
                                          "screenshots": shots[shot_cursor:]})
                shot_cursor = len(shots)
                last_sent = time.monotonic()
            if time.monotonic() - last_sent > 15:
                yield ": keepalive\n\n"
                last_sent = time.monotonic()

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Remote ingest - phones/other devices stream audio chunks here
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).resolve().parent / "static"

_ingest_lock = threading.Lock()
_ingest_manager = None


def _ingest():
    """Lazily create the IngestManager (loads no model until a session starts)."""
    global _ingest_manager
    if not settings.get("ingest_enabled", True):
        raise HTTPException(status_code=403, detail="ingest is disabled "
                            "(ingest_enabled setting)")
    with _ingest_lock:
        if _ingest_manager is None:
            from ingest import IngestManager
            _ingest_manager = IngestManager(settings, watcher.recordings_dir)
            atexit.register(_ingest_manager.shutdown)
    return _ingest_manager


@app.on_event("shutdown")
async def _shutdown_ingest():
    """uvicorn's graceful shutdown - finalize any remote session and kill a
    spawned whisper-server (atexit alone is not reliable under uvicorn)."""
    with _ingest_lock:
        manager = _ingest_manager
    if manager is not None:
        await asyncio.to_thread(manager.shutdown)


@app.get("/record")
async def record_page():
    page = STATIC_DIR / "record.html"
    if not page.is_file():
        return PlainTextResponse("record.html missing\n", status_code=404)
    return FileResponse(page, media_type="text/html")


@app.post("/ingest/start")
async def ingest_start(request: Request):
    manager = _ingest()
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    try:
        return await asyncio.to_thread(
            manager.start_session,
            str(body.get("name", "")),
            body.get("chunk_seconds", 15),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/chunk")
async def ingest_chunk(request: Request, session: str = Query(...),
                       offset: Optional[float] = Query(None, ge=0)):
    manager = _ingest()
    data = await request.body()
    try:
        return await asyncio.to_thread(manager.add_chunk, session, data, offset)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e.args[0]))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ingest/photo")
async def ingest_photo(request: Request, session: str = Query(...)):
    manager = _ingest()
    data = await request.body()
    content_type = request.headers.get("content-type", "")
    try:
        return await asyncio.to_thread(manager.add_photo, session, data,
                                       content_type)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e.args[0]))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ingest/stop")
async def ingest_stop(session: str = Query(...)):
    manager = _ingest()
    try:
        return await asyncio.to_thread(manager.stop_session, session)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e.args[0]))


@app.get("/ingest/status")
async def ingest_status():
    if not settings.get("ingest_enabled", True):
        return {"enabled": False}
    with _ingest_lock:
        manager = _ingest_manager
    if manager is None:
        return {"enabled": True, "active_session": None,
                "backend_loaded": False}
    return {"enabled": True, **manager.status()}


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def resolve_host(settings_obj: SettingsManager) -> str:
    """Explicit live_host setting, else the tailscale IP, else localhost."""
    host = settings_obj.get("live_host", "")
    if host:
        return host
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.TimeoutExpired):
        pass
    print("[LiveServer] WARNING: could not detect tailscale IP; binding to 127.0.0.1")
    return "127.0.0.1"


def start_in_thread(settings_obj: SettingsManager) -> Optional[threading.Thread]:
    """Start the server as a daemon thread inside the GUI app.

    Returns the thread, or None if the port is already taken (e.g. a
    standalone instance of this server is running).
    """
    import uvicorn

    host = resolve_host(settings_obj)
    port = int(settings_obj.get("live_port", 8010))

    with socket.socket() as probe:
        try:
            probe.bind((host, port))
        except OSError:
            print(f"[LiveServer] {host}:{port} already in use - "
                  f"assuming a live server is already running, skipping")
            return None

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="live-server")
    thread.start()
    print(f"[LiveServer] Live transcript API: http://{host}:{port}")
    return thread


def main() -> int:
    import uvicorn

    parser = argparse.ArgumentParser(description="Live transcript web server")
    parser.add_argument("--host", default="", help="Bind address (default: tailscale IP)")
    parser.add_argument("--port", type=int, default=0, help="Port (default: live_port setting)")
    parser.add_argument("--dir", default="", help="Recordings directory to watch")
    args = parser.parse_args()

    if args.dir:
        watcher.recordings_dir = Path(args.dir)

    host = args.host or resolve_host(settings)
    port = args.port or int(settings.get("live_port", 8010))

    print(f"Watching: {watcher.recordings_dir}")
    print(f"Serving:  http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
