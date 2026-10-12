"""
Remote Audio Ingest - lets other devices (phones, laptops) stream audio
chunks to this machine over the tailnet and have them transcribed here.

A remote session is a normal recording session: it gets its own folder under
recordings/ with a session.json manifest, a growing .wav and a transcript
rewritten after every chunk - so the live transcript API (status, segments,
timeline, SSE stream) serves remote recordings exactly like local ones,
with no extra wiring.

Chunks are self-contained audio files (WAV from the /record web page, or
anything ffmpeg can decode from native clients). Transcription reuses the
same inference backends as local recording (`inference_engine` setting),
with a separate whisper-server port so a local GUI recording and a remote
session can run at the same time. The backend stays warm between sessions
and is shut down after 10 minutes idle.
"""
import io
import re
import json
import shutil
import struct
import subprocess
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from transcription_engine import FasterWhisperBackend, WhisperCppBackend

NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")

BACKEND_IDLE_UNLOAD = 600  # stop a warm backend after this many idle seconds


def _fmt_hms(total: float) -> str:
    h = int(total // 3600)
    m = int((total % 3600) // 60)
    s = int(total % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _sniff_kind(data: bytes) -> str:
    """Best-effort container detection so temp files get a useful suffix."""
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] == b"\x1aE\xdf\xa3":  # EBML - webm/mkv
        return "webm"
    if data[:4] == b"OggS":
        return "ogg"
    if data[4:8] == b"ftyp":  # mp4/m4a
        return "m4a"
    return "bin"


def _parse_wav(data: bytes) -> Tuple[int, int, int, bytes]:
    """Return (rate, channels, sampwidth, pcm_frames) from a WAV file."""
    with wave.open(io.BytesIO(data), "rb") as wf:
        return (wf.getframerate(), wf.getnchannels(), wf.getsampwidth(),
                wf.readframes(wf.getnframes()))


def _wav_header(rate: int, channels: int, sampwidth: int, data_len: int) -> bytes:
    byte_rate = rate * channels * sampwidth
    return (b"RIFF" + struct.pack("<I", 36 + data_len) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, rate,
                                    byte_rate, channels * sampwidth,
                                    sampwidth * 8)
            + b"data" + struct.pack("<I", data_len))


def _append_pcm(wav_path: Path, params: Tuple[int, int, int], pcm: bytes) -> None:
    """Append PCM frames to a WAV file, keeping the header valid after every
    chunk so the file is playable even if the session is never stopped."""
    if not wav_path.exists():
        wav_path.write_bytes(_wav_header(*params, len(pcm)) + pcm)
        return
    with open(wav_path, "r+b") as f:
        f.seek(0, 2)
        f.write(pcm)
        size = f.tell()
        f.seek(4)
        f.write(struct.pack("<I", size - 8))
        f.seek(40)
        f.write(struct.pack("<I", size - 44))


class IngestSession:
    """State for one remote recording session."""

    def __init__(self, session_dir: Path, name: str, chunk_seconds: int):
        self.dir = Path(session_dir)
        self.name = name
        self.chunk_seconds = chunk_seconds
        self.wav_path = self.dir / f"{name}.wav"
        self.transcript_path = self.dir / f"{name}.txt"
        self.started = datetime.now()
        self.ended: Optional[datetime] = None
        self.status = "recording"
        self.segments: List[dict] = []   # recording-clock absolute times
        self.photos: List[dict] = []
        self.chunks_received = 0
        self.audio_seconds = 0.0         # running recording clock
        self.wav_params: Optional[Tuple[int, int, int]] = None
        self.last_chunk_time = time.time()
        self.warnings: List[str] = []


class IngestManager:
    """One active remote session at a time, plus a warm inference backend."""

    def __init__(self, settings, recordings_dir):
        self.settings = settings
        self.recordings_dir = Path(recordings_dir)
        self.backend = None
        self.session: Optional[IngestSession] = None
        self._lock = threading.RLock()
        self._backend_last_used = 0.0
        self.ffmpeg = shutil.which("ffmpeg")
        self._shutdown = False
        watchdog = threading.Thread(target=self._watchdog_loop, daemon=True,
                                    name="ingest-watchdog")
        watchdog.start()

    # -- backend -------------------------------------------------------

    def _build_backend(self):
        engine = self.settings.get("inference_engine", "faster-whisper")
        if engine == "whisper-cpp":
            # Own port so a simultaneous local GUI recording (which spawns
            # and kills its own whisper-server) can't pull it out from
            # under us. An explicit whisper_cpp_url is shared as-is.
            return WhisperCppBackend(
                server_bin=self.settings.get("whisper_cpp_server", ""),
                model_path=self.settings.get("whisper_cpp_model", ""),
                port=self.settings.get("whisper_cpp_ingest_port", 8011),
                url=self.settings.get("whisper_cpp_url", ""),
            )
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except ImportError:
            pass
        return FasterWhisperBackend(
            model_size=self.settings.get("whisper_model", "base"),
            device=device,
            compute_type="int8" if device == "cpu" else "float16",
        )

    def _ensure_backend(self):
        if self.backend is None:
            self.backend = self._build_backend()
            print(f"[Ingest] Inference backend: {self.backend.describe()}")
        if not self.backend.load():
            self.backend = None
            raise RuntimeError("ingest inference backend failed to load "
                               "(see server console)")
        self._backend_last_used = time.time()

    # -- session lifecycle ---------------------------------------------

    def start_session(self, name: str = "", chunk_seconds: int = 15) -> dict:
        with self._lock:
            if self.session is not None:
                self._finalize(self.session, note="superseded by new session")

            safe = NAME_RE.sub("-", (name or "phone")).strip("-")[:40] or "phone"
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base = f"{timestamp}_{safe}"
            session_dir = self.recordings_dir / base
            session_dir.mkdir(parents=True, exist_ok=True)

            try:
                chunk_seconds = max(5, min(120, int(chunk_seconds)))
            except (TypeError, ValueError):
                chunk_seconds = 15

            self._ensure_backend()

            session = IngestSession(session_dir, base, chunk_seconds)
            self.session = session
            self._write_manifest(session)
            self._write_transcript(session)
            print(f"[Ingest] Session started: {base} "
                  f"({chunk_seconds}s chunks, {self.backend.describe()})")
            return {
                "ok": True,
                "session": base,
                "chunk_seconds": chunk_seconds,
                "engine": self.backend.describe(),
                "started": session.started.isoformat(timespec="seconds"),
            }

    def _require(self, session_id: str) -> IngestSession:
        session = self.session
        if session is None or session.name != session_id:
            raise KeyError(f"no active ingest session named {session_id!r}")
        return session

    def add_chunk(self, session_id: str, data: bytes,
                  client_offset: Optional[float] = None) -> dict:
        with self._lock:
            session = self._require(session_id)
            if not data:
                raise ValueError("empty chunk body")
            session.chunks_received += 1
            session.last_chunk_time = time.time()

            kind = _sniff_kind(data)
            chunk_path = session.dir / f".chunk.{kind}"
            chunk_path.write_bytes(data)

            pcm = None
            params = None
            duration = None
            warning = ""

            if kind == "wav":
                try:
                    rate, channels, sampwidth, pcm = _parse_wav(data)
                    params = (rate, channels, sampwidth)
                except (wave.Error, EOFError) as e:
                    raise ValueError(f"bad WAV chunk: {e}")

            # Anything non-WAV (or WAV that doesn't match the session file)
            # goes through ffmpeg so the archived audio stays consistent
            target = session.wav_params or (params if kind == "wav" else (16000, 1, 2))
            if params != target:
                converted = self._convert(chunk_path, target)
                if converted is not None:
                    chunk_path = converted
                    rate, channels, sampwidth, pcm = _parse_wav(
                        converted.read_bytes())
                    params = (rate, channels, sampwidth)
                else:
                    pcm = None
                    warning = ("ffmpeg unavailable or failed - chunk "
                               "transcribed but not archived to the session WAV")
                    session.warnings.append(warning)

            offset = float(client_offset) if client_offset is not None \
                else session.audio_seconds

            if pcm:
                if session.wav_params is None:
                    session.wav_params = params
                _append_pcm(session.wav_path, session.wav_params, pcm)
                rate, channels, sampwidth = params
                duration = len(pcm) / (channels * sampwidth) / rate
            if duration is None:
                duration = float(session.chunk_seconds)

            error = ""
            new_segments: List[dict] = []
            try:
                raw = self.backend.transcribe(str(chunk_path))
                for seg in raw:
                    text = (seg.get("text") or "").strip()
                    if not text:
                        continue
                    new_segments.append({
                        "start": round(offset + seg["start"], 2),
                        "end": round(offset + seg["end"], 2),
                        "start_hms": _fmt_hms(offset + seg["start"]),
                        "end_hms": _fmt_hms(offset + seg["end"]),
                        "text": text,
                    })
            except Exception as e:
                error = f"transcription failed: {e}"
                session.warnings.append(error)
                print(f"[Ingest] {error}")

            session.segments.extend(new_segments)
            session.audio_seconds = max(session.audio_seconds, offset + duration)
            self._write_transcript(session)
            self._backend_last_used = time.time()

            return {
                "ok": not error,
                "session": session.name,
                "chunk": session.chunks_received,
                "segments": new_segments,
                "segment_count": len(session.segments),
                "audio_seconds": round(session.audio_seconds, 2),
                "warning": warning,
                "error": error,
            }

    def add_photo(self, session_id: str, data: bytes,
                  content_type: str = "") -> dict:
        """Attach a photo (phone camera, whiteboard, ...) to the session,
        stamped with the recording clock - served exactly like screenshots."""
        with self._lock:
            session = self._require(session_id)
            if not data:
                raise ValueError("empty photo body")
            if data[:8] == b"\x89PNG\r\n\x1a\n":
                ext = ".png"
            elif data[:3] == b"\xff\xd8\xff" or "jpeg" in content_type:
                ext = ".jpg"
            elif "png" in content_type:
                ext = ".png"
            else:
                raise ValueError("photo must be JPEG or PNG")

            elapsed = session.audio_seconds
            hms = _fmt_hms(elapsed)
            filename = (f"photo_{len(session.photos) + 1:03d}_"
                        f"{hms.replace(':', '-')}{ext}")
            (session.dir / filename).write_bytes(data)

            entry = {
                "file": filename,
                "elapsed_seconds": round(elapsed, 1),
                "elapsed_hms": hms,
                "taken_at": datetime.now().isoformat(timespec="seconds"),
            }
            session.photos.append(entry)
            session.last_chunk_time = time.time()
            self._write_manifest(session)
            return {"ok": True, "session": session.name, "photo": entry,
                    "photo_count": len(session.photos)}

    def stop_session(self, session_id: str) -> dict:
        with self._lock:
            session = self._require(session_id)
            self._finalize(session)
            return {
                "ok": True,
                "session": session.name,
                "duration": _fmt_hms(session.audio_seconds),
                "chunks": session.chunks_received,
                "segment_count": len(session.segments),
                "photo_count": len(session.photos),
                "warnings": session.warnings,
            }

    def status(self) -> dict:
        with self._lock:
            session = self.session
            return {
                "active_session": session.name if session else None,
                "audio_seconds": round(session.audio_seconds, 2) if session else 0,
                "chunks_received": session.chunks_received if session else 0,
                "segment_count": len(session.segments) if session else 0,
                "backend": self.backend.describe() if self.backend else None,
                "backend_loaded": self.backend is not None,
                "ffmpeg": bool(self.ffmpeg),
            }

    def shutdown(self) -> None:
        self._shutdown = True
        with self._lock:
            if self.session is not None:
                self._finalize(self.session, note="server shutdown")
            if self.backend is not None:
                self.backend.stop()
                self.backend = None

    # -- internals -------------------------------------------------------

    def _finalize(self, session: IngestSession, note: str = "") -> None:
        session.status = "stopped"
        session.ended = datetime.now()
        if note:
            session.warnings.append(note)
        self._write_transcript(session, final=True)
        self._write_manifest(session)
        for tmp in session.dir.glob(".chunk.*"):
            try:
                tmp.unlink()
            except OSError:
                pass
        if self.session is session:
            self.session = None
        extra = f" ({note})" if note else ""
        print(f"[Ingest] Session stopped: {session.name} - "
              f"{_fmt_hms(session.audio_seconds)}, "
              f"{len(session.segments)} segments{extra}")

    def _convert(self, in_path: Path, params: Tuple[int, int, int]) -> Optional[Path]:
        """ffmpeg-decode any audio container to a WAV matching `params`."""
        if not self.ffmpeg:
            return None
        rate, channels, _ = params
        out_path = in_path.with_name(".chunk.converted.wav")
        cmd = [self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
               "-i", str(in_path),
               "-ac", str(channels), "-ar", str(rate), "-sample_fmt", "s16",
               str(out_path)]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0 or not out_path.exists():
            print(f"[Ingest] ffmpeg convert failed: "
                  f"{result.stderr.decode(errors='replace')[:200]}")
            return None
        return out_path

    def _write_manifest(self, session: IngestSession) -> None:
        """Same schema CaptureManager writes, so SessionWatcher needs no changes."""
        manifest = {
            "version": 1,
            "name": session.name,
            "status": session.status,
            "started": session.started.isoformat(timespec="seconds"),
            "ended": session.ended.isoformat(timespec="seconds")
                     if session.ended else None,
            "chunk_duration": session.chunk_seconds,
            "audio_file": session.wav_path.name,
            "transcript_file": session.transcript_path.name,
            "screenshots": list(session.photos),
            "source": "remote-ingest",
        }
        try:
            tmp = session.dir / "session.json.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            tmp.replace(session.dir / "session.json")
        except OSError as e:
            print(f"[Ingest] Error writing manifest: {e}")

    def _write_transcript(self, session: IngestSession, final: bool = False) -> None:
        """Same line format TranscriptionEngine writes ([HH:MM:SS - HH:MM:SS])
        so the live server's parser and any downstream tooling just work."""
        backend_desc = self.backend.describe() if self.backend else "unknown"
        try:
            with open(session.transcript_path, "w", encoding="utf-8") as f:
                if final:
                    f.write("# Meeting Transcript (remote recording)\n\n")
                    f.write(f"Date: {session.started:%Y-%m-%d}\n")
                    f.write(f"Start time: {session.started:%H:%M:%S}\n")
                    if session.ended:
                        f.write(f"End time: {session.ended:%H:%M:%S}\n")
                    f.write(f"Duration: {_fmt_hms(session.audio_seconds)}\n")
                    f.write(f"Model: {backend_desc}\n")
                    f.write(f"Chunks: {session.chunks_received}\n")
                    f.write("\n---\n\n")
                else:
                    f.write(f"=== {session.name} (remote recording) ===\n\n")
                    f.write(f"Recording started: "
                            f"{session.started:%Y-%m-%d %H:%M:%S}\n\n")
                    f.write("-" * 80 + "\n\n")
                for seg in session.segments:
                    f.write(f"[{seg['start_hms']} - {seg['end_hms']}]  "
                            f"{seg['text']}\n")
                if final:
                    f.write("\n---\n\nEnd of transcript.\n")
                else:
                    f.write("\n" + "-" * 80 + "\n")
                    f.write(f"\nTranscript generated by {backend_desc} "
                            f"(remote ingest)\n")
        except OSError as e:
            print(f"[Ingest] Error writing transcript: {e}")

    def _watchdog_loop(self) -> None:
        """Finalize sessions whose device went silent; unload an idle backend."""
        while not self._shutdown:
            time.sleep(15)
            with self._lock:
                session = self.session
                if session is not None:
                    idle_limit = max(120, 3 * session.chunk_seconds)
                    if time.time() - session.last_chunk_time > idle_limit:
                        self._finalize(session, note="idle timeout - "
                                       "no chunks from device")
                elif (self.backend is not None and self._backend_last_used
                      and time.time() - self._backend_last_used
                      > BACKEND_IDLE_UNLOAD):
                    print("[Ingest] Backend idle - unloading")
                    self.backend.stop()
                    self.backend = None
