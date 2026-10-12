"""
Transcription Engine - Processes audio chunks with Whisper AI

Two inference backends, selected by the `inference_engine` setting:
  faster-whisper - CTranslate2, CPU (int8) or NVIDIA CUDA. The default.
  whisper-cpp    - a local whisper.cpp server (GPU via Vulkan - works on AMD
                   iGPUs like the Radeon 8060S, which CTranslate2 cannot use).
"""
import json
import os
import re
import socket
import subprocess
import threading
import queue
import time
import urllib.request
import uuid
from pathlib import Path
from typing import List, Optional, Callable
from datetime import datetime, timedelta

# Suppress HuggingFace Hub warnings on Windows
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'


# --- Whisper hallucination filtering --------------------------------------
# Whisper (both faster-whisper and whisper.cpp) emits stock phrases on silence,
# music, or non-speech: video outros, subtitle credits, a bare "thank you", etc.
# We drop any segment containing a spam substring, and - crucially - if the ENTIRE
# output of a chunk/utterance is nothing but a stock filler phrase we treat the
# whole thing as a hallucination and discard it. Real speech that merely *contains*
# one of these words is left untouched, so this is safe on genuine meeting audio.
_HALLUCINATION_SUBSTRINGS = (
    "thanks for watching", "thank you for watching", "thank for watching",
    "please subscribe", "don't forget to subscribe", "like and subscribe",
    "subscribe to my channel", "see you in the next video", "see you next time",
    "subtitles by", "subtitles created by", "captions by", "amara.org",
    "transcription by", "subtitling by", "translated by", "[music]", "[applause]",
    "[laughter]", "[blank_audio]", "♪", "www.", ".com/",
)


def _norm_phrase(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace - for phrase matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w ]+", " ", text).strip().lower())


# If a chunk's WHOLE normalized output equals one of these, it's a silence
# hallucination. Kept conservative (no bare "so"/"okay"/"yeah") to avoid dropping
# real one-word backchannels.
_HALLUCINATION_WHOLE = {
    _norm_phrase(p) for p in (
        "you", "thank you", "thank you very much", "thank you so much",
        "thanks", "thanks a lot", "bye", "bye bye", "goodbye",
    )
}


def clean_segments(segments: List[dict]) -> List[dict]:
    """Drop hallucinated segments. Returns a filtered [{start,end,text}] list.

    Used by BOTH the 30s-chunk path and the streaming path (each whisper backend
    runs it), so hallucination handling is identical everywhere.
    """
    if not segments:
        return segments
    kept = []
    for s in segments:
        t = (s.get("text") or "").strip()
        low = t.lower()
        if any(sub in low for sub in _HALLUCINATION_SUBSTRINGS):
            continue
        if not _norm_phrase(t):  # pure punctuation / music note
            continue
        kept.append(s)
    whole = _norm_phrase(" ".join((s.get("text") or "") for s in kept))
    if whole in _HALLUCINATION_WHOLE:
        return []
    return kept


class FasterWhisperBackend:
    """CPU/CUDA inference via faster-whisper (CTranslate2)."""

    def __init__(self, model_size: str, device: str, compute_type: str,
                 language: str = "en", vad_filter: bool = True):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.vad_filter = vad_filter  # Silero VAD gates out non-speech (needs onnxruntime)
        self.model = None

    def describe(self) -> str:
        return f"faster-whisper ({self.model_size}, {self.device}/{self.compute_type})"

    def load(self) -> bool:
        if self.model is not None:
            return True
        try:
            from faster_whisper import WhisperModel
            print(f"Loading Whisper model: {self.model_size} "
                  f"({self.device}, {self.compute_type})...")
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            print("Whisper model loaded.")
            return True
        except Exception as e:
            print(f"ERROR loading Whisper model: {e}")
            return False

    def transcribe(self, chunk_file: str) -> List[dict]:
        """Transcribe one chunk; returns [{start, end, text}] (chunk-relative)."""
        # Decode-stability settings that suppress hallucinations on silence/music:
        #  - condition_on_previous_text=False stops a bad guess from snowballing;
        #  - the thresholds treat low-confidence / no-speech audio as empty;
        #  - VAD gates out non-speech before decoding.
        kwargs = dict(
            beam_size=5,
            language=self.language,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        if self.vad_filter:
            kwargs["vad_filter"] = True
            kwargs["vad_parameters"] = {"min_silence_duration_ms": 500}
        try:
            segments, info = self.model.transcribe(chunk_file, **kwargs)
        except Exception as e:
            # e.g. onnxruntime missing for VAD, or an older lib rejecting a kwarg -
            # fall back to a plain decode once and stop trying VAD.
            print(f"faster-whisper transcribe fell back (no VAD): {e}")
            self.vad_filter = False
            segments, info = self.model.transcribe(
                chunk_file, beam_size=5, language=self.language,
                condition_on_previous_text=False)
        result = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                result.append({"start": segment.start, "end": segment.end,
                               "text": text})
        return clean_segments(result)

    def stop(self) -> None:
        pass


class WhisperCppBackend:
    """GPU inference via a local whisper.cpp server (Vulkan on AMD/Intel).

    Either connects to an already-running server (`url`) or spawns
    `whisper-server` itself and shuts it down when recording stops.
    """

    def __init__(self, server_bin: str = "", model_path: str = "",
                 port: int = 8009, url: str = "", language: str = "en"):
        self.server_bin = server_bin
        self.model_path = model_path
        self.port = int(port)
        self.url = url.rstrip("/") if url else ""
        self.language = language
        self.process: Optional[subprocess.Popen] = None

    def describe(self) -> str:
        model = Path(self.model_path).stem if self.model_path else "external server"
        return f"whisper.cpp ({model}, Vulkan GPU)"

    def _base_url(self) -> str:
        return self.url or f"http://127.0.0.1:{self.port}"

    def _server_ready(self, timeout: float = 1.0) -> bool:
        """whisper-server loads the model before it binds, so a successful
        TCP connect means it is ready for inference."""
        from urllib.parse import urlparse
        parsed = urlparse(self._base_url())
        try:
            with socket.create_connection(
                    (parsed.hostname, parsed.port or 80), timeout=timeout):
                return True
        except OSError:
            return False

    def load(self) -> bool:
        if self._server_ready():
            print(f"whisper.cpp server already running at {self._base_url()}")
            return True
        if self.url:
            print(f"ERROR: whisper.cpp server not reachable at {self.url}")
            return False

        if not self.server_bin or not os.path.isfile(self.server_bin):
            print(f"ERROR: whisper-server binary not found: {self.server_bin}")
            return False
        if not self.model_path or not os.path.isfile(self.model_path):
            print(f"ERROR: whisper.cpp model not found: {self.model_path}")
            return False

        cmd = [
            self.server_bin,
            "-m", self.model_path,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--convert",  # accept 44.1kHz WAV chunks (uses ffmpeg)
        ]
        print(f"Starting whisper.cpp server: {' '.join(cmd)}")
        # NOTE: no PR_SET_PDEATHSIG here - Linux delivers it when the
        # spawning THREAD exits, not the process, and callers (GUI worker
        # threads) are short-lived. Cleanup is stop()'s job; live_server
        # additionally kills its ingest server in its shutdown hook.
        try:
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as e:
            print(f"ERROR starting whisper-server: {e}")
            return False

        # Wait for the model to load (large models can take a while)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                print(f"ERROR: whisper-server exited (code {self.process.returncode})")
                return False
            if self._server_ready():
                print("whisper.cpp server ready.")
                return True
            time.sleep(0.5)
        print("ERROR: whisper-server did not become ready in time")
        self.stop()
        return False

    def transcribe(self, chunk_file: str, timeout: float = 120.0) -> List[dict]:
        """POST the chunk to /inference; returns [{start, end, text}].

        `timeout` must cover the whole server-side decode - the default
        suits ~30s live chunks; whole-file callers scale it to duration.
        """
        with open(chunk_file, "rb") as f:
            file_data = f.read()

        boundary = uuid.uuid4().hex
        fields = {
            "response_format": "verbose_json",
            "language": self.language,
            "temperature": "0.0",
            # Anti-hallucination decode params (recent whisper.cpp servers honor
            # these; older builds simply ignore unknown fields).
            "no_speech_thold": "0.6",
            "no_context": "true",  # don't carry prior text across chunks
        }
        parts = []
        for name, value in fields.items():
            parts.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n".encode()
            )
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; '
            f'filename="{Path(chunk_file).name}"\r\n'
            f"Content-Type: audio/wav\r\n\r\n".encode()
            + file_data + b"\r\n"
        )
        parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(parts)

        req = urllib.request.Request(
            f"{self._base_url()}/inference",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))

        return clean_segments(self._parse_segments(payload))

    @staticmethod
    def _parse_segments(payload: dict) -> List[dict]:
        """Normalize whisper.cpp server JSON into [{start, end, text}]."""
        result = []
        for seg in payload.get("segments", []):
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            if "start" in seg and "end" in seg:  # OpenAI-style, seconds
                start, end = float(seg["start"]), float(seg["end"])
            elif "t0" in seg and "t1" in seg:  # whisper.cpp native, centiseconds
                start, end = seg["t0"] / 100.0, seg["t1"] / 100.0
            elif "offsets" in seg:  # {"from": ms, "to": ms}
                start = seg["offsets"].get("from", 0) / 1000.0
                end = seg["offsets"].get("to", 0) / 1000.0
            else:
                start = end = 0.0
            result.append({"start": start, "end": end, "text": text})
        if not result and (payload.get("text") or "").strip():
            # Server returned plain text only - keep it, without timestamps
            result.append({"start": 0.0, "end": 0.0,
                           "text": payload["text"].strip()})
        return result

    def stop(self) -> None:
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            self.process = None


class TranscriptionEngine:
    """Handles real-time transcription of audio chunks using faster-whisper"""

    def __init__(
        self,
        output_file: str,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "default",
        chunk_duration: int = 30,
        on_transcript_update: Optional[Callable[[str], None]] = None,
        engine: str = "faster-whisper",
        engine_options: Optional[dict] = None
    ):
        """
        Initialize transcription engine

        Args:
            output_file: Path to output transcript file
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (cpu, cuda, auto)
            compute_type: Compute type (default, int8, float16, int8_float16)
            chunk_duration: Duration of each audio chunk in seconds
            on_transcript_update: Callback when new transcript is available
            engine: "faster-whisper" (default) or "whisper-cpp"
            engine_options: whisper-cpp settings (server_bin, model_path, port, url)
        """
        self.output_file = output_file
        self.model_size = model_size
        self.chunk_duration = chunk_duration
        self.on_transcript_update = on_transcript_update

        # Determine device
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        # Adjust compute type based on device
        if self.device == "cpu":
            self.compute_type = "int8"
        elif compute_type == "default":
            self.compute_type = "float16"
        else:
            self.compute_type = compute_type

        # Pick the inference backend
        opts = engine_options or {}
        if engine == "whisper-cpp":
            self.backend = WhisperCppBackend(
                server_bin=opts.get("server_bin", ""),
                model_path=opts.get("model_path", ""),
                port=opts.get("port", 8009),
                url=opts.get("url", ""),
            )
        else:
            self.backend = FasterWhisperBackend(
                model_size=self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        print(f"Inference backend: {self.backend.describe()}")
        self.model_loaded = False

        # Processing queue and thread
        self.chunk_queue = queue.Queue()
        self.is_processing = False
        self.process_thread = None

        # Transcript storage
        self.transcripts = []  # List of chunk entries with per-segment timestamps
        self.start_timestamp = None
        self.end_timestamp = None

    def _load_model(self) -> bool:
        """
        Load/start the inference backend

        Returns:
            True if loaded successfully
        """
        if self.model_loaded:
            return True

        print(f"\n{'='*60}")
        print(f"Starting inference backend: {self.backend.describe()}")
        print(f"{'='*60}\n")

        if self.backend.load():
            self.model_loaded = True
            print(f"Backend ready: {self.backend.describe()}")
            return True
        return False

    def start(self) -> bool:
        """
        Start transcription processing

        Returns:
            True if started successfully
        """
        if self.is_processing:
            return False

        # Load model if not already loaded
        if not self._load_model():
            return False

        self.is_processing = True
        self.start_timestamp = datetime.now()
        self.transcripts = []

        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()

        return True

    def stop(self) -> Optional[str]:
        """
        Stop transcription processing and save the final consolidated transcript.

        Returns:
            Path to the final transcript file, or None.
        """
        if not self.is_processing:
            return None

        self.is_processing = False

        # Wait for processing thread to finish
        if self.process_thread:
            self.process_thread.join(timeout=30.0)

        # Shut down the backend (kills a spawned whisper.cpp server)
        self.backend.stop()

        # Save the consolidated LLM-friendly transcript
        return self.save_final_transcript()

    def add_chunk(self, chunk_file: str) -> None:
        """
        Add audio chunk to processing queue

        Args:
            chunk_file: Path to audio chunk file
        """
        self.chunk_queue.put(chunk_file)

    def _process_chunk_item(self, chunk_file: str) -> None:
        """Process a single chunk from the queue."""
        chunk_index = len(self.transcripts)
        chunk_offset = chunk_index * self.chunk_duration

        result = self._transcribe_chunk(chunk_file, chunk_offset)

        if result:
            timestamp = self.start_timestamp + timedelta(seconds=chunk_offset)
            self.transcripts.append({
                "timestamp": timestamp,
                "text": result["text"],
                "segments": result["segments"],
                "chunk_file": chunk_file,
            })

            self._save_transcript()

            if self.on_transcript_update:
                self.on_transcript_update(result["text"])

            print(f"Transcribed chunk {chunk_index}: {result['text'][:100]}...")

    def _process_loop(self) -> None:
        """Main processing loop (runs in separate thread)"""
        while self.is_processing:
            try:
                try:
                    chunk_file = self.chunk_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                self._process_chunk_item(chunk_file)
                self.chunk_queue.task_done()

            except Exception as e:
                print(f"Error in transcription processing loop: {e}")

        # Process any remaining chunks
        while not self.chunk_queue.empty():
            try:
                chunk_file = self.chunk_queue.get_nowait()
                self._process_chunk_item(chunk_file)
                self.chunk_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                print(f"Error processing remaining chunk: {e}")

    def _transcribe_chunk(self, chunk_file: str, chunk_offset: float) -> Optional[dict]:
        """
        Transcribe a single audio chunk, preserving per-segment timestamps.

        Args:
            chunk_file: Path to audio chunk file
            chunk_offset: Time offset (in seconds) of this chunk from recording start

        Returns:
            Dict with 'text' (full text) and 'segments' (list of {start, end, text}) or None
        """
        try:
            raw_segments = self.backend.transcribe(chunk_file)

            segment_list = []
            text_parts = []
            for seg in raw_segments:
                seg_text = seg["text"].strip()
                if seg_text:
                    segment_list.append({
                        "start": chunk_offset + seg["start"],
                        "end": chunk_offset + seg["end"],
                        "text": seg_text,
                    })
                    text_parts.append(seg_text)

            if not text_parts:
                return None

            return {
                "text": " ".join(text_parts),
                "segments": segment_list,
            }

        except Exception as e:
            print(f"Error transcribing chunk {chunk_file}: {e}")
            return None

    @staticmethod
    def _format_seconds(total_seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        h = int(total_seconds // 3600)
        m = int((total_seconds % 3600) // 60)
        s = int(total_seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _save_transcript(self) -> None:
        """Save current transcripts to output file (incremental, with per-segment timestamps)."""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write("=== Teams Call Transcript ===\n\n")

                if self.start_timestamp:
                    f.write(f"Recording started: {self.start_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("-" * 80 + "\n\n")

                for entry in self.transcripts:
                    for seg in entry.get("segments", []):
                        start_str = self._format_seconds(seg["start"])
                        end_str = self._format_seconds(seg["end"])
                        f.write(f"[{start_str} - {end_str}]  {seg['text']}\n")
                    f.write("\n")

                f.write("-" * 80 + "\n")
                f.write(f"\nTranscript generated by {self.backend.describe()}\n")

        except Exception as e:
            print(f"Error saving transcript: {e}")

    def save_final_transcript(self) -> Optional[str]:
        """
        Save a consolidated, LLM-friendly transcript file when recording stops.

        Returns:
            Path to the final transcript file, or None on error.
        """
        self.end_timestamp = datetime.now()

        try:
            # Compute total duration
            if self.start_timestamp and self.end_timestamp:
                total_duration = (self.end_timestamp - self.start_timestamp).total_seconds()
            else:
                total_duration = 0

            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write("# Meeting Transcript\n\n")
                if self.start_timestamp:
                    f.write(f"Date: {self.start_timestamp.strftime('%Y-%m-%d')}\n")
                    f.write(f"Start time: {self.start_timestamp.strftime('%H:%M:%S')}\n")
                if self.end_timestamp:
                    f.write(f"End time: {self.end_timestamp.strftime('%H:%M:%S')}\n")
                f.write(f"Duration: {self._format_seconds(total_duration)}\n")
                f.write(f"Model: {self.backend.describe()}\n")
                f.write(f"Chunks: {len(self.transcripts)}\n")
                f.write("\n---\n\n")

                for entry in self.transcripts:
                    for seg in entry.get("segments", []):
                        start_str = self._format_seconds(seg["start"])
                        end_str = self._format_seconds(seg["end"])
                        f.write(f"[{start_str} - {end_str}]  {seg['text']}\n")

                f.write("\n---\n\nEnd of transcript.\n")

            print(f"Final transcript saved: {self.output_file}")
            return self.output_file

        except Exception as e:
            print(f"Error saving final transcript: {e}")
            return None

    def get_full_transcript(self) -> str:
        """
        Get complete transcript as single string

        Returns:
            Full transcript text
        """
        return "\n".join([entry["text"] for entry in self.transcripts])

    def get_transcript_count(self) -> int:
        """
        Get number of transcribed chunks

        Returns:
            Number of chunks processed
        """
        return len(self.transcripts)
