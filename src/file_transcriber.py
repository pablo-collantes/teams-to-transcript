"""
File Transcriber - whole-file transcription shared by the CLI and the GUI.

Local files use the same pluggable engines as live recording
(`inference_engine` setting):

  whisper-cpp    - whisper.cpp server, Vulkan GPU (AMD/Intel iGPUs).
                   The whole file goes through in one pass on the GPU.
  faster-whisper - CTranslate2, CPU int8 or NVIDIA CUDA. Segments stream
                   in as they are decoded.

The input (m4a, mp3, wav, ...) is converted once with ffmpeg to 16 kHz mono
WAV, and both whisper and pyannote (speaker diarization) read that copy.
"""
import os
import re
import subprocess
import tempfile
import wave
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional


ProgressFn = Callable[[str], None]

# Whole-file decoding is done in independent chunks. Whisper can fall into a
# repetition loop on silence/music/faint audio, and in a single whole-file
# request that loop snowballs to the end of the file (observed: 39 minutes of
# two alternating lines). A fresh request per chunk hard-resets decoder state,
# so a loop cannot outlive its chunk; suspect chunks are re-run in small
# pieces, and a collapse pass cleans whatever survives.
CHUNK_SECONDS = 600
RETRY_PIECE_SECONDS = 60
SUSPECT_UNIQUE_RATIO = 0.6   # below this, a chunk is considered looping
MIN_SEGMENTS_FOR_RATIO = 12  # don't judge tiny chunks
LOOP_SPAN_MIN = 8            # >=8 segments ping-ponging <=2 texts = a loop
IDENTICAL_RUN_KEEP = 2       # "I see. I see." is real; a third repeat is not


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w ]+", " ", text).strip().lower())


def _unique_ratio(segments: List[dict]) -> float:
    texts = [_norm_text(s["text"]) for s in segments if s["text"].strip()]
    if len(texts) < MIN_SEGMENTS_FOR_RATIO:
        return 1.0
    return len(set(texts)) / len(texts)


def _dedupe_runs(segments: List[dict], keep: int = IDENTICAL_RUN_KEEP) -> List[dict]:
    """Collapse runs of consecutive identical lines down to `keep`."""
    out, run_key, run_len = [], None, 0
    for seg in segments:
        key = _norm_text(seg["text"])
        run_len = run_len + 1 if key == run_key else 1
        run_key = key
        if run_len <= keep:
            out.append(seg)
    return out


def _collapse_loops(segments: List[dict]) -> List[dict]:
    """Collapse hallucination loops - long spans alternating between at most
    two distinct texts become one instance of each plus a marker line."""
    out, i, n = [], 0, len(segments)
    while i < n:
        uniq, j = set(), i
        while j < n:
            key = _norm_text(segments[j]["text"])
            if key not in uniq and len(uniq) == 2:
                break
            uniq.add(key)
            j += 1
        if j - i >= LOOP_SPAN_MIN:
            seen = set()
            for seg in segments[i:j]:
                key = _norm_text(seg["text"])
                if key not in seen:
                    seen.add(key)
                    out.append(seg)
            out.append({"start": segments[i]["start"],
                        "end": segments[j - 1]["end"],
                        "text": "[repetitive output collapsed - likely "
                                "silence, music or faint audio]"})
            i = j
        else:
            out.append(segments[i])
            i += 1
    return out


def _clean_chunk(segments: List[dict]) -> List[dict]:
    return _collapse_loops(_dedupe_runs(segments))


def format_seconds(total: float) -> str:
    h = int(total // 3600)
    m = int((total % 3600) // 60)
    s = int(total % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def segment_line(seg: dict) -> str:
    speaker = seg.get("speaker", "")
    prefix = f"{speaker}: " if speaker else ""
    return (f"[{format_seconds(seg['start'])} - {format_seconds(seg['end'])}] "
            f"{prefix}{seg['text']}")


def write_transcript(output_path: str, source_path: str, duration: float,
                     engine_desc: str, segments: List[dict],
                     num_speakers: int = 0) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Audio Transcript\n\n")
        f.write(f"Source: {os.path.basename(source_path)}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {format_seconds(duration)}\n")
        f.write(f"Model: {engine_desc}\n")
        if num_speakers:
            f.write(f"Speakers: {num_speakers}\n")
        f.write("\n---\n\n")
        for seg in segments:
            f.write(segment_line(seg) + "\n")
        f.write("\n---\n\nEnd of transcript.\n")


class FileTranscriber:
    """Whole-file transcription with the engine picked in settings.

    Create once, transcribe several files (the model/server stays warm),
    then close().
    """

    def __init__(self, settings, model_size: str = "", language: str = "",
                 progress: Optional[ProgressFn] = None):
        self.settings = settings
        self.model_size = model_size or settings.get("web_model", "large-v3")
        self.language = language  # "" = auto-detect
        self.progress = progress or print
        self.backend = None
        self.engine = self._pick_engine()

    def _pick_engine(self) -> str:
        if self.settings.get("inference_engine") == "whisper-cpp":
            if self.settings.get("whisper_cpp_url", ""):
                return "whisper-cpp"
            server = self.settings.get("whisper_cpp_server", "")
            model = self.settings.get("whisper_cpp_model", "")
            if os.path.isfile(server) and os.path.isfile(model):
                return "whisper-cpp"
            self.progress("whisper-cpp is not configured; using faster-whisper")
        return "faster-whisper"

    def describe(self) -> str:
        if self.backend is not None:
            return self.backend.describe()
        if self.engine == "whisper-cpp":
            return "whisper.cpp (Vulkan GPU)"
        return f"faster-whisper ({self.model_size})"

    def load(self) -> None:
        """Start the backend (spawns whisper-server / loads the model)."""
        if self.backend is not None:
            return
        if self.engine == "whisper-cpp":
            from transcription_engine import WhisperCppBackend
            backend = WhisperCppBackend(
                server_bin=self.settings.get("whisper_cpp_server", ""),
                model_path=self.settings.get("whisper_cpp_model", ""),
                port=self.settings.get("whisper_cpp_file_port", 8012),
                url=self.settings.get("whisper_cpp_url", ""),
                language=self.language or "auto",
            )
            self.progress(f"Starting {backend.describe()}...")
            if not backend.load():
                raise RuntimeError(
                    "whisper.cpp server failed to start (check "
                    "whisper_cpp_server / whisper_cpp_model in settings)")
        else:
            from transcription_engine import FasterWhisperBackend
            device = "cpu"
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
            except ImportError:
                pass
            compute_type = "float16" if device == "cuda" else "int8"
            backend = FasterWhisperBackend(
                model_size=self.model_size, device=device,
                compute_type=compute_type, language=self.language or None,
                vad_filter=False)
            self.progress(f"Loading {backend.describe()}...")
            if not backend.load():
                raise RuntimeError(
                    f"could not load whisper model {self.model_size}")
        self.backend = backend
        self.progress(f"Backend ready: {backend.describe()}")

    def close(self) -> None:
        if self.backend is not None:
            self.backend.stop()
            self.backend = None

    def prepare_wav(self, audio_path: str):
        """Return (wav_path, duration_seconds, is_temp).

        Converts to 16 kHz mono s16 WAV - the format whisper and pyannote
        both consume directly.
        """
        fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="transcribe_")
        os.close(fd)
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", audio_path,
               "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            os.unlink(wav_path)
            if audio_path.lower().endswith(".wav"):
                with wave.open(audio_path, "rb") as w:
                    duration = w.getnframes() / w.getframerate()
                return audio_path, duration, False
            raise RuntimeError("ffmpeg is required to decode non-WAV audio")
        if proc.returncode != 0:
            os.unlink(wav_path)
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[:300]}")
        with wave.open(wav_path, "rb") as w:
            duration = w.getnframes() / w.getframerate()
        return wav_path, duration, True

    def transcribe_wav(self, wav_path: str, duration: float,
                       on_segment: Optional[ProgressFn] = None,
                       is_cancelled: Optional[Callable[[], bool]] = None,
                       ) -> List[dict]:
        """Transcribe a prepared WAV; returns [{start, end, text}]."""
        self.load()
        if self.engine == "whisper-cpp":
            segments = self._transcribe_whisper_cpp(
                wav_path, duration, on_segment, is_cancelled)
        else:
            segments = self._transcribe_faster_whisper(
                wav_path, on_segment, is_cancelled)
        # Final cross-chunk guard (idempotent on clean output)
        return _clean_chunk(segments)

    @staticmethod
    def _cut_wav(wav_path: str, start: float, length: float) -> str:
        """Write [start, start+length) of a WAV to a temp file; return path."""
        with wave.open(wav_path, "rb") as src:
            rate = src.getframerate()
            src.setpos(min(int(start * rate), src.getnframes()))
            frames = src.readframes(int(length * rate))
            params = src.getparams()
        fd, piece_path = tempfile.mkstemp(suffix=".wav", prefix="chunk_")
        os.close(fd)
        with wave.open(piece_path, "wb") as dst:
            dst.setparams(params)
            dst.writeframes(frames)
        return piece_path

    def _transcribe_piece(self, wav_path, start, length):
        piece = self._cut_wav(wav_path, start, length)
        try:
            return self.backend.transcribe(
                piece, timeout=max(120.0, length * 1.5))
        finally:
            try:
                os.unlink(piece)
            except OSError:
                pass

    def _transcribe_whisper_cpp(self, wav_path, duration, on_segment,
                                is_cancelled):
        # Independent request per chunk - see the note on CHUNK_SECONDS.
        n_chunks = max(1, -(-int(duration) // CHUNK_SECONDS))
        est = max(30, int(duration / 12))
        self.progress(f"Transcribing {format_seconds(duration)} of audio on "
                      f"the GPU in {n_chunks} chunk(s) "
                      f"(~{est // 60}m{est % 60:02d}s expected)...")
        all_segments = []
        for i in range(n_chunks):
            if is_cancelled and is_cancelled():
                break
            start = i * CHUNK_SECONDS
            length = min(CHUNK_SECONDS, duration - start)
            segments = self._transcribe_piece(wav_path, start, length)

            ratio = _unique_ratio(segments)
            if ratio < SUSPECT_UNIQUE_RATIO:
                self.progress(
                    f"  chunk {i + 1}/{n_chunks}: repetitive output "
                    f"(unique ratio {ratio:.2f}) - re-running in "
                    f"{RETRY_PIECE_SECONDS}s pieces...")
                segments = self._retry_chunk(wav_path, start, length,
                                             is_cancelled)

            segments = _clean_chunk(segments)
            for seg in segments:
                seg["start"] += start
                seg["end"] += start
                if on_segment:
                    on_segment(segment_line(seg))
            all_segments.extend(segments)
            self.progress(f"  chunk {i + 1}/{n_chunks} "
                          f"[{format_seconds(start)} - "
                          f"{format_seconds(start + length)}]: "
                          f"{len(segments)} segments")
        return all_segments

    def _retry_chunk(self, wav_path, chunk_start, chunk_length, is_cancelled):
        """Re-run a suspect chunk in small pieces so a repetition loop cannot
        span more than one piece; returns chunk-relative segments."""
        segments = []
        offset = 0.0
        while offset < chunk_length:
            if is_cancelled and is_cancelled():
                break
            length = min(RETRY_PIECE_SECONDS, chunk_length - offset)
            piece_segments = self._transcribe_piece(
                wav_path, chunk_start + offset, length)
            for seg in _clean_chunk(piece_segments):
                segments.append({"start": seg["start"] + offset,
                                 "end": seg["end"] + offset,
                                 "text": seg["text"]})
            offset += length
        return segments

    def _transcribe_faster_whisper(self, wav_path, on_segment, is_cancelled):
        from transcription_engine import clean_segments
        kwargs = dict(beam_size=5, condition_on_previous_text=False)
        if self.language:
            kwargs["language"] = self.language
        self.progress("Transcribing...")
        segments_gen, info = self.backend.model.transcribe(wav_path, **kwargs)
        result = []
        for seg in segments_gen:
            if is_cancelled and is_cancelled():
                break
            text = seg.text.strip()
            if not text:
                continue
            entry = {"start": seg.start, "end": seg.end, "text": text}
            result.append(entry)
            if on_segment:
                on_segment(segment_line(entry))
        return clean_segments(result)


def run_pipeline(audio_path: str, settings, *, model_size: str = "",
                 language: str = "", diarize: bool = False,
                 num_speakers: int = 0, output_path: str = "",
                 transcriber: Optional[FileTranscriber] = None,
                 diarization_engine=None,
                 progress: Optional[ProgressFn] = None,
                 on_segment: Optional[ProgressFn] = None,
                 is_cancelled: Optional[Callable[[], bool]] = None) -> dict:
    """Full local-file job: convert -> transcribe -> (diarize) -> write.

    Pass a preloaded `transcriber` / `diarization_engine` to keep them warm
    across a batch of files; otherwise they are created and torn down here.
    A diarization failure never discards the transcription - the transcript
    is written without speakers and the error is reported in the result.
    """
    progress = progress or print
    audio_path = os.path.abspath(audio_path)
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(audio_path)

    if not output_path:
        out_dir = Path(settings.get("output_directory")) / "local_transcripts"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(out_dir / f"{Path(audio_path).stem}_{stamp}.txt")

    owns_transcriber = transcriber is None
    if owns_transcriber:
        transcriber = FileTranscriber(settings, model_size=model_size,
                                      language=language, progress=progress)

    wav_path, duration, is_temp = transcriber.prepare_wav(audio_path)
    diarization_error = ""
    speakers_found = 0
    try:
        segments = transcriber.transcribe_wav(
            wav_path, duration, on_segment=on_segment,
            is_cancelled=is_cancelled)
        engine_desc = transcriber.describe()
        if is_cancelled and is_cancelled():
            progress("Cancelled.")
            return {"cancelled": True, "output_path": "", "segments": segments,
                    "duration": duration, "num_speakers": 0,
                    "engine": engine_desc, "diarization_error": ""}
        progress(f"Transcription done: {len(segments)} segments, "
                 f"duration {format_seconds(duration)}")

        final_segments = segments
        if diarize and segments:
            try:
                from diarization_engine import DiarizationEngine
                if diarization_engine is None:
                    diarization_engine = DiarizationEngine(
                        hf_token=settings.get("hf_token", ""))
                dev = "GPU" if diarization_engine.device == "cuda" else \
                    "CPU - this takes a while on long files"
                progress(f"Running speaker diarization ({dev})...")
                turns = diarization_engine.diarize(
                    wav_path,
                    num_speakers=num_speakers if num_speakers > 0 else None,
                    progress_callback=lambda m: progress(f"  {m}"))
                labeled = DiarizationEngine.align_with_transcript(
                    segments, turns)
                final_segments = DiarizationEngine.merge_consecutive(labeled)
                speakers_found = len(
                    {s.get("speaker") for s in final_segments})
                progress(f"Diarization done: {speakers_found} speakers")
            except Exception as exc:
                diarization_error = str(exc)
                progress("Diarization FAILED - writing transcript without "
                         f"speakers.\n{diarization_error}")
    finally:
        if is_temp:
            try:
                os.unlink(wav_path)
            except OSError:
                pass
        if owns_transcriber:
            transcriber.close()

    write_transcript(output_path, audio_path, duration, engine_desc,
                     final_segments, speakers_found)
    progress(f"Saved: {output_path}")
    return {"cancelled": False, "output_path": output_path,
            "segments": final_segments, "duration": duration,
            "num_speakers": speakers_found, "engine": engine_desc,
            "diarization_error": diarization_error}
