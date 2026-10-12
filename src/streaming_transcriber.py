"""
Streaming transcriber - near-real-time transcription via a growing-window
LocalAgreement scheme (opt-in; the classic non-overlapping chunk path is
untouched and remains the default).

Idea (same family as whisper_streaming / LocalAgreement-2):
  * Capture audio continuously into a per-utterance buffer.
  * Every `hop_seconds`, re-transcribe the WHOLE growing buffer (so Whisper always
    sees full context - no bare 5s clips that hallucinate).
  * Commit only the words that AGREE between the last two hypotheses (the stable
    prefix). The still-unstable tail is held back until the next hop confirms it.
  * On a speech pause (VAD) or a max length, flush the utterance and reset.

This gives low-latency, self-correcting transcription: words appear a couple of
seconds after they are spoken, and a word is only written once two consecutive
re-transcriptions agree on it. Output is written in the exact
`[HH:MM:SS - HH:MM:SS]  text` format the live server + rest of the app expect, so
nothing downstream changes.

Audio source is either live PulseAudio/PipeWire (`parec`) or, for testing, a WAV
file replayed in real time (`file_source=`). It reuses the whisper backends from
transcription_engine (faster-whisper or whisper.cpp/Vulkan).
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

_STRIP = re.compile(r"[^\w']+", re.UNICODE)


def _norm(word: str) -> str:
    """Normalize a word for agreement comparison (case/punctuation-insensitive)."""
    return _STRIP.sub("", word.lower())


def _common_prefix(a: List[str], b: List[str]) -> int:
    """Length of the longest common prefix of two normalized word lists."""
    n = 0
    for x, y in zip(a, b):
        if x and x == y:
            n += 1
        else:
            break
    return n


def _rms(pcm: bytes) -> float:
    """RMS energy of signed-16-bit PCM (all channels lumped together)."""
    if not pcm:
        return 0.0
    if np is not None:
        a = np.frombuffer(pcm, dtype=np.int16)
        if a.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(a.astype(np.float32) ** 2)))
    # Pure-Python fallback (audioop was removed in Python 3.13).
    import array
    a = array.array("h")
    a.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not a:
        return 0.0
    return (sum(x * x for x in a) / len(a)) ** 0.5


def _hhmmss(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class StreamingTranscriber:
    def __init__(
        self,
        output_file: str,
        backend,
        *,
        on_transcript_update: Optional[Callable[[str], None]] = None,
        audio_source: str = "",
        mic_source: str = "",
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
        # streaming behavior
        hop_seconds: float = 2.0,
        min_speech_seconds: float = 1.0,
        max_utterance_seconds: float = 24.0,
        end_silence_seconds: float = 0.8,
        silence_rms: float = 300.0,
        preroll_seconds: float = 0.3,
        # reliability
        commit_holdback_words: int = 1,   # tail-keep: hold back the last agreed word(s)
        stall_backoff: bool = True,       # space hops out when decoding is slower than realtime
        max_consec_fail: int = 3,         # warn after this many failed hops in a row
        hop_watchdog_secs: float = 20.0,  # warn if a single hop runs longer than this
        on_status: Optional[Callable[[str], None]] = None,
        # testing: replay a WAV file instead of capturing live
        file_source: str = "",
        realtime_file: bool = True,
        session_wav: str = "",
    ):
        self.output_file = output_file
        self.backend = backend
        self.on_transcript_update = on_transcript_update
        self.audio_source = audio_source
        self.mic_source = mic_source
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.bytes_per_sec = sample_rate * channels * sample_width

        self.hop_seconds = hop_seconds
        self.min_speech_seconds = min_speech_seconds
        self.max_utterance_seconds = max_utterance_seconds
        self.end_silence_seconds = end_silence_seconds
        self.silence_rms = silence_rms
        self.preroll_bytes = int(preroll_seconds * self.bytes_per_sec)

        # reliability
        self.commit_holdback_words = max(0, int(commit_holdback_words))
        self.stall_backoff = stall_backoff
        self.max_consec_fail = max(1, int(max_consec_fail))
        self.hop_watchdog_secs = float(hop_watchdog_secs)
        self.on_status = on_status
        self._effective_hop = hop_seconds       # grows when decoding lags realtime
        self._last_hop_secs = 0.0
        self._consec_fail = 0
        self._hop_inflight_since = 0.0           # monotonic ts of an in-flight hop (0 = idle)
        self._watchdog: Optional[threading.Thread] = None

        self.file_source = file_source
        self.realtime_file = realtime_file
        self.session_wav = session_wav

        self._tmp_wav = str(Path(output_file).with_suffix(".partial.wav"))
        self._model_loaded = False

        # capture buffer (filled by reader, drained by processor)
        self._buf = bytearray()
        self._buf_lock = threading.Lock()

        # per-utterance state
        self._utt = bytearray()
        self._has_speech = False
        self._silence_run = 0.0
        self._since_hop = 0.0
        self._committed: List[str] = []       # committed words (original case)
        self._committed_norm: List[str] = []
        self._prev_norm: List[str] = []

        self._start_time: Optional[float] = None
        self._last_commit_elapsed = 0.0
        self._header_written = False
        self._file_lock = threading.Lock()

        self._running = False
        self._reader: Optional[threading.Thread] = None
        self._processor: Optional[threading.Thread] = None
        self._parec: Optional[subprocess.Popen] = None
        self._mic_parec: Optional[subprocess.Popen] = None
        self._mic_buf = bytearray()
        self._mic_lock = threading.Lock()
        self._wav_writer: Optional[wave.Wave_write] = None

    # -- lifecycle -----------------------------------------------------
    def _status(self, msg: str):
        print(f"StreamingTranscriber: {msg}")
        if self.on_status:
            try:
                self.on_status(msg)
            except Exception:
                pass

    def start(self) -> bool:
        if self._running:
            return False
        if not self.backend.load():
            print("StreamingTranscriber: backend failed to load")
            return False
        self._model_loaded = True
        self._start_time = time.time()
        if self.session_wav:
            try:
                self._wav_writer = wave.open(self.session_wav, "wb")
                self._wav_writer.setnchannels(self.channels)
                self._wav_writer.setsampwidth(self.sample_width)
                self._wav_writer.setframerate(self.sample_rate)
            except Exception as e:
                print(f"StreamingTranscriber: could not open session wav: {e}")
                self._wav_writer = None

        self._running = True
        target = self._read_file if self.file_source else self._read_live
        self._reader = threading.Thread(target=target, daemon=True)
        self._processor = threading.Thread(target=self._process_loop, daemon=True)
        self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._reader.start()
        self._processor.start()
        self._watchdog.start()
        return True

    def stop(self) -> Optional[str]:
        if not self._running:
            return self.output_file
        self._running = False
        for p in (self._parec, self._mic_parec):
            if p:
                try:
                    p.terminate()
                except Exception:
                    pass
        if self._reader:
            self._reader.join(timeout=5)
        if self._processor:
            self._processor.join(timeout=max(10.0, self.max_utterance_seconds))
        # flush any trailing utterance
        self._finalize_utterance()
        self.backend.stop()
        if self._wav_writer:
            try:
                self._wav_writer.close()
            except Exception:
                pass
        try:
            Path(self._tmp_wav).unlink(missing_ok=True)
        except Exception:
            pass
        return self.output_file

    # -- audio readers -------------------------------------------------
    def _start_parec(self, source: str) -> subprocess.Popen:
        cmd = ["parec", "--format=s16le", f"--rate={self.sample_rate}",
               f"--channels={self.channels}", f"--device={source}"]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _mic_reader(self):
        try:
            while self._running and self._mic_parec and self._mic_parec.stdout:
                data = self._mic_parec.stdout.read(4096)
                if not data:
                    break
                with self._mic_lock:
                    self._mic_buf.extend(data)
        except Exception as e:
            print(f"StreamingTranscriber mic reader error: {e}")

    def _take_mic(self, n: int) -> bytes:
        with self._mic_lock:
            if len(self._mic_buf) >= n:
                data = bytes(self._mic_buf[:n])
                del self._mic_buf[:n]
                return data
            data = bytes(self._mic_buf) + b"\x00" * (n - len(self._mic_buf))
            self._mic_buf.clear()
            return data

    @staticmethod
    def _mix(sys_data: bytes, mic_data: bytes) -> bytes:
        if np is not None:
            s = np.frombuffer(sys_data, dtype=np.int16).astype(np.int32)
            m = np.frombuffer(mic_data, dtype=np.int16).astype(np.int32)
            return np.clip(s + m, -32768, 32767).astype(np.int16).tobytes()
        import array
        s = array.array("h"); s.frombytes(sys_data[: len(sys_data) - (len(sys_data) % 2)])
        m = array.array("h"); m.frombytes(mic_data[: len(mic_data) - (len(mic_data) % 2)])
        n = min(len(s), len(m))
        out = array.array("h", s[:n])
        for i in range(n):
            v = s[i] + m[i]
            out[i] = 32767 if v > 32767 else -32768 if v < -32768 else v
        return out.tobytes()

    def _read_live(self):
        source = self.audio_source
        if not source:
            print("StreamingTranscriber: no audio source")
            self._running = False
            return
        try:
            self._parec = self._start_parec(source)
            if self.mic_source:
                self._mic_parec = self._start_parec(self.mic_source)
                threading.Thread(target=self._mic_reader, daemon=True).start()
            while self._running and self._parec.stdout:
                data = self._parec.stdout.read(4096)
                if not data:
                    break
                if self.mic_source:
                    data = self._mix(data, self._take_mic(len(data)))
                self._ingest(data)
        except Exception as e:
            print(f"StreamingTranscriber live reader error: {e}")
        finally:
            self._running = False

    def _read_file(self):
        """Replay a WAV file as if it were live (for testing)."""
        try:
            with wave.open(self.file_source, "rb") as wf:
                self.sample_rate = wf.getframerate()
                self.channels = wf.getnchannels()
                self.sample_width = wf.getsampwidth()
                self.bytes_per_sec = self.sample_rate * self.channels * self.sample_width
                block_frames = int(self.sample_rate * 0.1)  # 100 ms
                while self._running:
                    frames = wf.readframes(block_frames)
                    if not frames:
                        break
                    self._ingest(frames)
                    if self.realtime_file:
                        time.sleep(0.1)
        except Exception as e:
            print(f"StreamingTranscriber file reader error: {e}")
        finally:
            # allow the processor to drain + finalize, then stop
            time.sleep(0.2)
            self._running = False

    def _ingest(self, data: bytes):
        if self._wav_writer:
            try:
                self._wav_writer.writeframes(data)
            except Exception:
                pass
        with self._buf_lock:
            self._buf.extend(data)

    def _drain(self) -> bytes:
        with self._buf_lock:
            if not self._buf:
                return b""
            data = bytes(self._buf)
            self._buf.clear()
            return data

    # -- processing ----------------------------------------------------
    def _process_loop(self):
        while self._running or self._buf:
            block = self._drain()
            if block:
                self._feed(block)
            else:
                time.sleep(0.03)
                continue

            utt_dur = len(self._utt) / self.bytes_per_sec
            if self._has_speech and self._silence_run >= self.end_silence_seconds:
                self._finalize_utterance()
            elif utt_dur >= self.max_utterance_seconds:
                self._finalize_utterance()
            elif (self._has_speech and self._since_hop >= self._effective_hop
                    and utt_dur >= self.min_speech_seconds):
                self._hop()
                self._since_hop = 0.0

        # Processing ended (reader stopped + buffer drained). Flush any trailing
        # utterance here so it's captured even if the reader set _running=False
        # before stop() runs (e.g. file replay). Idempotent: a second finalize in
        # stop() is a no-op once _has_speech is cleared.
        self._finalize_utterance()

    def _feed(self, block: bytes):
        dur = len(block) / self.bytes_per_sec
        energy = _rms(block)
        self._utt.extend(block)
        self._since_hop += dur
        if energy >= self.silence_rms:
            if not self._has_speech:
                self._has_speech = True
                self._utt_start_elapsed = self._elapsed()
            self._silence_run = 0.0
        else:
            self._silence_run += dur
            # Trim leading silence to a small pre-roll until speech starts.
            if not self._has_speech and len(self._utt) > self.preroll_bytes:
                self._utt = bytearray(self._utt[-self.preroll_bytes:])

    def _hop(self):
        t0 = time.monotonic()
        self._hop_inflight_since = t0
        try:
            words = self._transcribe(bytes(self._utt))
        finally:
            self._hop_inflight_since = 0.0
        dt = time.monotonic() - t0
        self._last_hop_secs = dt

        if words is None:
            self._consec_fail += 1
            if self._consec_fail == self.max_consec_fail:
                self._status(f"transcription backend stalling "
                             f"({self._consec_fail} failed hops in a row)")
            return
        if self._consec_fail >= self.max_consec_fail:
            self._status("transcription backend recovered")
        self._consec_fail = 0

        # Stall-backoff: if a hop takes longer than the audio cadence, decoding is
        # slower than realtime - space hops out so we don't thrash on an ever-growing
        # buffer (max_utterance_seconds still bounds the worst case).
        if self.stall_backoff:
            self._effective_hop = max(
                self.hop_seconds,
                min(dt * 1.2, self.max_utterance_seconds * 0.5))

        norm = [_norm(w) for w in words]
        k = _common_prefix(self._prev_norm, norm)
        # Tail-keep / word-boundary snap: hold back the last agreed word(s). The tail
        # of even the agreed region can still shift as more audio arrives, so committing
        # it early is what produces visible "corrections". It flushes on finalize.
        commit_k = max(0, k - self.commit_holdback_words)
        stable_words = words[:commit_k]
        stable_norm = norm[:commit_k]
        if len(stable_norm) > len(self._committed_norm):
            new = stable_words[len(self._committed_norm):]
            self._emit(new)
            self._committed = stable_words
            self._committed_norm = stable_norm
        self._prev_norm = norm

    def _finalize_utterance(self):
        if self._has_speech and len(self._utt) > int(0.2 * self.bytes_per_sec):
            words = self._transcribe(bytes(self._utt))
            if words and len(words) > len(self._committed):
                self._emit(words[len(self._committed):])
        # reset per-utterance state
        self._utt = bytearray()
        self._has_speech = False
        self._silence_run = 0.0
        self._since_hop = 0.0
        self._effective_hop = self.hop_seconds   # fresh utterance -> reset backoff
        self._committed = []
        self._committed_norm = []
        self._prev_norm = []

    def _watchdog_loop(self):
        """Warn (once per stall) if a single transcription hop runs too long -
        the GPU/backend is likely overloaded. Can't safely cancel a blocking
        decode, but it surfaces the stall instead of leaving the UI silent."""
        warned = False
        while self._running:
            time.sleep(1.0)
            since = self._hop_inflight_since
            if since and (time.monotonic() - since) > self.hop_watchdog_secs:
                if not warned:
                    self._status(f"a transcription hop has run >{self.hop_watchdog_secs:.0f}s; "
                                 "the GPU/backend may be overloaded")
                    warned = True
            else:
                warned = False

    def _transcribe(self, pcm: bytes) -> Optional[List[str]]:
        if not pcm:
            return None
        try:
            with wave.open(self._tmp_wav, "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.sample_width)
                wf.setframerate(self.sample_rate)
                wf.writeframes(pcm)
            segments = self.backend.transcribe(self._tmp_wav)
        except Exception as e:
            print(f"StreamingTranscriber transcribe error: {e}")
            return None
        text = " ".join((s.get("text") or "").strip() for s in segments).strip()
        return text.split() if text else []

    def _elapsed(self) -> float:
        return time.time() - self._start_time if self._start_time else 0.0

    def _emit(self, words: List[str]):
        if not words:
            return
        text = " ".join(words).strip()
        if not text:
            return
        start = self._last_commit_elapsed
        end = self._elapsed()
        if end < start:
            end = start
        self._last_commit_elapsed = end
        line = f"[{_hhmmss(start)} - {_hhmmss(end)}]  {text}\n"
        with self._file_lock:
            try:
                if not self._header_written:
                    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(self.output_file, "w", encoding="utf-8") as f:
                        f.write("=== Teams Call Transcript (streaming) ===\n\n")
                        f.write(f"Recording started: {started}\n\n")
                        f.write("-" * 80 + "\n\n")
                    self._header_written = True
                with open(self.output_file, "a", encoding="utf-8") as f:
                    f.write(line)
            except Exception as e:
                print(f"StreamingTranscriber write error: {e}")
        if self.on_transcript_update:
            try:
                self.on_transcript_update(text)
            except Exception:
                pass
