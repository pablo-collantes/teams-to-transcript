"""
Audio Recorder - captures system audio (+ optional mic) into WAV chunks.

Backends:
  PulseAudioRecorder     - Linux, via parec/pactl (PipeWire or PulseAudio)
  SoundcardAudioRecorder - Windows (WASAPI loopback) / macOS, via the
                           `soundcard` library (audio_recorder_soundcard.py)

`AudioRecorder` is aliased to the right backend for the current platform at
the bottom of this module; callers only ever import AudioRecorder.
"""
import subprocess
import sys
import wave
import threading
import time
from pathlib import Path
from typing import Optional, Callable, List, Tuple

import numpy as np


class BaseAudioRecorder:
    """Shared chunking, mic mixing and WAV writing; backends implement capture."""

    def __init__(
        self,
        output_file: str,
        chunk_dir: Path,
        sample_rate: int = 44100,
        channels: int = 2,
        chunk_duration: int = 30,
        audio_source: str = "",
        mic_source: str = "",
        on_chunk_complete: Optional[Callable[[str], None]] = None
    ):
        self.output_file = output_file
        self.chunk_dir = chunk_dir
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.audio_source = audio_source
        self.mic_source = mic_source
        self.on_chunk_complete = on_chunk_complete

        self.is_recording = False
        self.record_thread = None

        # Mic recording state (mic thread fills the buffer, capture loop drains it)
        self._mic_buffer = bytearray()
        self._mic_lock = threading.Lock()
        self._mic_thread: Optional[threading.Thread] = None

        # s16le = 2 bytes per sample
        self.sample_width = 2

        # Chunk tracking
        self.chunk_counter = 0
        self.start_time = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start recording audio. Returns True if started successfully."""
        if self.is_recording:
            return False

        self.is_recording = True
        self.chunk_counter = 0
        self.start_time = time.time()

        self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.record_thread.start()

        return True

    def stop(self) -> None:
        """Stop recording"""
        if not self.is_recording:
            return

        self.is_recording = False
        self._stop_capture()

        if self._mic_thread:
            self._mic_thread.join(timeout=5.0)

        if self.record_thread:
            self.record_thread.join(timeout=5.0)

    def _record_loop(self) -> None:
        """Backend capture loop (runs in separate thread)."""
        raise NotImplementedError

    def _stop_capture(self) -> None:
        """Backend hook: unblock/terminate capture streams on stop."""

    def cleanup(self) -> None:
        """Backend hook: release capture resources."""

    # ------------------------------------------------------------------
    # Shared mic buffer / mixing / WAV helpers
    # ------------------------------------------------------------------

    def _get_mic_data(self, num_bytes: int) -> bytes:
        """Take num_bytes from the mic buffer, padding with silence if needed."""
        with self._mic_lock:
            available = len(self._mic_buffer)
            if available >= num_bytes:
                data = bytes(self._mic_buffer[:num_bytes])
                del self._mic_buffer[:num_bytes]
                return data
            else:
                data = bytes(self._mic_buffer) + b'\x00' * (num_bytes - available)
                self._mic_buffer.clear()
                return data

    def _mix_audio(self, sys_data: bytes, mic_data: bytes) -> bytes:
        """Mix system audio and mic audio by summing with clipping."""
        sys_samples = np.frombuffer(sys_data, dtype=np.int16).astype(np.int32)
        mic_samples = np.frombuffer(mic_data, dtype=np.int16).astype(np.int32)
        mixed = np.clip(sys_samples + mic_samples, -32768, 32767).astype(np.int16)
        return mixed.tobytes()

    def _save_chunk(self, raw_pcm_data: bytes) -> None:
        """Save raw PCM data as a WAV chunk file and notify callback"""
        if not raw_pcm_data:
            return

        try:
            chunk_file = self.chunk_dir / f"chunk_{self.chunk_counter:04d}.wav"

            with wave.open(str(chunk_file), 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.sample_width)
                wf.setframerate(self.sample_rate)
                wf.writeframes(raw_pcm_data)

            print(f"Saved audio chunk: {chunk_file}")

            if self.on_chunk_complete:
                self.on_chunk_complete(str(chunk_file))

            self.chunk_counter += 1

        except Exception as e:
            print(f"Error saving audio chunk: {e}")

    def save_final_audio(self, all_chunks: list) -> None:
        """Combine all chunks into final audio file"""
        try:
            with wave.open(self.output_file, 'wb') as outfile:
                outfile.setnchannels(self.channels)
                outfile.setsampwidth(self.sample_width)
                outfile.setframerate(self.sample_rate)

                for chunk_path in sorted(all_chunks):
                    try:
                        with wave.open(chunk_path, 'rb') as infile:
                            outfile.writeframes(infile.readframes(infile.getnframes()))
                    except Exception as e:
                        print(f"Error reading chunk {chunk_path}: {e}")

            print(f"Saved final audio: {self.output_file}")

        except Exception as e:
            print(f"Error saving final audio: {e}")

    def get_elapsed_time(self) -> float:
        """Get elapsed recording time in seconds"""
        if self.start_time and self.is_recording:
            return time.time() - self.start_time
        return 0.0

    # ------------------------------------------------------------------
    # Device listing (backend statics)
    # ------------------------------------------------------------------

    @staticmethod
    def list_audio_sources() -> List[Tuple[str, str]]:
        return []

    @staticmethod
    def list_mic_sources() -> List[Tuple[str, str]]:
        return []

    @staticmethod
    def get_default_monitor_source() -> str:
        return ""


class PulseAudioRecorder(BaseAudioRecorder):
    """Records system audio using parec (PipeWire/PulseAudio, Linux)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parec_process: Optional[subprocess.Popen] = None
        self.mic_process: Optional[subprocess.Popen] = None

    def _stop_capture(self) -> None:
        # Terminate parec processes so blocked stdout reads return
        for proc in (self.parec_process, self.mic_process):
            if proc:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()

    def _start_parec(self, source: str) -> subprocess.Popen:
        """Start a parec process for the given source."""
        cmd = [
            "parec",
            "--format=s16le",
            f"--rate={self.sample_rate}",
            f"--channels={self.channels}",
            f"--device={source}",
        ]
        print(f"Starting parec: {' '.join(cmd)}")
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _mic_read_loop(self) -> None:
        """Read mic audio into a shared buffer (runs in separate thread)."""
        try:
            while self.is_recording and self.mic_process:
                data = self.mic_process.stdout.read(4096)
                if not data:
                    break
                with self._mic_lock:
                    self._mic_buffer.extend(data)
        except Exception as e:
            print(f"Error in mic recording: {e}")

    def _record_loop(self) -> None:
        """Main recording loop (runs in separate thread)"""
        try:
            source = self.audio_source or self.get_default_monitor_source()
            if not source:
                print("Error: No audio source available")
                self.is_recording = False
                return

            self.parec_process = self._start_parec(source)

            # Start mic capture if configured
            if self.mic_source:
                print(f"Mic recording enabled: {self.mic_source}")
                self.mic_process = self._start_parec(self.mic_source)
                self._mic_buffer = bytearray()
                self._mic_thread = threading.Thread(target=self._mic_read_loop, daemon=True)
                self._mic_thread.start()

            bytes_per_chunk = (
                self.sample_rate
                * self.channels
                * self.sample_width
                * self.chunk_duration
            )

            buffer = b""

            while self.is_recording:
                data = self.parec_process.stdout.read(4096)
                if not data:
                    break

                buffer += data

                if len(buffer) >= bytes_per_chunk:
                    chunk_data = buffer[:bytes_per_chunk]
                    buffer = buffer[bytes_per_chunk:]

                    if self.mic_source and self.mic_process:
                        mic_data = self._get_mic_data(len(chunk_data))
                        chunk_data = self._mix_audio(chunk_data, mic_data)

                    self._save_chunk(chunk_data)

            # Save any remaining audio as a final chunk
            if buffer:
                if self.mic_source and self.mic_process:
                    mic_data = self._get_mic_data(len(buffer))
                    buffer = self._mix_audio(buffer, mic_data)
                self._save_chunk(buffer)

        except Exception as e:
            print(f"Error in audio recording: {e}")
            self.is_recording = False

    def cleanup(self) -> None:
        """Clean up resources"""
        for proc in (self.parec_process, self.mic_process):
            if proc:
                try:
                    proc.kill()
                except Exception:
                    pass

    @staticmethod
    def list_audio_sources() -> List[Tuple[str, str]]:
        """
        List available PulseAudio sources.

        Returns:
            List of (source_name, description) tuples
        """
        try:
            result = subprocess.run(
                ["pactl", "list", "sources", "short"],
                capture_output=True, text=True, timeout=5
            )
            sources = []
            for line in result.stdout.strip().splitlines():
                parts = line.split('\t')
                if len(parts) >= 2:
                    source_name = parts[1]
                    label = source_name
                    if ".monitor" in source_name:
                        label += " (monitor)"
                    sources.append((source_name, label))
            return sources
        except Exception as e:
            print(f"Error listing audio sources: {e}")
            return []

    @staticmethod
    def list_mic_sources() -> List[Tuple[str, str]]:
        """
        List available microphone/input sources (excluding monitor sources).

        Returns:
            List of (source_name, description) tuples
        """
        all_sources = PulseAudioRecorder.list_audio_sources()
        return [(name, label) for name, label in all_sources if ".monitor" not in name]

    @staticmethod
    def get_default_monitor_source() -> str:
        """
        Get the default sink's monitor source name.

        Returns:
            Monitor source name string, or empty string on failure
        """
        try:
            result = subprocess.run(
                ["pactl", "get-default-sink"],
                capture_output=True, text=True, timeout=5
            )
            default_sink = result.stdout.strip()
            if default_sink:
                return f"{default_sink}.monitor"
        except Exception as e:
            print(f"Error getting default monitor source: {e}")
        return ""


# Pick the backend for this platform. Callers import AudioRecorder only.
if sys.platform in ("win32", "darwin"):
    from audio_recorder_soundcard import SoundcardAudioRecorder as AudioRecorder
else:
    AudioRecorder = PulseAudioRecorder
