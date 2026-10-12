"""
Soundcard backend - system audio capture for Windows (and macOS).

Uses the `soundcard` library: on Windows, system audio is captured through
WASAPI loopback devices (every speaker has a matching loopback "microphone");
on macOS there is no OS loopback, so system capture needs a virtual audio
device (e.g. BlackHole) selected as the audio source. Microphone capture
works the same on both.

Unlike the PulseAudio backend, which chunks a continuous byte stream, WASAPI
loopback delivers NO packets while the system is silent. This backend
therefore paces chunks on the wall clock and pads any shortfall with silence,
so chunk N always covers [N*chunk_duration, (N+1)*chunk_duration] on the
recording clock and transcript timestamps stay aligned.
"""
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from audio_recorder import BaseAudioRecorder

_BLOCK_FRAMES = 512


def _soundcard():
    """Import the soundcard library, or None with a hint if unavailable."""
    try:
        import soundcard
        return soundcard
    except Exception as e:
        print(f"soundcard library unavailable: {e}")
        print("Install it with: pip install soundcard")
        return None


class SoundcardAudioRecorder(BaseAudioRecorder):
    """Records system audio via WASAPI loopback (Windows) / CoreAudio (macOS)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sys_buffer = bytearray()
        self._sys_lock = threading.Lock()
        self._sys_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def _resolve_source(self, sc, source_id: str):
        """Find the capture device: explicit id, else default speaker's loopback."""
        try:
            mics = sc.all_microphones(include_loopback=True)
        except Exception as e:
            print(f"Error listing capture devices: {e}")
            return None

        if source_id:
            for mic in mics:
                if str(mic.id) == source_id:
                    return mic
            print(f"Error: Audio source not found: {source_id}")
            return None

        loopbacks = [m for m in mics if getattr(m, "isloopback", False)]
        if not loopbacks:
            print("Error: No loopback/monitor capture device found. "
                  "On macOS install a virtual device (e.g. BlackHole) and "
                  "select it as the audio source.")
            return None
        try:
            speaker_name = sc.default_speaker().name
        except Exception:
            speaker_name = ""
        for mic in loopbacks:
            if speaker_name and speaker_name in mic.name:
                return mic
        return loopbacks[0]

    def _capture_loop(self, device, buffer: bytearray, lock: threading.Lock,
                      label: str) -> None:
        """Read float32 frames from a device into a shared s16le byte buffer."""
        try:
            with device.recorder(samplerate=self.sample_rate,
                                 channels=self.channels,
                                 blocksize=_BLOCK_FRAMES) as rec:
                while self.is_recording:
                    frames = rec.record(numframes=_BLOCK_FRAMES)
                    if frames is None or len(frames) == 0:
                        continue
                    data = np.clip(frames * 32767.0, -32768, 32767) \
                             .astype(np.int16).tobytes()
                    with lock:
                        buffer.extend(data)
        except Exception as e:
            print(f"Error capturing {label} audio: {e}")

    def _take_system_bytes(self, num_bytes: int) -> bytes:
        """Take up to num_bytes of system audio, silence-padded to num_bytes.

        Padding keeps the chunk clock aligned when WASAPI loopback stalls
        during system silence. Surplus stays buffered for the next chunk.
        """
        with self._sys_lock:
            take = min(len(self._sys_buffer), num_bytes)
            data = bytes(self._sys_buffer[:take])
            del self._sys_buffer[:take]
        if len(data) < num_bytes:
            data += b"\x00" * (num_bytes - len(data))
        return data

    def _flush_chunk(self, num_bytes: int) -> None:
        chunk_data = self._take_system_bytes(num_bytes)
        if self.mic_source:
            chunk_data = self._mix_audio(chunk_data, self._get_mic_data(len(chunk_data)))
        self._save_chunk(chunk_data)

    def _record_loop(self) -> None:
        """Main recording loop: wall-clock paced chunk flushing."""
        sc = _soundcard()
        if sc is None:
            self.is_recording = False
            return

        try:
            system_device = self._resolve_source(sc, self.audio_source)
            if system_device is None:
                self.is_recording = False
                return
            print(f"Capturing system audio from: {system_device.name}")

            self._sys_buffer = bytearray()
            self._sys_thread = threading.Thread(
                target=self._capture_loop,
                args=(system_device, self._sys_buffer, self._sys_lock, "system"),
                daemon=True)
            self._sys_thread.start()

            if self.mic_source:
                mic_device = self._resolve_source(sc, self.mic_source)
                if mic_device is not None:
                    print(f"Mic recording enabled: {mic_device.name}")
                    self._mic_buffer = bytearray()
                    self._mic_thread = threading.Thread(
                        target=self._capture_loop,
                        args=(mic_device, self._mic_buffer, self._mic_lock, "mic"),
                        daemon=True)
                    self._mic_thread.start()
                else:
                    self.mic_source = ""

            bytes_per_chunk = (
                self.sample_rate
                * self.channels
                * self.sample_width
                * self.chunk_duration
            )

            next_flush = time.monotonic() + self.chunk_duration
            while self.is_recording:
                time.sleep(0.05)
                if time.monotonic() >= next_flush:
                    self._flush_chunk(bytes_per_chunk)
                    next_flush += self.chunk_duration

            # Save whatever real audio remains as a final partial chunk
            frame_bytes = self.channels * self.sample_width
            with self._sys_lock:
                remainder = bytes(self._sys_buffer)
                self._sys_buffer.clear()
            remainder = remainder[:len(remainder) // frame_bytes * frame_bytes]
            if remainder:
                if self.mic_source:
                    remainder = self._mix_audio(remainder, self._get_mic_data(len(remainder)))
                self._save_chunk(remainder)

        except Exception as e:
            print(f"Error in audio recording: {e}")
            self.is_recording = False

    def stop(self) -> None:
        super().stop()
        # Capture threads are daemons; a WASAPI read blocked on total silence
        # may not return promptly, so don't wait long
        if self._sys_thread:
            self._sys_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Device listing
    # ------------------------------------------------------------------

    @staticmethod
    def list_audio_sources() -> List[Tuple[str, str]]:
        """List capture devices; loopbacks of speakers are marked (monitor)."""
        sc = _soundcard()
        if sc is None:
            return []
        try:
            sources = []
            for mic in sc.all_microphones(include_loopback=True):
                label = mic.name
                if getattr(mic, "isloopback", False):
                    label += " (monitor)"
                sources.append((str(mic.id), label))
            return sources
        except Exception as e:
            print(f"Error listing audio sources: {e}")
            return []

    @staticmethod
    def list_mic_sources() -> List[Tuple[str, str]]:
        """List real input devices (no loopbacks)."""
        sc = _soundcard()
        if sc is None:
            return []
        try:
            return [(str(mic.id), mic.name)
                    for mic in sc.all_microphones(include_loopback=False)]
        except Exception as e:
            print(f"Error listing mic sources: {e}")
            return []

    @staticmethod
    def get_default_monitor_source() -> str:
        """Id of the default speaker's loopback device, or empty string."""
        sc = _soundcard()
        if sc is None:
            return ""
        try:
            mics = sc.all_microphones(include_loopback=True)
            loopbacks = [m for m in mics if getattr(m, "isloopback", False)]
            if not loopbacks:
                return ""
            try:
                speaker_name = sc.default_speaker().name
            except Exception:
                speaker_name = ""
            for mic in loopbacks:
                if speaker_name and speaker_name in mic.name:
                    return str(mic.id)
            return str(loopbacks[0].id)
        except Exception as e:
            print(f"Error getting default monitor source: {e}")
            return ""
