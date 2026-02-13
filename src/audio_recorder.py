"""
Audio Recorder - Captures system audio via PulseAudio monitor sources
"""
import subprocess
import wave
import threading
import time
from pathlib import Path
from typing import Optional, Callable, List, Tuple


class AudioRecorder:
    """Records system audio using parec with support for chunked output"""

    def __init__(
        self,
        output_file: str,
        chunk_dir: Path,
        sample_rate: int = 44100,
        channels: int = 2,
        chunk_duration: int = 30,
        audio_source: str = "",
        on_chunk_complete: Optional[Callable[[str], None]] = None
    ):
        self.output_file = output_file
        self.chunk_dir = chunk_dir
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.audio_source = audio_source
        self.on_chunk_complete = on_chunk_complete

        self.is_recording = False
        self.record_thread = None
        self.parec_process: Optional[subprocess.Popen] = None

        # s16le = 2 bytes per sample
        self.sample_width = 2

        # Chunk tracking
        self.chunk_counter = 0
        self.start_time = 0.0

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

        # Terminate parec process
        if self.parec_process:
            try:
                self.parec_process.terminate()
                self.parec_process.wait(timeout=3)
            except Exception:
                self.parec_process.kill()

        if self.record_thread:
            self.record_thread.join(timeout=5.0)

    def _record_loop(self) -> None:
        """Main recording loop (runs in separate thread)"""
        try:
            source = self.audio_source or self.get_default_monitor_source()
            if not source:
                print("Error: No audio source available")
                self.is_recording = False
                return

            cmd = [
                "parec",
                f"--format=s16le",
                f"--rate={self.sample_rate}",
                f"--channels={self.channels}",
                f"--device={source}",
            ]

            print(f"Starting parec: {' '.join(cmd)}")
            self.parec_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

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
                    self._save_chunk(buffer[:bytes_per_chunk])
                    buffer = buffer[bytes_per_chunk:]

            # Save any remaining audio as a final chunk
            if buffer:
                self._save_chunk(buffer)

        except Exception as e:
            print(f"Error in audio recording: {e}")
            self.is_recording = False

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

    def cleanup(self) -> None:
        """Clean up resources"""
        if self.parec_process:
            try:
                self.parec_process.kill()
            except Exception:
                pass

    def get_elapsed_time(self) -> float:
        """Get elapsed recording time in seconds"""
        if self.start_time and self.is_recording:
            return time.time() - self.start_time
        return 0.0

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
