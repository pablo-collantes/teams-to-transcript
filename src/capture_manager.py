"""
Capture Manager - Orchestrates audio capture and transcription
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable
from audio_recorder import AudioRecorder
from transcription_engine import TranscriptionEngine
from streaming_transcriber import StreamingTranscriber
from settings_manager import SettingsManager

_MONTH_ABBR = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def _ordinal_suffix(day: int) -> str:
    if 11 <= day <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def generate_category_filename(category: str) -> str:
    """Generate filename like 'Mar-10th_MUSI' from current date and category."""
    now = datetime.now()
    month = _MONTH_ABBR[now.month]
    day = now.day
    suffix = _ordinal_suffix(day)
    return f"{month}-{day}{suffix}_{category}"


class CaptureManager:
    """Manages the complete recording workflow"""

    def __init__(self, settings: SettingsManager):
        self.settings = settings
        self.is_recording = False

        # Component instances
        self.audio_recorder: Optional[AudioRecorder] = None
        self.transcription_engine: Optional[TranscriptionEngine] = None
        self.streaming_transcriber: Optional[StreamingTranscriber] = None

        # Current recording session (one folder per recording)
        self.current_audio_file = None
        self.current_transcript_file = None
        self.session_dir: Optional[Path] = None
        self.session_name = ""
        self.session_status = "idle"
        self.session_started: Optional[datetime] = None
        self.session_ended: Optional[datetime] = None
        self.screenshots = []
        self._manifest_lock = threading.Lock()
        self.audio_chunks = []
        self.start_time = 0.0

        # Callbacks
        self.on_status_update: Optional[Callable[[str], None]] = None
        self.on_transcript_update: Optional[Callable[[str], None]] = None

    def start_recording(self) -> bool:
        """Start recording audio and transcription. Returns True if started successfully."""
        if self.is_recording:
            self._update_status("Already recording")
            return False

        try:
            # Each recording lives in its own folder: audio, transcript,
            # screenshots and session.json all together
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base_name = f"{timestamp}_teams-call"

            recordings_dir = self.settings.get_recordings_dir()
            temp_dir = self.settings.get_temp_dir()

            self.session_dir = recordings_dir / base_name
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self.session_name = base_name
            self.screenshots = []
            self.session_started = datetime.now()
            self.session_ended = None

            self.current_audio_file = str(self.session_dir / f"{base_name}.wav")
            self.current_transcript_file = str(self.session_dir / f"{base_name}.txt")

            # Announce the session immediately so the live server broadcasts
            # it before the (slow) model load and first chunk
            self.session_status = "starting"
            self._write_manifest()

            # Clear temporary directory
            self._clear_temp_dir(temp_dir)
            self.audio_chunks = []

            # Streaming mode: one component does capture + growing-window
            # transcription. The classic chunked path below is left unchanged.
            if self.settings.get("streaming_mode", False):
                return self._start_streaming_recording()

            # Initialize transcription engine
            try:
                self._update_status("Initializing transcription engine...")
                self.transcription_engine = TranscriptionEngine(
                    output_file=self.current_transcript_file,
                    model_size=self.settings.get("whisper_model"),
                    device="auto",
                    chunk_duration=self.settings.get("chunk_duration"),
                    on_transcript_update=self._on_transcript_chunk,
                    engine=self.settings.get("inference_engine", "faster-whisper"),
                    engine_options={
                        "server_bin": self.settings.get("whisper_cpp_server", ""),
                        "model_path": self.settings.get("whisper_cpp_model", ""),
                        "port": self.settings.get("whisper_cpp_port", 8009),
                        "url": self.settings.get("whisper_cpp_url", ""),
                    }
                )

                if not self.transcription_engine.start():
                    self._update_status("Failed to start transcription engine")
                    self._mark_session_error()
                    return False
            except Exception as e:
                import traceback
                print(f"ERROR in transcription engine: {e}")
                print(traceback.format_exc())
                self._update_status(f"Transcription engine error: {e}")
                self._mark_session_error()
                return False

            # Initialize audio recorder
            try:
                self._update_status("Initializing audio recorder...")
                mic_source = ""
                if self.settings.get("enable_mic", False):
                    mic_source = self.settings.get("mic_source", "")
                    if mic_source:
                        self._update_status(f"Mic recording enabled: {mic_source}")

                self.audio_recorder = AudioRecorder(
                    output_file=self.current_audio_file,
                    chunk_dir=temp_dir,
                    sample_rate=self.settings.get("audio_sample_rate"),
                    channels=self.settings.get("audio_channels"),
                    chunk_duration=self.settings.get("chunk_duration"),
                    audio_source=self.settings.get("audio_source", ""),
                    mic_source=mic_source,
                    on_chunk_complete=self._on_audio_chunk
                )

                if not self.audio_recorder.start():
                    self._update_status("Failed to start audio recorder")
                    self.transcription_engine.stop()
                    self._mark_session_error()
                    return False
            except Exception as e:
                import traceback
                print(f"ERROR in audio recorder: {e}")
                print(traceback.format_exc())
                self._update_status(f"Audio recorder error: {e}")
                if self.transcription_engine:
                    self.transcription_engine.stop()
                self._mark_session_error()
                return False

            # The transcript clock (and screenshot timestamps) start now,
            # when audio actually begins flowing
            self.start_time = time.time()
            self.session_started = datetime.now()
            self.session_status = "recording"
            self._write_manifest()

            self.is_recording = True
            self._update_status(f"Recording started: {base_name}")
            return True

        except Exception as e:
            import traceback
            print(f"ERROR in start_recording: {e}")
            print(traceback.format_exc())
            self._update_status(f"Error starting recording: {e}")
            return False

    def _start_streaming_recording(self) -> bool:
        """Start the near-real-time streaming path (growing-window LocalAgreement).
        Uses one StreamingTranscriber that captures audio and transcribes it itself."""
        from transcription_engine import FasterWhisperBackend, WhisperCppBackend

        engine = self.settings.get("inference_engine", "faster-whisper")
        language = self.settings.get("language", "en")
        try:
            if engine == "whisper-cpp":
                backend = WhisperCppBackend(
                    server_bin=self.settings.get("whisper_cpp_server", ""),
                    model_path=self.settings.get("whisper_cpp_model", ""),
                    port=self.settings.get("whisper_cpp_port", 8009),
                    url=self.settings.get("whisper_cpp_url", ""),
                    language=language,
                )
            else:
                device, compute = "cpu", "int8"
                try:
                    import torch
                    if torch.cuda.is_available():
                        device, compute = "cuda", "float16"
                except Exception:
                    pass
                backend = FasterWhisperBackend(
                    model_size=self.settings.get("whisper_model"),
                    device=device, compute_type=compute, language=language,
                )

            mic_source = ""
            if self.settings.get("enable_mic", False):
                mic_source = self.settings.get("mic_source", "")

            self.streaming_transcriber = StreamingTranscriber(
                output_file=self.current_transcript_file,
                backend=backend,
                on_transcript_update=self._on_transcript_chunk,
                audio_source=self.settings.get("audio_source", ""),
                mic_source=mic_source,
                sample_rate=self.settings.get("audio_sample_rate", 16000),
                channels=self.settings.get("audio_channels", 1),
                hop_seconds=self.settings.get("stream_hop_seconds", 2.0),
                min_speech_seconds=self.settings.get("stream_min_speech_seconds", 1.0),
                max_utterance_seconds=self.settings.get("stream_max_utterance_seconds", 24.0),
                end_silence_seconds=self.settings.get("stream_end_silence_seconds", 0.8),
                silence_rms=self.settings.get("stream_silence_rms", 300.0),
                commit_holdback_words=self.settings.get("stream_commit_holdback_words", 1),
                stall_backoff=self.settings.get("stream_stall_backoff", True),
                on_status=self._update_status,
                session_wav=self.current_audio_file,
            )
            if not self.streaming_transcriber.start():
                self._update_status("Failed to start streaming transcriber")
                self._mark_session_error()
                return False
        except Exception as e:
            import traceback
            print(f"ERROR starting streaming transcriber: {e}")
            print(traceback.format_exc())
            self._update_status(f"Streaming transcriber error: {e}")
            self._mark_session_error()
            return False

        self.start_time = time.time()
        self.session_started = datetime.now()
        self.session_status = "recording"
        self._write_manifest()
        self.is_recording = True
        self._update_status(f"Recording started (streaming): {self.session_name}")
        return True

    def stop_recording(self) -> bool:
        """Stop recording and finalize all files. Returns True if stopped successfully."""
        if not self.is_recording:
            self._update_status("Not currently recording")
            return False

        try:
            self._update_status("Stopping recording...")

            # Stop streaming transcriber (does its own capture + final flush)
            if self.streaming_transcriber:
                self.streaming_transcriber.stop()
                self.streaming_transcriber = None

            # Stop audio recorder
            if self.audio_recorder:
                self.audio_recorder.stop()
                self.audio_recorder.save_final_audio(self.audio_chunks)
                self.audio_recorder.cleanup()

            # Wait for transcription to complete
            self._update_status("Finalizing transcription...")
            if self.transcription_engine:
                self.transcription_engine.stop()

            # Clean up temp directory
            temp_dir = self.settings.get_temp_dir()
            self._clear_temp_dir(temp_dir)

            self.session_status = "stopped"
            self.session_ended = datetime.now()
            self._write_manifest()

            self.is_recording = False
            self._update_status(
                f"Recording saved: {Path(self.current_transcript_file).name}"
            )

            self._show_summary()

            return True

        except Exception as e:
            self._update_status(f"Error stopping recording: {e}")
            return False

    def rename_and_move_files(self, category: str) -> bool:
        """Move the whole session folder into recordings/<CATEGORY>/<title>/."""
        try:
            if not self.session_dir or not self.session_dir.exists():
                self._update_status("No recording session to move")
                return False

            recordings_dir = self.settings.get_recordings_dir()
            cat_dir = recordings_dir / category
            cat_dir.mkdir(parents=True, exist_ok=True)

            old_base = self.session_name
            new_base = generate_category_filename(category)
            new_base = self._deduplicate_dirname(cat_dir, new_base)
            dest_dir = cat_dir / new_base

            shutil.move(str(self.session_dir), str(dest_dir))

            # Rename audio/transcript inside the folder to match the new title;
            # screenshots keep their elapsed-clock names
            for ext in (".wav", ".txt"):
                old_path = dest_dir / f"{old_base}{ext}"
                if old_path.exists():
                    old_path.rename(dest_dir / f"{new_base}{ext}")

            self.session_dir = dest_dir
            self.session_name = new_base
            self.current_audio_file = str(dest_dir / f"{new_base}.wav")
            self.current_transcript_file = str(dest_dir / f"{new_base}.txt")
            self._write_manifest()

            self._update_status(f"Recording saved to {category}/{new_base}/")
            return True
        except Exception as e:
            self._update_status(f"Error moving files: {e}")
            return False

    @staticmethod
    def _deduplicate_dirname(directory: Path, base_name: str) -> str:
        """Append _2, _3, etc. if a folder with base_name already exists."""
        if not (directory / base_name).exists():
            return base_name
        counter = 2
        while (directory / f"{base_name}_{counter}").exists():
            counter += 1
        return f"{base_name}_{counter}"

    # ------------------------------------------------------------------
    # Screenshots
    # ------------------------------------------------------------------

    def take_screenshot(self, mode: str = "region") -> Optional[dict]:
        """Capture a screenshot into the session folder, stamped with the
        elapsed recording clock.

        Args:
            mode: "region" (interactive slurp selection) or "full" (whole screen)

        Returns:
            The manifest entry dict, or None if cancelled/failed.
        """
        if not self.is_recording or not self.session_dir:
            self._update_status("Screenshot skipped: not recording")
            return None

        # Elapsed is captured at click time so it matches what the user saw
        elapsed = max(0.0, time.time() - self.start_time)
        hms = self._format_elapsed(elapsed)
        filename = f"screenshot_{len(self.screenshots) + 1:03d}_{hms.replace(':', '-')}.png"
        dest = self.session_dir / filename

        try:
            if sys.platform == "win32":
                # PowerShell full-screen capture; no region select on Windows
                if mode == "region":
                    self._update_status("Region select not available on Windows; "
                                        "capturing full screen")
                ps_script = (
                    "Add-Type -AssemblyName System.Windows.Forms; "
                    "Add-Type -AssemblyName System.Drawing; "
                    "$b=[System.Windows.Forms.SystemInformation]::VirtualScreen; "
                    "$bmp=New-Object System.Drawing.Bitmap($b.Width,$b.Height); "
                    "$g=[System.Drawing.Graphics]::FromImage($bmp); "
                    "$g.CopyFromScreen($b.Left,$b.Top,0,0,$bmp.Size); "
                    f"$bmp.Save('{dest}',[System.Drawing.Imaging.ImageFormat]::Png); "
                    "$g.Dispose(); $bmp.Dispose()"
                )
                cmd = ["powershell", "-NoProfile", "-Command", ps_script]
            elif mode == "region":
                sel = subprocess.run(["slurp"], capture_output=True, text=True,
                                     timeout=120)
                if sel.returncode != 0 or not sel.stdout.strip():
                    self._update_status("Screenshot cancelled")
                    return None
                cmd = ["grim", "-g", sel.stdout.strip(), str(dest)]
            else:
                cmd = ["grim", str(dest)]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0 or not dest.exists():
                err = result.stderr.strip() or "screenshot command error"
                self._update_status(f"Screenshot failed: {err}")
                return None
        except FileNotFoundError as e:
            hint = "" if sys.platform == "win32" else " (install grim/slurp)"
            self._update_status(f"Screenshot failed: {e}{hint}")
            return None
        except subprocess.TimeoutExpired:
            self._update_status("Screenshot timed out")
            return None

        entry = {
            "file": filename,
            "elapsed_seconds": round(elapsed, 1),
            "elapsed_hms": hms,
            "taken_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.screenshots.append(entry)
        self._write_manifest()
        self._update_status(f"Screenshot saved at {hms} ({filename})")
        return entry

    # ------------------------------------------------------------------
    # Session manifest
    # ------------------------------------------------------------------

    def _write_manifest(self) -> None:
        """Write session.json atomically so the live server never sees a partial file."""
        if not self.session_dir:
            return
        manifest = {
            "version": 1,
            "name": self.session_name,
            "status": self.session_status,
            "started": self.session_started.isoformat(timespec="seconds")
                       if self.session_started else None,
            "ended": self.session_ended.isoformat(timespec="seconds")
                     if self.session_ended else None,
            "chunk_duration": self.settings.get("chunk_duration", 30),
            "audio_file": Path(self.current_audio_file).name
                          if self.current_audio_file else None,
            "transcript_file": Path(self.current_transcript_file).name
                               if self.current_transcript_file else None,
            "screenshots": list(self.screenshots),
        }
        try:
            with self._manifest_lock:
                tmp = self.session_dir / "session.json.tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
                tmp.replace(self.session_dir / "session.json")
        except OSError as e:
            print(f"Error writing session manifest: {e}")

    def _mark_session_error(self) -> None:
        self.session_status = "error"
        self.session_ended = datetime.now()
        self._write_manifest()

    @staticmethod
    def _format_elapsed(total: float) -> str:
        """Format seconds as HH:MM:SS."""
        h = int(total // 3600)
        m = int((total % 3600) // 60)
        s = int(total % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _on_audio_chunk(self, chunk_file: str) -> None:
        """Callback when audio chunk is saved"""
        self.audio_chunks.append(chunk_file)

        if self.transcription_engine:
            self.transcription_engine.add_chunk(chunk_file)

    def _on_transcript_chunk(self, transcript: str) -> None:
        """Callback when new transcript is available"""
        if self.on_transcript_update:
            self.on_transcript_update(transcript)

    def _update_status(self, message: str) -> None:
        """Update status message"""
        print(f"[CaptureManager] {message}")
        if self.on_status_update:
            self.on_status_update(message)

    def _clear_temp_dir(self, temp_dir: Path) -> None:
        """Clear temporary directory"""
        try:
            if temp_dir.exists():
                for file in temp_dir.glob("*.wav"):
                    file.unlink()
        except Exception as e:
            print(f"Error clearing temp directory: {e}")

    def _show_summary(self) -> None:
        """Display recording summary"""
        try:
            summary = "\n=== Recording Summary ===\n"

            if self.current_audio_file and os.path.exists(self.current_audio_file):
                audio_size = os.path.getsize(self.current_audio_file) / (1024 * 1024)
                summary += f"Audio: {Path(self.current_audio_file).name} ({audio_size:.2f} MB)\n"

            if self.current_transcript_file and os.path.exists(self.current_transcript_file):
                with open(self.current_transcript_file, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                summary += f"Transcript: {Path(self.current_transcript_file).name} ({lines} lines)\n"

            if self.transcription_engine:
                chunk_count = self.transcription_engine.get_transcript_count()
                summary += f"Chunks transcribed: {chunk_count}\n"

            if self.screenshots:
                summary += f"Screenshots: {len(self.screenshots)}\n"

            elapsed = time.time() - self.start_time if self.start_time else 0
            summary += f"Duration: {elapsed:.1f}s\n"

            summary += "=" * 30

            self._update_status(summary)

        except Exception as e:
            print(f"Error generating summary: {e}")

    def get_recording_status(self) -> dict:
        """Get current recording status"""
        status = {
            "is_recording": self.is_recording,
            "audio_file": self.current_audio_file,
            "transcript_file": self.current_transcript_file,
            "chunks_processed": 0,
            "elapsed_time": 0.0,
        }

        if self.transcription_engine:
            status["chunks_processed"] = self.transcription_engine.get_transcript_count()

        if self.start_time and self.is_recording:
            status["elapsed_time"] = time.time() - self.start_time

        return status

    def set_status_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for status updates"""
        self.on_status_update = callback

    def set_transcript_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for transcript updates"""
        self.on_transcript_update = callback
