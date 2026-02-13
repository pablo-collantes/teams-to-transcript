"""
Capture Manager - Orchestrates audio capture and transcription
"""
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable
from audio_recorder import AudioRecorder
from transcription_engine import TranscriptionEngine
from settings_manager import SettingsManager


class CaptureManager:
    """Manages the complete recording workflow"""

    def __init__(self, settings: SettingsManager):
        self.settings = settings
        self.is_recording = False

        # Component instances
        self.audio_recorder: Optional[AudioRecorder] = None
        self.transcription_engine: Optional[TranscriptionEngine] = None

        # Current recording files
        self.current_audio_file = None
        self.current_transcript_file = None
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
            # Generate output filenames
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base_name = f"{timestamp}_teams-call"

            recordings_dir = self.settings.get_recordings_dir()
            temp_dir = self.settings.get_temp_dir()

            self.current_audio_file = str(recordings_dir / f"{base_name}.wav")
            self.current_transcript_file = str(recordings_dir / f"{base_name}.txt")

            # Clear temporary directory
            self._clear_temp_dir(temp_dir)
            self.audio_chunks = []
            self.start_time = time.time()

            # Initialize transcription engine
            try:
                self._update_status("Initializing transcription engine...")
                self.transcription_engine = TranscriptionEngine(
                    output_file=self.current_transcript_file,
                    model_size=self.settings.get("whisper_model"),
                    device="auto",
                    on_transcript_update=self._on_transcript_chunk
                )

                if not self.transcription_engine.start():
                    self._update_status("Failed to start transcription engine")
                    return False
            except Exception as e:
                import traceback
                print(f"ERROR in transcription engine: {e}")
                print(traceback.format_exc())
                self._update_status(f"Transcription engine error: {e}")
                return False

            # Initialize audio recorder
            try:
                self._update_status("Initializing audio recorder...")
                self.audio_recorder = AudioRecorder(
                    output_file=self.current_audio_file,
                    chunk_dir=temp_dir,
                    sample_rate=self.settings.get("audio_sample_rate"),
                    channels=self.settings.get("audio_channels"),
                    chunk_duration=self.settings.get("chunk_duration"),
                    audio_source=self.settings.get("audio_source", ""),
                    on_chunk_complete=self._on_audio_chunk
                )

                if not self.audio_recorder.start():
                    self._update_status("Failed to start audio recorder")
                    self.transcription_engine.stop()
                    return False
            except Exception as e:
                import traceback
                print(f"ERROR in audio recorder: {e}")
                print(traceback.format_exc())
                self._update_status(f"Audio recorder error: {e}")
                if self.transcription_engine:
                    self.transcription_engine.stop()
                return False

            self.is_recording = True
            self._update_status(f"Recording started: {base_name}")
            return True

        except Exception as e:
            import traceback
            print(f"ERROR in start_recording: {e}")
            print(traceback.format_exc())
            self._update_status(f"Error starting recording: {e}")
            return False

    def stop_recording(self) -> bool:
        """Stop recording and finalize all files. Returns True if stopped successfully."""
        if not self.is_recording:
            self._update_status("Not currently recording")
            return False

        try:
            self._update_status("Stopping recording...")

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

            self.is_recording = False
            self._update_status(
                f"Recording saved: {Path(self.current_transcript_file).name}"
            )

            self._show_summary()

            return True

        except Exception as e:
            self._update_status(f"Error stopping recording: {e}")
            return False

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
