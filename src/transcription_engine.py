"""
Transcription Engine - Processes audio chunks with Whisper AI
"""
import os
import threading
import queue
import time
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime, timedelta
from faster_whisper import WhisperModel

# Suppress HuggingFace Hub warnings on Windows
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'


class TranscriptionEngine:
    """Handles real-time transcription of audio chunks using faster-whisper"""

    def __init__(
        self,
        output_file: str,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "default",
        on_transcript_update: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize transcription engine

        Args:
            output_file: Path to output transcript file
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (cpu, cuda, auto)
            compute_type: Compute type (default, int8, float16, int8_float16)
            on_transcript_update: Callback when new transcript is available
        """
        self.output_file = output_file
        self.model_size = model_size
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

        # Initialize model (will download on first use)
        print(f"Loading Whisper model: {model_size} on {self.device}")
        self.model = None
        self.model_loaded = False

        # Processing queue and thread
        self.chunk_queue = queue.Queue()
        self.is_processing = False
        self.process_thread = None

        # Transcript storage
        self.transcripts = []
        self.start_timestamp = None

    def _load_model(self) -> bool:
        """
        Load Whisper model

        Returns:
            True if loaded successfully
        """
        if self.model_loaded:
            return True

        try:
            print(f"\n{'='*60}")
            print(f"Loading Whisper model: {self.model_size}")
            print(f"Device: {self.device} | Compute type: {self.compute_type}")
            print(f"\nNOTE: First run will download the model (~140MB for base)")
            print(f"      This may take 1-2 minutes depending on your connection.")
            print(f"      Subsequent runs will be much faster (5-10 seconds).")
            print(f"\nPlease wait...")
            print(f"{'='*60}\n")

            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            self.model_loaded = True
            print(f"\n{'='*60}")
            print(f"Whisper model loaded successfully!")
            print(f"{'='*60}\n")
            return True
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"ERROR loading Whisper model: {e}")
            print(f"{'='*60}\n")
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

    def stop(self) -> None:
        """Stop transcription processing and save final transcript"""
        if not self.is_processing:
            return

        self.is_processing = False

        # Wait for processing thread to finish
        if self.process_thread:
            self.process_thread.join(timeout=30.0)  # Give time for final chunks

        # Save final transcript
        self._save_transcript()

    def add_chunk(self, chunk_file: str) -> None:
        """
        Add audio chunk to processing queue

        Args:
            chunk_file: Path to audio chunk file
        """
        self.chunk_queue.put(chunk_file)

    def _process_loop(self) -> None:
        """Main processing loop (runs in separate thread)"""
        while self.is_processing:
            try:
                # Get chunk from queue (timeout to allow checking is_processing)
                try:
                    chunk_file = self.chunk_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Process the chunk
                transcript_text = self._transcribe_chunk(chunk_file)

                if transcript_text:
                    # Calculate timestamp for this chunk
                    chunk_index = len(self.transcripts)
                    timestamp = self.start_timestamp + timedelta(seconds=chunk_index * 30)

                    # Store transcript
                    self.transcripts.append({
                        "timestamp": timestamp,
                        "text": transcript_text,
                        "chunk_file": chunk_file
                    })

                    # Save incrementally
                    self._save_transcript()

                    # Notify callback
                    if self.on_transcript_update:
                        self.on_transcript_update(transcript_text)

                    print(f"Transcribed chunk {chunk_index}: {transcript_text[:100]}...")

                self.chunk_queue.task_done()

            except Exception as e:
                print(f"Error in transcription processing loop: {e}")

        # Process any remaining chunks
        while not self.chunk_queue.empty():
            try:
                chunk_file = self.chunk_queue.get_nowait()
                transcript_text = self._transcribe_chunk(chunk_file)

                if transcript_text:
                    chunk_index = len(self.transcripts)
                    timestamp = self.start_timestamp + timedelta(seconds=chunk_index * 30)

                    self.transcripts.append({
                        "timestamp": timestamp,
                        "text": transcript_text,
                        "chunk_file": chunk_file
                    })

                self.chunk_queue.task_done()

            except queue.Empty:
                break
            except Exception as e:
                print(f"Error processing remaining chunk: {e}")

    def _transcribe_chunk(self, chunk_file: str) -> Optional[str]:
        """
        Transcribe a single audio chunk

        Args:
            chunk_file: Path to audio chunk file

        Returns:
            Transcribed text or None on error
        """
        try:
            # Transcribe with Whisper
            segments, info = self.model.transcribe(
                chunk_file,
                beam_size=5,
                language="en"  # Can be made configurable
            )

            # Combine all segments into single text
            transcript_parts = []
            for segment in segments:
                transcript_parts.append(segment.text.strip())

            transcript_text = " ".join(transcript_parts)
            return transcript_text

        except Exception as e:
            print(f"Error transcribing chunk {chunk_file}: {e}")
            return None

    def _save_transcript(self) -> None:
        """Save current transcripts to output file"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write("=== Teams Call Transcript ===\n\n")

                if self.start_timestamp:
                    f.write(f"Recording started: {self.start_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("-" * 80 + "\n\n")

                for entry in self.transcripts:
                    timestamp_str = entry["timestamp"].strftime('%H:%M:%S')
                    f.write(f"[{timestamp_str}]\n")
                    f.write(f"{entry['text']}\n\n")

                f.write("-" * 80 + "\n")
                f.write(f"\nTranscript generated by faster-whisper ({self.model_size} model)\n")

        except Exception as e:
            print(f"Error saving transcript: {e}")

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
