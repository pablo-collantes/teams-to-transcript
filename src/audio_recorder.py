"""
Audio Recorder - Captures system audio and microphone input
"""
import pyaudio
import wave
import numpy as np
import threading
import queue
import time
from pathlib import Path
from typing import Optional, Callable


class AudioRecorder:
    """Records system audio and microphone with support for chunked output"""

    def __init__(
        self,
        output_file: str,
        chunk_dir: Path,
        sample_rate: int = 44100,
        channels: int = 2,
        chunk_duration: int = 30,
        on_chunk_complete: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize audio recorder

        Args:
            output_file: Path to final audio output file
            chunk_dir: Directory to save audio chunks for transcription
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1=mono, 2=stereo)
            chunk_duration: Duration of each audio chunk in seconds
            on_chunk_complete: Callback function when chunk is saved
        """
        self.output_file = output_file
        self.chunk_dir = chunk_dir
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.on_chunk_complete = on_chunk_complete

        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.record_thread = None

        # PyAudio setup
        self.audio = pyaudio.PyAudio()
        self.format = pyaudio.paInt16

        # Chunk tracking
        self.chunk_counter = 0
        self.current_chunk_frames = []

    def start(self) -> bool:
        """
        Start recording audio

        Returns:
            True if started successfully
        """
        if self.is_recording:
            return False

        self.is_recording = True
        self.chunk_counter = 0
        self.current_chunk_frames = []

        # Start recording thread
        self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.record_thread.start()

        return True

    def stop(self) -> None:
        """Stop recording and save final audio file"""
        if not self.is_recording:
            return

        self.is_recording = False

        # Wait for recording thread to finish
        if self.record_thread:
            self.record_thread.join(timeout=5.0)

        # Save any remaining chunk
        if self.current_chunk_frames:
            self._save_chunk()

    def _record_loop(self) -> None:
        """Main recording loop (runs in separate thread)"""
        try:
            # Open audio stream
            # Note: This captures the default input device
            # For system audio (loopback), you may need to configure
            # Windows stereo mix or use alternative libraries like soundcard
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024,
                stream_callback=self._audio_callback
            )

            stream.start_stream()

            # Keep stream active while recording
            while self.is_recording and stream.is_active():
                time.sleep(0.1)

            stream.stop_stream()
            stream.close()

        except Exception as e:
            print(f"Error in audio recording: {e}")
            self.is_recording = False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback for audio stream

        Args:
            in_data: Input audio data
            frame_count: Number of frames
            time_info: Time information
            status: Stream status

        Returns:
            Tuple of (None, continue flag)
        """
        if status:
            print(f"Audio stream status: {status}")

        # Add frames to current chunk
        self.current_chunk_frames.append(in_data)

        # Check if we've accumulated enough for a chunk
        frames_per_chunk = int(self.sample_rate * self.chunk_duration)
        current_frames = len(self.current_chunk_frames) * frame_count

        if current_frames >= frames_per_chunk:
            self._save_chunk()

        return (None, pyaudio.paContinue)

    def _save_chunk(self) -> None:
        """Save current audio chunk to file and notify callback"""
        if not self.current_chunk_frames:
            return

        try:
            # Generate chunk filename
            chunk_file = self.chunk_dir / f"chunk_{self.chunk_counter:04d}.wav"

            # Save chunk as WAV file
            with wave.open(str(chunk_file), 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.current_chunk_frames))

            print(f"Saved audio chunk: {chunk_file}")

            # Notify callback
            if self.on_chunk_complete:
                self.on_chunk_complete(str(chunk_file))

            # Reset for next chunk
            self.chunk_counter += 1
            self.current_chunk_frames = []

        except Exception as e:
            print(f"Error saving audio chunk: {e}")

    def save_final_audio(self, all_chunks: list) -> None:
        """
        Combine all chunks into final audio file

        Args:
            all_chunks: List of chunk file paths
        """
        try:
            with wave.open(self.output_file, 'wb') as outfile:
                outfile.setnchannels(self.channels)
                outfile.setsampwidth(self.audio.get_sample_size(self.format))
                outfile.setframerate(self.sample_rate)

                # Append all chunks
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
        """Clean up PyAudio resources"""
        self.audio.terminate()

    @staticmethod
    def list_audio_devices() -> list:
        """
        List available audio input devices

        Returns:
            List of (index, name) tuples
        """
        audio = pyaudio.PyAudio()
        devices = []

        for i in range(audio.get_device_count()):
            try:
                info = audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    devices.append((i, info['name']))
            except Exception:
                pass

        audio.terminate()
        return devices


class SystemAudioRecorder:
    """
    Windows system audio recorder using WASAPI loopback

    Note: This requires Windows Stereo Mix to be enabled or
    alternative implementation using soundcard library
    """

    def __init__(self, sample_rate: int = 44100, channels: int = 2):
        """
        Initialize system audio recorder

        Args:
            sample_rate: Audio sample rate
            channels: Number of channels
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.frames = []

    def start(self) -> bool:
        """Start recording system audio"""
        # This is a placeholder for WASAPI loopback implementation
        # In production, you would use Windows-specific APIs or
        # libraries like soundcard
        print("System audio recording requires Stereo Mix or soundcard library")
        return True

    def stop(self) -> list:
        """
        Stop recording and return frames

        Returns:
            List of audio frames
        """
        return self.frames
