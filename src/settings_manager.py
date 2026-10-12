"""
Settings Manager - Handles configuration storage and defaults
"""
import json
import os
from pathlib import Path
from typing import Dict, Any


class SettingsManager:
    """Manages application settings with persistence to JSON file"""

    DEFAULT_SETTINGS = {
        "output_directory": str(Path.home() / "Documents" / "TeamRecordings"),
        "whisper_model": "tiny",  # tiny, base, small, medium, large
        "audio_source": "",  # PulseAudio source name (empty = default monitor)
        "enable_mic": False,  # Record microphone alongside system audio
        "mic_source": "",  # PulseAudio mic source name
        "chunk_duration": 30,  # seconds
        "enable_gpu": True,
        "audio_sample_rate": 44100,
        "audio_channels": 2,
        "hf_token": "",
        "web_host": "127.0.0.1",  # web_server.py bind address; it has no auth
        "web_port": 8000,
        "web_model": "large-v3",
        "live_server_enabled": True,  # serve the live transcript over HTTP
        "live_host": "",  # empty = auto-detect tailscale IP (fallback 127.0.0.1)
        "live_port": 8010,
        "screenshot_mode": "region",  # "region" (slurp select) or "full"
        "inference_engine": "faster-whisper",  # or "whisper-cpp" (GPU via Vulkan)
        "whisper_cpp_server": "",  # path to whisper-server binary
        "whisper_cpp_model": "",   # path to ggml-*.bin model file
        "whisper_cpp_port": 8009,
        "whisper_cpp_url": "",     # connect to an existing server instead of spawning
        "ingest_enabled": True,    # accept remote audio (phones) via /ingest + /record
        "whisper_cpp_ingest_port": 8011,  # separate server so local + remote can overlap
        "whisper_cpp_file_port": 8012,    # server for whole-file transcription (CLI/GUI)
        # --- Near-real-time streaming mode (opt-in; classic chunked path is default) ---
        "streaming_mode": False,   # growing-window LocalAgreement transcription
        "language": "en",          # decode language for streaming backends
        "stream_hop_seconds": 2.0,        # re-transcribe the growing buffer this often
        "stream_min_speech_seconds": 1.0, # need this much speech before first commit
        "stream_max_utterance_seconds": 24.0,  # force flush + reset past this length
        "stream_end_silence_seconds": 0.8,     # silence that ends an utterance
        "stream_silence_rms": 300.0,           # RMS below this counts as silence (tune per mic)
        "stream_commit_holdback_words": 1,     # tail-keep: hold back the last agreed word(s)
        "stream_stall_backoff": True,          # space hops out when decoding lags realtime
        # --- In-app AI companion (local LLM advisor, right-hand dock) ---
        "companion_enabled": False,            # show the AI Companion dock + warm the model
        "companion_llama_bin": "",             # empty -> llama-server on PATH
        "companion_model": "",                 # path to a GGUF chat model (required)
        "companion_host": "127.0.0.1",
        "companion_port": 8090,                # separate from whisper.cpp (8009) + live server (8010)
        "companion_ctx": 32768,
        "companion_ngl": 999,                  # offload all layers to the iGPU
        "companion_threads": 8,
        "companion_ready_timeout": 240,
        "companion_max_tokens": 320,
        "companion_temperature": 0.3,
        "companion_context_dir": "",           # folder of .txt/.md reference docs (+ optional system.txt)
        "companion_max_transcript_chars": 20000,
        "companion_ambient": True,             # brief note when the speaker pauses
        "companion_pause_secs": 7,
        "companion_ambient_min_chars": 140,
        "companion_ambient_cooldown": 12,
        "categories": ["MUSI", "WORK", "MISC"],
    }

    def __init__(self, settings_file: str = "settings.json"):
        """
        Initialize settings manager

        Args:
            settings_file: Path to settings JSON file
        """
        self.settings_file = settings_file
        self.settings = self._load_settings()
        self._ensure_directories()

    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from file or create with defaults"""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    loaded_settings = json.load(f)
                # Merge with defaults to handle new settings
                settings = self.DEFAULT_SETTINGS.copy()
                settings.update(loaded_settings)
                return settings
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading settings: {e}. Using defaults.")
                return self.DEFAULT_SETTINGS.copy()
        else:
            return self.DEFAULT_SETTINGS.copy()

    def save_settings(self) -> bool:
        """
        Save current settings to file

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
            return True
        except IOError as e:
            print(f"Error saving settings: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value

        Args:
            key: Setting key
            default: Default value if key doesn't exist

        Returns:
            Setting value or default
        """
        return self.settings.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a setting value

        Args:
            key: Setting key
            value: Setting value
        """
        self.settings[key] = value
        self.save_settings()

    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        output_dir = Path(self.get("output_directory"))
        recordings_dir = output_dir / "recordings"
        temp_dir = output_dir / "temp"

        for directory in [output_dir, recordings_dir, temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def get_recordings_dir(self) -> Path:
        """Get the recordings directory path"""
        return Path(self.get("output_directory")) / "recordings"

    def get_temp_dir(self) -> Path:
        """Get the temporary files directory path"""
        return Path(self.get("output_directory")) / "temp"

    def add_category(self, category: str) -> None:
        """Add a new category (uppercased, deduplicated) and persist."""
        category = category.upper().strip()
        if not category:
            return
        categories = self.settings.get("categories", [])
        if category not in categories:
            categories.append(category)
            self.settings["categories"] = categories
            self.save_settings()

    def validate(self) -> bool:
        """
        Validate current settings

        Returns:
            True if all settings are valid
        """
        # Check output directory is writable
        output_dir = Path(self.get("output_directory"))
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                return False

        # Check whisper model is valid
        valid_models = ["tiny", "base", "small", "medium", "large",
                        "large-v2", "large-v3", "large-v3-turbo", "distil-large-v3"]
        if self.get("whisper_model") not in valid_models:
            return False

        return True
