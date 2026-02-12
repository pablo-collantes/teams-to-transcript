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
        "audio_device_index": None,  # None = default system device
        "video_fps": 24,  # Lower default FPS for better performance
        "video_resolution": "1920x1080",
        "chunk_duration": 30,  # seconds
        "enable_gpu": True,
        "capture_mode": "fullscreen",  # fullscreen or window
        "audio_sample_rate": 44100,
        "audio_channels": 2,
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

    def get_video_resolution(self) -> tuple:
        """
        Get video resolution as tuple

        Returns:
            (width, height) tuple
        """
        resolution = self.get("video_resolution", "1920x1080")
        width, height = resolution.split('x')
        return (int(width), int(height))

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
        valid_models = ["tiny", "base", "small", "medium", "large"]
        if self.get("whisper_model") not in valid_models:
            return False

        # Check FPS is reasonable
        fps = self.get("video_fps")
        if not isinstance(fps, int) or fps < 1 or fps > 60:
            return False

        return True
