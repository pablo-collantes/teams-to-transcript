"""
Main Entry Point - Teams Call Transcriber
"""
import sys
import shutil
from PyQt5.QtWidgets import QApplication
from settings_manager import SettingsManager
from capture_manager import CaptureManager
from ui.main_window import MainWindow


def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []

    try:
        import PyQt5
    except ImportError:
        missing.append("PyQt5")

    try:
        import faster_whisper
    except ImportError:
        missing.append("faster-whisper")

    if not shutil.which("pactl"):
        missing.append("pactl (install pipewire-pulse or pulseaudio)")

    if not shutil.which("parec"):
        missing.append("parec (install pipewire-pulse or pulseaudio)")

    if missing:
        print("\n" + "!" * 60)
        print("ERROR: Missing required dependencies!")
        print("!" * 60)
        print("\nMissing:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall Python packages:")
        print("  pip install -r requirements.txt")
        print("\nInstall system packages (Arch Linux):")
        print("  sudo pacman -S pipewire-pulse")
        print("\n" + "!" * 60 + "\n")
        return False

    return True


def main():
    """Main application entry point"""
    print("=" * 60)
    print("Teams Call Transcriber")
    print("=" * 60)

    # Check dependencies
    print("\nChecking dependencies...")
    if not check_dependencies():
        input("Press Enter to exit...")
        return 1

    # Initialize settings
    print("\nLoading settings...")
    settings = SettingsManager()

    if not settings.validate():
        print("ERROR: Invalid settings configuration")
        return 1

    print(f"Output directory: {settings.get('output_directory')}")
    print(f"Whisper model: {settings.get('whisper_model')}")

    audio_source = settings.get("audio_source")
    if audio_source:
        print(f"Audio source: {audio_source}")
    else:
        from audio_recorder import AudioRecorder
        default = AudioRecorder.get_default_monitor_source()
        print(f"Audio source: {default} (default monitor)")

    # Initialize capture manager
    print("Initializing capture manager...")
    capture_manager = CaptureManager(settings)

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Teams Call Transcriber")

    # Create and show main window
    print("Starting UI...")
    window = MainWindow(capture_manager, settings)
    window.show()

    print("Application ready!")
    print("-" * 60)

    # Run application
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
