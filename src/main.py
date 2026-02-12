"""
Main Entry Point - Teams Call Recorder with Transcription
"""
import sys
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
        import pyaudio
    except ImportError:
        missing.append("pyaudio")

    try:
        import mss
    except ImportError:
        missing.append("mss")

    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")

    try:
        import faster_whisper
    except ImportError:
        missing.append("faster-whisper")

    if missing:
        print("\n" + "!" * 60)
        print("ERROR: Missing required dependencies!")
        print("!" * 60)
        print("\nMissing packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nPlease install missing packages:")
        if "pyaudio" in missing:
            print("  pip install pyaudio")
            print("  (If this fails, see README for Windows PyAudio installation)")
        print(f"  pip install {' '.join([p for p in missing if p != 'pyaudio'])}")
        print("\n" + "!" * 60 + "\n")
        return False

    return True


def main():
    """Main application entry point"""
    print("=" * 60)
    print("Teams Call Recorder with Live Transcription")
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
    print(f"Video FPS: {settings.get('video_fps')}")

    # Initialize capture manager
    print("Initializing capture manager...")
    capture_manager = CaptureManager(settings)

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Teams Call Recorder")

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
