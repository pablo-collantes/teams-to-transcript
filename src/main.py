"""
Main Entry Point - Teams Call Transcriber

Usage:
  GUI mode (default):
    python main.py

  Transcribe a local file:
    python main.py /path/to/audio.m4a
    python main.py recording.mp3 --model large-v3 --diarize --num-speakers 2
    python main.py meeting.wav --output /tmp/meeting_transcript.txt
"""
import argparse
import os
import sys
import shutil

from settings_manager import SettingsManager


VALID_MODELS = [
    "tiny", "base", "small", "medium",
    "large", "large-v2", "large-v3", "large-v3-turbo", "distil-large-v3",
]


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

    if sys.platform in ("win32", "darwin"):
        try:
            import soundcard  # noqa: F401
        except Exception:
            missing.append("soundcard (pip install soundcard)")
    else:
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
        if sys.platform not in ("win32", "darwin"):
            print("\nInstall system packages (Arch Linux):")
            print("  sudo pacman -S pipewire-pulse")
        print("\n" + "!" * 60 + "\n")
        return False

    return True


def transcribe_file(audio_path: str, model_size: str, diarize: bool = False,
                     num_speakers: int = 0, language: str = "",
                     output_path: str = "") -> int:
    """Transcribe a local audio file and write the transcript to disk.

    Uses the engine from settings (`inference_engine`) - whisper.cpp on the
    GPU if configured, faster-whisper otherwise.
    """
    from file_transcriber import run_pipeline

    settings = SettingsManager()

    print(f"File:    {os.path.abspath(audio_path)}")
    print(f"Engine:  {settings.get('inference_engine')}")
    print()

    try:
        result = run_pipeline(
            audio_path, settings,
            model_size=model_size, language=language,
            diarize=diarize, num_speakers=num_speakers,
            output_path=output_path,
            progress=print,
            on_segment=lambda line: print(f"  {line}"),
        )
    except FileNotFoundError:
        print(f"ERROR: File not found: {audio_path}")
        return 1
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"\nTranscript saved: {result['output_path']}")
    return 0


def gui_main():
    """Launch the GUI application."""
    from PyQt5.QtWidgets import QApplication
    from capture_manager import CaptureManager
    from ui.main_window import MainWindow

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

    if settings.get("enable_mic", False):
        mic_source = settings.get("mic_source", "")
        print(f"Mic recording: enabled ({mic_source or 'no source selected'})")
    else:
        print(f"Mic recording: disabled")

    # Start the live transcript web server (read-only file watcher)
    if settings.get("live_server_enabled", True):
        try:
            from live_server import start_in_thread
            start_in_thread(settings)
        except Exception as e:
            print(f"WARNING: Could not start live transcript server: {e}")

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


def main():
    parser = argparse.ArgumentParser(
        description="Teams Call Transcriber — GUI or local file transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python main.py                              # Launch GUI\n"
               "  python main.py recording.m4a                # Transcribe file\n"
               "  python main.py meeting.mp3 --model large-v3 --diarize\n",
    )
    parser.add_argument("file", nargs="?", default=None,
                        help="Audio file to transcribe (m4a, mp3, wav, etc). "
                             "Omit to launch the GUI.")
    parser.add_argument("--model", "-m", default="",
                        help=f"Whisper model (default: from settings). "
                             f"Options: {', '.join(VALID_MODELS)}")
    parser.add_argument("--diarize", "-d", action="store_true",
                        help="Run speaker diarization (requires hf_token in settings)")
    parser.add_argument("--num-speakers", "-n", type=int, default=0,
                        help="Hint for number of speakers (0 = auto-detect)")
    parser.add_argument("--language", "-l", default="",
                        help="Language code (e.g. en, es). Empty = auto-detect")
    parser.add_argument("--output", "-o", default="",
                        help="Output transcript path (default: auto-generated)")

    args = parser.parse_args()

    if args.file is None:
        return gui_main()

    # Resolve model
    settings = SettingsManager()
    model_size = args.model if args.model and args.model in VALID_MODELS else settings.get("web_model", "large-v3")

    print("=" * 60)
    print("Teams Call Transcriber — Local File Mode")
    print("=" * 60 + "\n")

    return transcribe_file(
        audio_path=args.file,
        model_size=model_size,
        diarize=args.diarize,
        num_speakers=args.num_speakers,
        language=args.language,
        output_path=args.output,
    )


if __name__ == "__main__":
    sys.exit(main())
