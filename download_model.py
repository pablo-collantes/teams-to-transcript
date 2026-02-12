"""
Pre-download Whisper model to avoid UI freeze on first run
"""
import os
import sys

# Suppress warnings
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

def download_model(model_size="base"):
    """
    Download Whisper model in advance

    Args:
        model_size: Model size (tiny, base, small, medium, large)
    """
    print("=" * 60)
    print("Whisper Model Downloader")
    print("=" * 60)
    print()
    print(f"This will download the '{model_size}' Whisper model.")
    print("This prevents the UI from freezing on first recording.")
    print()

    model_sizes = {
        "tiny": "~75 MB",
        "base": "~140 MB",
        "small": "~465 MB",
        "medium": "~1.5 GB",
        "large": "~3 GB"
    }

    print(f"Model size: {model_sizes.get(model_size, 'Unknown')}")
    print()

    response = input(f"Download '{model_size}' model? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return

    try:
        print()
        print("Downloading model... This may take a few minutes.")
        print()

        from faster_whisper import WhisperModel

        # Determine device
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        compute_type = "int8" if device == "cpu" else "float16"

        print(f"Device: {device}")
        print(f"Compute type: {compute_type}")
        print()

        # Download/load model
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

        print()
        print("=" * 60)
        print("Model downloaded successfully!")
        print("=" * 60)
        print()
        print("You can now run the application with:")
        print("  cd src")
        print("  python main.py")
        print()

    except Exception as e:
        print()
        print("=" * 60)
        print(f"ERROR: {e}")
        print("=" * 60)
        print()
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    model = "base"
    if len(sys.argv) > 1:
        model = sys.argv[1]

    download_model(model)
