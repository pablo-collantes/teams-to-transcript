"""
Test script to diagnose faster-whisper and system audio dependencies
"""
import os
import sys
import shutil

# Suppress warnings
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

print("=" * 60)
print("Testing teams-to-transcript dependencies")
print("=" * 60)
print()

# Test 1: PulseAudio tools
print("Test 1: Checking PulseAudio tools...")
pactl_ok = shutil.which("pactl") is not None
parec_ok = shutil.which("parec") is not None

if pactl_ok:
    print("  pactl: found")
else:
    print("  pactl: NOT FOUND")

if parec_ok:
    print("  parec: found")
else:
    print("  parec: NOT FOUND")

if not pactl_ok or not parec_ok:
    print("\n  Install PulseAudio tools:")
    print("    sudo pacman -S pipewire-pulse   # (Arch Linux)")
    print("    sudo apt install pulseaudio-utils  # (Debian/Ubuntu)")
    sys.exit(1)

print()

# Test 2: List audio sources
print("Test 2: Listing PulseAudio sources...")
import subprocess
result = subprocess.run(["pactl", "list", "sources", "short"],
                       capture_output=True, text=True, timeout=5)
if result.stdout.strip():
    for line in result.stdout.strip().splitlines():
        print(f"  {line}")
else:
    print("  No sources found (is PulseAudio/PipeWire running?)")
print()

# Test 3: Import faster_whisper
print("Test 3: Importing faster_whisper...")
try:
    import faster_whisper
    print("  faster_whisper imported successfully")
    print(f"  Version: {faster_whisper.__version__ if hasattr(faster_whisper, '__version__') else 'unknown'}")
except Exception as e:
    print(f"  Failed to import faster_whisper: {e}")
    sys.exit(1)

print()

# Test 4: Import ctranslate2 (dependency)
print("Test 4: Importing ctranslate2...")
try:
    import ctranslate2
    print("  ctranslate2 imported successfully")
    print(f"  Version: {ctranslate2.__version__ if hasattr(ctranslate2, '__version__') else 'unknown'}")
except Exception as e:
    print(f"  Failed to import ctranslate2: {e}")
    print("\n  Try: pip install --upgrade ctranslate2")
    sys.exit(1)

print()

# Test 5: Create WhisperModel instance
print("Test 5: Creating WhisperModel instance...")
print("  This will download the tiny model if not already downloaded.")
print("  Please wait...")
print()

try:
    from faster_whisper import WhisperModel

    print("  Creating model with tiny/cpu/int8...")
    model = WhisperModel("tiny", device="cpu", compute_type="int8")

    print("  WhisperModel created successfully!")
    print()

    print("=" * 60)
    print("SUCCESS: All dependencies are working correctly!")
    print("=" * 60)
    print()
    print("Run the application with:")
    print("  cd src && python main.py")

except Exception as e:
    print(f"  Failed to create WhisperModel: {e}")
    print()
    import traceback
    traceback.print_exc()
    print()
    print("=" * 60)
    print("DIAGNOSIS:")
    print("=" * 60)
    print()
    print("faster-whisper or ctranslate2 is having issues.")
    print()
    print("Try these fixes:")
    print("1. Reinstall faster-whisper:")
    print("   pip uninstall faster-whisper ctranslate2 -y")
    print("   pip install faster-whisper")
    print()
    print("2. If that doesn't work, try installing specific versions:")
    print("   pip install faster-whisper==0.9.0")
    sys.exit(1)
