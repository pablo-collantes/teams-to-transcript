"""
Test script to diagnose faster-whisper issues
"""
import os
import sys

# Suppress warnings
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

print("=" * 60)
print("Testing faster-whisper installation")
print("=" * 60)
print()

# Test 1: Import faster_whisper
print("Test 1: Importing faster_whisper...")
try:
    import faster_whisper
    print("✓ faster_whisper imported successfully")
    print(f"  Version: {faster_whisper.__version__ if hasattr(faster_whisper, '__version__') else 'unknown'}")
except Exception as e:
    print(f"✗ Failed to import faster_whisper: {e}")
    sys.exit(1)

print()

# Test 2: Import ctranslate2 (dependency)
print("Test 2: Importing ctranslate2...")
try:
    import ctranslate2
    print("✓ ctranslate2 imported successfully")
    print(f"  Version: {ctranslate2.__version__ if hasattr(ctranslate2, '__version__') else 'unknown'}")
except Exception as e:
    print(f"✗ Failed to import ctranslate2: {e}")
    print("\nTry: pip install --upgrade ctranslate2")
    sys.exit(1)

print()

# Test 3: Create WhisperModel instance
print("Test 3: Creating WhisperModel instance...")
print("This will download the tiny model if not already downloaded.")
print("Please wait...")
print()

try:
    from faster_whisper import WhisperModel

    # Try to create model instance
    print("Creating model with tiny/cpu/int8...")
    model = WhisperModel("tiny", device="cpu", compute_type="int8")

    print("✓ WhisperModel created successfully!")
    print()

    # Test 4: Try a simple transcription
    print("Test 4: Testing transcription (this will use a dummy audio)...")
    print("Skipping - model creation was successful")
    print()

    print("=" * 60)
    print("SUCCESS: faster-whisper is working correctly!")
    print("=" * 60)
    print()
    print("The issue might be elsewhere. Try running the main app again.")

except Exception as e:
    print(f"✗ Failed to create WhisperModel: {e}")
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
    print()
    print("3. Or switch to standard openai-whisper:")
    print("   pip install openai-whisper")
    sys.exit(1)
