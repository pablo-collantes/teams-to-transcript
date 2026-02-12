# Teams Call Recorder with Live Transcription

A Windows desktop application that records Teams calls (or any screen content) with synchronized audio and video capture, while providing near-real-time transcription using OpenAI's Whisper AI model running locally.

## Features

- **Screen & Audio Recording**: Captures full screen video and system audio
- **Live Transcription**: Real-time speech-to-text using faster-whisper
- **Non-Intrusive**: Runs in background without interfering with Teams calls
- **Fast Processing**: Transcripts available within 1-2 minutes of call completion
- **Privacy-Focused**: All processing happens locally on your machine
- **Customizable**: Configure video quality, Whisper model size, and output location

## System Requirements

- Windows 10 or 11
- Python 3.9 or higher
- Minimum 8GB RAM (16GB recommended for larger Whisper models)
- NVIDIA GPU with CUDA support (optional, for faster transcription)

## Installation

### 1. Install Python

Download and install Python 3.9+ from [python.org](https://www.python.org/downloads/)

Make sure to check "Add Python to PATH" during installation.

### 2. Clone or Download Repository

```bash
git clone <repository-url>
cd teams-to-transcript
```

### 3. Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install PyAudio (Windows)

PyAudio can be tricky on Windows. If the above fails, try:

```bash
pip install pipwin
pipwin install pyaudio
```

Or download the appropriate wheel file from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) and install:

```bash
pip install PyAudio-0.2.14-cp39-cp39-win_amd64.whl
```

### 6. (Optional) Install CUDA for GPU Acceleration

For faster transcription with GPU:

1. Install [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Install cuDNN compatible with CUDA 11.8

The application will automatically use GPU if available.

## Usage

### Starting the Application

```bash
cd src
python main.py
```

### Recording a Call

1. Launch the application
2. (Optional) Configure settings:
   - Output directory
   - Whisper model size (base is default, larger = more accurate but slower)
   - Video FPS (30 is default)
3. Click **"Start Recording"** when your Teams call begins
4. The application will:
   - Record your screen
   - Capture audio
   - Process transcription in 30-second chunks
5. Click **"Stop Recording"** when the call ends
6. Wait a moment for final processing to complete

### Viewing Results

Recorded files are saved to your configured output directory (default: `Documents/TeamRecordings/recordings/`):

- **Video file**: `YYYY-MM-DD_HH-MM-SS_teams-call.mp4`
- **Transcript**: `YYYY-MM-DD_HH-MM-SS_teams-call.txt`

## Configuration

### Whisper Models

Choose based on your hardware and accuracy needs:

- **tiny**: Fastest, lowest accuracy (~1GB RAM)
- **base**: Good balance, default (~1GB RAM) ✓ Recommended
- **small**: Better accuracy (~2GB RAM)
- **medium**: High accuracy (~5GB RAM, GPU recommended)
- **large**: Best accuracy (~10GB RAM, GPU required)

### Video Quality

- **15 FPS**: Smaller file size, suitable for static content
- **30 FPS**: Good balance (default) ✓
- **60 FPS**: Smoothest, larger files

## Audio Capture Notes

### Windows System Audio

The application captures audio from your default recording device. To record Teams audio:

**Option 1: Enable Stereo Mix (Recommended)**
1. Right-click speaker icon in taskbar → Sounds
2. Go to "Recording" tab
3. Right-click empty area → Show Disabled Devices
4. Right-click "Stereo Mix" → Enable
5. Set as default recording device

**Option 2: Use VB-Audio Virtual Cable**
1. Download [VB-CABLE](https://vb-audio.com/Cable/)
2. Install and set as default playback device
3. Configure Teams to use VB-CABLE
4. Set VB-CABLE output as recording source

## Troubleshooting

### "Failed to start audio recorder"

- Ensure you have a working microphone or Stereo Mix enabled
- Check that no other application is using the audio device
- Try different audio device in settings

### "Failed to start video recorder"

- Check disk space in output directory
- Verify you have write permissions to output directory
- Try reducing video FPS or resolution

### "Error loading Whisper model"

- First run downloads the model (may take several minutes)
- Ensure stable internet connection for initial download
- Check available disk space (~1-10GB depending on model)
- Models are cached in `~/.cache/huggingface/`

### Transcription is slow

- Use a smaller Whisper model (tiny or base)
- Install CUDA and cuDNN for GPU acceleration
- Close other resource-intensive applications

### Video and audio out of sync

- Reduce video FPS to 15 or 24
- Check CPU usage - close background applications
- Try recording to SSD instead of HDD

## Privacy & Security

- All processing happens locally on your machine
- No data is sent to external servers
- Audio and transcripts are stored only on your computer
- Always obtain consent before recording calls

## Known Limitations

- Currently captures full screen only (window-specific capture in development)
- Requires Stereo Mix or virtual audio cable for system audio
- Transcription accuracy depends on audio quality and model size
- No speaker diarization (all speakers transcribed together)

## Future Enhancements

- Automatic Teams call detection
- Speaker diarization (identify different speakers)
- Multiple output formats (SRT, VTT, JSON)
- Cloud backup integration
- Hotkey support for start/stop
- Window-specific capture

## Technical Architecture

- **UI**: PyQt5
- **Audio Capture**: PyAudio with WASAPI loopback
- **Video Capture**: mss (screen capture) + OpenCV (encoding)
- **Transcription**: faster-whisper (optimized Whisper implementation)
- **Processing**: Multi-threaded for concurrent audio/video/transcription

## License

[Your License Here]

## Support

For issues, questions, or feature requests, please open an issue on GitHub.

## Credits

- OpenAI Whisper for speech recognition
- faster-whisper by Guillaume Klein
- PyQt5 for UI framework
- mss for screen capture
- OpenCV for video encoding
