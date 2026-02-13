# Teams Call Transcriber

A Linux desktop application that captures audio from Teams calls (via Chrome or teams-for-linux) and provides near-real-time transcription using OpenAI's Whisper AI model running locally.

## Features

- **System Audio Capture**: Records audio from any application via PulseAudio/PipeWire monitor sources
- **Live Transcription**: Real-time speech-to-text using faster-whisper
- **Non-Intrusive**: Runs in background without interfering with calls
- **Privacy-Focused**: All processing happens locally on your machine
- **Customizable**: Configure Whisper model size, audio source, and output location

## System Requirements

- Arch Linux (or any Linux distro with PulseAudio/PipeWire)
- Python 3.9 or higher
- PipeWire with PulseAudio compatibility (`pipewire-pulse`) or PulseAudio
- Minimum 8GB RAM (16GB recommended for larger Whisper models)
- NVIDIA GPU with CUDA support (optional, for faster transcription)

## Installation

### 1. Install System Dependencies

```bash
# Arch Linux
sudo pacman -S python python-pyqt5 pipewire-pulse

# Debian/Ubuntu
sudo apt install python3 python3-pyqt5 pulseaudio-utils
```

### 2. Clone or Download Repository

```bash
git clone <repository-url>
cd teams-to-transcript
```

### 3. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. (Optional) Pre-download Whisper Model

```bash
python download_model.py
```

### 6. (Optional) Install CUDA for GPU Acceleration

For faster transcription with an NVIDIA GPU, install CUDA and cuDNN. The application will automatically use GPU if available.

## Usage

### Starting the Application

```bash
cd src
python main.py
```

### Recording a Call

1. Launch the application
2. Select your audio source from the dropdown (monitor sources capture system output)
3. Click **"Start Recording"** when your call begins
4. The application will capture audio and process transcription in 30-second chunks
5. Click **"Stop Recording"** when the call ends

### Viewing Results

Recorded files are saved to your configured output directory (default: `~/Documents/TeamRecordings/recordings/`):

- **Audio file**: `YYYY-MM-DD_HH-MM-SS_teams-call.wav`
- **Transcript**: `YYYY-MM-DD_HH-MM-SS_teams-call.txt`

## Audio Source Setup

The application captures audio using PulseAudio monitor sources. A **monitor source** mirrors the audio output of a sink (speaker/headphones), letting you record what you hear.

### How It Works

When you select a monitor source (e.g. `alsa_output.pci-0000_00_1f.3.analog-stereo.monitor`), the application uses `parec` to capture the raw PCM audio stream from that sink.

### Finding the Right Source

1. Use the **Audio Source** dropdown in the app (click **Refresh** to update)
2. Sources ending in `.monitor` capture system audio output
3. The default monitor source is auto-detected from your default audio sink
4. If using Teams in Chrome, the audio goes through your default sink — the default monitor source will capture it

### Verifying Audio Sources

```bash
# List all sources
pactl list sources short

# Check default sink
pactl get-default-sink
```

## Configuration

### Whisper Models

Choose based on your hardware and accuracy needs:

| Model | RAM | Speed | Accuracy |
|-------|-----|-------|----------|
| tiny | ~1GB | Fastest | Lowest |
| base | ~1GB | Fast | Good (recommended) |
| small | ~2GB | Moderate | Better |
| medium | ~5GB | Slow (GPU recommended) | High |
| large | ~10GB | Slowest (GPU required) | Best |

## Troubleshooting

### "pactl/parec not found"

Install PulseAudio tools:
```bash
sudo pacman -S pipewire-pulse   # Arch Linux
```

### "No audio sources found"

Ensure PipeWire/PulseAudio is running:
```bash
systemctl --user status pipewire-pulse
```

### "Failed to start audio recorder"

- Check that the selected audio source exists: `pactl list sources short`
- Ensure no permission issues with PulseAudio

### Transcription is slow

- Use a smaller Whisper model (tiny or base)
- Install CUDA and cuDNN for GPU acceleration

### Run diagnostics

```bash
python test_whisper.py
```

## Privacy & Security

- All processing happens locally on your machine
- No data is sent to external servers
- Audio and transcripts are stored only on your computer
- Always obtain consent before recording calls

## Technical Architecture

- **UI**: PyQt5
- **Audio Capture**: `parec` (PulseAudio recording utility) via subprocess
- **Transcription**: faster-whisper (optimized Whisper implementation)
- **Processing**: Multi-threaded for concurrent audio capture and transcription

## Credits

- OpenAI Whisper for speech recognition
- faster-whisper by Guillaume Klein
- PyQt5 for UI framework
