# Teams Call Transcriber

A desktop application that captures audio from Teams calls (via Chrome or teams-for-linux) and provides near-real-time transcription using OpenAI's Whisper AI model running locally. Runs on Linux and Windows.

## Features

- **System Audio Capture**: Records audio from any application — PulseAudio/PipeWire monitor sources on Linux, WASAPI loopback on Windows
- **Live Transcription**: Real-time speech-to-text using faster-whisper
- **Non-Intrusive**: Runs in background without interfering with calls
- **Privacy-Focused**: All processing happens locally on your machine
- **Customizable**: Configure Whisper model size, audio source, and output location

## System Requirements

- Linux (any distro with PulseAudio/PipeWire) or Windows 10/11
- Python 3.10 or higher
- Linux: PipeWire with PulseAudio compatibility (`pipewire-pulse`) or PulseAudio
- Minimum 8GB RAM (16GB recommended for larger Whisper models)
- NVIDIA GPU with CUDA support (optional, for faster transcription)

Platform differences on Windows: screenshots capture the full screen only (no
region select), and speaker diarization needs the optional extras below.
macOS mostly works too, but capturing *system* audio there requires a virtual
audio device (e.g. BlackHole) selected as the audio source.

## Installation

### Windows (quick start)

1. Install Python 3.10+ from [python.org](https://www.python.org/downloads/)
   — check **"Add python.exe to PATH"** during install
2. Download/clone this repository
3. Double-click `launch.bat` — the first run creates the environment and
   installs dependencies, then starts the app

### Linux

### 1. Install System Dependencies

```bash
# Arch Linux
sudo pacman -S python python-pyqt5 pipewire-pulse

# Debian/Ubuntu
sudo apt install python3 python3-pyqt5 pulseaudio-utils
```

### 2. Clone or Download Repository

```bash
git clone https://github.com/pablo-collantes/teams-to-transcript.git
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

# Optional: speaker diarization (heavy - pulls in PyTorch)
pip install -r requirements-diarization.txt
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

### Inference Engines

Live transcription supports two backends via the `inference_engine` setting:

- **`faster-whisper`** (default) — CTranslate2. CPU (int8) everywhere, or
  NVIDIA CUDA if available. No AMD GPU support.
- **`whisper-cpp`** — a local [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
  server. Its Vulkan backend runs on AMD and Intel GPUs (e.g. Radeon iGPUs),
  which CTranslate2 cannot use. Build whisper.cpp with `-DGGML_VULKAN=1`,
  download a GGML model (`models/download-ggml-model.sh large-v3`), then set:

```json
{
    "inference_engine": "whisper-cpp",
    "whisper_cpp_server": "/path/to/whisper.cpp/build/bin/whisper-server",
    "whisper_cpp_model": "/path/to/whisper.cpp/models/ggml-large-v3.bin"
}
```

The app spawns the server when recording starts and stops it afterwards.
Alternatively set `whisper_cpp_url` to use a server you manage yourself.

### Transcribing existing audio files

Already-recorded files (m4a, mp3, wav, ...) can be transcribed from the CLI
or from the **Transcribe Files** tab in the GUI:

```bash
python src/main.py ~/Downloads/recording.m4a --diarize
```

File transcription uses the same `inference_engine` as live recording, so
with `whisper-cpp` configured the file runs through the GPU (roughly 15x
realtime for large-v3 on a Radeon iGPU — an hour of audio in ~5 minutes).
Long files are decoded in independent 10-minute chunks so a whisper
repetition loop (silence/music can trigger one) can never snowball through
the rest of the file; a chunk that comes back repetitive is automatically
re-run in 60-second pieces, and any residue is collapsed with a marker
line. `whisper_cpp_file_port` (default 8012) keeps the file server
separate from a simultaneous live recording. Transcripts land in
`<output_directory>/local_transcripts/`.

Speaker diarization (`--diarize`, or the GUI checkbox) labels segments by
speaker using pyannote. One-time setup: install the extras
(`pip install -r requirements-diarization.txt`), create a free Hugging Face
token, accept the model conditions at
[pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
while logged in as that token's account, and set `"hf_token"` in
`settings.json`. If diarization fails, the transcript is still written —
just without speaker labels.

### Near-real-time streaming mode (opt-in)

By default the app transcribes in fixed non-overlapping chunks (`chunk_duration`,
30 s), so a line appears up to 30 s after it is spoken. Streaming mode replaces
that with a **growing-window LocalAgreement** scheme for near-real-time output:

- Audio is captured continuously into a per-utterance buffer.
- Every `stream_hop_seconds` the **whole growing buffer** is re-transcribed (so
  Whisper always sees full sentence context - no bare 5 s clips that hallucinate).
- Only the words that **agree between the last two transcriptions** are committed;
  the still-unstable tail is held back until the next hop confirms it. A word is
  written once, self-correcting, typically 2-4 s after it is spoken.
- A speech pause (VAD) or `stream_max_utterance_seconds` flushes the utterance.

Output is the same `[HH:MM:SS - HH:MM:SS]  text` transcript (just finer-grained),
so the Live Transcript API and everything downstream work unchanged. Enable it:

```json
{
    "streaming_mode": true,
    "inference_engine": "whisper-cpp",
    "stream_hop_seconds": 2.0,
    "stream_end_silence_seconds": 0.8,
    "stream_silence_rms": 300.0
}
```

Tuning: lower `stream_hop_seconds` (e.g. 1.0) for snappier commits at more GPU
load; raise/lower `stream_silence_rms` to match your mic (higher = more
aggressive silence detection). `stream_max_utterance_seconds` caps how long a
single monologue grows before a forced flush. GPU (whisper-cpp/Vulkan) is
strongly recommended since it re-transcribes every hop. The classic chunked path
is untouched and remains the default (`streaming_mode: false`).

### Recording sessions & screenshots

Each recording gets its own folder — `recordings/<title>/` — holding the audio,
the transcript, any screenshots, and a `session.json` manifest written the
moment recording starts. Categorizing a recording moves and renames the whole
folder (`recordings/<CATEGORY>/<Jul-10th_WORK>/`).

While recording, the **📸 Screenshot** button captures the screen (region
select via slurp, or full screen via grim — requires `grim`/`slurp`) into the
session folder. Each screenshot is stamped with the elapsed recording clock in
its filename and in the manifest (`screenshot_001_00-15-32.png`), so it can be
lined up against the transcript timestamps.

### Live Transcript API (tailnet)

The app serves the live session over HTTP so LLMs or scripts on your tailnet
can follow the meeting — text and screenshots — as it happens. A new session is
announced the instant recording starts; segments stream in as each ~30s chunk
is transcribed. It starts automatically with the GUI (`live_server_enabled`,
default true) and binds to your tailscale IP (auto-detected; override with
`live_host`) on `live_port` (default 8010). It can also run standalone:

```bash
python src/live_server.py            # watches the recordings dir, read-only
```

The API has **no authentication**: anything that can reach the port can read
the transcript and screenshots and post audio to the ingest endpoints. Keep it
on a tailnet you trust, set `live_host` to `127.0.0.1` to keep it on this
machine, or set `live_server_enabled` to `false` to turn it off.

Endpoints:

| Endpoint | Purpose |
|----------|---------|
| `GET /` | API description + current status |
| `GET /status` | Session name, recording state, segment/screenshot counts |
| `GET /transcript` | Full transcript as plain text (`?tail=N` for last N segments) |
| `GET /transcript.json` | All segments and screenshots as JSON |
| `GET /segments?since=N` | Incremental fetch; returns `next_since` cursor |
| `GET /screenshots` | Screenshot list with elapsed-clock timestamps and URLs |
| `GET /screenshots/{file}` | The screenshot image itself |
| `GET /timeline` | Segments + screenshots interleaved by recording clock — multimodal LLM context |
| `GET /stream` | SSE: `session` on recording start, `segments` per chunk, `screenshot` on capture, `status` on state change |
| `GET /record` | Web page: record on a phone or another computer, chunks stream here |
| `POST /ingest/start` | Start a remote session (JSON: `{name, chunk_seconds}`) |
| `POST /ingest/chunk?session=ID` | Body = one audio chunk (WAV, or anything ffmpeg decodes) |
| `POST /ingest/photo?session=ID` | Attach a JPEG/PNG photo, stamped on the recording clock |
| `POST /ingest/stop?session=ID` | Finalize the remote session |
| `GET /ingest/status` | Remote ingest state |

### Recording from a phone (remote ingest)

Open `/record` on any device on the tailnet and hit **Start Recording** — the
page captures the microphone and streams gapless WAV chunks to the PC, which
transcribes them (on the GPU, if `whisper-cpp` is configured) and serves the
session through all the endpoints above, live. Remote sessions are ordinary
session folders: manifest, growing `.wav`, transcript, and photos attached
from the phone camera appear in `/timeline` just like desktop screenshots.
Sessions finalize automatically if the device goes silent (idle timeout), and
the inference backend unloads after 10 minutes idle.

Phone browsers only expose the microphone over HTTPS, so front the live
server once with Tailscale's built-in TLS:

```bash
sudo tailscale serve --bg http://<tailscale-ip>:8010
# then on the phone: https://<machine>.<tailnet>.ts.net/record
```

To record a **VOIP call** from the phone, put the call on speakerphone — the
mic picks up both sides. (Android blocks apps and browsers from capturing
another app's call audio directly.) Set `ingest_enabled` to `false` to turn
the ingest API off; `whisper_cpp_ingest_port` (default 8011) keeps the remote
session's whisper-server separate from a simultaneous local recording.

### Upload web UI

`python src/web_server.py` serves a browser page for uploading audio files and
downloading their transcripts (with optional diarization) on
`http://127.0.0.1:8000`. It has no authentication either, so it only listens on
this machine by default. To reach it from another device, set `web_host` to
your tailscale IP (or `0.0.0.0` for the LAN), and only on a network you trust.
`web_port` changes the port.

### AI Companion (optional)

An optional dock next to the live transcript runs a local LLM as a silent
meeting advisor. It answers typed questions, gives **Talking points** on what
is being said right now, and can drop a short **ambient note** whenever the
speaker pauses. It reads the running transcript in-process, so nothing leaves
your machine.

It needs a llama.cpp `llama-server` binary and a GGUF chat model (a 7B-14B
instruct model works well on a GPU). Enable it in `settings.json`:

```json
{
    "companion_enabled": true,
    "companion_model": "/path/to/model.gguf",
    "companion_llama_bin": "/path/to/llama.cpp/build/bin/llama-server",
    "companion_context_dir": "/path/to/meeting-notes"
}
```

- `companion_llama_bin` can be left empty if `llama-server` is on your `PATH`.
  If a server is already answering on `companion_port` (default 8090), the
  app reuses it rather than starting its own.
- `companion_context_dir` is optional. Every `.txt` and `.md` file in it (an
  agenda, a spec, notes from the last meeting) is loaded as reference material
  the advisor grounds its answers in and cites by file name.
- A `system.txt` in that folder replaces the built-in advisor prompt. Use it
  to say who you are, what the meeting is for, and what to warn you about.
- The documents and transcript must fit in `companion_ctx` (default 32768
  tokens). The app warns when the documents alone exceed it.
- **View → Pin companion overlay** mirrors the advice in a small always-on-top
  window. On Hyprland, the window rules to make it float, pin and not steal
  focus are in `src/ui/overlay_window.py`.

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
- The live transcript API and upload web UI have no authentication (see
  their sections above)
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

## License

MIT. See [LICENSE](LICENSE).
