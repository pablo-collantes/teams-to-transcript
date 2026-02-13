"""
Main Window - Primary UI for the application
"""
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QGroupBox,
    QComboBox, QFileDialog, QLineEdit, QMessageBox, QProgressDialog,
    QSlider, QSpinBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QFont
from .styles import MAIN_WINDOW_STYLE


class RecordingWorker(QThread):
    """Worker thread for starting recording without blocking UI"""

    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, capture_manager):
        super().__init__()
        self.capture_manager = capture_manager

    def run(self):
        """Run the recording start in background"""
        try:
            success = self.capture_manager.start_recording()
            if success:
                self.finished.emit(True, "Recording started successfully")
            else:
                self.finished.emit(False, "Failed to start recording")
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.finished.emit(False, error_msg)


class MainWindow(QMainWindow):
    """Main application window"""

    # Signals
    start_recording_signal = pyqtSignal()
    stop_recording_signal = pyqtSignal()

    def __init__(self, capture_manager, settings):
        super().__init__()
        self.capture_manager = capture_manager
        self.settings = settings

        # Connect callbacks
        self.capture_manager.set_status_callback(self.update_status)
        self.capture_manager.set_transcript_callback(self.update_transcript)

        # Setup UI
        self.init_ui()

        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_recording_stats)
        self.status_timer.start(1000)  # Update every second

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Teams Call Transcriber")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet(MAIN_WINDOW_STYLE)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Title
        title = QLabel("Teams Call Transcriber")
        title.setFont(QFont("Sans Serif", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Recording controls
        controls_layout = self.create_controls_section()
        layout.addLayout(controls_layout)

        # Status section
        self.status_label = QLabel("Ready to record")
        self.status_label.setObjectName("statusLabel")
        layout.addWidget(self.status_label)

        # Stats section
        self.stats_label = QLabel("")
        layout.addWidget(self.stats_label)

        # Transcript preview
        transcript_group = QGroupBox("Live Transcript Preview")
        transcript_layout = QVBoxLayout()
        transcript_group.setLayout(transcript_layout)

        self.transcript_text = QTextEdit()
        self.transcript_text.setReadOnly(True)
        self.transcript_text.setPlaceholderText("Transcripts will appear here during recording...")
        transcript_layout.addWidget(self.transcript_text)

        layout.addWidget(transcript_group)

        # Settings section
        settings_layout = self.create_settings_section()
        layout.addLayout(settings_layout)

    def create_controls_section(self) -> QHBoxLayout:
        """Create recording controls section"""
        layout = QHBoxLayout()

        # Start button
        self.start_button = QPushButton("Start Recording")
        self.start_button.setObjectName("startButton")
        self.start_button.clicked.connect(self.on_start_recording)
        layout.addWidget(self.start_button)

        # Stop button
        self.stop_button = QPushButton("Stop Recording")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.clicked.connect(self.on_stop_recording)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)

        return layout

    def create_settings_section(self) -> QVBoxLayout:
        """Create settings section"""
        layout = QVBoxLayout()

        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()
        settings_group.setLayout(settings_layout)

        # Output directory
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Directory:"))

        self.output_dir_input = QLineEdit()
        self.output_dir_input.setText(self.settings.get("output_directory"))
        self.output_dir_input.setReadOnly(True)
        output_layout.addWidget(self.output_dir_input)

        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_output_directory)
        output_layout.addWidget(browse_button)

        settings_layout.addLayout(output_layout)

        # Whisper model
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Whisper Model:"))

        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText(self.settings.get("whisper_model"))
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)

        model_layout.addWidget(QLabel("(Larger = more accurate, slower)"))
        model_layout.addStretch()

        settings_layout.addLayout(model_layout)

        # Audio Source
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Audio Source:"))

        self.source_combo = QComboBox()
        self.source_combo.setMinimumWidth(300)
        self.source_combo.currentIndexChanged.connect(self.on_source_changed)
        source_layout.addWidget(self.source_combo)

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._populate_audio_sources)
        source_layout.addWidget(refresh_button)

        source_layout.addStretch()

        settings_layout.addLayout(source_layout)

        # Populate audio sources
        self._populate_audio_sources()

        # Audio chunk duration
        chunk_layout = QVBoxLayout()
        chunk_header = QHBoxLayout()
        chunk_header.addWidget(QLabel("Audio Chunk Duration:"))

        self.chunk_spinbox = QSpinBox()
        self.chunk_spinbox.setRange(5, 120)
        self.chunk_spinbox.setSuffix(" seconds")
        self.chunk_spinbox.setValue(self.settings.get("chunk_duration", 30))
        self.chunk_spinbox.setFixedWidth(120)
        chunk_header.addWidget(self.chunk_spinbox)

        chunk_header.addStretch()
        chunk_layout.addLayout(chunk_header)

        self.chunk_slider = QSlider(Qt.Horizontal)
        self.chunk_slider.setRange(5, 120)
        self.chunk_slider.setValue(self.settings.get("chunk_duration", 30))
        self.chunk_slider.setTickPosition(QSlider.TicksBelow)
        self.chunk_slider.setTickInterval(15)
        chunk_layout.addWidget(self.chunk_slider)

        chunk_labels_layout = QHBoxLayout()
        chunk_labels_layout.addWidget(QLabel("5s - Faster preview"))
        chunk_labels_layout.addStretch()
        chunk_labels_layout.addWidget(QLabel("120s - Fewer, larger chunks"))
        chunk_layout.addLayout(chunk_labels_layout)

        chunk_hint = QLabel(
            "Controls how often audio is sent to Whisper for transcription. "
            "Shorter = faster live preview. No audio is lost between chunks."
        )
        chunk_hint.setWordWrap(True)
        chunk_hint.setStyleSheet("color: #888888; font-size: 11px;")
        chunk_layout.addWidget(chunk_hint)

        # Sync slider and spinbox
        self.chunk_slider.valueChanged.connect(self.chunk_spinbox.setValue)
        self.chunk_spinbox.valueChanged.connect(self.chunk_slider.setValue)
        self.chunk_spinbox.valueChanged.connect(self.on_chunk_duration_changed)

        settings_layout.addLayout(chunk_layout)

        layout.addWidget(settings_group)

        return layout

    def _populate_audio_sources(self):
        """Populate audio source dropdown from PulseAudio"""
        import sys
        sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
        from audio_recorder import AudioRecorder

        self.source_combo.blockSignals(True)
        self.source_combo.clear()

        # Add default option
        self.source_combo.addItem("(Default monitor source)", "")

        sources = AudioRecorder.list_audio_sources()
        for source_name, label in sources:
            self.source_combo.addItem(label, source_name)

        # Select saved source
        saved_source = self.settings.get("audio_source", "")
        if saved_source:
            idx = self.source_combo.findData(saved_source)
            if idx >= 0:
                self.source_combo.setCurrentIndex(idx)

        self.source_combo.blockSignals(False)

    def on_source_changed(self, index):
        """Handle audio source change"""
        source = self.source_combo.itemData(index)
        if source is not None:
            self.settings.set("audio_source", source)

    def on_start_recording(self):
        """Handle start recording button click"""
        self.start_button.setEnabled(False)

        # Show loading dialog
        self.progress = QProgressDialog(
            "Initializing recording...\n\n"
            "Loading Whisper model...\n"
            "This may take 5-30 seconds.\n\n"
            "Please wait...",
            None,
            0,
            0,
            self
        )
        self.progress.setWindowTitle("Starting Recording")
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setCancelButton(None)
        self.progress.setMinimumDuration(0)
        self.progress.show()

        # Create and start worker thread
        self.recording_worker = RecordingWorker(self.capture_manager)
        self.recording_worker.finished.connect(self.on_recording_started)
        self.recording_worker.start()

    @pyqtSlot(bool, str)
    def on_recording_started(self, success, message):
        """Called when recording worker finishes"""
        if hasattr(self, 'progress'):
            self.progress.close()

        if success:
            self.stop_button.setEnabled(True)
            self.transcript_text.clear()
            self.update_status("Recording started")
        else:
            self.start_button.setEnabled(True)
            self.update_status("Failed to start recording")

            QMessageBox.critical(
                self,
                "Recording Error",
                f"Failed to start recording:\n\n{message}\n\n"
                f"Check console for details."
            )

    def on_stop_recording(self):
        """Handle stop recording button click"""
        if self.capture_manager.stop_recording():
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def update_status(self, message: str):
        """Update status label"""
        self.status_label.setText(message)

    def update_transcript(self, transcript: str):
        """Update transcript preview"""
        self.transcript_text.append(transcript)
        self.transcript_text.append("")  # Add blank line

        # Auto-scroll to bottom
        scrollbar = self.transcript_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_recording_stats(self):
        """Update recording statistics"""
        if self.capture_manager.is_recording:
            status = self.capture_manager.get_recording_status()

            elapsed = status.get('elapsed_time', 0)
            minutes = int(elapsed) // 60
            seconds = int(elapsed) % 60

            stats_text = (
                f"Recording... | "
                f"Chunks: {status.get('chunks_processed', 0)} | "
                f"Time: {minutes:02d}:{seconds:02d}"
            )

            self.stats_label.setText(stats_text)
        else:
            self.stats_label.setText("")

    def browse_output_directory(self):
        """Open directory browser for output directory"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.settings.get("output_directory")
        )

        if directory:
            self.settings.set("output_directory", directory)
            self.output_dir_input.setText(directory)

    def on_model_changed(self, model: str):
        """Handle Whisper model change"""
        self.settings.set("whisper_model", model)

    def on_chunk_duration_changed(self, value: int):
        """Handle chunk duration change"""
        self.settings.set("chunk_duration", value)

    def closeEvent(self, event):
        """Handle window close event"""
        if self.capture_manager.is_recording:
            self.capture_manager.stop_recording()

        event.accept()
