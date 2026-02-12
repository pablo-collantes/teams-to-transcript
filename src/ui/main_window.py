"""
Main Window - Primary UI for the application
"""
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QGroupBox,
    QComboBox, QFileDialog, QLineEdit, QMessageBox, QProgressDialog
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
        """
        Initialize main window

        Args:
            capture_manager: CaptureManager instance
            settings: SettingsManager instance
        """
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
        self.setWindowTitle("Teams Call Recorder & Transcriber")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet(MAIN_WINDOW_STYLE)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Title
        title = QLabel("Teams Call Recorder with Live Transcription")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
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
        """
        Create recording controls section

        Returns:
            QHBoxLayout with controls
        """
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
        """
        Create settings section

        Returns:
            QVBoxLayout with settings
        """
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

        # Video FPS
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Video FPS:"))

        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["15", "24", "30", "60"])
        self.fps_combo.setCurrentText(str(self.settings.get("video_fps")))
        self.fps_combo.currentTextChanged.connect(self.on_fps_changed)
        fps_layout.addWidget(self.fps_combo)

        fps_layout.addStretch()

        settings_layout.addLayout(fps_layout)

        layout.addWidget(settings_group)

        return layout

    def on_start_recording(self):
        """Handle start recording button click"""
        # Disable button to prevent double-click
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
        # Close progress dialog
        if hasattr(self, 'progress'):
            self.progress.close()

        if success:
            # Recording started successfully
            self.stop_button.setEnabled(True)
            self.transcript_text.clear()
            self.update_status("Recording started")
        else:
            # Recording failed
            self.start_button.setEnabled(True)
            self.update_status("Failed to start recording")

            # Show error dialog
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
        """
        Update status label

        Args:
            message: Status message
        """
        self.status_label.setText(message)

    def update_transcript(self, transcript: str):
        """
        Update transcript preview

        Args:
            transcript: New transcript text
        """
        self.transcript_text.append(transcript)
        self.transcript_text.append("")  # Add blank line

        # Auto-scroll to bottom
        scrollbar = self.transcript_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_recording_stats(self):
        """Update recording statistics"""
        if self.capture_manager.is_recording:
            status = self.capture_manager.get_recording_status()

            stats_text = (
                f"Recording... | "
                f"Frames: {status.get('frames_recorded', 0)} | "
                f"FPS: {status.get('actual_fps', 0):.1f} | "
                f"Chunks: {status.get('chunks_processed', 0)} | "
                f"Time: {status.get('elapsed_time', 0):.1f}s"
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
        """
        Handle Whisper model change

        Args:
            model: Selected model name
        """
        self.settings.set("whisper_model", model)

    def on_fps_changed(self, fps: str):
        """
        Handle FPS change

        Args:
            fps: Selected FPS value
        """
        self.settings.set("video_fps", int(fps))

    def closeEvent(self, event):
        """Handle window close event"""
        # Stop recording if active
        if self.capture_manager.is_recording:
            self.capture_manager.stop_recording()

        event.accept()
