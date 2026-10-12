"""
Main Window - Primary UI for the application
"""
import os
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QGroupBox,
    QComboBox, QFileDialog, QLineEdit, QMessageBox, QProgressDialog,
    QSlider, QSpinBox, QCheckBox, QTabWidget, QListWidget,
    QListWidgetItem, QProgressBar, QAbstractItemView, QDockWidget
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QFont
from .styles import MAIN_WINDOW_STYLE
from .category_dialog import CategoryDialog
from .companion_panel import CompanionPanel


VALID_MODELS = [
    "tiny", "base", "small", "medium",
    "large", "large-v2", "large-v3", "large-v3-turbo", "distil-large-v3",
]


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


class ScreenshotWorker(QThread):
    """Worker thread for screenshots (slurp blocks while the user selects a region)"""

    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, capture_manager, mode):
        super().__init__()
        self.capture_manager = capture_manager
        self.mode = mode

    def run(self):
        try:
            entry = self.capture_manager.take_screenshot(self.mode)
            if entry:
                self.finished.emit(
                    True, f"Screenshot at {entry['elapsed_hms']} ({entry['file']})")
            else:
                self.finished.emit(False, "Screenshot cancelled or failed")
        except Exception as e:
            self.finished.emit(False, f"Screenshot error: {e}")


class FileTranscriptionWorker(QThread):
    """Worker thread that transcribes a queue of local audio files."""

    progress = pyqtSignal(str)           # log/status messages
    segment = pyqtSignal(str)            # individual transcript lines
    file_started = pyqtSignal(int, str)  # index, filename
    file_finished = pyqtSignal(int, str) # index, output_path
    file_error = pyqtSignal(int, str)    # index, error message
    all_done = pyqtSignal()

    def __init__(self, file_paths, model_size, diarize, num_speakers, language,
                 output_dir, settings):
        super().__init__()
        self.file_paths = file_paths
        self.model_size = model_size
        self.diarize = diarize
        self.num_speakers = num_speakers
        self.language = language
        self.output_dir = output_dir
        self.settings = settings
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        from file_transcriber import FileTranscriber, run_pipeline

        transcriber = FileTranscriber(
            self.settings, model_size=self.model_size,
            language=self.language, progress=self.progress.emit)

        diarization_engine = None
        if self.diarize:
            from diarization_engine import DiarizationEngine
            diarization_engine = DiarizationEngine(
                hf_token=self.settings.get("hf_token", ""))

        try:
            for idx, fpath in enumerate(self.file_paths):
                if self._cancelled:
                    self.progress.emit("Cancelled.")
                    break

                fname = os.path.basename(fpath)
                self.file_started.emit(idx, fname)
                self.progress.emit(
                    f"\n--- [{idx+1}/{len(self.file_paths)}] {fname} ---")

                stem = Path(fpath).stem
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(
                    self.output_dir, f"{stem}_{timestamp}.txt")

                try:
                    result = run_pipeline(
                        fpath, self.settings,
                        model_size=self.model_size, language=self.language,
                        diarize=self.diarize, num_speakers=self.num_speakers,
                        output_path=output_path,
                        transcriber=transcriber,
                        diarization_engine=diarization_engine,
                        progress=self.progress.emit,
                        on_segment=self.segment.emit,
                        is_cancelled=lambda: self._cancelled,
                    )
                    if result["cancelled"]:
                        break
                    self.file_finished.emit(idx, result["output_path"])
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    self.file_error.emit(idx, str(exc))
                    self.progress.emit(f"ERROR: {exc}")
        finally:
            transcriber.close()

        self.all_done.emit()


class MainWindow(QMainWindow):
    """Main application window"""

    # Signals for thread-safe UI updates (callbacks come from background threads)
    _status_signal = pyqtSignal(str)
    _transcript_signal = pyqtSignal(str)

    def __init__(self, capture_manager, settings):
        super().__init__()
        self.capture_manager = capture_manager
        self.settings = settings
        self._file_worker = None
        self._screenshot_worker = None

        # Connect signals to UI update slots (ensures updates run on GUI thread)
        self._status_signal.connect(self._do_update_status)
        self._transcript_signal.connect(self._do_update_transcript)

        # Connect callbacks (these emit signals, safe from any thread)
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
        self.setGeometry(100, 100, 900, 700)
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

        # Tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # --- Live Recording tab ---
        self.recording_tab = QWidget()
        self._build_recording_tab()
        self.tabs.addTab(self.recording_tab, "Live Recording")

        # --- Local Files tab ---
        self.files_tab = QWidget()
        self._build_files_tab()
        self.tabs.addTab(self.files_tab, "Local Files")

        # --- AI Companion dock (right side, next to the transcript) ---
        self._build_companion_dock()

    def _build_companion_dock(self):
        """Add the local-LLM advisor as a dockable panel on the right, and start
        warming the model so it's ready by the time recording begins. Opt-in via
        the `companion_enabled` setting (default on)."""
        self.companion_panel = None
        if not self.settings.get("companion_enabled", True):
            return

        self.companion_panel = CompanionPanel(self.settings, parent=self)
        dock = QDockWidget("AI Companion", self)
        dock.setObjectName("companionDock")
        dock.setWidget(self.companion_panel)
        dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.companion_dock = dock

        # Toggle it from the View menu.
        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(dock.toggleViewAction())
        overlay_action = view_menu.addAction("Pin companion overlay")
        overlay_action.setCheckable(True)
        overlay_action.toggled.connect(self.companion_panel.toggle_overlay)
        # keep the menu item in sync if toggled from the dock checkbox
        self.companion_panel.overlay_check.toggled.connect(overlay_action.setChecked)

        # Wider default window so the transcript + companion sit side by side.
        self.resize(1360, 760)

        # Warm the model in the background right away (~30-60s), so it's ready
        # before the meeting starts.
        self.companion_panel.start_engine()

    # ------------------------------------------------------------------
    # Live Recording tab
    # ------------------------------------------------------------------

    def _build_recording_tab(self):
        layout = QVBoxLayout()
        self.recording_tab.setLayout(layout)

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

        # Screenshot button (enabled while recording); saves into the session
        # folder stamped with the elapsed recording clock
        self.screenshot_button = QPushButton("📸 Screenshot")
        self.screenshot_button.setObjectName("screenshotButton")
        self.screenshot_button.clicked.connect(self.on_take_screenshot)
        self.screenshot_button.setEnabled(False)
        layout.addWidget(self.screenshot_button)

        self.screenshot_mode_combo = QComboBox()
        self.screenshot_mode_combo.addItem("Select region", "region")
        self.screenshot_mode_combo.addItem("Full screen", "full")
        saved_mode = self.settings.get("screenshot_mode", "region")
        mode_idx = self.screenshot_mode_combo.findData(saved_mode)
        if mode_idx >= 0:
            self.screenshot_mode_combo.setCurrentIndex(mode_idx)
        self.screenshot_mode_combo.currentIndexChanged.connect(
            self.on_screenshot_mode_changed)
        layout.addWidget(self.screenshot_mode_combo)

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

        # Microphone input
        mic_layout = QHBoxLayout()

        self.mic_checkbox = QCheckBox("Record Microphone")
        self.mic_checkbox.setChecked(self.settings.get("enable_mic", False))
        self.mic_checkbox.toggled.connect(self.on_mic_toggled)
        mic_layout.addWidget(self.mic_checkbox)

        self.mic_combo = QComboBox()
        self.mic_combo.setMinimumWidth(300)
        self.mic_combo.setEnabled(self.mic_checkbox.isChecked())
        self.mic_combo.currentIndexChanged.connect(self.on_mic_source_changed)
        mic_layout.addWidget(self.mic_combo)

        mic_refresh_button = QPushButton("Refresh")
        mic_refresh_button.clicked.connect(self._populate_mic_sources)
        mic_layout.addWidget(mic_refresh_button)

        mic_layout.addStretch()
        settings_layout.addLayout(mic_layout)

        mic_hint = QLabel(
            "When enabled, your microphone audio is mixed with system audio "
            "so the transcription captures both sides of the conversation."
        )
        mic_hint.setWordWrap(True)
        mic_hint.setStyleSheet("color: #888888; font-size: 11px;")
        settings_layout.addWidget(mic_hint)

        # Populate mic sources
        self._populate_mic_sources()

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

    # ------------------------------------------------------------------
    # Local Files tab
    # ------------------------------------------------------------------

    def _build_files_tab(self):
        layout = QVBoxLayout()
        self.files_tab.setLayout(layout)

        # --- File selection ---
        file_group = QGroupBox("Audio Files")
        file_layout = QVBoxLayout()
        file_group.setLayout(file_layout)

        btn_row = QHBoxLayout()
        add_files_btn = QPushButton("Add Files...")
        add_files_btn.clicked.connect(self._on_add_files)
        btn_row.addWidget(add_files_btn)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._on_remove_files)
        btn_row.addWidget(remove_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._on_clear_files)
        btn_row.addWidget(clear_btn)

        btn_row.addStretch()
        file_layout.addLayout(btn_row)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_list.setMinimumHeight(100)
        file_layout.addWidget(self.file_list)

        layout.addWidget(file_group)

        # --- Options ---
        opts_group = QGroupBox("Transcription Options")
        opts_layout = QVBoxLayout()
        opts_group.setLayout(opts_layout)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.ft_model_combo = QComboBox()
        self.ft_model_combo.addItems(VALID_MODELS)
        self.ft_model_combo.setCurrentText(self.settings.get("web_model", "large-v3"))
        model_row.addWidget(self.ft_model_combo)
        model_row.addStretch()
        opts_layout.addLayout(model_row)

        diar_row = QHBoxLayout()
        self.ft_diarize_cb = QCheckBox("Speaker diarization")
        diar_row.addWidget(self.ft_diarize_cb)

        diar_row.addWidget(QLabel("Speakers:"))
        self.ft_speakers_spin = QSpinBox()
        self.ft_speakers_spin.setRange(0, 20)
        self.ft_speakers_spin.setSpecialValueText("auto")
        self.ft_speakers_spin.setFixedWidth(80)
        diar_row.addWidget(self.ft_speakers_spin)

        diar_row.addStretch()
        opts_layout.addLayout(diar_row)

        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("Language:"))
        self.ft_lang_input = QLineEdit()
        self.ft_lang_input.setPlaceholderText("auto-detect (or e.g. en, es, fr)")
        self.ft_lang_input.setMaximumWidth(250)
        lang_row.addWidget(self.ft_lang_input)
        lang_row.addStretch()
        opts_layout.addLayout(lang_row)

        layout.addWidget(opts_group)

        # --- Controls ---
        ctrl_row = QHBoxLayout()

        self.ft_transcribe_btn = QPushButton("Transcribe All")
        self.ft_transcribe_btn.setObjectName("startButton")
        self.ft_transcribe_btn.clicked.connect(self._on_start_file_transcription)
        ctrl_row.addWidget(self.ft_transcribe_btn)

        self.ft_cancel_btn = QPushButton("Cancel")
        self.ft_cancel_btn.setObjectName("stopButton")
        self.ft_cancel_btn.setEnabled(False)
        self.ft_cancel_btn.clicked.connect(self._on_cancel_file_transcription)
        ctrl_row.addWidget(self.ft_cancel_btn)

        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # --- Progress ---
        self.ft_progress_bar = QProgressBar()
        self.ft_progress_bar.setValue(0)
        layout.addWidget(self.ft_progress_bar)

        self.ft_status_label = QLabel("No files queued")
        self.ft_status_label.setObjectName("statusLabel")
        layout.addWidget(self.ft_status_label)

        # --- Transcript output ---
        out_group = QGroupBox("Transcript Output")
        out_layout = QVBoxLayout()
        out_group.setLayout(out_layout)

        self.ft_output_text = QTextEdit()
        self.ft_output_text.setReadOnly(True)
        self.ft_output_text.setPlaceholderText("Transcript output will appear here...")
        out_layout.addWidget(self.ft_output_text)

        layout.addWidget(out_group)

    # --- file list management ---

    def _on_add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files",
            str(Path.home()),
            "Audio Files (*.m4a *.mp3 *.wav *.flac *.ogg *.opus *.wma *.aac *.webm);;All Files (*)",
        )
        for f in files:
            # avoid duplicates
            existing = [self.file_list.item(i).data(Qt.UserRole)
                        for i in range(self.file_list.count())]
            if f not in existing:
                item = QListWidgetItem(os.path.basename(f))
                item.setData(Qt.UserRole, f)
                item.setToolTip(f)
                self.file_list.addItem(item)
        self._update_file_count_label()

    def _on_remove_files(self):
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))
        self._update_file_count_label()

    def _on_clear_files(self):
        self.file_list.clear()
        self._update_file_count_label()

    def _update_file_count_label(self):
        n = self.file_list.count()
        if n == 0:
            self.ft_status_label.setText("No files queued")
        else:
            self.ft_status_label.setText(f"{n} file{'s' if n != 1 else ''} queued")

    # --- transcription control ---

    def _on_start_file_transcription(self):
        if self.file_list.count() == 0:
            QMessageBox.information(self, "No Files", "Add audio files first.")
            return

        file_paths = [self.file_list.item(i).data(Qt.UserRole)
                      for i in range(self.file_list.count())]

        output_dir = os.path.join(self.settings.get("output_directory"), "local_transcripts")
        os.makedirs(output_dir, exist_ok=True)

        self.ft_transcribe_btn.setEnabled(False)
        self.ft_cancel_btn.setEnabled(True)
        self.ft_output_text.clear()
        self.ft_progress_bar.setMaximum(len(file_paths))
        self.ft_progress_bar.setValue(0)

        self._file_worker = FileTranscriptionWorker(
            file_paths=file_paths,
            model_size=self.ft_model_combo.currentText(),
            diarize=self.ft_diarize_cb.isChecked(),
            num_speakers=self.ft_speakers_spin.value(),
            language=self.ft_lang_input.text().strip(),
            output_dir=output_dir,
            settings=self.settings,
        )
        self._file_worker.progress.connect(self._ft_on_progress)
        self._file_worker.segment.connect(self._ft_on_segment)
        self._file_worker.file_started.connect(self._ft_on_file_started)
        self._file_worker.file_finished.connect(self._ft_on_file_finished)
        self._file_worker.file_error.connect(self._ft_on_file_error)
        self._file_worker.all_done.connect(self._ft_on_all_done)
        self._file_worker.start()

    def _on_cancel_file_transcription(self):
        if self._file_worker:
            self._file_worker.cancel()
            self.ft_status_label.setText("Cancelling...")

    @pyqtSlot(str)
    def _ft_on_progress(self, msg):
        self.ft_output_text.append(msg)
        sb = self.ft_output_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    @pyqtSlot(str)
    def _ft_on_segment(self, line):
        self.ft_output_text.append(line)
        sb = self.ft_output_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    @pyqtSlot(int, str)
    def _ft_on_file_started(self, idx, fname):
        self.ft_status_label.setText(f"Transcribing [{idx+1}/{self.file_list.count()}]: {fname}")
        item = self.file_list.item(idx)
        if item:
            item.setText(f">> {fname}")

    @pyqtSlot(int, str)
    def _ft_on_file_finished(self, idx, output_path):
        self.ft_progress_bar.setValue(idx + 1)
        item = self.file_list.item(idx)
        if item:
            fname = os.path.basename(item.data(Qt.UserRole))
            item.setText(f"[done] {fname}")

    @pyqtSlot(int, str)
    def _ft_on_file_error(self, idx, error):
        item = self.file_list.item(idx)
        if item:
            fname = os.path.basename(item.data(Qt.UserRole))
            item.setText(f"[ERROR] {fname}")
        self.ft_progress_bar.setValue(idx + 1)

    @pyqtSlot()
    def _ft_on_all_done(self):
        self.ft_transcribe_btn.setEnabled(True)
        self.ft_cancel_btn.setEnabled(False)
        self.ft_status_label.setText("Done")
        self._file_worker = None

    # ------------------------------------------------------------------
    # Shared / Live Recording callbacks
    # ------------------------------------------------------------------

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

    def _populate_mic_sources(self):
        """Populate mic source dropdown (input devices only, no monitors)"""
        import sys
        sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
        from audio_recorder import AudioRecorder

        self.mic_combo.blockSignals(True)
        self.mic_combo.clear()

        sources = AudioRecorder.list_mic_sources()
        if not sources:
            self.mic_combo.addItem("(No microphones found)", "")
        else:
            for source_name, label in sources:
                self.mic_combo.addItem(label, source_name)

        # Select saved source
        saved_source = self.settings.get("mic_source", "")
        if saved_source:
            idx = self.mic_combo.findData(saved_source)
            if idx >= 0:
                self.mic_combo.setCurrentIndex(idx)

        self.mic_combo.blockSignals(False)

        # Persist the currently displayed source so it's never empty when items exist
        current_source = self.mic_combo.currentData()
        if current_source:
            self.settings.set("mic_source", current_source)

    def on_mic_toggled(self, checked):
        """Handle mic checkbox toggle"""
        self.mic_combo.setEnabled(checked)
        self.settings.set("enable_mic", checked)

    def on_mic_source_changed(self, index):
        """Handle mic source change"""
        source = self.mic_combo.itemData(index)
        if source is not None:
            self.settings.set("mic_source", source)

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
            self.screenshot_button.setEnabled(True)
            self.transcript_text.clear()
            self.update_status("Recording started")
            # New meeting -> reset the companion's running transcript.
            if getattr(self, "companion_panel", None):
                self.companion_panel.reset()
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
        if not self.capture_manager.stop_recording():
            return

        self.stop_button.setEnabled(False)
        self.screenshot_button.setEnabled(False)

        # Show category dialog
        categories = self.settings.get("categories", [])
        dialog = CategoryDialog(categories, parent=self)
        if dialog.exec_() == CategoryDialog.Accepted:
            category = dialog.get_selected_category()
            self.settings.add_category(category)
            self.capture_manager.rename_and_move_files(category)

        self.start_button.setEnabled(True)

    def on_take_screenshot(self):
        """Handle screenshot button click"""
        if not self.capture_manager.is_recording:
            return

        self.screenshot_button.setEnabled(False)
        mode = self.screenshot_mode_combo.currentData() or "region"

        self._screenshot_worker = ScreenshotWorker(self.capture_manager, mode)
        self._screenshot_worker.finished.connect(self.on_screenshot_done)
        self._screenshot_worker.start()

    @pyqtSlot(bool, str)
    def on_screenshot_done(self, success, message):
        """Called when the screenshot worker finishes"""
        if self.capture_manager.is_recording:
            self.screenshot_button.setEnabled(True)
        self.update_status(message)
        if success:
            self.transcript_text.append(f"--- 📸 {message} ---")
            self.transcript_text.append("")
        self._screenshot_worker = None

    def on_screenshot_mode_changed(self, index):
        """Handle screenshot mode change"""
        mode = self.screenshot_mode_combo.itemData(index)
        if mode:
            self.settings.set("screenshot_mode", mode)

    def update_status(self, message: str):
        """Thread-safe status update — emits signal so the GUI thread does the work."""
        self._status_signal.emit(message)

    def update_transcript(self, transcript: str):
        """Thread-safe transcript update — emits signal so the GUI thread does the work."""
        self._transcript_signal.emit(transcript)

    @pyqtSlot(str)
    def _do_update_status(self, message: str):
        """Actually update the status label (runs on GUI thread)."""
        self.status_label.setText(message)

    @pyqtSlot(str)
    def _do_update_transcript(self, transcript: str):
        """Actually update the transcript widget (runs on GUI thread)."""
        self.transcript_text.append(transcript)
        self.transcript_text.append("")  # Add blank line

        # Auto-scroll to bottom
        scrollbar = self.transcript_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

        # Feed the same line to the in-app AI companion (append-only).
        if getattr(self, "companion_panel", None):
            self.companion_panel.ingest(transcript)

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

        if self._file_worker and self._file_worker.isRunning():
            self._file_worker.cancel()
            self._file_worker.wait(5000)

        # Shut down the companion's llama-server if we spawned it.
        if getattr(self, "companion_panel", None):
            self.companion_panel.stop_engine()

        event.accept()
