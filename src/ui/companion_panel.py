"""
Companion panel - the in-app UI for the local-LLM meeting advisor.

Hosts a CompanionEngine and marshals its background-thread events onto the GUI
thread through Qt signals. Dropped into the main window as a right-hand dock so
the advisor sits next to the live transcript. The transcript is fed to the engine
in-process (see MainWindow._do_update_transcript) - no HTTP polling.
"""
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QLineEdit,
    QPushButton, QCheckBox,
)

from companion_engine import CompanionConfig, CompanionEngine
from .overlay_window import OverlayWindow


class CompanionPanel(QWidget):
    """Chat-style advisor panel driven by a local llama.cpp model."""

    # Signals so the engine's worker threads can safely touch widgets.
    _status_sig = pyqtSignal(str)
    _ready_sig = pyqtSignal(bool)
    _resp_start_sig = pyqtSignal(str)
    _token_sig = pyqtSignal(str)
    _resp_end_sig = pyqtSignal()
    _error_sig = pyqtSignal(str)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._started = False

        # Rendered-markdown display state. Each response is kept as raw markdown so
        # the whole document can be re-rendered via QTextEdit.setMarkdown() as tokens
        # stream in (Qt 5.14+ renders GitHub-flavored markdown natively).
        self._turns = []            # finalized responses, as markdown strings
        self._cur_header = None     # header of the in-progress response (None if idle)
        self._cur_body = ""         # accumulating raw markdown of the in-progress response
        self._overlay = None        # lazily created glanceable OverlayWindow
        self._last_line = ""        # most recent transcript line (shown in the overlay)

        self.engine = CompanionEngine(CompanionConfig(settings), self._emit)

        self._status_sig.connect(self._on_status)
        self._ready_sig.connect(self._on_ready)
        self._resp_start_sig.connect(self._on_response_start)
        self._token_sig.connect(self._on_token)
        self._resp_end_sig.connect(self._on_response_end)
        self._error_sig.connect(self._on_error)

        # Coalesce bursts of streamed tokens into ~11 fps re-renders so setMarkdown
        # isn't called on every single token.
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(90)
        self._render_timer.timeout.connect(self._render)

        self._build_ui()

    # -- engine -> Qt bridge (called from engine threads) --------------
    def _emit(self, kind: str, payload=None):
        if kind == "status":
            self._status_sig.emit(str(payload))
        elif kind == "ready":
            self._ready_sig.emit(bool(payload))
        elif kind == "response_start":
            self._resp_start_sig.emit(str(payload))
        elif kind == "token":
            self._token_sig.emit(str(payload))
        elif kind == "response_end":
            self._resp_end_sig.emit()
        elif kind == "error":
            self._error_sig.emit(str(payload))

    # -- UI ------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        self.setLayout(layout)

        header = QLabel("AI Companion")
        header.setFont(QFont("Sans Serif", 12, QFont.Bold))
        layout.addWidget(header)

        self.status_label = QLabel("Idle - starts with the app.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.status_label)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText(
            "The advisor's notes appear here.\n\n"
            "• Ask a question about the meeting or your reference docs at any time.\n"
            "• Talking points -> guidance on what's being said right now.\n"
            "• Ambient notes -> a brief nudge whenever the speaker pauses.")
        self.output.setStyleSheet("font-size: 13px;")
        layout.addWidget(self.output, 1)

        # question input
        ask_row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask the model… (Enter to send)")
        self.input.returnPressed.connect(self._on_ask_clicked)
        self.input.setEnabled(False)
        ask_row.addWidget(self.input, 1)
        self.ask_button = QPushButton("Ask")
        self.ask_button.clicked.connect(self._on_ask_clicked)
        self.ask_button.setEnabled(False)
        ask_row.addWidget(self.ask_button)
        layout.addLayout(ask_row)

        # controls
        ctrl_row = QHBoxLayout()
        self.talking_button = QPushButton("Talking points")
        self.talking_button.setToolTip("Guidance on what's being discussed right now")
        self.talking_button.clicked.connect(self._on_talking_clicked)
        self.talking_button.setEnabled(False)
        ctrl_row.addWidget(self.talking_button)

        self.ambient_check = QCheckBox("Ambient notes")
        self.ambient_check.setToolTip("Drop a brief note whenever the speaker pauses")
        self.ambient_check.setChecked(bool(self.settings.get("companion_ambient", True)))
        self.ambient_check.stateChanged.connect(self._on_ambient_toggled)
        ctrl_row.addWidget(self.ambient_check)

        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self._on_clear)
        ctrl_row.addWidget(clear_button)
        ctrl_row.addStretch(1)
        layout.addLayout(ctrl_row)

        # overlay controls (a pinned, glanceable floating mirror over Teams)
        overlay_row = QHBoxLayout()
        self.overlay_check = QCheckBox("Pin overlay")
        self.overlay_check.setToolTip(
            "Show a frameless, always-on-top window you can keep over Teams")
        self.overlay_check.stateChanged.connect(self._on_overlay_toggled)
        overlay_row.addWidget(self.overlay_check)

        self.lock_check = QCheckBox("Click-through")
        self.lock_check.setToolTip(
            "Let clicks pass through the overlay (true pass-through on X11; on Wayland "
            "it stops the overlay grabbing its own clicks). Toggle from here - the dock "
            "stays clickable so you can always undo it.")
        self.lock_check.setEnabled(False)
        self.lock_check.stateChanged.connect(self._on_lock_toggled)
        overlay_row.addWidget(self.lock_check)
        overlay_row.addStretch(1)
        layout.addLayout(overlay_row)

    # -- lifecycle -----------------------------------------------------
    def start_engine(self):
        """Begin spawning/warming the model (safe to call once)."""
        if self._started:
            return
        self._started = True
        self.status_label.setText("Starting the local model…")
        self.engine.start()

    def stop_engine(self):
        self.engine.stop()
        if self._overlay is not None:
            self._overlay.close()
            self._overlay = None

    def ingest(self, text: str):
        """Feed a new transcript line to the engine (from the capture manager)."""
        self.engine.ingest(text)
        t = (text or "").strip()
        if t:
            self._last_line = t
            if self._overlay_visible():
                self._overlay.set_transcript_line(t)

    def reset(self, name: str = ""):
        self.engine.reset(name)

    # -- button handlers ----------------------------------------------
    def _controls_enabled(self, on: bool):
        self.input.setEnabled(on)
        self.ask_button.setEnabled(on)
        self.talking_button.setEnabled(on)

    def _on_ask_clicked(self):
        q = self.input.text().strip()
        if not q or not self.engine.ready:
            return
        self.input.clear()
        self.engine.ask_question(q)

    def _on_talking_clicked(self):
        if self.engine.ready:
            self.engine.talking_points()

    def _on_ambient_toggled(self, state):
        enabled = state == Qt.Checked
        self.engine.set_ambient(enabled)
        self.settings.set("companion_ambient", enabled)

    # -- overlay -------------------------------------------------------
    def toggle_overlay(self, on=None):
        """Show/hide the glanceable overlay (called from the checkbox or View menu)."""
        want = (not self._overlay_visible()) if on is None else bool(on)
        if want and self._overlay is None:
            self._overlay = OverlayWindow()
        if want:
            self._overlay.set_status(self.status_label.text())
            self._overlay.set_transcript_line(self._last_line)
            self._render()          # push current content
            self._overlay.show()
        elif self._overlay is not None:
            self._overlay.hide()
        # keep the checkbox + lock control in sync without re-triggering
        self.overlay_check.blockSignals(True)
        self.overlay_check.setChecked(want)
        self.overlay_check.blockSignals(False)
        self.lock_check.setEnabled(want)
        if not want:
            self.lock_check.blockSignals(True)
            self.lock_check.setChecked(False)
            self.lock_check.blockSignals(False)

    def _overlay_visible(self) -> bool:
        return self._overlay is not None and self._overlay.isVisible()

    def _on_overlay_toggled(self, state):
        self.toggle_overlay(state == Qt.Checked)

    def _on_lock_toggled(self, state):
        if self._overlay is not None:
            self._overlay.set_click_through(state == Qt.Checked)

    # -- slots (GUI thread) -------------------------------------------
    @pyqtSlot(str)
    def _on_status(self, msg: str):
        self.status_label.setText(msg)
        if self._overlay_visible():
            self._overlay.set_status(msg)

    @pyqtSlot(bool)
    def _on_ready(self, ready: bool):
        self._controls_enabled(ready)

    @pyqtSlot(str)
    def _on_response_start(self, header: str):
        self._cur_header = header
        self._cur_body = ""
        self._render()  # show the header immediately

    @pyqtSlot(str)
    def _on_token(self, piece: str):
        self._cur_body += piece
        if not self._render_timer.isActive():
            self._render_timer.start()  # coalesced re-render

    @pyqtSlot()
    def _on_response_end(self):
        self._render_timer.stop()
        if self._cur_header is not None:
            self._turns.append(self._format_turn(self._cur_header, self._cur_body))
            self._cur_header = None
            self._cur_body = ""
        self._render()

    @pyqtSlot(str)
    def _on_error(self, msg: str):
        note = f"**⚠ {_md_escape(msg)}**"
        if self._cur_header is not None:
            # error mid-stream: fold it into the answer in progress
            self._cur_body += ("\n\n" if self._cur_body else "") + note
        else:
            self._turns.append(note)
        self.status_label.setText(msg)
        self._render_timer.stop()
        self._render()

    # -- rendering -----------------------------------------------------
    def _format_turn(self, header: str, body: str) -> str:
        return f"**{_md_escape(header)}**\n\n{body}".rstrip()

    def _render(self):
        """Re-render the whole transcript of responses as markdown."""
        parts = list(self._turns)
        if self._cur_header is not None:
            parts.append(self._format_turn(self._cur_header, self._cur_body))
        md = "\n\n---\n\n".join(p for p in parts if p.strip())

        sb = self.output.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        self.output.setMarkdown(md)
        if at_bottom:
            sb.setValue(sb.maximum())

        if self._overlay_visible():
            self._overlay.set_markdown(md)

    def _on_clear(self):
        self._render_timer.stop()
        self._turns = []
        self._cur_header = None
        self._cur_body = ""
        self.output.clear()


def _md_escape(s: str) -> str:
    """Escape markdown-significant characters in text we don't want interpreted
    (response headers built from the user's typed question, error messages)."""
    for ch in ("\\", "`", "*", "_", "[", "]", "#", "~"):
        s = s.replace(ch, "\\" + ch)
    return s
