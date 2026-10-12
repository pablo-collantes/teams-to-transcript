"""
Glanceable companion overlay - a frameless, translucent, always-on-top window that
mirrors the AI Companion's output so you can keep it pinned over Teams and read it
at a glance while talking.

It is deliberately dumb: it owns no engine and holds no state beyond what it shows.
The CompanionPanel pushes rendered markdown / status / the latest transcript line to
it, so the overlay and the docked panel always agree.

Wayland / Hyprland notes
------------------------
* Always-on-top: Qt.WindowStaysOnTopHint is honored by most compositors; on Hyprland
  the reliable way is a window rule (see HYPRLAND_RULES below), which also floats/pins it.
* Non-focus-stealing: Qt.Tool + WindowDoesNotAcceptFocus keeps typing focus in Teams.
* Click-through: true pass-through to *another* app is not portable on Wayland. The
  "lock" toggle sets WA_TransparentForMouseEvents (real click-through on X11; on Wayland
  it at least stops the overlay grabbing its own clicks). It is driven from the docked
  panel, which always stays clickable, so you can never lock yourself out.
"""
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QFont, QTextCursor, QTextCharFormat, QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QFrame,
)

# Paste into ~/.config/hypr/hyprland.conf for a pinned, floating, non-focus overlay.
HYPRLAND_RULES = """\
# AI Companion overlay (teams-to-transcript)
windowrulev2 = float, title:^(AI Companion Overlay)$
windowrulev2 = pin, title:^(AI Companion Overlay)$
windowrulev2 = nofocus, title:^(AI Companion Overlay)$
windowrulev2 = noblur, title:^(AI Companion Overlay)$
"""


class OverlayWindow(QWidget):
    """A pinned, glanceable mirror of the companion output."""

    WINDOW_TITLE = "AI Companion Overlay"

    def __init__(self, parent=None):
        # No parent on purpose: a parented Tool window only floats over its parent,
        # but we want it above every window (Teams included).
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setWindowTitle(self.WINDOW_TITLE)
        self.setObjectName("companionOverlay")
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setWindowOpacity(0.94)
        self.resize(430, 340)

        self._drag_offset = None
        self._build_ui()

    # -- UI ------------------------------------------------------------
    def _build_ui(self):
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        self.setLayout(outer)

        # A rounded card gives us a visible surface under the translucent top-level.
        self.card = QFrame()
        self.card.setObjectName("overlayCard")
        self.card.setStyleSheet("""
            #overlayCard {
                background: rgba(20, 22, 28, 235);
                border: 1px solid rgba(120, 130, 150, 120);
                border-radius: 12px;
            }
            QLabel { color: #cfd6e4; }
            QTextEdit {
                background: transparent; border: none; color: #e8ecf4;
                font-size: 14px;
            }
        """)
        outer.addWidget(self.card)

        lay = QVBoxLayout()
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        self.card.setLayout(lay)

        # header: title + status + a drag hint
        head = QHBoxLayout()
        title = QLabel("● AI Companion")
        title.setFont(QFont("Sans Serif", 11, QFont.Bold))
        title.setStyleSheet("color: #7fd1ff;")
        head.addWidget(title)
        head.addStretch(1)
        self.status = QLabel("")
        self.status.setStyleSheet("color: #8b94a7; font-size: 10px;")
        head.addWidget(self.status)
        lay.addLayout(head)

        # body: markdown-rendered advisor output
        self.body = QTextEdit()
        self.body.setReadOnly(True)
        self.body.setFocusPolicy(Qt.NoFocus)
        self.body.setPlaceholderText("Waiting for the model…")
        lay.addWidget(self.body, 1)

        # footer: the latest transcript line, dim
        self.footer = QLabel("")
        self.footer.setWordWrap(True)
        self.footer.setStyleSheet("color: #6f7889; font-size: 11px; font-style: italic;")
        lay.addWidget(self.footer)

    # -- content pushes (called by CompanionPanel) --------------------
    def set_markdown(self, md: str):
        sb = self.body.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        self.body.setMarkdown(md)
        self._highlight_warnings()
        if at_bottom:
            sb.setValue(sb.maximum())

    def set_status(self, text: str):
        self.status.setText(text or "")

    def set_transcript_line(self, text: str):
        t = (text or "").strip()
        if t:
            self.footer.setText("🎙 " + t[-160:])

    def _highlight_warnings(self):
        """Color each 'WARN' run (to end of its line) red so it pops at a glance."""
        doc = self.body.document()
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#ff6b6b"))
        fmt.setFontWeight(QFont.Bold)
        cur = doc.find("WARN")
        while not cur.isNull():
            cur.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
            cur.mergeCharFormat(fmt)
            cur = doc.find("WARN", cur)

    # -- click-through (driven from the dock, which stays clickable) ---
    def set_click_through(self, on: bool):
        self.setAttribute(Qt.WA_TransparentForMouseEvents, bool(on))

    # -- frameless dragging -------------------------------------------
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_offset = e.globalPos() - self.frameGeometry().topLeft()
            e.accept()

    def mouseMoveEvent(self, e):
        if self._drag_offset is not None and (e.buttons() & Qt.LeftButton):
            self.move(e.globalPos() - self._drag_offset)
            e.accept()

    def mouseReleaseEvent(self, e):
        self._drag_offset = None
