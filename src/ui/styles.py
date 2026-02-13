"""
UI Styles - Application styling
"""

MAIN_WINDOW_STYLE = """
QMainWindow {
    background-color: #2b2b2b;
}

QWidget {
    color: #ffffff;
    font-family: 'Noto Sans', 'DejaVu Sans', sans-serif;
    font-size: 10pt;
}

QPushButton {
    background-color: #3d3d3d;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 8px 16px;
    color: #ffffff;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #4a4a4a;
}

QPushButton:pressed {
    background-color: #2a2a2a;
}

QPushButton:disabled {
    background-color: #333333;
    color: #777777;
}

QPushButton#startButton {
    background-color: #28a745;
}

QPushButton#startButton:hover {
    background-color: #34ce57;
}

QPushButton#stopButton {
    background-color: #dc3545;
}

QPushButton#stopButton:hover {
    background-color: #e04555;
}

QLabel {
    color: #ffffff;
}

QLabel#statusLabel {
    padding: 8px;
    background-color: #3d3d3d;
    border-radius: 4px;
    border: 1px solid #555555;
}

QTextEdit {
    background-color: #1e1e1e;
    border: 1px solid #555555;
    border-radius: 4px;
    color: #d4d4d4;
    padding: 8px;
    font-family: 'DejaVu Sans Mono', 'Liberation Mono', monospace;
    font-size: 9pt;
}

QGroupBox {
    border: 1px solid #555555;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: #ffffff;
}

QComboBox {
    background-color: #3d3d3d;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 4px 8px;
    color: #ffffff;
}

QComboBox:hover {
    background-color: #4a4a4a;
}

QComboBox::drop-down {
    border: none;
}

QComboBox QAbstractItemView {
    background-color: #3d3d3d;
    color: #ffffff;
    selection-background-color: #4a4a4a;
}

QLineEdit {
    background-color: #3d3d3d;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 4px 8px;
    color: #ffffff;
}

QLineEdit:focus {
    border: 1px solid #007acc;
}

QProgressBar {
    border: 1px solid #555555;
    border-radius: 4px;
    background-color: #2b2b2b;
    text-align: center;
    color: #ffffff;
}

QProgressBar::chunk {
    background-color: #007acc;
    border-radius: 3px;
}

QSpinBox {
    background-color: #3d3d3d;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 4px 8px;
    color: #ffffff;
}

QSpinBox:focus {
    border: 1px solid #007acc;
}

QSpinBox::up-button, QSpinBox::down-button {
    background-color: #4a4a4a;
    border: none;
    width: 16px;
}

QSpinBox::up-button:hover, QSpinBox::down-button:hover {
    background-color: #555555;
}

QSpinBox::up-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid #ffffff;
    width: 0;
    height: 0;
}

QSpinBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #ffffff;
    width: 0;
    height: 0;
}

QSlider::groove:horizontal {
    border: 1px solid #555555;
    height: 6px;
    background-color: #3d3d3d;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background-color: #007acc;
    border: 1px solid #005a9e;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background-color: #1a8ad4;
}

QSlider::sub-page:horizontal {
    background-color: #007acc;
    border-radius: 3px;
}

QProgressDialog {
    background-color: #2b2b2b;
}

QProgressDialog QLabel {
    color: #ffffff;
}

QMessageBox {
    background-color: #2b2b2b;
}

QMessageBox QLabel {
    color: #ffffff;
}

QDialog {
    background-color: #2b2b2b;
    color: #ffffff;
}

QFileDialog {
    background-color: #2b2b2b;
    color: #ffffff;
}

QFileDialog QTreeView,
QFileDialog QListView {
    background-color: #1e1e1e;
    color: #ffffff;
    border: 1px solid #555555;
    selection-background-color: #007acc;
    selection-color: #ffffff;
}

QFileDialog QTreeView::item,
QFileDialog QListView::item {
    color: #ffffff;
    padding: 2px;
}

QFileDialog QTreeView::item:hover,
QFileDialog QListView::item:hover {
    background-color: #3d3d3d;
}

QFileDialog QTreeView::item:selected,
QFileDialog QListView::item:selected {
    background-color: #007acc;
    color: #ffffff;
}

QFileDialog QHeaderView::section {
    background-color: #3d3d3d;
    color: #ffffff;
    border: 1px solid #555555;
    padding: 4px;
}

QFileDialog QLineEdit {
    background-color: #3d3d3d;
    color: #ffffff;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 4px 8px;
}

QFileDialog QComboBox {
    background-color: #3d3d3d;
    color: #ffffff;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 4px 8px;
}

QFileDialog QComboBox QAbstractItemView {
    background-color: #3d3d3d;
    color: #ffffff;
    selection-background-color: #007acc;
}

QFileDialog QToolButton {
    background-color: #3d3d3d;
    border: 1px solid #555555;
    border-radius: 4px;
    color: #ffffff;
    padding: 4px;
}

QFileDialog QToolButton:hover {
    background-color: #4a4a4a;
}

QFileDialog QPushButton {
    background-color: #3d3d3d;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 6px 16px;
    color: #ffffff;
}

QFileDialog QPushButton:hover {
    background-color: #4a4a4a;
}

QFileDialog QSplitter::handle {
    background-color: #555555;
}

QFileDialog QFrame {
    background-color: #2b2b2b;
    color: #ffffff;
}

QScrollBar:vertical {
    background-color: #2b2b2b;
    width: 12px;
    border: none;
}

QScrollBar::handle:vertical {
    background-color: #555555;
    border-radius: 4px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #666666;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
    border: none;
}

QScrollBar:horizontal {
    background-color: #2b2b2b;
    height: 12px;
    border: none;
}

QScrollBar::handle:horizontal {
    background-color: #555555;
    border-radius: 4px;
    min-width: 20px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #666666;
}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0;
    border: none;
}

QMenu {
    background-color: #3d3d3d;
    color: #ffffff;
    border: 1px solid #555555;
}

QMenu::item:selected {
    background-color: #007acc;
}
"""
