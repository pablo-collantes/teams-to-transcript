"""
UI Styles - Application styling
"""

MAIN_WINDOW_STYLE = """
QMainWindow {
    background-color: #2b2b2b;
}

QWidget {
    color: #ffffff;
    font-family: 'Segoe UI', Arial, sans-serif;
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
    font-family: 'Consolas', 'Courier New', monospace;
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
"""
