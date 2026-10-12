"""
Category Dialog - Modal dialog for selecting/creating a recording category
"""
import re
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QLineEdit, QPushButton,
)
from PyQt5.QtCore import Qt
from .styles import MAIN_WINDOW_STYLE


class CategoryDialog(QDialog):
    """Dialog to pick or create a category when stopping a recording."""

    def __init__(self, categories: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Category")
        self.setModal(True)
        self.setFixedWidth(360)
        self.setStyleSheet(MAIN_WINDOW_STYLE)

        self._selected_category = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Category dropdown
        layout.addWidget(QLabel("Category:"))
        self.combo = QComboBox()
        self.combo.addItems(sorted(categories))
        layout.addWidget(self.combo)

        # New category row
        layout.addWidget(QLabel("Or create new:"))
        new_row = QHBoxLayout()

        self.new_input = QLineEdit()
        self.new_input.setPlaceholderText("e.g. CSCI")
        self.new_input.setMaxLength(20)
        new_row.addWidget(self.new_input)

        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._on_add_category)
        new_row.addWidget(add_btn)

        layout.addLayout(new_row)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #dc3545; font-size: 9pt;")
        layout.addWidget(self.error_label)

        # Save / Cancel
        btn_row = QHBoxLayout()

        save_btn = QPushButton("Save")
        save_btn.setObjectName("startButton")
        save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(save_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        layout.addLayout(btn_row)

    def _on_add_category(self):
        """Validate and add a new category to the dropdown."""
        text = self.new_input.text().strip().upper()
        if not text:
            return
        if not re.match(r'^[A-Z0-9]+$', text):
            self.error_label.setText("Only letters and numbers allowed.")
            return
        self.error_label.setText("")
        # Add if not already present
        if self.combo.findText(text) == -1:
            self.combo.addItem(text)
        self.combo.setCurrentText(text)
        self.new_input.clear()

    def _on_save(self):
        """Accept with the currently selected category."""
        if self.combo.currentText():
            self._selected_category = self.combo.currentText()
            self.accept()

    def get_selected_category(self) -> str | None:
        """Return the selected category, or None if cancelled."""
        return self._selected_category
