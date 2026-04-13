"""
Theme Module
============
Dark theme QSS stylesheet for the application.
Off-black backgrounds, orange gradient accents, white text, soft rounded corners.
"""

# --- Color palette ---
BG_DARK = "#1a1a1a"
BG_PANEL = "#242424"
BG_INPUT = "#2a2a2a"
BORDER = "#333333"
BORDER_FOCUS = "#FF9F43"

TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#B0B0B0"
TEXT_DISABLED = "#666666"

ORANGE_LIGHT = "#FF9F43"
ORANGE_DARK = "#E67E22"
ORANGE_HOVER = "#FFB366"
ORANGE_PRESSED = "#CC6A1A"

RED_ERROR = "#E74C3C"
GREEN_SUCCESS = "#2ECC71"
YELLOW_WARNING = "#F1C40F"
BLUE_INFO = "#3498DB"

RADIUS = "8px"
RADIUS_SMALL = "6px"


def get_stylesheet():
    """Return the full application QSS stylesheet."""
    return f"""
    /* ===== Global ===== */
    QWidget {{
        background-color: {BG_DARK};
        color: {TEXT_PRIMARY};
        font-family: "Segoe UI", "Arial", sans-serif;
        font-size: 13px;
    }}

    /* ===== Main Window ===== */
    QMainWindow {{
        background-color: {BG_DARK};
    }}

    /* ===== Labels ===== */
    QLabel {{
        background-color: transparent;
        color: {TEXT_PRIMARY};
    }}

    QLabel[secondary="true"] {{
        color: {TEXT_SECONDARY};
    }}

    /* ===== Buttons ===== */
    QPushButton {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 {ORANGE_LIGHT}, stop:1 {ORANGE_DARK});
        color: {TEXT_PRIMARY};
        border: none;
        border-radius: {RADIUS};
        padding: 8px 20px;
        font-weight: bold;
        font-size: 13px;
    }}

    QPushButton:hover {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 {ORANGE_HOVER}, stop:1 {ORANGE_LIGHT});
    }}

    QPushButton:pressed {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 {ORANGE_DARK}, stop:1 {ORANGE_PRESSED});
    }}

    QPushButton:disabled {{
        background: {BG_INPUT};
        color: {TEXT_DISABLED};
    }}

    QPushButton[flat="true"] {{
        background: transparent;
        color: {TEXT_SECONDARY};
        border: 1px solid {BORDER};
    }}

    QPushButton[flat="true"]:hover {{
        color: {ORANGE_LIGHT};
        border-color: {ORANGE_LIGHT};
    }}

    /* ===== Inputs ===== */
    QLineEdit, QSpinBox, QDoubleSpinBox {{
        background-color: {BG_INPUT};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SMALL};
        padding: 6px 10px;
    }}

    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {ORANGE_LIGHT};
    }}

    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
        color: {TEXT_DISABLED};
        background-color: {BG_DARK};
    }}

    /* ===== Dropdowns ===== */
    QComboBox {{
        background-color: {BG_INPUT};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SMALL};
        padding: 6px 10px;
        min-width: 100px;
    }}

    QComboBox:focus {{
        border-color: {ORANGE_LIGHT};
    }}

    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}

    QComboBox::down-arrow {{
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid {TEXT_SECONDARY};
        margin-right: 8px;
    }}

    QComboBox QAbstractItemView {{
        background-color: {BG_PANEL};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        selection-background-color: {ORANGE_DARK};
        selection-color: {TEXT_PRIMARY};
        border-radius: {RADIUS_SMALL};
    }}

    /* ===== Checkboxes ===== */
    QCheckBox {{
        background-color: transparent;
        spacing: 8px;
    }}

    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border: 1px solid {BORDER};
        border-radius: 4px;
        background-color: {BG_INPUT};
    }}

    QCheckBox::indicator:checked {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 {ORANGE_LIGHT}, stop:1 {ORANGE_DARK});
        border-color: {ORANGE_DARK};
    }}

    /* ===== Panels / Frames ===== */
    QFrame[panel="true"] {{
        background-color: {BG_PANEL};
        border: 1px solid {BORDER};
        border-radius: {RADIUS};
    }}

    /* ===== Sidebar (wizard step list) ===== */
    /* NOTE: sidebar item colors are controlled by main_window._update_sidebar_styles */
    QListWidget#sidebar {{
        background-color: {BG_PANEL};
        border: none;
        border-right: 1px solid {BORDER};
        outline: none;
        font-size: 14px;
    }}

    QListWidget#sidebar::item {{
        padding: 14px 16px;
        border: none;
    }}

    QListWidget#sidebar::item:selected {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {ORANGE_DARK}, stop:0.6 transparent);
        border-left: 3px solid {ORANGE_LIGHT};
    }}

    /* Generic QListWidget (used elsewhere, e.g. project dialog) */
    QListWidget {{
        background-color: {BG_PANEL};
        border: none;
        outline: none;
    }}

    /* ===== Tables ===== */
    QTableWidget, QTableView {{
        background-color: {BG_PANEL};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SMALL};
        gridline-color: {BORDER};
        selection-background-color: {ORANGE_DARK};
        selection-color: {TEXT_PRIMARY};
    }}

    QHeaderView::section {{
        background-color: {BG_DARK};
        color: {TEXT_SECONDARY};
        border: none;
        border-bottom: 1px solid {BORDER};
        border-right: 1px solid {BORDER};
        padding: 6px 10px;
        font-weight: bold;
    }}

    /* ===== Scroll bars ===== */
    QScrollBar:vertical {{
        background: {BG_DARK};
        width: 10px;
        border: none;
    }}

    QScrollBar::handle:vertical {{
        background: {BORDER};
        border-radius: 5px;
        min-height: 30px;
    }}

    QScrollBar::handle:vertical:hover {{
        background: {TEXT_DISABLED};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    QScrollBar:horizontal {{
        background: {BG_DARK};
        height: 10px;
        border: none;
    }}

    QScrollBar::handle:horizontal {{
        background: {BORDER};
        border-radius: 5px;
        min-width: 30px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background: {TEXT_DISABLED};
    }}

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}

    /* ===== Tab widget ===== */
    QTabWidget::pane {{
        background-color: {BG_PANEL};
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SMALL};
        top: -1px;
    }}

    QTabBar::tab {{
        background-color: {BG_DARK};
        color: {TEXT_SECONDARY};
        border: 1px solid {BORDER};
        border-bottom: none;
        border-top-left-radius: {RADIUS_SMALL};
        border-top-right-radius: {RADIUS_SMALL};
        padding: 8px 18px;
        margin-right: 2px;
    }}

    QTabBar::tab:selected {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 {ORANGE_DARK}, stop:1 {BG_PANEL});
        color: {TEXT_PRIMARY};
    }}

    QTabBar::tab:hover:!selected {{
        color: {ORANGE_LIGHT};
    }}

    /* ===== Progress bar ===== */
    QProgressBar {{
        background-color: {BG_INPUT};
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SMALL};
        text-align: center;
        color: {TEXT_PRIMARY};
        height: 22px;
    }}

    QProgressBar::chunk {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {ORANGE_DARK}, stop:1 {ORANGE_LIGHT});
        border-radius: {RADIUS_SMALL};
    }}

    /* ===== Splitter ===== */
    QSplitter::handle {{
        background-color: {BORDER};
    }}

    QSplitter::handle:horizontal {{
        width: 2px;
    }}

    QSplitter::handle:vertical {{
        height: 2px;
    }}

    /* ===== Dock widget (log panel) ===== */
    QDockWidget {{
        color: {TEXT_PRIMARY};
        titlebar-close-icon: none;
    }}

    QDockWidget::title {{
        background-color: {BG_PANEL};
        border: 1px solid {BORDER};
        padding: 6px;
        text-align: left;
    }}

    /* ===== Log text area ===== */
    QPlainTextEdit {{
        background-color: {BG_DARK};
        color: {TEXT_SECONDARY};
        border: none;
        font-family: "Consolas", "Courier New", monospace;
        font-size: 12px;
    }}

    /* ===== Group boxes ===== */
    QGroupBox {{
        background-color: {BG_PANEL};
        border: 1px solid {BORDER};
        border-radius: {RADIUS};
        margin-top: 12px;
        padding-top: 16px;
    }}

    QGroupBox::title {{
        color: {TEXT_SECONDARY};
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 8px;
        left: 12px;
    }}

    /* ===== Tool tips ===== */
    QToolTip {{
        background-color: {BG_PANEL};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 4px 8px;
    }}

    /* ===== Dialog ===== */
    QDialog {{
        background-color: {BG_DARK};
    }}

    /* ===== Menu ===== */
    QMenuBar {{
        background-color: {BG_DARK};
        color: {TEXT_PRIMARY};
        border-bottom: 1px solid {BORDER};
    }}

    QMenuBar::item:selected {{
        background-color: {ORANGE_DARK};
    }}

    QMenu {{
        background-color: {BG_PANEL};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
    }}

    QMenu::item:selected {{
        background-color: {ORANGE_DARK};
    }}

    /* ===== Message box ===== */
    QMessageBox {{
        background-color: {BG_DARK};
    }}

    QMessageBox QLabel {{
        color: {TEXT_PRIMARY};
    }}
    """
