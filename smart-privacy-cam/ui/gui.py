"""
Author: Ethan O'Brien
Date: 4th July 2025
License: Open license, free to be redistributed

MainWindow (gui.py)
-------------------
Modern, professional GUI with intuitive design and advanced features
"""

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QCheckBox, QGroupBox, QComboBox, QButtonGroup, QRadioButton, QMenu, QMenuBar, QAction,
    QFrame, QSlider, QProgressBar, QTabWidget, QGridLayout, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap, QIcon
from typing import Optional
import sys

class ModernButton(QPushButton):
    """Custom modern button with hover effects"""
    def __init__(self, text, primary=False):
        super().__init__(text)
        self.primary = primary
        self.setMinimumHeight(40)
        self.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_style()
    
    def _update_style(self):
        if self.primary:
            self.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #4a90e2, stop:1 #357abd);
                    border: none;
                    border-radius: 6px;
                    color: white;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #5ba0f2, stop:1 #4a90e2);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #357abd, stop:1 #2d6da3);
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background: #2d3748;
                    border: 1px solid #4a5568;
                    border-radius: 6px;
                    color: #e2e8f0;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background: #4a5568;
                    border-color: #718096;
                }
                QPushButton:pressed {
                    background: #1a202c;
                }
            """)

class ModernCheckBox(QCheckBox):
    """Custom modern checkbox"""
    def __init__(self, text):
        super().__init__(text)
        self.setFont(QFont("Segoe UI", 9))
        self.setStyleSheet("""
            QCheckBox {
                color: #e2e8f0;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #4a5568;
                border-radius: 3px;
                background: #2d3748;
            }
            QCheckBox::indicator:checked {
                background: #4a90e2;
                border-color: #4a90e2;
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iOSIgdmlld0JveD0iMCAwIDEyIDkiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0xIDQuNUw0LjUgOEwxMSAxIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPgo8L3N2Zz4K);
            }
            QCheckBox::indicator:hover {
                border-color: #718096;
            }
        """)

class ModernRadioButton(QRadioButton):
    """Custom modern radio button"""
    def __init__(self, text):
        super().__init__(text)
        self.setFont(QFont("Segoe UI", 9))
        self.setStyleSheet("""
            QRadioButton {
                color: #e2e8f0;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #4a5568;
                border-radius: 8px;
                background: #2d3748;
            }
            QRadioButton::indicator:checked {
                background: #4a90e2;
                border-color: #4a90e2;
            }
            QRadioButton::indicator:checked::after {
                content: "";
                width: 6px;
                height: 6px;
                border-radius: 3px;
                background: white;
                margin: 3px;
            }
        """)

class ModernGroupBox(QGroupBox):
    """Custom modern group box"""
    def __init__(self, title):
        super().__init__(title)
        self.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.setStyleSheet("""
            QGroupBox {
                color: #e2e8f0;
                border: 1px solid #4a5568;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 8px;
                background: #2d3748;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px 0 8px;
                background: #1a202c;
                border-radius: 4px;
            }
        """)

class MainWindow(QMainWindow):
    """
    MainWindow.__init__
    -------------------
    Initialize modern, professional GUI with dark theme
    """
    def __init__(self):
        super().__init__()
        self.controller: Optional[object] = None
        self.setWindowTitle('Smart Privacy Cam - Professional Edition')
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(1000, 700)
        
        # Set dark theme
        self._set_dark_theme()
        self._init_ui()

    def _set_dark_theme(self):
        """Apply modern dark theme"""
        self.setStyleSheet("""
            QMainWindow {
                background: #1a202c;
            }
            QWidget {
                background: #1a202c;
                color: #e2e8f0;
            }
            QLabel {
                color: #e2e8f0;
            }
            QTabWidget::pane {
                border: 1px solid #4a5568;
                background: #2d3748;
            }
            QTabBar::tab {
                background: #2d3748;
                color: #e2e8f0;
                padding: 8px 16px;
                border: 1px solid #4a5568;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #4a90e2;
                color: white;
            }
            QTabBar::tab:hover {
                background: #4a5568;
            }
        """)

    def _init_ui(self):
        """Initialize modern UI layout"""
        # Create central widget with main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Video feed section (left side)
        self._create_video_section(main_layout)
        
        # Controls section (right side)
        self._create_controls_section(main_layout)
        
        # Set layout proportions
        main_layout.setStretch(0, 2)  # Video section
        main_layout.setStretch(1, 1)  # Controls section

    def _create_video_section(self, parent_layout):
        """Create video feed section"""
        video_frame = QFrame()
        video_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        video_frame.setStyleSheet("""
            QFrame {
                border: 2px solid #4a5568;
                border-radius: 12px;
                background: #2d3748;
            }
        """)
        
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(16, 16, 16, 16)
        
        # Video title
        video_title = QLabel("Live Camera Feed")
        video_title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        video_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_title.setStyleSheet("color: #4a90e2; margin-bottom: 8px;")
        video_layout.addWidget(video_title)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                background: #1a202c;
                border: 1px solid #4a5568;
                border-radius: 8px;
                color: #718096;
                font-size: 16px;
            }
        """)
        self.video_label.setText("Initializing Camera...")
        video_layout.addWidget(self.video_label)
        
        # Status bar
        self.status_bar = QLabel("System Ready")
        self.status_bar.setFont(QFont("Segoe UI", 9))
        self.status_bar.setStyleSheet("color: #a0aec0; padding: 8px; background: #2d3748; border-radius: 4px;")
        self.status_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_layout.addWidget(self.status_bar)
        
        parent_layout.addWidget(video_frame)

    def _create_controls_section(self, parent_layout):
        """Create controls section with tabs"""
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        controls_frame.setStyleSheet("""
            QFrame {
                border: 2px solid #4a5568;
                border-radius: 12px;
                background: #2d3748;
            }
        """)
        
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setContentsMargins(16, 16, 16, 16)
        
        # Create tab widget
        tab_widget = QTabWidget()
        tab_widget.setFont(QFont("Segoe UI", 10))
        
        # Main controls tab
        main_tab = self._create_main_controls_tab()
        tab_widget.addTab(main_tab, "Main Controls")
        
        # Privacy tab
        privacy_tab = self._create_privacy_tab()
        tab_widget.addTab(privacy_tab, "Privacy & Effects")
        
        # Advanced tab
        advanced_tab = self._create_advanced_tab()
        tab_widget.addTab(advanced_tab, "Advanced")
        
        controls_layout.addWidget(tab_widget)
        parent_layout.addWidget(controls_frame)

    def _create_main_controls_tab(self):
        """Create main controls tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)
        
        # Quick Actions
        quick_group = ModernGroupBox("Quick Actions")
        quick_layout = QVBoxLayout(quick_group)
        
        self.mute_override_btn = ModernButton("🎤 Mic Override: OFF", primary=True)
        self.mute_override_btn.setCheckable(True)
        self.mute_override_btn.clicked.connect(self.toggle_mute_override)
        quick_layout.addWidget(self.mute_override_btn)
        
        # Emergency Stop
        self.emergency_stop_btn = ModernButton("🛑 Emergency Stop", primary=True)
        self.emergency_stop_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e53e3e, stop:1 #c53030);
                border: none;
                border-radius: 6px;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f56565, stop:1 #e53e3e);
            }
        """)
        quick_layout.addWidget(self.emergency_stop_btn)
        
        layout.addWidget(quick_group)
        
        # Mood Detection
        mood_group = ModernGroupBox("Mood Analysis")
        mood_layout = QVBoxLayout(mood_group)
        
        self.mood_label = QLabel("Mood: Neutral")
        self.mood_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.mood_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mood_label.setStyleSheet("""
            padding: 12px;
            background: #4a5568;
            border-radius: 8px;
            border: 2px solid #718096;
        """)
        mood_layout.addWidget(self.mood_label)
        
        # Mood confidence indicator
        self.mood_confidence = QProgressBar()
        self.mood_confidence.setRange(0, 100)
        self.mood_confidence.setValue(75)
        self.mood_confidence.setStyleSheet("""
            QProgressBar {
                border: 1px solid #4a5568;
                border-radius: 4px;
                text-align: center;
                background: #2d3748;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4a90e2, stop:1 #38a169);
                border-radius: 3px;
            }
        """)
        mood_layout.addWidget(self.mood_confidence)
        
        layout.addWidget(mood_group)
        
        # System Status
        status_group = ModernGroupBox("System Status")
        status_layout = QGridLayout(status_group)
        
        status_layout.addWidget(QLabel("Camera:"), 0, 0)
        self.camera_status = QLabel("🟢 Connected")
        self.camera_status.setStyleSheet("color: #38a169;")
        status_layout.addWidget(self.camera_status, 0, 1)
        
        status_layout.addWidget(QLabel("Audio:"), 1, 0)
        self.audio_status = QLabel("🟢 Active")
        self.audio_status.setStyleSheet("color: #38a169;")
        status_layout.addWidget(self.audio_status, 1, 1)
        
        # Mic status indicator with LED-style light
        status_layout.addWidget(QLabel("Mic Status:"), 2, 0)
        self.mic_status_light = QLabel("🔴")
        self.mic_status_light.setStyleSheet("""
            QLabel {
                font-size: 16px;
                padding: 4px;
                border-radius: 8px;
                background: #2d3748;
            }
        """)
        status_layout.addWidget(self.mic_status_light, 2, 1)
        
        status_layout.addWidget(QLabel("Privacy:"), 3, 0)
        self.privacy_status = QLabel("🔴 Disabled")
        self.privacy_status.setStyleSheet("color: #e53e3e;")
        status_layout.addWidget(self.privacy_status, 3, 1)
        
        layout.addWidget(status_group)
        
        layout.addStretch()
        return tab

    def _create_privacy_tab(self):
        """Create privacy and effects tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)
        
        # Privacy Mode
        privacy_group = ModernGroupBox("Privacy Protection")
        privacy_layout = QVBoxLayout(privacy_group)
        
        self.privacy_none_radio = ModernRadioButton("None (No Privacy)")
        self.privacy_blur_radio = ModernRadioButton("Face Blur")
        self.privacy_anon_radio = ModernRadioButton("Anonymous Mode")
        self.privacy_none_radio.setChecked(True)
        
        self.privacy_mode_group = QButtonGroup()
        self.privacy_mode_group.setExclusive(True)
        self.privacy_mode_group.addButton(self.privacy_none_radio, 0)
        self.privacy_mode_group.addButton(self.privacy_blur_radio, 1)
        self.privacy_mode_group.addButton(self.privacy_anon_radio, 2)
        
        privacy_layout.addWidget(self.privacy_none_radio)
        privacy_layout.addWidget(self.privacy_blur_radio)
        privacy_layout.addWidget(self.privacy_anon_radio)
        
        layout.addWidget(privacy_group)
        
        # Fun Effects
        effects_group = ModernGroupBox("Fun Effects")
        effects_layout = QVBoxLayout(effects_group)
        
        self.mustache_checkbox = ModernCheckBox("Add Mustache")
        self.mustache_checkbox.setStyleSheet("""
            QCheckBox::indicator:checked {
                background: #9f7aea;
                border-color: #9f7aea;
            }
        """)
        effects_layout.addWidget(self.mustache_checkbox)
        
        self.glasses_checkbox = ModernCheckBox("Add Glasses")
        self.glasses_checkbox.setStyleSheet("""
            QCheckBox::indicator:checked {
                background: #ed8936;
                border-color: #ed8936;
            }
        """)
        effects_layout.addWidget(self.glasses_checkbox)
        
        self.hat_checkbox = ModernCheckBox("Add Hat")
        self.hat_checkbox.setStyleSheet("""
            QCheckBox::indicator:checked {
                background: #38a169;
                border-color: #38a169;
            }
        """)
        effects_layout.addWidget(self.hat_checkbox)
        
        layout.addWidget(effects_group)
        
        # Visual Effects
        visual_group = ModernGroupBox("Visual Effects")
        visual_layout = QVBoxLayout(visual_group)
        
        # Gamma correction slider
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("Gamma:"))
        self.gamma_slider = QSlider(Qt.Orientation.Horizontal)
        self.gamma_slider.setRange(50, 200)
        self.gamma_slider.setValue(100)
        self.gamma_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #4a5568;
                height: 8px;
                background: #2d3748;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4a90e2;
                border: 1px solid #4a90e2;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        gamma_layout.addWidget(self.gamma_slider)
        visual_layout.addLayout(gamma_layout)
        
        # Brightness slider
        brightness_layout = QHBoxLayout()
        brightness_layout.addWidget(QLabel("Brightness:"))
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(50, 200)
        self.brightness_slider.setValue(100)
        self.brightness_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #4a5568;
                height: 8px;
                background: #2d3748;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #f6ad55;
                border: 1px solid #f6ad55;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        brightness_layout.addWidget(self.brightness_slider)
        visual_layout.addLayout(brightness_layout)
        
        layout.addWidget(visual_group)
        layout.addStretch()
        return tab

    def _create_advanced_tab(self):
        """Create advanced settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)
        
        # Developer Options
        dev_group = ModernGroupBox("Developer Options")
        dev_layout = QVBoxLayout(dev_group)
        
        self.dev_mode_checkbox = ModernCheckBox("Developer Mode (Show Bounding Boxes)")
        dev_layout.addWidget(self.dev_mode_checkbox)
        
        self.debug_checkbox = ModernCheckBox("Debug Mode (Show Landmarks)")
        dev_layout.addWidget(self.debug_checkbox)
        
        self.performance_checkbox = ModernCheckBox("Performance Mode (Faster Processing)")
        dev_layout.addWidget(self.performance_checkbox)
        
        layout.addWidget(dev_group)
        
        # Camera Settings
        camera_group = ModernGroupBox("Camera Settings")
        camera_layout = QGridLayout(camera_group)
        
        camera_layout.addWidget(QLabel("Resolution:"), 0, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "1280x720", "1920x1080"])
        self.resolution_combo.setCurrentText("1280x720")  # Set default to HD
        self.resolution_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #4a5568;
                border-radius: 4px;
                padding: 6px;
                background: #2d3748;
                color: #e2e8f0;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #718096;
            }
        """)
        camera_layout.addWidget(self.resolution_combo, 0, 1)
        
        camera_layout.addWidget(QLabel("FPS:"), 1, 0)
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["15", "30", "60", "Comic Book Mode"])
        self.fps_combo.setCurrentText("60")  # Set default to 60 FPS
        self.fps_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #4a5568;
                border-radius: 4px;
                padding: 6px;
                background: #2d3748;
                color: #e2e8f0;
            }
        """)
        camera_layout.addWidget(self.fps_combo, 1, 1)
        
        layout.addWidget(camera_group)
        
        # AI Settings
        ai_group = ModernGroupBox("AI Settings")
        ai_layout = QVBoxLayout(ai_group)
        
        # Mood sensitivity slider
        mood_layout = QHBoxLayout()
        mood_layout.addWidget(QLabel("Mood Sensitivity:"))
        self.mood_sensitivity = QSlider(Qt.Orientation.Horizontal)
        self.mood_sensitivity.setRange(1, 10)
        self.mood_sensitivity.setValue(5)
        self.mood_sensitivity.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #4a5568;
                height: 8px;
                background: #2d3748;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #9f7aea;
                border: 1px solid #9f7aea;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        mood_layout.addWidget(self.mood_sensitivity)
        ai_layout.addLayout(mood_layout)
        
        # Face tracking sensitivity
        tracking_layout = QHBoxLayout()
        tracking_layout.addWidget(QLabel("Tracking Sensitivity:"))
        self.tracking_sensitivity = QSlider(Qt.Orientation.Horizontal)
        self.tracking_sensitivity.setRange(1, 10)
        self.tracking_sensitivity.setValue(7)
        self.tracking_sensitivity.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #4a5568;
                height: 8px;
                background: #2d3748;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #38a169;
                border: 1px solid #38a169;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        tracking_layout.addWidget(self.tracking_sensitivity)
        ai_layout.addLayout(tracking_layout)
        
        layout.addWidget(ai_group)
        
        # System Info
        info_group = ModernGroupBox("System Information")
        info_layout = QGridLayout(info_group)
        
        info_layout.addWidget(QLabel("Version:"), 0, 0)
        info_layout.addWidget(QLabel("1.0.0 Professional"), 0, 1)
        
        info_layout.addWidget(QLabel("Build:"), 1, 0)
        info_layout.addWidget(QLabel("2025.07.04"), 1, 1)
        
        info_layout.addWidget(QLabel("License:"), 2, 0)
        info_layout.addWidget(QLabel("Open Source"), 2, 1)
        
        layout.addWidget(info_group)
        layout.addStretch()
        return tab

    # Signal handlers
    def toggle_mute_override(self):
        if self.mute_override_btn.isChecked():
            self.mute_override_btn.setText("🎤 Mic Override: ON")
            self.mute_override_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #38a169, stop:1 #2f855a);
                    border: none;
                    border-radius: 6px;
                    color: white;
                    padding: 8px 16px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #48bb78, stop:1 #38a169);
                }
            """)
        else:
            self.mute_override_btn.setText("🎤 Mic Override: OFF")
            self.mute_override_btn._update_style()

    def privacy_mode_changed(self):
        mode_id = self.privacy_mode_group.checkedId()
        if mode_id == 0:
            self.privacy_status.setText("🔴 Disabled")
            self.privacy_status.setStyleSheet("color: #e53e3e;")
        elif mode_id == 1:
            self.privacy_status.setText("🟡 Face Blur")
            self.privacy_status.setStyleSheet("color: #d69e2e;")
        else:
            self.privacy_status.setText("🟢 Anonymous")
            self.privacy_status.setStyleSheet("color: #38a169;")

    def mustache_changed(self):
        enabled = self.mustache_checkbox.isChecked()

    def update_mic_status(self, is_muted: bool):
        """Update the mic status light indicator"""
        if is_muted:
            self.mic_status_light.setText("🔴")
            self.mic_status_light.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    padding: 4px;
                    border-radius: 8px;
                    background: #2d3748;
                    color: #e53e3e;
                }
            """)
        else:
            self.mic_status_light.setText("🟢")
            self.mic_status_light.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    padding: 4px;
                    border-radius: 8px;
                    background: #2d3748;
                    color: #38a169;
                }
            """)

    def update_mood(self, mood: str):
        self.mood_label.setText(f'Mood: {mood}')
        # Enhanced color coding
        if mood == 'Happy':
            self.mood_label.setStyleSheet("""
                padding: 12px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #38a169, stop:1 #48bb78);
                border-radius: 8px;
                border: 2px solid #2f855a;
                color: white;
                font-weight: bold;
            """)
            self.mood_confidence.setValue(95)
        elif mood == 'Sad':
            self.mood_label.setStyleSheet("""
                padding: 12px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3182ce, stop:1 #4299e1);
                border-radius: 8px;
                border: 2px solid #2c5282;
                color: white;
                font-weight: bold;
            """)
            self.mood_confidence.setValue(85)
        elif mood == 'Angry':
            self.mood_label.setStyleSheet("""
                padding: 12px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #e53e3e, stop:1 #f56565);
                border-radius: 8px;
                border: 2px solid #c53030;
                color: white;
                font-weight: bold;
            """)
            self.mood_confidence.setValue(90)
        elif mood == 'Surprised':
            self.mood_label.setStyleSheet("""
                padding: 12px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #d69e2e, stop:1 #ed8936);
                border-radius: 8px;
                border: 2px solid #b7791f;
                color: white;
                font-weight: bold;
            """)
            self.mood_confidence.setValue(80)
        elif mood == 'Confused':
            self.mood_label.setStyleSheet("""
                padding: 12px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #9f7aea, stop:1 #b794f4);
                border-radius: 8px;
                border: 2px solid #805ad5;
                color: white;
                font-weight: bold;
            """)
            self.mood_confidence.setValue(70)
        elif mood == 'Disgusted':
            self.mood_label.setStyleSheet("""
                padding: 12px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #d69e2e, stop:1 #ed8936);
                border-radius: 8px;
                border: 2px solid #b7791f;
                color: white;
                font-weight: bold;
            """)
            self.mood_confidence.setValue(75)
        elif mood.endswith('?'):  # Low confidence
            self.mood_label.setStyleSheet("""
                padding: 12px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ed8936, stop:1 #f6ad55);
                border-radius: 8px;
                border: 2px solid #dd6b20;
                color: white;
                font-weight: bold;
            """)
            self.mood_confidence.setValue(50)
        else:  # Neutral
            self.mood_label.setStyleSheet("""
                padding: 12px;
                background: #4a5568;
                border-radius: 8px;
                border: 2px solid #718096;
                color: #e2e8f0;
                font-weight: bold;
            """)
            self.mood_confidence.setValue(60)

    def closeEvent(self, event):
        if self.controller is not None:
            try:
                self.controller.cleanup()
            except AttributeError:
                pass
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 