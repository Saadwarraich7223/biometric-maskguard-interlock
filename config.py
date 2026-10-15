"""
MaskGuard AI - Configuration Module
Dynamically resolves all project paths and system settings.
No hardcoded drive letters or absolute paths.
"""

import os
from pathlib import Path

# Base workspace directory
BASE_DIR = Path(__file__).resolve().parent

# Model & Haar Cascade paths
MODEL_PATH = BASE_DIR / "Face Mask Detection Model.h5"
CASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"
DATASET_PATH = BASE_DIR / "Face Dataset"

# Static & Database paths
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
VIOLATIONS_DIR = STATIC_DIR / "violations"
DB_PATH = BASE_DIR / "maskguard.db"

# Ensure runtime directories exist
STATIC_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
VIOLATIONS_DIR.mkdir(exist_ok=True)

# Detection & Model Hyperparameters
IMG_SIZE = (150, 150)
CLASSES = ["Mask", "No Mask"]
CONFIDENCE_THRESHOLD = 0.50
RECT_SCALE_FACTOR = 4  # Downscale factor for high FPS face detection
FRAME_PERSISTENCE_LIMIT = 5  # Minimum consecutive frames before state change

# IoT / Access Control Settings
DEFAULT_COM_PORT = "COM9"
DEFAULT_RELAY_PIN = 13
VIRTUAL_HARDWARE_MODE = True  # Auto-falls back if physical hardware is absent
