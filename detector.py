"""
MaskGuard AI - AI Detection & Inference Pipeline
Handles face localization (Haar Cascade), pre-processing (150x150),
CNN inference, confidence computation, and temporal state smoothing.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from config import MODEL_PATH, CASCADE_PATH, IMG_SIZE, CLASSES, CONFIDENCE_THRESHOLD, RECT_SCALE_FACTOR, FRAME_PERSISTENCE_LIMIT
from hardware_controller import hardware
import logging

logger = logging.getLogger("FaceMaskDetector")

class FaceMaskDetector:
    def __init__(self):
        logger.info(f"Loading Haar Cascade from: {CASCADE_PATH}")
        if not CASCADE_PATH.exists():
            raise FileNotFoundError(f"Cascade file missing at {CASCADE_PATH}")
        self.face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))

        logger.info(f"Loading Keras Model from: {MODEL_PATH}")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file missing at {MODEL_PATH}")
        self.model = load_model(str(MODEL_PATH))
        logger.info("Face Mask Detection Model loaded successfully.")

        # Dynamic configurable hyperparameters
        self.confidence_threshold = float(CONFIDENCE_THRESHOLD)
        self.persistence_limit = int(FRAME_PERSISTENCE_LIMIT)

        # Temporal persistence counters
        self.mask_streak = 0
        self.no_mask_streak = 0
        self.last_confirmed_state = None  # "Mask" or "No Mask" or None

        # Colors in BGR: Green for Mask, Red for No Mask
        self.COLOR_MASK = (46, 204, 113)     # Emerald Green
        self.COLOR_NO_MASK = (60, 60, 231)    # Coral Red
        self.COLOR_TEXT = (255, 255, 255)

    def process_frame(self, frame, auto_control_gate=True):
        """
        Process a single video frame.
        Returns:
            annotated_frame (np.ndarray): Frame with bounding boxes & labels
            detections (list): Structured info for each detected face
            active_state (dict): Summary telemetry for the current frame
        """
        if frame is None or frame.size == 0:
            return frame, [], {"status": "NO_FEED", "faces": 0}

        # Create working copies
        display_frame = frame.copy()
        h_orig, w_orig = frame.shape[:2]

        # Adaptive scale factor: downscale only large frames (e.g. 1080p) to keep high FPS,
        # but keep small/medium images (like dataset crops or static uploads) at full resolution!
        if w_orig > 900 or h_orig > 900:
            scale_factor = 2
            small_frame = cv2.resize(frame, (w_orig // scale_factor, h_orig // scale_factor))
        else:
            scale_factor = 1
            small_frame = frame

        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Detect faces with Haar Cascade
        detected_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Filter out tiny noise artifacts
        faces = [f for f in detected_faces if f[2] >= 40 and f[3] >= 40]

        # If no face localized by Haar (e.g. tight portrait crop or mask occludes nose/mouth features),
        # treat the primary portrait frame as the face candidate!
        if len(faces) == 0:
            pad_x = int(0.08 * w_orig)
            pad_y = int(0.08 * h_orig)
            face_w = w_orig - (2 * pad_x)
            face_h = h_orig - (2 * pad_y)
            faces = [[pad_x // scale_factor, pad_y // scale_factor, face_w // scale_factor, face_h // scale_factor]]

        detections = []
        has_mask_in_frame = False
        has_no_mask_in_frame = False

        for f in faces:
            # Scale coordinates back to original frame dimensions
            x, y, w, h = [v * scale_factor for v in f]

            # Ensure coordinates stay within frame bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, w_orig - x)
            h = min(h, h_orig - y)

            if w < 20 or h < 20:
                continue

            # Crop face area
            face_crop = frame[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue

            # Model expects RGB
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, IMG_SIZE)
            face_norm = face_resized.astype("float32") / 255.0
            face_tensor = np.expand_dims(face_norm, axis=0)

            # Predict
            raw_pred = float(self.model.predict(face_tensor, verbose=0)[0][0])

            # In the trained model:
            # pred <= 0.50 -> 'Mask'
            # pred >  0.50 -> 'No Mask'
            if raw_pred <= self.confidence_threshold:
                label = "Mask"
                confidence = (1.0 - raw_pred) * 100.0
                is_mask = True
                box_color = self.COLOR_MASK
                has_mask_in_frame = True
            else:
                label = "No Mask"
                confidence = raw_pred * 100.0
                is_mask = False
                box_color = self.COLOR_NO_MASK
                has_no_mask_in_frame = True

            detection_info = {
                "box": [int(x), int(y), int(w), int(h)],
                "label": label,
                "confidence": round(confidence, 1),
                "is_mask": is_mask,
                "face_crop": face_crop
            }
            detections.append(detection_info)

            # Draw stylish rounded/chamfered bounding box
            self._draw_detection_overlay(display_frame, x, y, w, h, label, confidence, box_color)

        # Temporal smoothing and gate control
        if len(detections) > 0:
            if has_no_mask_in_frame:
                self.no_mask_streak += 1
                self.mask_streak = 0
                if self.no_mask_streak >= self.persistence_limit:
                    self.last_confirmed_state = "No Mask"
                    if auto_control_gate:
                        hardware.set_gate(open_gate=False)  # Lock gate
            elif has_mask_in_frame:
                self.mask_streak += 1
                self.no_mask_streak = 0
                if self.mask_streak >= self.persistence_limit:
                    self.last_confirmed_state = "Mask"
                    if auto_control_gate:
                        hardware.set_gate(open_gate=True)   # Unlock gate
        else:
            # No face detected
            self.mask_streak = 0
            self.no_mask_streak = 0

        active_telemetry = {
            "faces_detected": len(detections),
            "confirmed_state": self.last_confirmed_state,
            "mask_streak": self.mask_streak,
            "no_mask_streak": self.no_mask_streak,
            "gate_status": hardware.get_status()["status_text"]
        }

        return display_frame, detections, active_telemetry

    def _draw_detection_overlay(self, img, x, y, w, h, label, confidence, color):
        """Draws a modern HUD bounding box with pill badge"""
        # Bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

        # Corner accents for sleek cyber/AI look
        corner_len = min(20, w // 4, h // 4)
        thickness = 4
        # Top-left
        cv2.line(img, (x, y), (x + corner_len, y), color, thickness)
        cv2.line(img, (x, y), (x, y + corner_len), color, thickness)
        # Top-right
        cv2.line(img, (x + w, y), (x + w - corner_len, y), color, thickness)
        cv2.line(img, (x + w, y), (x + w, y + corner_len), color, thickness)
        # Bottom-left
        cv2.line(img, (x, y + h), (x + corner_len, y + h), color, thickness)
        cv2.line(img, (x, y + h), (x, y + h - corner_len), color, thickness)
        # Bottom-right
        cv2.line(img, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
        cv2.line(img, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)

        # Header Badge
        badge_text = f" {label.upper()} {confidence:.1f}% "
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thick = 2
        (text_w, text_h), baseline = cv2.getTextSize(badge_text, font, font_scale, font_thick)

        if y - text_h - 10 < 0:
            badge_y1 = y
            badge_y2 = y + text_h + 10
            text_y = y + text_h + 3
        else:
            badge_y1 = y - text_h - 10
            badge_y2 = y
            text_y = y - 5

        cv2.rectangle(img, (x, badge_y1), (x + text_w + 10, badge_y2), color, -1)
        cv2.putText(img, badge_text, (x + 5, text_y), font, font_scale, self.COLOR_TEXT, font_thick, cv2.LINE_AA)
