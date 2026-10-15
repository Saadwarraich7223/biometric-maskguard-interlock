"""
MaskGuard AI - Main Flask Web Application Server
Serves live video surveillance, access control telemetry,
incident logging, analytics, and diagnostics.
"""

import os
import cv2
import time
import json
import base64
from flask import Flask, render_template, Response, jsonify, request, send_from_directory, send_file
from config import BASE_DIR, STATIC_DIR, VIOLATIONS_DIR, UPLOADS_DIR
from detector import FaceMaskDetector
from hardware_controller import hardware
from database import db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MaskGuardApp")

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(STATIC_DIR))
app.config["SECRET_KEY"] = "maskguard-secret-fyp-2026"

# Initialize AI Detector
detector = FaceMaskDetector()

# Global camera / feed capture
class VideoCamera:
    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.warning(f"Could not open camera {source}. Synthetic feed will be generated if unavailable.")
            self.has_real_feed = False
        else:
            self.has_real_feed = True
            logger.info(f"Camera opened successfully on source {source}.")

    def get_frame(self):
        if self.has_real_feed and self.cap and self.cap.isOpened():
            success, frame = self.cap.read()
            if success and frame is not None and frame.size > 0:
                frame = cv2.flip(frame, 1)
                return frame
            else:
                self.has_real_feed = False
        return self._generate_synthetic_feed()

    def _generate_synthetic_feed(self):
        """Generates a clean synthetic surveillance stream if camera hardware is locked or offline"""
        img = 20 * np.ones((480, 640, 3), dtype=np.uint8)
        # Grid lines
        for y in range(0, 480, 40):
            cv2.line(img, (0, y), (640, y), (35, 35, 35), 1)
        for x in range(0, 640, 40):
            cv2.line(img, (x, 0), (x, 480), (35, 35, 35), 1)

        t = time.time()
        # Animated radar scan line
        scan_y = int((t * 120) % 480)
        cv2.line(img, (0, scan_y), (640, scan_y), (0, 200, 255), 2)

        cv2.putText(img, "SIMULATED SURVEILLANCE FEED", (140, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        cv2.putText(img, "Stand in front of active webcam or connect USB camera", (100, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
        cv2.putText(img, time.strftime("%Y-%m-%d %H:%M:%S"), (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        return img

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

import numpy as np
camera = VideoCamera(0)

# Global runtime telemetry
live_telemetry = {
    "total_faces": 0,
    "last_label": "Scanning",
    "confidence": 0.0,
    "gate_status": "LOCKED",
    "alert_triggered": False,
    "fps": 0
}

def generate_video_stream():
    """Generator function for streaming video frames via MJPEG"""
    prev_time = time.time()

    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.03)
            continue

        # AI Detection Pipeline
        annotated_frame, detections, telemetry = detector.process_frame(frame, auto_control_gate=True)

        # FPS computation
        now = time.time()
        fps = int(1.0 / (now - prev_time)) if (now - prev_time) > 0 else 30
        prev_time = now

        # Draw HUD overlays on frame
        gate_state = hardware.get_status()["status_text"]
        gate_color = (46, 204, 113) if gate_state == "UNLOCKED" else (60, 60, 231)

        cv2.putText(annotated_frame, f"FPS: {fps}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
        cv2.putText(annotated_frame, f"GATE: {gate_state}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, gate_color, 2)

        # Process detections & log violations
        alert_triggered = False
        for det in detections:
            label = det["label"]
            conf = det["confidence"]
            face_crop = det.get("face_crop")

            # Update live telemetry
            live_telemetry["total_faces"] = len(detections)
            live_telemetry["last_label"] = label
            live_telemetry["confidence"] = conf
            live_telemetry["fps"] = fps

            # If persistent violation occurs, log snapshot to database
            if label == "No Mask" and telemetry.get("no_mask_streak", 0) >= 5:
                alert_triggered = True
                db.log_incident(
                    label="No Mask",
                    confidence=conf,
                    full_frame=frame,
                    face_crop=face_crop,
                    gate_status="LOCKED",
                    camera_id="CAM-01-ENTRANCE"
                )
                db.record_scan_count(is_mask=False)

            elif label == "Mask" and telemetry.get("mask_streak", 0) >= 5:
                db.record_scan_count(is_mask=True)

        live_telemetry["alert_triggered"] = alert_triggered
        live_telemetry["gate_status"] = gate_state

        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ---------------- WEB ROUTES ---------------- #

@app.route("/")
def index():
    """Live Surveillance & Access Control Monitor"""
    stats = db.get_summary_stats()
    gate_telemetry = hardware.get_status()
    recent_incidents = db.get_recent_incidents(limit=5)
    return render_template("index.html", stats=stats, gate=gate_telemetry, incidents=recent_incidents)

@app.route("/video_feed")
def video_feed():
    """MJPEG Live Streaming Endpoint"""
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/telemetry")
def api_telemetry():
    """Returns live JSON metrics for real-time HUD updates"""
    stats = db.get_summary_stats()
    gate = hardware.get_status()
    return jsonify({
        "live": live_telemetry,
        "stats": stats,
        "gate": gate
    })

@app.route("/api/camera_status")
def api_camera_status():
    """Returns whether a physical server camera feed is active"""
    return jsonify({
        "has_real_feed": camera.has_real_feed,
        "source": camera.source
    })

@app.route("/api/process_client_frame", methods=["POST"])
def process_client_frame():
    """Processes a frame streamed from the user's laptop/browser webcam"""
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    image_data = data["image"]
    if "," in image_data:
        image_data = image_data.split(",")[1]

    try:
        decoded_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(decoded_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Could not decode frame"}), 400

        # AI Detection Pipeline
        annotated_frame, detections, telemetry = detector.process_frame(frame, auto_control_gate=True)
        gate_state = hardware.get_status()["status_text"]

        alert_triggered = False
        for det in detections:
            label = det["label"]
            conf = det["confidence"]
            face_crop = det.get("face_crop")

            if label == "No Mask" and telemetry.get("no_mask_streak", 0) >= 5:
                alert_triggered = True
                db.log_incident(
                    label="No Mask",
                    confidence=conf,
                    full_frame=frame,
                    face_crop=face_crop,
                    gate_status="LOCKED",
                    camera_id="LAPTOP-WEBCAM"
                )
                db.record_scan_count(is_mask=False)
            elif label == "Mask" and telemetry.get("mask_streak", 0) >= 5:
                db.record_scan_count(is_mask=True)

        clean_detections = []
        for d in detections:
            clean_detections.append({
                "box": d["box"],
                "label": d["label"],
                "confidence": d["confidence"],
                "is_mask": d["is_mask"]
            })

        ret, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        encoded_annotated = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "success": True,
            "annotated_image": encoded_annotated,
            "detections": clean_detections,
            "telemetry": telemetry,
            "gate_status": gate_state,
            "alert_triggered": alert_triggered
        })
    except Exception as e:
        logger.error(f"Error processing client frame: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/gate/toggle", methods=["POST"])
def api_toggle_gate():
    """Manual Emergency Gate Override"""
    current_status = hardware.get_status()["gate_open"]
    new_status = not current_status
    hardware.set_gate(open_gate=new_status)
    return jsonify({"success": True, "gate_open": new_status, "status_text": "UNLOCKED" if new_status else "LOCKED"})

@app.route("/violations")
def violations():
    """Historical Incident & Violation Audit Log"""
    incidents = db.get_recent_incidents(limit=100)
    return render_template("violations.html", incidents=incidents)

@app.route("/api/violations")
def api_violations():
    """Returns recent incidents in JSON format for live auto-refresh"""
    limit = request.args.get("limit", 100, type=int)
    incidents = db.get_recent_incidents(limit=limit)
    stats = db.get_summary_stats()
    return jsonify({
        "success": True,
        "count": len(incidents),
        "incidents": incidents,
        "stats": stats
    })

@app.route("/api/violations/resolve/<int:incident_id>", methods=["POST"])
def api_resolve_incident(incident_id):
    """Marks an incident as acknowledged and resolved in database"""
    db.resolve_incident(incident_id, status=1)
    return jsonify({"success": True, "incident_id": incident_id, "status": "resolved"})

@app.route("/api/violations/ignore/<int:incident_id>", methods=["POST"])
def api_ignore_incident(incident_id):
    """Marks an incident as dismissed / ignored (e.g. false alert or authorized waiver)"""
    db.resolve_incident(incident_id, status=2)
    return jsonify({"success": True, "incident_id": incident_id, "status": "ignored"})

@app.route("/api/violations/reopen/<int:incident_id>", methods=["POST"])
def api_reopen_incident(incident_id):
    """Re-opens an incident back to active critical alert state"""
    db.reopen_incident(incident_id)
    return jsonify({"success": True, "incident_id": incident_id, "status": "active"})

@app.route("/api/violations/export")
def export_violations_csv():
    """Exports all violation records to CSV for health & security audits"""
    import io
    import csv
    incidents = db.get_recent_incidents(limit=1000)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Timestamp", "Date", "Time", "Status", "Confidence (%)", "Gate Action", "Camera ID"])

    for inc in incidents:
        writer.writerow([
            inc["id"], inc["timestamp"], inc["date"], inc["time"],
            inc["label"], inc["confidence"], inc["gate_status"], inc["camera_id"]
        ])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=MaskGuard_Violations_Audit.csv"}
    )

@app.route("/analytics")
def analytics():
    """Compliance Analytics & Graphs"""
    stats = db.get_summary_stats()
    hourly = db.get_hourly_analytics()
    return render_template("analytics.html", stats=stats, hourly_json=json.dumps(hourly))

@app.route("/diagnostics")
def diagnostics():
    """Model Architecture, Confusion Matrix & Viva Defense Cheat Sheet"""
    return render_template("diagnostics.html")

@app.route("/scanner")
def scanner():
    """Offline Photo & Video CCTV Media Scanner"""
    return render_template("scanner.html")

@app.route("/api/scan_image", methods=["POST"])
def scan_image():
    """Scans an uploaded image or sample image for face mask compliance"""
    import uuid
    try:
        frame = None
        json_data = request.get_json(silent=True) or {}

        # 1. Check for uploaded files (multipart/form-data)
        if request.files:
            file = request.files.get("file") or next(iter(request.files.values()), None)
            if file and file.filename != "":
                raw_bytes = file.read()
                if raw_bytes and len(raw_bytes) > 0:
                    file_bytes = np.frombuffer(raw_bytes, np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    # Fallback to Pillow if OpenCV cannot decode (e.g. WebP, RGBA PNG, CMYK, HEIC)
                    if frame is None:
                        try:
                            import io
                            from PIL import Image, ImageOps
                            pil_img = Image.open(io.BytesIO(raw_bytes))
                            pil_img = ImageOps.exif_transpose(pil_img)
                            pil_img = pil_img.convert("RGB")
                            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                        except Exception as pe:
                            logger.error(f"Pillow upload decode failed: {pe}")

        # 2. Check for base64 encoded image in JSON
        elif "image" in json_data:
            try:
                img_str = json_data["image"]
                if "," in img_str:
                    img_str = img_str.split(",")[1]
                raw_bytes = base64.b64decode(img_str)
                file_bytes = np.frombuffer(raw_bytes, np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if frame is None:
                    import io
                    from PIL import Image, ImageOps
                    pil_img = Image.open(io.BytesIO(raw_bytes))
                    pil_img = ImageOps.exif_transpose(pil_img)
                    pil_img = pil_img.convert("RGB")
                    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as be:
                logger.error(f"Base64 image decode error: {be}")

        # 3. Check for pre-loaded dataset sample
        elif "sample" in json_data or "sample" in request.form:
            sample_name = json_data.get("sample") or request.form.get("sample")
            sample_path = STATIC_DIR / "samples" / sample_name
            if sample_path.exists():
                frame = cv2.imread(str(sample_path))
                if frame is None:
                    from PIL import Image
                    pil_img = Image.open(str(sample_path)).convert("RGB")
                    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        if frame is None or frame.size == 0:
            return jsonify({"error": "Unable to read image. Please ensure the file is a valid JPG, PNG, or WEBP image."}), 400

        # Process through detector
        annotated_frame, detections, _ = detector.process_frame(frame, auto_control_gate=False)

        mask_count = sum(1 for d in detections if d["is_mask"])
        no_mask_count = sum(1 for d in detections if not d["is_mask"])
        total_faces = len(detections)
        compliance_rate = round((mask_count / total_faces * 100), 1) if total_faces > 0 else 100.0

        # Save annotated result
        res_filename = f"scan_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
        res_path = UPLOADS_DIR / res_filename
        cv2.imwrite(str(res_path), annotated_frame)

        clean_detections = []
        for d in detections:
            clean_detections.append({
                "box": d["box"],
                "label": d["label"],
                "confidence": d["confidence"],
                "is_mask": d["is_mask"]
            })

        return jsonify({
            "success": True,
            "result_image_url": f"/static/uploads/{res_filename}",
            "total_faces": total_faces,
            "mask_count": mask_count,
            "no_mask_count": no_mask_count,
            "compliance_rate": compliance_rate,
            "detections": clean_detections,
            "gate_recommendation": "GRANT_ACCESS" if no_mask_count == 0 and total_faces > 0 else "DENY_ACCESS"
        })
    except Exception as e:
        logger.error(f"Image scan error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    """Retrieve or dynamically adjust AI sensitivity and temporal persistence limits"""
    if request.method == "POST":
        try:
            data = request.get_json() or {}
            threshold = data.get("confidence_threshold")
            persistence = data.get("persistence_limit")
            
            if threshold is not None:
                detector.confidence_threshold = float(threshold) / 100.0
            if persistence is not None:
                detector.persistence_limit = max(1, int(persistence))
                
            logger.info(f"Updated AI settings: threshold={detector.confidence_threshold}, persistence={detector.persistence_limit}")
            return jsonify({
                "success": True,
                "confidence_threshold": int(round(detector.confidence_threshold * 100)),
                "persistence_limit": detector.persistence_limit,
                "message": "AI Sensitivity settings applied successfully."
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 400

    return jsonify({
        "success": True,
        "confidence_threshold": int(round(detector.confidence_threshold * 100)),
        "persistence_limit": detector.persistence_limit
    })

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)


if __name__ == "__main__":
    # Run on port 5002 to avoid conflicting with existing port 5001
    logger.info("Starting MaskGuard AI Web Surveillance Server on http://localhost:5002 ...")
    app.run(host="0.0.0.0", port=5002, debug=False, threaded=True)
