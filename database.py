"""
MaskGuard AI - Incident Database & Snapshot Logger
SQLite-backed storage for tracking compliance, violations, and photographic evidence.
Includes automated image saving and anti-spam snapshot throttling.
"""

import sqlite3
import time
import cv2
import uuid
from datetime import datetime
from pathlib import Path
from config import DB_PATH, VIOLATIONS_DIR
import logging

logger = logging.getLogger("Database")

# Ensure violations directory exists
VIOLATIONS_DIR.mkdir(parents=True, exist_ok=True)

class DatabaseManager:
    def __init__(self, db_path=DB_PATH):
        self.db_path = str(db_path)
        self.last_snapshot_time = 0
        self.snapshot_cooldown = 3.0  # Minimum 3 seconds between snapshots to prevent flooding
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes tables for incidents and audit tracking"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    snapshot_path TEXT NOT NULL,
                    face_crop_path TEXT,
                    gate_status TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scan_telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    date TEXT NOT NULL,
                    hour INTEGER NOT NULL,
                    mask_count INTEGER DEFAULT 0,
                    no_mask_count INTEGER DEFAULT 0
                )
            """)
            conn.commit()
            logger.info("Database initialized successfully.")

    def log_incident(self, label, confidence, full_frame, face_crop=None, gate_status="LOCKED", camera_id="CAM-01-ENTRANCE"):
        """
        Saves snapshot images and records violation incident in the database.
        Returns incident_id if logged, None if skipped due to cooldown.
        """
        now = time.time()
        # Enforce cooldown only for the same violation stream
        if label == "No Mask" and (now - self.last_snapshot_time < self.snapshot_cooldown):
            return None

        self.last_snapshot_time = now
        dt = datetime.now()
        timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        date_str = dt.strftime("%Y-%m-%d")
        time_str = dt.strftime("%H:%M:%S")
        unique_id = uuid.uuid4().hex[:8]

        # File names
        snap_filename = f"violation_{dt.strftime('%Y%m%d_%H%M%S')}_{unique_id}.jpg"
        crop_filename = f"crop_{dt.strftime('%Y%m%d_%H%M%S')}_{unique_id}.jpg"

        snap_full_path = VIOLATIONS_DIR / snap_filename
        crop_full_path = VIOLATIONS_DIR / crop_filename

        # Save images
        try:
            if full_frame is not None and full_frame.size > 0:
                cv2.imwrite(str(snap_full_path), full_frame)
            if face_crop is not None and face_crop.size > 0:
                cv2.imwrite(str(crop_full_path), face_crop)
        except Exception as e:
            logger.error(f"Failed to save snapshot image: {e}")
            return None

        # Relative paths for web serving
        rel_snap_path = f"violations/{snap_filename}"
        rel_crop_path = f"violations/{crop_filename}" if face_crop is not None else ""

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO incidents (
                    timestamp, date, time, label, confidence,
                    snapshot_path, face_crop_path, gate_status, camera_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp_str, date_str, time_str, label, confidence,
                rel_snap_path, rel_crop_path, gate_status, camera_id
            ))
            incident_id = cursor.lastrowid
            conn.commit()

        logger.info(f"Logged Incident #{incident_id}: [{label.upper()} {confidence:.1f}%] - Gate: {gate_status}")
        return incident_id

    def record_scan_count(self, is_mask=True):
        """Records hourly telemetry for compliance charts"""
        dt = datetime.now()
        date_str = dt.strftime("%Y-%m-%d")
        hour = dt.hour
        timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, mask_count, no_mask_count FROM scan_telemetry
                WHERE date = ? AND hour = ?
            """, (date_str, hour))
            row = cursor.fetchone()

            if row:
                if is_mask:
                    cursor.execute("UPDATE scan_telemetry SET mask_count = mask_count + 1 WHERE id = ?", (row["id"],))
                else:
                    cursor.execute("UPDATE scan_telemetry SET no_mask_count = no_mask_count + 1 WHERE id = ?", (row["id"],))
            else:
                m = 1 if is_mask else 0
                nm = 1 if not is_mask else 0
                cursor.execute("""
                    INSERT INTO scan_telemetry (timestamp, date, hour, mask_count, no_mask_count)
                    VALUES (?, ?, ?, ?, ?)
                """, (timestamp_str, date_str, hour, m, nm))
            conn.commit()

    def get_recent_incidents(self, limit=20):
        """Fetches latest violation incidents"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM incidents
                ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def resolve_incident(self, incident_id, status=1):
        """Marks an incident as resolved (1) or ignored (2)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE incidents SET resolved = ? WHERE id = ?", (status, incident_id))
            conn.commit()
            return True

    def reopen_incident(self, incident_id):
        """Re-opens an incident back to active state (0)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE incidents SET resolved = 0 WHERE id = ?", (incident_id,))
            conn.commit()
            return True

    def get_summary_stats(self):
        """Calculates global and today's summary KPIs"""
        today_str = datetime.now().strftime("%Y-%m-%d")
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Today's scan telemetry
            cursor.execute("""
                SELECT SUM(mask_count) as total_mask, SUM(no_mask_count) as total_no_mask
                FROM scan_telemetry WHERE date = ?
            """, (today_str,))
            row = cursor.fetchone()
            total_mask = row["total_mask"] or 0
            total_no_mask = row["total_no_mask"] or 0

            # Total violations logged today
            cursor.execute("""
                SELECT COUNT(*) as violations_today FROM incidents
                WHERE date = ? AND label = 'No Mask'
            """, (today_str,))
            violations_today = cursor.fetchone()["violations_today"]

            # Fallback if live scans haven't accumulated yet
            if total_no_mask < violations_today:
                total_no_mask = violations_today

            total_scans = total_mask + total_no_mask
            compliance_rate = round((total_mask / total_scans * 100), 1) if total_scans > 0 else 100.0

            return {
                "total_scans": total_scans,
                "total_mask": total_mask,
                "total_no_mask": total_no_mask,
                "violations_today": violations_today,
                "compliance_rate": compliance_rate
            }

    def get_hourly_analytics(self):
        """Returns hourly stats for today for Chart.js"""
        today_str = datetime.now().strftime("%Y-%m-%d")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT hour, mask_count, no_mask_count
                FROM scan_telemetry
                WHERE date = ?
                ORDER BY hour ASC
            """, (today_str,))
            rows = cursor.fetchall()

            hours = [f"{h:02d}:00" for h in range(24)]
            mask_data = [0] * 24
            no_mask_data = [0] * 24

            for r in rows:
                h = r["hour"]
                if 0 <= h < 24:
                    mask_data[h] = r["mask_count"]
                    no_mask_data[h] = r["no_mask_count"]

            return {
                "labels": hours,
                "mask_data": mask_data,
                "no_mask_data": no_mask_data
            }


# Singleton instance for application use
db = DatabaseManager()
