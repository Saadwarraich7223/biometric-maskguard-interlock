"""
MaskGuard AI - Hardware Abstraction Layer
Supports real Arduino (pin 13 Relay / Solenoid lock)
Gracefully falls back to Virtual Relay Mode if hardware is not connected.
Ensures non-blocking fallback and uninterrupted operation across diverse deployment environments.
"""

import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HardwareController")

class HardwareController:
    def __init__(self, port="COM9", pin=13, virtual_mode=False):
        self.port = port
        self.pin = pin
        self.board = None
        self.is_virtual = virtual_mode
        self.gate_open = False
        self.last_state_change = time.time()
        self.connection_message = "Initializing..."

        if not self.is_virtual:
            self._connect()
        else:
            self.connection_message = "Running in Virtual Relay Mode (Simulation Active)"
            logger.info(self.connection_message)

    def _connect(self):
        try:
            import pyfirmata
            logger.info(f"Attempting connection to Arduino on {self.port}...")
            self.board = pyfirmata.Arduino(self.port)
            self.is_virtual = False
            self.connection_message = f"Connected to Arduino on {self.port} (Pin {self.pin})"
            logger.info(self.connection_message)
            # Ensure gate starts in locked state
            self.set_gate(open_gate=False)
        except Exception as e:
            self.is_virtual = True
            self.board = None
            self.connection_message = f"Arduino not detected on {self.port} ({str(e)}). Fallback: Virtual Access Control Enabled."
            logger.warning(self.connection_message)
            self.set_gate(open_gate=False)

    def set_gate(self, open_gate: bool):
        """
        open_gate == True  -> Gate Unlocked (Pin 13 = 1 / Virtual = Open)
        open_gate == False -> Gate Locked   (Pin 13 = 0 / Virtual = Locked)
        """
        self.gate_open = open_gate
        self.last_state_change = time.time()

        if not self.is_virtual and self.board:
            try:
                # pyfirmata write 1 or 0
                val = 1 if open_gate else 0
                self.board.digital[self.pin].write(val)
            except Exception as e:
                logger.error(f"Hardware write error: {e}. Switching to virtual mode.")
                self.is_virtual = True

        status_str = "UNLOCKED (Access Granted)" if open_gate else "LOCKED (Access Denied)"
        mode_str = "Hardware Relay" if not self.is_virtual else "Virtual Solenoid"
        logger.info(f"[{mode_str}] Gate Status: {status_str}")

    def get_status(self):
        """Returns the current gate and connection telemetry dictionary"""
        return {
            "gate_open": self.gate_open,
            "status_text": "UNLOCKED" if self.gate_open else "LOCKED",
            "is_virtual": self.is_virtual,
            "port": self.port,
            "pin": self.pin,
            "message": self.connection_message,
            "last_updated": self.last_state_change
        }

    def close(self):
        if self.board:
            try:
                self.board.digital[self.pin].write(0)
            except:
                pass


# Global singleton instance for app-wide access
hardware = HardwareController()
