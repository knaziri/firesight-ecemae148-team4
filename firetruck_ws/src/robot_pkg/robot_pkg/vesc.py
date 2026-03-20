import pyvesc
from pyvesc.VESC.messages import SetRPM, SetCurrent, SetServoPosition, GetValues
import serial
import logging
import time

from robot_pkg.config import (
    VESC_PORT,
    VESC_BAUDRATE,
    MAX_RPM,
    MAX_CURRENT,
)

logger = logging.getLogger(__name__)


class VESC:

    def __init__(self):
        self.ser = None
        self._connect()

    def _connect(self):
        try:
            self.ser = serial.Serial(VESC_PORT, VESC_BAUDRATE, timeout=0.05)
            time.sleep(0.1)
            logger.info(f"Connected to VESC on {VESC_PORT}")
        except serial.SerialException as e:
            logger.error(f"Failed to connect to VESC: {e}")
            self.ser = None

    def _send(self, message):
        if self.ser is None or not self.ser.is_open:
            logger.warning("Serial port not available")
            return False
        try:
            self.ser.write(pyvesc.encode(message))
            return True
        except Exception as e:
            logger.error(f"Serial write failed: {e}")
            return False

    def set_throttle_rpm(self, value: float):
        value = max(-1.0, min(1.0, value))
        rpm   = int(value * MAX_RPM)
        self._send(SetRPM(rpm))
        logger.debug(f"RPM command: {rpm}")

    def set_throttle_current(self, value: float):
        value   = max(-1.0, min(1.0, value))
        current = value * MAX_CURRENT
        self._send(SetCurrent(current))
        logger.debug(f"Current command: {current:.2f}A")

    def set_steering(self, value: float):
        value     = max(-1.0, min(1.0, value))
        servo_pos = (value + 1.0) / 2.0
        servo_pos = 1.0 - servo_pos          # invert direction
        self._send(SetServoPosition(servo_pos))
        logger.debug(f"Steering: {value:.2f} -> Servo pos: {servo_pos:.2f}")

    def get_rpm(self):
        if self.ser is None or not self.ser.is_open:
            return None
        try:
            self.ser.reset_input_buffer()
            self.ser.write(pyvesc.encode_request(GetValues))
            time.sleep(0.05)
            if self.ser.in_waiting > 0:
                response = self.ser.read(self.ser.in_waiting)
                msg, _   = pyvesc.decode(response)
                if isinstance(msg, GetValues):
                    return abs(int(msg.rpm))
        except Exception as e:
            logger.error(f"RPM read failed: {e}")
        return None

    def get_telemetry(self):
        if self.ser is None or not self.ser.is_open:
            return None
        try:
            self.ser.reset_input_buffer()
            self.ser.write(pyvesc.encode_request(GetValues))
            time.sleep(0.1)
            if self.ser.in_waiting > 0:
                response = self.ser.read(self.ser.in_waiting)
                msg, _   = pyvesc.decode(response)
                if isinstance(msg, GetValues):
                    return {
                        'voltage':  msg.v_in,
                        'rpm':      msg.rpm,
                        'current':  msg.avg_motor_current,
                        'temp_fet': msg.temp_fet_filtered,
                    }
        except Exception as e:
            logger.error(f"Telemetry read failed: {e}")
        return None

    def neutral(self):
        logger.info("Setting VESC to neutral")
        for _ in range(3):
            self._send(SetCurrent(0.0))
            time.sleep(0.02)
        for _ in range(3):
            self._send(SetRPM(0))
            time.sleep(0.02)
        for _ in range(3):
            self._send(SetServoPosition(0.5))
            time.sleep(0.02)

    def stop(self):
        self._send(SetCurrent(0.0))
        self._send(SetRPM(0))
        logger.info("VESC stopped")

    def close(self):
        logger.info("Closing VESC connection")
        self.neutral()
        time.sleep(0.1)
        if self.ser and self.ser.is_open:
            self.ser.close()
            logger.info("Serial port closed")
