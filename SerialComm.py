"""Minimal serial control helpers for HiDrive hardware.

Command protocol:
    D,<c-rate>  battery discharge
    C,<c-rate>  regenerative charge
    N,0.0000    idle/stop

This module contains no charging-supply, electronic-load replay, CSV logging,
or battery-aging logic.
"""

import csv
import struct
import threading
import time
from pathlib import Path

try:
    import serial
    import serial.tools.list_ports
except ImportError:  # Allow the SOC model to run without serial hardware support.
    serial = None

try:
    from pymodbus.client import ModbusSerialClient
except ImportError:
    ModbusSerialClient = None


DEFAULT_PORT = "COM23"
DEFAULT_BAUDRATE = 115200
DEFAULT_TIMEOUT_S = 1.0
ET54_PORT = "COM22"
RD6006_PORT = "COM9"
CHARGE_VOLTAGE_V = 51.0
EXPERIMENT_CAPACITY_AH = 1.985
MIN_PACK_VOLTAGE_V = 40.0
MAX_PACK_VOLTAGE_V = 53.0
DEFAULT_REPLAY_FILE = "HiDrive_SOC_Timeseries.csv"


class BMSData:
    def __init__(self, cell_voltages_mv, total_voltage_mv, current_a):
        self.cell_voltages = cell_voltages_mv
        self.total_voltage = total_voltage_mv
        self.current = current_a


def parse_battery_data(frame):
    """Parse one 34-byte BMS frame; return None for an invalid frame."""
    if len(frame) != 34 or frame[:2] != b"\xAA\x10" or frame[33] != 0x0D:
        return None
    if (sum(frame[1:32]) & 0xFF) != frame[32]:
        return None
    cells = [(frame[i] << 8) | frame[i + 1] for i in range(2, 26, 2)]
    total_voltage = (frame[26] << 8) | frame[27]
    current = struct.unpack("<f", frame[28:32])[0]
    return BMSData(cells, total_voltage, current)


class SerialController:
    """Thread-safe wrapper around one serial connection."""

    def __init__(
        self,
        port=DEFAULT_PORT,
        baudrate=DEFAULT_BAUDRATE,
        timeout=DEFAULT_TIMEOUT_S,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        self._lock = threading.Lock()
        self._listener = None
        self._stop_event = threading.Event()
        self._receive_buffer = bytearray()

    @property
    def is_connected(self):
        return bool(self.serial_conn and self.serial_conn.is_open)

    def connect(self):
        if serial is None:
            print("pyserial is not installed; serial control is unavailable.")
            return False
        if self.is_connected:
            return True
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            print(f"Serial port connected: {self.port}")
            return True
        except (serial.SerialException, OSError) as exc:
            self.serial_conn = None
            print(f"Serial connection failed ({self.port}): {exc}")
            return False

    def disconnect(self):
        self.stop_listening()
        if self.serial_conn:
            try:
                if self.serial_conn.is_open:
                    self.serial_conn.close()
            finally:
                self.serial_conn = None
        print(f"Serial port disconnected: {self.port}")

    def send(self, data):
        if not self.is_connected and not self.connect():
            return False
        payload = data.encode("ascii") if isinstance(data, str) else bytes(data)
        try:
            with self._lock:
                self.serial_conn.write(payload)
                self.serial_conn.flush()
            return True
        except (serial.SerialException, OSError) as exc:
            print(f"Serial write failed: {exc}")
            return False

    def send_control(self, direction, c_rate=0.0):
        direction = str(direction).upper()
        if direction not in {"D", "C", "N"}:
            raise ValueError("direction must be 'D', 'C', or 'N'")
        if c_rate < 0:
            raise ValueError("c_rate must be non-negative")
        if direction == "N":
            c_rate = 0.0
        return self.send(f"{direction},{c_rate:.4f}\n")

    def receive(self):
        if not self.is_connected:
            return None

    def receive_bms(self, timeout=0.2):
        """Receive the latest valid BMS frame within the timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            data = self.receive()
            if data:
                self._receive_buffer.extend(data)
            while len(self._receive_buffer) >= 34:
                try:
                    start = self._receive_buffer.index(0xAA)
                except ValueError:
                    self._receive_buffer.clear()
                    break
                if start:
                    del self._receive_buffer[:start]
                if len(self._receive_buffer) < 34:
                    break
                frame = bytes(self._receive_buffer[:34])
                del self._receive_buffer[:34]
                parsed = parse_battery_data(frame)
                if parsed:
                    return parsed
            time.sleep(0.01)
        return None
        try:
            with self._lock:
                waiting = self.serial_conn.in_waiting
                return self.serial_conn.read(waiting) if waiting else None
        except (serial.SerialException, OSError) as exc:
            print(f"Serial read failed: {exc}")
            return None

    def start_listening(self, callback=None, poll_interval=0.01):
        if not self.is_connected and not self.connect():
            return None
        if self._listener and self._listener.is_alive():
            return self._listener

        self._stop_event.clear()

        def listen_loop():
            while not self._stop_event.is_set() and self.is_connected:
                data = self.receive()
                if data:
                    if callback:
                        callback(data)
                    else:
                        print(data.decode("utf-8", errors="replace").rstrip())
                time.sleep(poll_interval)

        self._listener = threading.Thread(target=listen_loop, daemon=True)
        self._listener.start()
        return self._listener

    def stop_listening(self):
        self._stop_event.set()
        if (
            self._listener
            and self._listener.is_alive()
            and self._listener is not threading.current_thread()
        ):
            self._listener.join(timeout=1.0)
        self._listener = None


_serial_controller = None


def get_serial_controller(port=DEFAULT_PORT):
    global _serial_controller
    if _serial_controller is None:
        _serial_controller = SerialController(port=port)
    return _serial_controller


def init_serial_connection(port=DEFAULT_PORT):
    """Open the shared serial connection."""
    return get_serial_controller(port).connect()


def close_serial_connection():
    """Stop the listener and close the shared serial connection."""
    global _serial_controller
    if _serial_controller is not None:
        _serial_controller.disconnect()
        _serial_controller = None


def send_direction_command(direction, c_rate=0.0):
    """Send a direction and C-rate command to the controller."""
    return get_serial_controller().send_control(direction, c_rate)


def send_crate_to_serial(current, capacity_ah):
    """Convert signed battery current to the D/C/N control protocol."""
    if capacity_ah <= 0:
        raise ValueError("capacity_ah must be positive")
    if current > 0:
        direction = "D"
    elif current < 0:
        direction = "C"  # RBS current is retained as charge direction.
    else:
        direction = "N"
    return send_direction_command(direction, abs(current) / capacity_ah)


def send_energy_status_to_serial(battery_power_w):
    """Send a simple discharge/charge/idle command from signed power."""
    if battery_power_w > 0:
        direction = "D"
    elif battery_power_w < 0:
        direction = "C"
    else:
        direction = "N"
    return send_direction_command(direction, 0.0)


def receive_from_serial():
    return get_serial_controller().receive()


def start_listening(callback=None):
    return get_serial_controller().start_listening(callback)


def list_available_ports():
    """Return available serial ports using English-only field names."""
    if serial is None:
        return []
    return [
        {
            "device": port.device,
            "description": port.description,
            "hardware_id": port.hwid,
        }
        for port in serial.tools.list_ports.comports()
    ]


class ET54ElectronicLoad:
    """Basic constant-current control for the ET54 electronic load."""

    def __init__(self, port=ET54_PORT, baudrate=9600, timeout=1.0):
        if serial is None:
            raise RuntimeError("pyserial is required for ET54 control")
        self.connection = serial.Serial(port, baudrate, timeout=timeout)

    def _command(self, command):
        self.connection.write(f"{command}\n".encode("ascii"))
        self.connection.flush()

    def set_current(self, current_a):
        self._command("CH:MODE CC")
        self._command(f"CURR:CC {max(0.0, current_a):.3f}")

    def enable(self, enabled=True):
        self._command("CH:SW ON" if enabled else "CH:SW OFF")

    def stop(self):
        self.enable(False)
        self.set_current(0.0)

    def close(self):
        if self.connection and self.connection.is_open:
            self.connection.close()


class RD6006PowerSupply:
    """Basic voltage/current/output control for the RD6006 power supply."""

    def __init__(self, port=RD6006_PORT, baudrate=115200, timeout=1.0, slave_id=1):
        if ModbusSerialClient is None:
            raise RuntimeError("pymodbus is required for RD6006 control")
        self.slave_id = slave_id
        self.client = ModbusSerialClient(
            port=port, baudrate=baudrate, timeout=timeout,
            parity="N", stopbits=1, bytesize=8,
        )
        if not self.client.connect():
            raise ConnectionError(f"Unable to connect to RD6006 on {port}")

    def _write(self, register, value):
        try:
            result = self.client.write_register(register, value, device_id=self.slave_id)
        except TypeError:  # Compatibility with older pymodbus releases.
            result = self.client.write_register(register, value, slave=self.slave_id)
        if result.isError():
            raise IOError(f"RD6006 register write failed: 0x{register:04X}")

    def set_voltage(self, voltage_v):
        self._write(0x0008, int(max(0.0, voltage_v) * 100))

    def set_current(self, current_a):
        self._write(0x0009, int(max(0.0, current_a) * 1000))

    def enable(self, enabled=True):
        self._write(0x0012, 1 if enabled else 0)

    def stop(self):
        self.enable(False)
        self.set_current(0.0)

    def close(self):
        self.client.close()


def _safe_stop(load, power):
    """Disable both current sources, including during error handling."""
    for device in (load, power):
        if device is not None:
            try:
                device.stop()
            except Exception as exc:
                print(f"Hardware stop failed: {exc}")


def replay_csv(
    csv_file=DEFAULT_REPLAY_FILE,
    bms_port=DEFAULT_PORT,
    et54_port=ET54_PORT,
    rd6006_port=RD6006_PORT,
):
    """Replay model current commands with voltage protection.

    Positive current drives ET54 discharge. Negative current is RBS charging
    and drives RD6006. Each row is held until the next ``t_idx`` value.
    """
    path = Path(csv_file)
    if not path.is_file():
        raise FileNotFoundError(f"Replay CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Replay CSV contains no data rows")
    required = {"t_idx", "current_A", "c_rate"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Replay CSV missing columns: {', '.join(sorted(missing))}")

    bms = get_serial_controller(bms_port)
    load = None
    power = None
    if not bms.connect():
        raise ConnectionError(f"Unable to connect to BMS on {bms_port}")

    try:
        load = ET54ElectronicLoad(et54_port)
        power = RD6006PowerSupply(rd6006_port)
        power.set_voltage(CHARGE_VOLTAGE_V)

        for index, row in enumerate(rows):
            model_current = float(row["current_A"])
            c_rate = abs(float(row["c_rate"]))
            hardware_current = c_rate * EXPERIMENT_CAPACITY_AH
            duration = 1.0
            if index + 1 < len(rows):
                duration = max(float(rows[index + 1]["t_idx"]) - float(row["t_idx"]), 0.0)

            if model_current > 0:
                power.stop()
                send_direction_command("D", c_rate)
                load.set_current(hardware_current)
                load.enable(True)
                mode = "DISCHARGE"
            elif model_current < 0:
                load.stop()
                send_direction_command("C", c_rate)
                power.set_current(hardware_current)
                power.enable(True)
                mode = "RBS_CHARGE"
            else:
                _safe_stop(load, power)
                send_direction_command("N", 0.0)
                mode = "IDLE"

            print(
                f"[{index + 1}/{len(rows)}] t={row['t_idx']} "
                f"mode={mode} current={hardware_current:.3f}A"
            )
            deadline = time.monotonic() + duration
            while time.monotonic() < deadline:
                battery = bms.receive_bms(timeout=min(0.2, max(deadline - time.monotonic(), 0.0)))
                if battery:
                    voltage_v = battery.total_voltage / 1000.0
                    if voltage_v < MIN_PACK_VOLTAGE_V or voltage_v > MAX_PACK_VOLTAGE_V:
                        raise RuntimeError(
                            f"Voltage protection triggered: {voltage_v:.2f} V "
                            f"(allowed {MIN_PACK_VOLTAGE_V:.2f}-{MAX_PACK_VOLTAGE_V:.2f} V)"
                        )

        print("CSV replay completed.")
    finally:
        _safe_stop(load, power)
        send_direction_command("N", 0.0)
        if load:
            load.close()
        if power:
            power.close()
        close_serial_connection()


if __name__ == "__main__":
    replay_csv()
