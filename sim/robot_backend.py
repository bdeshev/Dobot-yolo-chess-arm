"""Robot backends: run Cartesian arm commands on a simulated or real Dobot.

`DobotLabBackend` speaks the DobotLab Python API surface (the functions of
the official Dobot Magician coding manual) through pydobotlab, so the same
call sequence runs in three places:

  * pydobotlab's simulator, headless with no hardware (`simulator=True`),
  * a real Dobot Magician over serial (`port="/dev/ttyUSB0"`),
  * DobotLab's Python Lab, which exposes the same function names.

The interface uses meters and degrees, matching the PyBullet scene; the
conversion to the millimetres DobotLab expects happens here, at the boundary.

Usage:
    python sim/robot_backend.py --sim                 # demo moves, no hardware
    python sim/robot_backend.py --port /dev/ttyUSB0  # demo moves on a real arm
"""

import argparse
from abc import ABC, abstractmethod

MM = 1000.0

MODE_NAMES = {
    "JUMP": "JUMP_XYZ",
    "MOVJ": "MOVJ_XYZ",
    "MOVL": "MOVL_XYZ",
    "JUMP_MOVL": "JUMP_MOVL_XYZ",
}


class RobotBackend(ABC):
    """Arm interface used by the chess mover. Units: meters and degrees."""

    @abstractmethod
    def move_to(self, x, y, z, mode="MOVL"):
        """Move the tool tip to (x, y, z) and wait for the move to end."""

    @abstractmethod
    def home(self):
        """Move the arm to its home pose."""

    @abstractmethod
    def pose(self):
        """Current (x, y, z, r) pose, meters and degrees."""

    def joints(self):
        """Current joint angles in degrees, when the backend knows them."""
        return

    @abstractmethod
    def gripper(self, close):
        """Drive the jaw gripper: True closes, False opens."""

    @abstractmethod
    def suck(self, on):
        """Drive the suction cup: True grips, False releases."""

    @abstractmethod
    def grab(self, close):
        """Drive whichever end effector this backend is configured with."""

    @abstractmethod
    def close(self):
        """Release the connection to the arm."""


class DobotLabBackend(RobotBackend):
    """DobotLab API execution via pydobotlab, simulator or real Magician."""

    def __init__(
        self,
        port=None,
        simulator=False,
        tool="gripper",
        speed=(100.0, 100.0),
        jump=None,
        verbose=True,
    ):
        from pydobotlab import Magician, PTPMode

        self._ptp_mode = PTPMode
        self._tool = tool
        if simulator:
            from pydobotlab.simulator import install_simulator

            port = install_simulator(arms=1)[0]
        if port is None:
            raise ValueError("pass port=... or simulator=True")
        self._bot = Magician(port=port, verbose=verbose)
        self._bot.connect()
        self._bot.motion_params = tuple(speed)
        if jump is not None:
            self._bot.jump_params = tuple(jump)

    def move_to(self, x, y, z, mode="MOVL"):
        self._bot.move_to(
            x * MM, y * MM, z * MM, 0.0, mode=self._ptp_mode[MODE_NAMES[mode]]
        )

    def home(self):
        self._bot.set_home()

    def pose(self):
        p = self._bot.get_pose()
        return (p.x / MM, p.y / MM, p.z / MM, p.r)

    def joints(self):
        return list(self._bot.get_pose().joints)

    def gripper(self, close):
        return self._bot.set_endeffector_gripper(True, bool(close))

    def suck(self, on):
        return self._bot.set_endeffector_suctioncup(True, bool(on))

    def grab(self, close):
        if self._tool == "suction":
            return self.suck(close)
        return self.gripper(close)

    def wait(self, seconds):
        return self._bot.wait(seconds)

    def clear_alarm(self):
        return self._bot.clear_alarm()

    def close(self):
        self._bot.disconnect()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def demo(backend):
    """Home, then pick-and-drop motion in place to exercise the API."""
    backend.home()
    x, y, z, _ = backend.pose()
    print(f"home pose: x={x * MM:.0f} y={y * MM:.0f} z={z * MM:.0f} (mm)")
    backend.move_to(x, y, z + 0.05, mode="MOVL")
    backend.move_to(x, y, z, mode="MOVL")
    backend.grab(True)
    backend.wait(0.3)
    backend.move_to(x, y, z + 0.05, mode="MOVL")
    backend.grab(False)
    backend.wait(0.3)
    backend.home()
    print("demo complete")


def main():
    from pydobotlab import DobotError

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", action="store_true", help="pydobotlab simulator")
    parser.add_argument("--port", help="serial port of a real Magician")
    parser.add_argument(
        "--tool", choices=("gripper", "suction"), default="gripper"
    )
    parser.add_argument("--speed", type=float, nargs=2, default=(100.0, 100.0))
    args = parser.parse_args()

    try:
        with DobotLabBackend(
            port=args.port,
            simulator=args.sim,
            tool=args.tool,
            speed=tuple(args.speed),
        ) as backend:
            demo(backend)
    except DobotError as e:
        print(f"robot error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
