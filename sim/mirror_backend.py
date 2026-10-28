"""Mirror backend: wrap a RobotBackend and forward poses to a visual mirror.

Any object with `update(x, y, z, r, joints)` and `close()` can be a mirror
(`sim/pybullet_mirror.py` renders the DobotLab sim in the PyBullet scene).
After every blocking command the wrapper re-reads the inner pose and pushes it
to the mirror, so the view follows the DobotLab API (simulator or real arm).
"""

from robot_backend import RobotBackend


class MirrorBackend(RobotBackend):
    """Wraps a backend and mirrors its pose after every blocking command."""

    def __init__(self, inner, mirror):
        self._inner = inner
        self._mirror = mirror

    def _sync(self, mode="MOVL"):
        x, y, z, r = self._inner.pose()
        self._mirror.update(x, y, z, r, self._inner.joints(), mode=mode)

    def move_to(self, x, y, z, mode="MOVL"):
        self._inner.move_to(x, y, z, mode=mode)
        self._sync(mode)

    def home(self):
        self._inner.home()
        self._sync("MOVJ")

    def pose(self):
        return self._inner.pose()

    def joints(self):
        return self._inner.joints()

    def gripper(self, close):
        return self._inner.gripper(close)

    def suck(self, on):
        return self._inner.suck(on)

    def grab(self, close):
        result = self._inner.grab(close)
        set_gripper = getattr(self._mirror, "set_gripper", None)
        if callable(set_gripper):
            set_gripper(close)
        return result

    def wait(self, seconds):
        return self._inner.wait(seconds)

    def clear_alarm(self):
        return self._inner.clear_alarm()

    def close(self):
        self._mirror.close()
        self._inner.close()
