"""Robot backend + DobotLab-API mover tests, run on pydobotlab's simulator.

No hardware: `DobotLabBackend(simulator=True)` installs pydobotlab's
in-memory arm, so the exact call sequence used on a real Magician (and in
DobotLab's Python Lab) is exercised here.
"""

import os
import sys

import chess
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))

pytest.importorskip("pydobotlab")

import chess_dobotlab as cd
from pydobotlab.simulator import stop_all
from robot_backend import DobotLabBackend


@pytest.fixture(scope="module")
def backend():
    backend = DobotLabBackend(simulator=True)
    backend.home()
    yield backend
    backend.close()
    stop_all()


@pytest.fixture(autouse=True)
def clean_alarms(backend):
    backend.clear_alarm()
    yield


def test_move_and_pose_round_trip(backend):
    backend.move_to(0.15, 0.08, 0.06, mode="MOVL")
    x, y, z, _ = backend.pose()
    assert (x, y, z) == pytest.approx((0.15, 0.08, 0.06), abs=1e-3)


def test_move_rejects_out_of_workspace(backend):
    from pydobotlab import DobotKinematicError

    with pytest.raises(DobotKinematicError):
        backend.move_to(0.5, 0.5, 0.06, mode="MOVL")
    backend.clear_alarm()


def test_end_effectors_acknowledged(backend):
    assert backend.gripper(True)
    assert backend.gripper(False)
    assert backend.suck(True)
    assert backend.suck(False)


def test_mover_plays_move(backend):
    mover = cd.DobotLabMover(backend)
    board = mover.execute(chess.Move.from_uci("e2e4"))
    expected = chess.Board()
    expected.push(chess.Move.from_uci("e2e4"))
    assert board.fen() == expected.fen()
    assert mover.log


def test_mover_capture_fills_tray(backend):
    mover = cd.DobotLabMover(backend)
    for uci in ("e2e4", "d7d5", "e4d5"):
        mover.execute(chess.Move.from_uci(uci))
    assert mover.tray_used == 1
    assert any("captured p to tray" in entry for entry in mover.log)


def test_game_runner_demo(backend):
    import chess_robot_dobotlab as crd

    class StubEngine:
        engine = None

        def best_move(self, board):
            return chess.Move.from_uci("e7e5")

        def close(self):
            pass

    mover = cd.DobotLabMover(backend)
    robot = crd.DobotLabChessRobot(mover, StubEngine())
    robot.run_demo(["e4"])
    assert [m.uci() for m in robot.board.move_stack] == ["e2e4", "e7e5"]


def test_mirror_backend_forwards_pose(backend):
    import mirror_backend

    class Recorder:
        def __init__(self):
            self.updates = []
            self.gripper = []

        def update(self, x, y, z, r, joints=None, mode=None):
            self.updates.append((x, y, z, r, joints))

        def set_gripper(self, close):
            self.gripper.append(close)

        def close(self):
            pass

    recorder = Recorder()
    mirrored = mirror_backend.MirrorBackend(backend, recorder)
    mirrored.move_to(0.18, 0.02, 0.05, mode="MOVL")
    mirrored.grab(True)
    mirrored.wait(0.01)
    x, y, z, _ = mirrored.pose()
    last = recorder.updates[-1]
    assert last[:3] == pytest.approx((x, y, z), abs=1e-3)
    assert len(last[4]) == 4
    assert recorder.gripper == [True]


def test_mover_on_move_callback(backend):
    seen = []
    mover = cd.DobotLabMover(backend, on_move=seen.append)
    mover.execute(chess.Move.from_uci("d2d4"))
    assert seen and seen[-1].fen() == mover.board.fen()


def test_pybullet_mirror_headless():
    import chess_scene as cs
    import pybullet_mirror as pm

    mirror = pm.PyBulletMirror(gui=False, robot="simple")
    try:
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        mirror.sync(board)
        piece = cs.piece_at_square(mirror.scene, chess.E4)
        assert piece is not None
        assert (piece["name"], piece["color"]) == ("pawn", "w")
    finally:
        mirror.close()


def test_calibration_units_are_mm(backend):
    mover = cd.DobotLabMover(backend)
    x, y = mover.square_xy(chess.E4)
    assert x == pytest.approx(205.0 + 0.5 * 31.0)
    assert y == pytest.approx(0.0 + (chess.square_rank(chess.E4) - 3.5) * 31.0)
    backend.move_to(0.2, -0.1, 0.05, mode="MOVL")
    assert backend.pose()[0] == pytest.approx(0.2, abs=1e-3)
