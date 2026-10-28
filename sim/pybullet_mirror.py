"""PyBullet mirror: play the DobotLab game visually in the PyBullet scene.

The DobotLab API (pydobotlab simulator or a real Magician) executes the real
command stream; this mirror shows the game with the scene's own proven chess
motion. `MirrorBackend` calls `sync(board)` after each executed move, the move
is inferred, and `ChessMover` replays it smoothly: jaw gripper, carried pieces,
captures to the tray, castling and promotion.

Per-command pose updates (`update`, `set_gripper`) are accepted but ignored on
purpose: echoing every raw waypoint produced jerky, bobbing arm motion, while
the replay gives the clean paths the PyBullet scene was built around.

Usage (through the runner):
    python sim/chess_robot_dobotlab.py --sim --demo e4 Nf3           # GUI
    python sim/chess_robot_dobotlab.py --sim --mirror none --demo e4 # headless
"""

import os
import sys

import chess
import pybullet as p

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chess_game
import chess_motion
import chess_scene as cs


class PyBulletMirror:
    """Replays the DobotLab game in the PyBullet chess scene via ChessMover."""

    def __init__(self, gui=True, robot="simple", board=None):
        self.board = board.copy() if board is not None else chess.Board()
        self.scene = cs.build_chess_scene(gui=gui, robot=robot, board=self.board)
        for _ in range(60):
            p.stepSimulation()
        self.mover = chess_motion.ChessMover(self.scene, board=self.board)

    def update(self, x, y, z, r, joints=None, mode="MOVJ"):
        """Ignored: the visual replay in `sync` owns the arm paths."""

    def set_gripper(self, close):
        """Ignored: ChessMover drives the jaws during the replay."""

    def sync(self, board):
        """Replay the move that turns the mirrored board into `board`."""
        if board == self.board:
            return
        move, _diff = chess_game.detect_move(self.board, board)
        if move is None:
            self.board = board.copy()
            return
        self.mover.execute(move, board=self.board)
        self.board = self.mover.board

    def close(self):
        p.disconnect(self.scene["cid"])
