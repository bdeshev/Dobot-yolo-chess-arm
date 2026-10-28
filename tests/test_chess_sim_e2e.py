"""End-to-end test: arm plays a scripted game, pieces stay in sync.

Slow (each move is ~10-20 s of arm motion), so it is marked `slow`.
Run with:  pytest -m slow tests/test_chess_sim_e2e.py
"""

import os
import sys

import chess
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))

import chess_motion as cmot
import chess_scene as cs
import pybullet as p

pytestmark = pytest.mark.slow


def test_scripted_game_stays_in_sync():
    scene = cs.build_chess_scene(gui=False, robot="simple")
    for _ in range(60):
        p.stepSimulation()
    mover = cmot.ChessMover(scene, board=cs.START_BOARD)
    board = chess.Board()
    try:
        for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5c6", "d7c6"):
            move = chess.Move.from_uci(uci)
            mover.execute(move)
            board.push(move)
        mover.home()
        assert cmot.verify_positions(scene, board) == []
        assert mover.board.fen() == board.fen()
    finally:
        p.disconnect(scene["cid"])
