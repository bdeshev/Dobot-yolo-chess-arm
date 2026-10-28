"""Play chess on the DobotLab API: pydobotlab simulator or a real Magician.

The loop:
  1. you type a move (SAN or UCI); the DobotLab mover executes it
  2. Stockfish replies, and the reply is executed the same way
  3. captures go to the tray; castling and promotion are handled
  4. the arm homes and the game summary is printed

No camera here: pydobotlab's simulator models the arm only, so the board
state is the chess logic itself. The PyBullet chess scene renders the run by
default (`PyBulletMirror`): the arm follows the DobotLab poses via the scene's
IK while the pieces sync to the game. `--mirror none` runs headless.

Usage:
    python sim/chess_robot_dobotlab.py --sim                  # type moves, PyBullet view
    python sim/chess_robot_dobotlab.py --sim --demo e4 d5
    python sim/chess_robot_dobotlab.py --sim --mirror none --demo e4
    python sim/chess_robot_dobotlab.py --port /dev/ttyUSB0 --side black
"""

import argparse
import json
import os
import sys

import chess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chess_game
from chess_dobotlab import DobotLabMover
from robot_backend import DobotLabBackend


class DobotLabChessRobot:
    """Human input + Stockfish replies, executed by DobotLabMover."""

    def __init__(self, mover, engine, human_color=chess.WHITE):
        self.mover = mover
        self.board = mover.board
        self.engine = engine
        self.human_color = human_color

    def human_turn(self, text=None):
        if text is None:
            text = input("your move (e.g. e4, Nf3, O-O, e8=Q): ").strip()
        move = chess_game.parse_move(self.board, text)
        if move is None:
            print(f"not a legal move: {text}")
            return False
        print(f"human: {text}")
        self.mover.execute(move)
        return True

    def engine_turn(self):
        if self.board.is_game_over():
            return None
        reply = self.engine.best_move(self.board)
        if reply is None:
            return None
        print(f"robot plays: {chess_game.describe(self.board, reply)} ({reply.uci()})")
        self.mover.execute(reply)
        return reply

    def run_text(self):
        side = "white" if self.human_color else "black"
        print(f"you play {side}; stockfish: {self.stockfish_status()}")
        while not self.board.is_game_over():
            if self.board.turn == self.human_color:
                if not self.human_turn():
                    continue
            else:
                self.engine_turn()
            print(f"position: {self.board.fen()}")
        self.finish()

    def run_demo(self, moves):
        for text in moves:
            if self.board.is_game_over():
                break
            move = chess_game.parse_move(self.board, text)
            if move is None:
                print(f"not legal: {text}")
                break
            if self.board.turn != self.human_color:
                print(f"skip {text}: not your turn")
                continue
            print(f"human: {text}")
            self.mover.execute(move)
            if not self.board.is_game_over():
                self.engine_turn()
        self.finish()

    def stockfish_status(self):
        return "yes" if self.engine.engine else "random fallback"

    def finish(self):
        outcome = self.board.outcome()
        print(f"game over: {outcome.result() if outcome else '*'} ({self.board.fen()})")
        self.mover.backend.home()
        self.engine.close()


def main():
    from pydobotlab import DobotError

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sim", action="store_true", help="pydobotlab simulator, no hardware"
    )
    parser.add_argument("--port", help="serial port of a real Magician")
    parser.add_argument("--tool", choices=("gripper", "suction"), default="gripper")
    parser.add_argument(
        "--side",
        choices=("white", "black"),
        default="white",
        help="your color; robot takes the other",
    )
    parser.add_argument(
        "--elo", type=int, default=None, help="stockfish elo cap (default full strength)"
    )
    parser.add_argument("--movetime", type=float, default=0.5, help="engine seconds per move")
    parser.add_argument("--cal", help="JSON calibration file for the mover")
    parser.add_argument(
        "--mirror",
        choices=("pybullet", "none"),
        default="pybullet",
        help="pybullet shows the run in the chess scene (GUI); none is headless",
    )
    parser.add_argument(
        "--robot",
        choices=("simple", "ros"),
        default="simple",
        help="PyBullet arm model used by the mirror",
    )
    parser.add_argument(
        "--demo",
        nargs="*",
        default=None,
        help="moves to auto-play (e.g. e4 d5), then reply",
    )
    args = parser.parse_args()

    cal = None
    if args.cal:
        with open(args.cal) as f:
            cal = json.load(f)

    try:
        backend = DobotLabBackend(port=args.port, simulator=args.sim, tool=args.tool)
    except DobotError as e:
        print(f"robot error: {e}")
        return 1

    mirror = None
    if args.mirror == "pybullet":
        from mirror_backend import MirrorBackend
        from pybullet_mirror import PyBulletMirror

        try:
            mirror = PyBulletMirror(gui=True, robot=args.robot)
        except Exception as e:  # noqa: BLE001 - pybullet needs a display
            print(f"pybullet mirror unavailable ({e})")
            backend.close()
            return 1
        backend = MirrorBackend(backend, mirror)

    engine = chess_game.Engine(elo=args.elo, movetime=args.movetime)
    mover = DobotLabMover(
        backend, cal=cal, on_move=mirror.sync if mirror else None
    )
    robot = DobotLabChessRobot(
        mover, engine, human_color=chess.WHITE if args.side == "white" else chess.BLACK
    )
    try:
        if args.demo is not None:
            robot.run_demo(args.demo)
        else:
            robot.run_text()
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
