"""Play chess with the Dobot in the sim: you type moves, the robot sees and replies.

The loop:
  1. render the board, detect pieces with YOLO, fuse frames into a position
  2. compare with the internal board, infer your legal move, sanity check
  3. ask Stockfish for the reply
  4. execute both moves with the arm (captures to tray, castling, promotion)
  5. re-observe the board as `self` does and verify pieces are where expected

Usage:
    python sim/chess_robot_sim.py                # text mode: type normal moves (e4, Nf3)
    python sim/chess_robot_sim.py --gui          # PyBullet GUI + move log
    python sim/chess_robot_sim.py --demo e4      # run one human move + reply, headless
    python sim/chess_robot_sim.py --model chess_pieces.pt --elo 1500

Moves use normal notation (SAN): e4, Nf3, exd5, O-O, e8=Q. UCI (e2e4) also works.
"""

import argparse
import os
import sys

import chess
import pybullet as p

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chess_game as cg
import chess_motion as cmot
import chess_scene as cs
import chess_vision as cv
import dobot_vision_sim as dvs

DEFAULT_MODEL = os.path.join(
    os.path.dirname(HERE),
    "runs",
    "detect",
    "runs",
    "chess_detect_v2",
    "weights",
    "best.pt",
)


class ChessRobot:
    def __init__(self, args):
        self.args = args
        self.scene = cs.build_chess_scene(gui=args.gui, robot=args.robot)
        for _ in range(60):
            p.stepSimulation()
        self.mover = cmot.ChessMover(self.scene, board=cs.START_BOARD)
        self.board = chess.Board()
        self.engine = cg.Engine(elo=args.elo, movetime=args.movetime)
        self.model = None
        if os.path.exists(args.model):
            from ultralytics import YOLO

            self.model = YOLO(args.model)
        else:
            print(
                f"note: no model at {args.model}; vision will fall back to ground truth"
            )
        self.human_color = chess.WHITE if args.side == "white" else chess.BLACK

    # -- observation -------------------------------------------------------

    def observe(self):
        """Return the position seen by the camera (fused over frames).

        The arm is parked first so it does not occlude the board; pieces that
        are physically still in the gripper area would hide squares otherwise.
        """
        self.mover.home()
        fused, _raw, _dets = cv.observe(
            self.scene,
            self.model,
            frames=self.args.frames,
            conf=self.args.conf,
            imgsz=self.args.imgsz,
            device=self.args.device,
            min_conf=self.args.conf,
        )
        return fused

    def ground_truth(self):
        """The scene's true position, for headless tests without the model."""
        board = chess.Board(None)
        for piece in self.scene["pieces"]:
            if piece["square"] is None:
                continue
            symbol = {"knight": "N", "king": "K"}.get(
                piece["name"], piece["name"][0].upper()
            )
            if piece["color"] == "b":
                symbol = symbol.lower()
            board.set_piece_at(piece["square"], chess.Piece.from_symbol(symbol))
        return board

    def perceive(self):
        if self.model is None:
            return self.ground_truth()
        return self.observe()

    # -- one turn ----------------------------------------------------------

    def human_turn(self, text=None):
        """Ask for a move (or take one), verify it via vision, and execute it.

        Accepts normal notation (e4, Nf3, O-O, e8=Q) or UCI (e2e4).
        """
        if self.args.gui:
            print(
                f"your move (playing {'white' if self.human_color else 'black'}): ",
                end="",
                flush=True,
            )
        if text is None:
            text = input().strip()
        move = cg.parse_move(self.board, text)
        if move is None:
            print(f"not a legal move: {text}")
            return False

        before = self.board.copy()
        self.mover.execute(move)
        self.board.push(move)

        observed = self.perceive()
        diff = sum(
            observed.piece_at(sq) != self.board.piece_at(sq) for sq in chess.SQUARES
        )
        if diff:
            inferred, _tol = cg.detect_move(before, observed)
            print(f"vision: {diff} square(s) differ; board update -> {inferred}")
        return True

    def engine_turn(self):
        if self.board.is_game_over():
            return None
        reply = self.engine.best_move(self.board)
        if reply is None:
            return None
        print(f"robot plays: {cg.describe(self.board, reply)} ({reply.uci()})")
        self.mover.execute(reply)
        self.board.push(reply)
        observed = self.perceive()
        diff = sum(
            observed.piece_at(sq) != self.board.piece_at(sq) for sq in chess.SQUARES
        )
        if diff and self.model is not None:
            print(f"warning: vision sees {diff} differing square(s) after my move")
        return reply

    def run_text(self):
        print(f"stockfish: {'yes' if self.engine.engine else 'random fallback'}")
        print(f"vision: {'yes' if self.model else 'ground truth'}")
        while not self.board.is_game_over():
            if self.board.turn == self.human_color:
                if not self.human_turn():
                    continue
            else:
                self.engine_turn()
            print(f"position: {self.board.fen()}")
        self.finish()

    def run_demo(self, moves):
        """Play each given move (normal or UCI notation) as the human, replying after each."""
        for text in moves:
            if self.board.is_game_over():
                break
            move = cg.parse_move(self.board, text)
            if move is None:
                print(f"not legal: {text}")
                break
            if self.board.turn != self.human_color:
                print(f"skip {text}: not your turn")
                continue
            print(f"human: {text}")
            self.mover.execute(move)
            self.board.push(move)
            if not self.board.is_game_over():
                self.engine_turn()
        self.finish()

    def finish(self):
        outcome = self.board.outcome()
        print(f"game over: {outcome.result() if outcome else '*'} ({self.board.fen()})")
        self.mover.home()
        errors = cmot.verify_positions(self.scene, self.board)
        print("sync:", "OK" if not errors else errors[:5])
        self.engine.close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gui", action="store_true", help="show the PyBullet GUI")
    parser.add_argument(
        "--side",
        choices=("white", "black"),
        default="white",
        help="your color; robot takes the other",
    )
    parser.add_argument(
        "--elo",
        type=int,
        default=None,
        help="stockfish elo cap (default full strength)",
    )
    parser.add_argument(
        "--movetime", type=float, default=0.5, help="engine seconds per move"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="YOLO weights; empty string uses ground truth",
    )
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--frames", type=int, default=3, help="vision frames fused per observation"
    )
    parser.add_argument("--robot", choices=sorted(dvs.ROBOTS), default="simple")
    parser.add_argument(
        "--demo",
        nargs="*",
        default=None,
        help="moves to auto-play (e.g. e4 Nf3), then reply",
    )
    args = parser.parse_args()

    robot = ChessRobot(args)
    if args.gui:
        import cv2

        cv2.waitKey(1)
    if args.demo is not None:
        robot.run_demo(args.demo)
    else:
        robot.run_text()


if __name__ == "__main__":
    main()
