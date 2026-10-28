"""Chess mover on the DobotLab API: pydobotlab simulator or a real Magician.

Plans every move in board coordinates and drives the arm through
`RobotBackend`, so no PyBullet scene is needed: piece type and height come
from the board itself. The same call sequence runs headless in pydobotlab's
simulator (`--sim`), on a real Dobot Magician (`--port`), and inside
DobotLab's Python Lab.

Calibration places the board in the arm frame (metres in the sim, mm here):
    {
      "square_mm": 31.0,
      "center_mm": [205.0, 0.0],
      "tray_origin_mm": [105.0, -160.0],
      "tray_step_mm": 31.0,
      "transit_z_mm": 60.0
    }

Usage:
    python sim/chess_dobotlab.py --sim --moves e2e4 e7e5 g1f3
    python sim/chess_dobotlab.py --sim --moves "e4" "d5" "exd5"
    python sim/chess_dobotlab.py --port /dev/ttyUSB0 --moves e2e4
"""

import argparse
import json
import os
import sys

import chess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chess_game
from robot_backend import MM, DobotLabBackend

SQUARE_MM = 31.0
CENTER_MM = (205.0, 0.0)
TRAY_ORIGIN_MM = (105.0, -160.0)
TRAY_STEP_MM = 31.0
TRANSIT_Z_MM = 60.0
GRASP_FRACTION = 0.6
SPARE_QUEEN_SLOTS = (8, 9)

PIECE_HEIGHT_MM = {
    chess.PAWN: 27.5,
    chess.KNIGHT: 33.0,
    chess.BISHOP: 38.5,
    chess.ROOK: 30.25,
    chess.QUEEN: 46.75,
    chess.KING: 52.25,
}


class DobotLabMover:
    """Executes chess moves with the DobotLab arm API; the board is the truth."""

    def __init__(self, backend, board=None, cal=None, on_move=None):
        self.backend = backend
        self.board = board.copy() if board is not None else chess.Board()
        self.on_move = on_move
        cal = cal or {}
        self.square_mm = cal.get("square_mm", SQUARE_MM)
        self.center_mm = tuple(cal.get("center_mm", CENTER_MM))
        self.tray_origin_mm = tuple(cal.get("tray_origin_mm", TRAY_ORIGIN_MM))
        self.tray_step_mm = cal.get("tray_step_mm", TRAY_STEP_MM)
        self.transit_z_mm = cal.get("transit_z_mm", TRANSIT_Z_MM)
        self.tray_used = 0
        self.log = []

    def square_xy(self, square):
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        x = self.center_mm[0] + (file_idx - 3.5) * self.square_mm
        y = self.center_mm[1] + (rank_idx - 3.5) * self.square_mm
        return x, y

    def tray_slots(self):
        slots = []
        for row in range(2):
            for col in range(8):
                slots.append(
                    (
                        self.tray_origin_mm[0] + col * self.tray_step_mm,
                        self.tray_origin_mm[1] - row * self.tray_step_mm,
                    )
                )
        return slots

    def free_tray_slot(self):
        """Next tray slot for a captured piece, skipping the spare queens."""
        slots = [
            s for i, s in enumerate(self.tray_slots()) if i not in SPARE_QUEEN_SLOTS
        ]
        slot = slots[self.tray_used % len(slots)]
        self.tray_used += 1
        return slot

    def move_mm(self, x, y, z, mode="MOVL"):
        """Move in board-frame millimetres; the backend interface is metres."""
        self.backend.move_to(x / MM, y / MM, z / MM, mode=mode)

    def lift(self, x, y):
        self.move_mm(x, y, self.transit_z_mm)

    def move_above(self, x, y):
        px, py, _, _ = self.backend.pose()
        self.move_mm(px * MM, py * MM, self.transit_z_mm)
        self.move_mm(x, y, self.transit_z_mm)

    def descend(self, x, y, z):
        self.move_mm(x, y, z)

    def pick(self, x, y, height_mm):
        self.backend.grab(False)
        self.move_above(x, y)
        self.descend(x, y, height_mm * GRASP_FRACTION)
        self.backend.grab(True)
        self.backend.wait(0.3)
        self.lift(x, y)

    def place(self, x, y, z_drop=0.0):
        self.move_above(x, y)
        self.descend(x, y, z_drop)
        self.backend.grab(False)
        self.backend.wait(0.3)
        self.lift(x, y)

    def execute(self, move, board=None):
        """Play `move` (already legal on self.board) physically and internally."""
        board = board if board is not None else self.board
        if move not in board.legal_moves:
            raise ValueError(f"illegal move {move}")

        victim_sq = chess_game.captured_square(board, move)
        if victim_sq is not None:
            victim = board.piece_at(victim_sq)
            vx, vy = self.square_xy(victim_sq)
            self.pick(vx, vy, PIECE_HEIGHT_MM[victim.piece_type])
            tx, ty = self.free_tray_slot()
            self.place(tx, ty)
            self.log.append(
                f"captured {victim.symbol()} to tray ({tx:.0f}, {ty:.0f})"
            )

        piece = board.piece_at(move.from_square)
        if piece is None:
            raise ValueError(f"no piece on {chess.square_name(move.from_square)}")

        if (
            piece.piece_type == chess.KING
            and abs(
                chess.square_file(move.from_square)
                - chess.square_file(move.to_square)
            )
            == 2
        ):
            self.execute_castle(board, move)
        else:
            fx, fy = self.square_xy(move.from_square)
            self.pick(fx, fy, PIECE_HEIGHT_MM[piece.piece_type])
            tx, ty = self.square_xy(move.to_square)
            self.place(tx, ty)

        if move.promotion is not None:
            self.promote(move.to_square, piece.color)

        board.push(move)
        self.board = board
        self.log.append(f"played {move.uci()}")
        if self.on_move is not None:
            self.on_move(self.board)
        return board

    def execute_castle(self, board, move):
        rank = chess.square_rank(move.from_square)
        kingside = chess.square_file(move.to_square) > chess.square_file(
            move.from_square
        )
        rook_from = chess.square(7 if kingside else 0, rank)
        rook_to = chess.square(5 if kingside else 3, rank)

        king = board.piece_at(move.from_square)
        fx, fy = self.square_xy(move.from_square)
        self.pick(fx, fy, PIECE_HEIGHT_MM[king.piece_type])
        tx, ty = self.square_xy(move.to_square)
        self.place(tx, ty)

        rook = board.piece_at(rook_from)
        if rook is not None:
            rx, ry = self.square_xy(rook_from)
            self.pick(rx, ry, PIECE_HEIGHT_MM[rook.piece_type])
            gx, gy = self.square_xy(rook_to)
            self.place(gx, gy)


    def promote(self, square, color):
        """Pawn to the tray, spare queen from the tray onto the square."""
        px, py = self.square_xy(square)
        self.pick(px, py, PIECE_HEIGHT_MM[chess.PAWN])
        tx, ty = self.free_tray_slot()
        self.place(tx, ty)

        spare = self.tray_slots()[SPARE_QUEEN_SLOTS[0 if color else 1]]
        self.pick(spare[0], spare[1], PIECE_HEIGHT_MM[chess.QUEEN])
        self.place(px, py)
        self.log.append(f"promoted on {chess.square_name(square)}")


def run(moves, backend, board=None, cal=None, on_move=None):
    mover = DobotLabMover(backend, board=board, cal=cal, on_move=on_move)
    for text in moves:
        move = chess_game.parse_move(mover.board, text)
        if move is None:
            print(f"illegal: {text}")
            return 1
        mover.execute(move)
        print(f"executed {text}", flush=True)
    mover.backend.home()
    print(f"final board: {mover.board.fen()}")
    return 0


def main():
    from pydobotlab import DobotError

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sim", action="store_true", help="pydobotlab simulator, no hardware"
    )
    parser.add_argument("--port", help="serial port of a real Magician")
    parser.add_argument(
        "--tool", choices=("gripper", "suction"), default="gripper"
    )
    parser.add_argument("--moves", nargs="*", default=["e4", "e5", "Nf3"])
    parser.add_argument("--cal", help="JSON calibration file")
    args = parser.parse_args()

    cal = None
    if args.cal:
        with open(args.cal) as f:
            cal = json.load(f)

    try:
        backend = DobotLabBackend(
            port=args.port, simulator=args.sim, tool=args.tool
        )
    except DobotError as e:
        print(f"robot error: {e}")
        return 1
    try:
        return run(args.moves, backend, cal=cal)
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
