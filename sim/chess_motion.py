"""Motion: execute chess moves with the Dobot arm in the sim.

Reuses the motion primitives from dobot_vision_sim (IK, move_to_safe, carry).
Uses the jaw gripper: open, descend so the jaws straddle the piece, close,
grip (constraint-based hold), move, release. Handles captures (victim to tray
first), castling (two legs), en passant, and promotion (swap with a tray spare).
"""

import os
import sys

import chess
import pybullet as p

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chess_game
import chess_scene as cs
import dobot_vision_sim as dvs

GRASP_APPROACH = 0.035
TRANSIT_Z = 0.12
GRASP_OPENING = 0.011


class ChessMover:
    """Drives the arm and keeps scene pieces + a python-chess board in sync."""

    def __init__(self, scene, board=None):
        self.scene = scene
        self.board = board.copy() if board is not None else chess.Board()
        self.tray_used = 0
        self.log = []

    # -- piece bookkeeping -------------------------------------------------

    def piece_at(self, square):
        for piece in self.scene["pieces"]:
            if piece["square"] == square:
                return piece
        return None

    def piece_height(self, piece):
        return self.scene["piece_heights"].get(piece["name"], 0.03)

    def free_tray_slot(self):
        """First tray row, skipping the two spare-queen slots (8, 9)."""
        slots = [s for i, s in enumerate(self.scene["tray"]) if i not in (8, 9)]
        slot = slots[self.tray_used % len(slots)]
        self.tray_used += 1
        return slot

    def set_piece_square(self, piece, square):
        piece["square"] = square

    # -- motion primitives -------------------------------------------------

    def move_tip(self, x, y, z):
        if not dvs.move_to_safe(self.scene, x, y, z):
            raise RuntimeError(f"cannot reach ({x:.3f}, {y:.3f}, {z:.3f})")
        return True

    def carry_to(self, x, y, z):
        q = dvs.solve_ik(self.scene, x, y, z)
        if q is None:
            raise RuntimeError(f"no IK for carry ({x:.3f}, {y:.3f}, {z:.3f})")
        dvs.step_towards(self.scene, q, steps=2500)

    def _vertical(self, x, y, z_from, z_to, steps=8):
        """Descend/ascend in small IK waypoints so the tip stays vertical.

        Joint-space interpolation between two poses sweeps the tip along an
        arc that can clip a neighbouring piece, so walk the z axis instead.
        """
        for i in range(1, steps + 1):
            z = z_from + (z_to - z_from) * i / steps
            self.carry_to(x, y, z)
            p.stepSimulation()

    def _settle_on(self, piece, x, y, z_hover):
        """Move the held piece over (x, y) and down to the table, closed-loop.

        The grip is a rigid constraint on the rotating tool link, so the
        pickup-time offset goes stale when the yaw changes. Travel at
        TRANSIT_Z (above every piece), then measure the piece and nudge the
        tip until it is on target, lower until its base touches the table.
        """
        self.carry_to(x, y, TRANSIT_Z)
        self._vertical(x, y, TRANSIT_Z, z_hover)
        for _ in range(4):
            pos = p.getBasePositionAndOrientation(piece["body"])[0]
            ex, ey = x - pos[0], y - pos[1]
            if abs(ex) < 5e-4 and abs(ey) < 5e-4:
                break
            tip = dvs.tool_tip(self.scene)
            self.carry_to(tip[0] + ex, tip[1] + ey, tip[2])
        tip = dvs.tool_tip(self.scene)
        self._vertical(tip[0], tip[1], tip[2], GRASP_APPROACH + piece["height"] * 0.6)
        for _ in range(4):
            pos = p.getBasePositionAndOrientation(piece["body"])[0]
            target_z = cs.TABLE_Z + piece["height"] / 2
            ez = target_z - pos[2]
            if abs(ez) < 5e-4:
                break
            tip = dvs.tool_tip(self.scene)
            self.carry_to(tip[0], tip[1], tip[2] + ez)
        for _ in range(3):
            pos = p.getBasePositionAndOrientation(piece["body"])[0]
            ex, ey = x - pos[0], y - pos[1]
            if abs(ex) < 5e-4 and abs(ey) < 5e-4:
                break
            tip = dvs.tool_tip(self.scene)
            self.carry_to(tip[0] + ex, tip[1] + ey, tip[2])

    def pick(self, piece):
        """Open jaws, descend to the piece's grasp height, close, and grip."""
        if piece.get("square") is not None:
            x, y = cs.square_to_xy(
                piece["square"], self.scene["square_size"], self.scene["center"]
            )
        else:
            x, y = piece["xy"]
        grasp_z = piece["height"] * 0.6
        dvs.set_gripper(self.scene, dvs.GRIPPER_OPEN)
        dvs.release_grip(self.scene)
        self.move_tip(x, y, TRANSIT_Z)
        self._vertical(x, y, TRANSIT_Z, grasp_z)
        dvs.filter_gripper_piece(self.scene, piece["body"])
        dvs.set_gripper(self.scene, GRASP_OPENING)
        for _ in range(60):
            p.stepSimulation()
        dvs.grip_object(self.scene, piece)
        self._vertical(x, y, grasp_z, TRANSIT_Z)

    def place(self, piece, square):
        x, y = cs.square_to_xy(square, self.scene["square_size"], self.scene["center"])
        z_hover = piece["height"] * 0.6 + GRASP_APPROACH
        self._settle_on(piece, x, y, z_hover)
        dvs.release_grip(self.scene)
        dvs.set_gripper(self.scene, dvs.GRIPPER_OPEN)
        for _ in range(60):
            p.stepSimulation()
        self.move_tip(x, y, TRANSIT_Z)
        self.set_piece_square(piece, square)
        piece["xy"] = [x, y]

    # -- move execution ----------------------------------------------------

    def remove_to_tray(self, piece):
        """Carry a captured piece to the tray and forget it."""
        slot = self.free_tray_slot()
        self.pick(piece)
        z_hover = piece["height"] * 0.6 + GRASP_APPROACH
        self._settle_on(piece, slot[0], slot[1], z_hover)
        dvs.release_grip(self.scene)
        dvs.set_gripper(self.scene, dvs.GRIPPER_OPEN)
        for _ in range(60):
            p.stepSimulation()
        self.scene["pieces"].remove(piece)
        self.log.append(f"captured {piece['name']} to tray {slot}")

    def execute(self, move, board=None):
        """Play `move` (already legal on self.board) physically and internally."""
        board = board if board is not None else self.board
        if move not in board.legal_moves:
            raise ValueError(f"illegal move {move}")

        captured_sq = chess_game_captured_square(board, move)
        if captured_sq is not None:
            victim = self.piece_at(captured_sq)
            if victim is not None:
                self.remove_to_tray(victim)

        piece = self.piece_at(move.from_square)
        if piece is None:
            raise RuntimeError(f"no piece on {chess.square_name(move.from_square)}")

        moving_piece = board.piece_at(move.from_square)
        if (
            moving_piece is not None
            and moving_piece.piece_type == chess.KING
            and abs(
                chess.square_file(move.from_square) - chess.square_file(move.to_square)
            )
            == 2
        ):
            self.execute_castle(board, move)
        else:
            self.pick(piece)
            self.place(piece, move.to_square)

        if move.promotion is not None:
            self.promote(piece, move.to_square, move.promotion)

        board.push(move)
        self.board = board
        self.log.append(f"played {move.uci()}")
        return board

    def execute_castle(self, board, move):
        rank = chess.square_rank(move.from_square)
        king = self.piece_at(move.from_square)
        kingside = chess.square_file(move.to_square) > chess.square_file(
            move.from_square
        )
        rook_from = chess.square(7 if kingside else 0, rank)
        rook_to = chess.square(5 if kingside else 3, rank)
        rook = self.piece_at(rook_from)
        self.pick(king)
        self.place(king, move.to_square)
        if rook is not None:
            self.pick(rook)
            self.place(rook, rook_to)

    def promote(self, pawn, square, piece_type):
        """Swap the pawn for a spare from the tray (of the pawn's color)."""
        color = "w" if pawn["color"] == "w" else "b"
        spare = next(
            (
                s
                for s in self.scene["spares"]
                if s["color"] == color and s["square"] is None
            ),
            None,
        )
        self.remove_to_tray(pawn)
        if spare is None:
            return
        name = {
            chess.KNIGHT: "knight",
            chess.BISHOP: "bishop",
            chess.ROOK: "rook",
            chess.QUEEN: "queen",
        }[piece_type]
        spare["name"] = name
        self.pick(spare)
        self.place(spare, square)
        self.scene["spares"].remove(spare)
        self.scene["pieces"].append(spare)

    def home(self):
        for _ in range(400):
            if dvs.step_towards(self.scene, self.scene["rest_q"], steps=1):
                break


def chess_game_captured_square(board, move):
    if board.is_en_passant(move):
        return chess.square(
            chess.square_file(move.to_square), chess.square_rank(move.from_square)
        )
    if board.piece_at(move.to_square) is not None:
        return move.to_square
    return None


def verify_positions(scene, board):
    """Check that every scene piece's square matches the python-chess board."""
    errors = []
    for piece in scene["pieces"]:
        sq = piece["square"]
        want = board.piece_at(sq) if sq is not None else None
        if want is None:
            errors.append(
                f"extra {piece['name']} on {chess.square_name(sq) if sq is not None else None}"
            )
            continue
        got_name = {
            "p": "pawn",
            "n": "knight",
            "b": "bishop",
            "r": "rook",
            "q": "queen",
            "k": "king",
        }[want.symbol().lower()]
        got_color = "w" if want.color == chess.WHITE else "b"
        if piece["name"] != got_name or piece["color"] != got_color:
            assert sq is not None
            errors.append(
                f"{chess.square_name(sq)}: scene {piece['color']}{piece['name']} vs board {want.symbol()}"
            )
    for sq in chess.SQUARES:
        if board.piece_at(sq) is not None and not any(
            piece["square"] == sq for piece in scene["pieces"]
        ):
            errors.append(
                f"missing {board.piece_at(sq).symbol()} on {chess.square_name(sq)}"
            )
    return errors


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--moves",
        nargs="*",
        default=["e4", "e5", "Nf3", "Nc6", "Bb5"],
        help="sequence of moves in normal or UCI notation to execute",
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--sleep", type=float, default=0.0, help="delay between steps (GUI)"
    )
    args = parser.parse_args()

    scene = cs.build_chess_scene(gui=args.gui, robot="simple")
    for _ in range(60):
        p.stepSimulation()
    mover = ChessMover(scene, board=cs.START_BOARD)

    for text in args.moves:
        move = chess_game.parse_move(mover.board, text)
        if move is None:
            print(f"illegal: {text}")
            break
        mover.execute(move)
        print(f"executed {text}", flush=True)
    mover.home()

    errors = verify_positions(scene, mover.board)
    print(f"final board: {mover.board.fen()}")
    if errors:
        print("MISMATCHES:")
        for e in errors:
            print(" ", e)
        raise SystemExit(1)
    print("ALL PIECES IN SYNC")
    if args.gui:
        import time

        while True:
            dvs.hold_step(scene)
            p.stepSimulation()
            time.sleep(0.001)


if __name__ == "__main__":
    main()
