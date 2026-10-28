"""Game logic: infer the human's move from board changes and pick engine replies.

The engine is Stockfish (binary under local/, pip package `stockfish`), with a
pure-python chess.Board fallback if no binary is available.
"""

import os

import chess
import chess.engine

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DEFAULT_STOCKFISH = os.path.join(PROJ, "local", "stockfish-bin")


class Engine:
    """Stockfish wrapper: fixed elo or fixed think time."""

    def __init__(
        self, path=DEFAULT_STOCKFISH, elo=100, movetime=0.5, threads=2, hash_mb=64
    ):
        self.path = path
        self.elo = elo
        self.movetime = movetime
        self.engine = None
        if os.path.exists(path):
            self.engine = chess.engine.SimpleEngine.popen_uci(path)
            options = {"Threads": threads, "Hash": hash_mb}
            if elo is not None:
                bounds = self.engine.options.get("UCI_Elo")
                if bounds is not None:
                    if bounds.min is not None:
                        elo = max(bounds.min, elo)
                    if bounds.max is not None:
                        elo = min(bounds.max, elo)
                self.elo = elo
                options["UCI_LimitStrength"] = True
                options["UCI_Elo"] = elo
            self.engine.configure(options)

    def close(self):
        if self.engine is not None:
            self.engine.quit()
            self.engine = None

    def best_move(self, board, movetime=None):
        if self.engine is None:
            return random_move(board)
        limit = chess.engine.Limit(time=movetime or self.movetime)
        result = self.engine.play(board, limit)
        return result.move


def parse_move(board, text):
    """Parse a move in normal notation (SAN: e4, Nf3, exd5, O-O, e8=Q).

    Case-insensitive; falls back to UCI (e2e4) so scripted commands keep working.
    Returns a legal chess.Move or None.
    """
    text = (text or "").strip()
    if not text:
        return None
    candidates = [text]
    if text[0].islower():
        candidates.append(text[0].upper() + text[1:])
    if text.upper().startswith("O-O"):
        candidates.append("O-O" + text[3:])
    if text[-1:].isdigit() and "=" not in text:
        candidates.append(f"{text}=Q")  # promotion shorthand: e8 means e8=Q
    for candidate in candidates:
        try:
            return board.parse_san(candidate)
        except ValueError:
            pass
    try:
        move = chess.Move.from_uci(text)
    except ValueError:
        return None
    return move if move in board.legal_moves else None


def random_move(board):
    import random

    return random.choice(list(board.legal_moves))


def detect_move(previous, observed):
    """Infer the move that connects two positions, tolerating small observation noise.

    Tries exact match first, then a relaxed match ignoring up to `tolerance`
    disagreeing squares (for example a piece the camera missed).
    """
    candidates = list(previous.legal_moves)
    for tolerance in (0, 1, 2, 3):
        for move in candidates:
            test = previous.copy()
            test.push(move)
            diff = sum(
                test.piece_at(sq) != observed.piece_at(sq) for sq in chess.SQUARES
            )
            if diff <= tolerance:
                return move, diff
    return None, None


def apply_observed(previous, observed, tolerance=3):
    """Update the reference board by pushing the inferred legal move.

    Returns (board_after_move, move, diff) or (previous, None, None) if no
    legal move explains the observation.
    """
    move, diff = detect_move(previous, observed)
    if move is None:
        return previous, None, None
    board = previous.copy()
    board.push(move)
    return board, move, diff


def describe(board, move):
    """Human-readable move description including check/checkmate markers."""
    san = board.san(move)
    marker = ""
    probe = board.copy()
    probe.push(move)
    if probe.is_checkmate():
        marker = "#"
    elif probe.is_check():
        marker = "+"
    return f"{san}{marker}"


def play_reply(board, engine):
    """Have the engine play one move on a board copy; returns (board, move)."""
    move = engine.best_move(board)
    if move is None:
        return board, None
    board = board.copy()
    board.push(move)
    return board, move


def move_pairs(board, move):
    """Expand a move into (from_square, to_square, promo_piece) tuples for motion.

    Castling returns both the king and rook legs; en passant is a normal pair
    since the captured pawn is on a different square than the destination.
    """
    if (
        board.piece_at(move.from_square).piece_type == chess.KING
        and abs(chess.square_file(move.from_square) - chess.square_file(move.to_square))
        == 2
    ):
        rank = chess.square_rank(move.from_square)
        if chess.square_file(move.to_square) > chess.square_file(move.from_square):
            return [
                (move.from_square, move.to_square, None),
                (chess.square(7, rank), chess.square(5, rank), None),
            ]
        return [
            (move.from_square, move.to_square, None),
            (chess.square(0, rank), chess.square(3, rank), None),
        ]
    return [(move.from_square, move.to_square, move.promotion)]


def captured_square(board, move):
    """Square whose piece is removed by `move` (destination, or pawn for en passant)."""
    if board.is_en_passant(move):
        return chess.square(
            chess.square_file(move.to_square), chess.square_rank(move.from_square)
        )
    if board.piece_at(move.to_square) is not None:
        return move.to_square
    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--moves",
        nargs="*",
        default=["e4", "e5", "Nf3"],
        help="moves to play, in normal or UCI notation",
    )
    parser.add_argument("--elo", type=int, default=None)
    parser.add_argument("--movetime", type=float, default=0.3)
    args = parser.parse_args()

    engine = Engine(elo=args.elo, movetime=args.movetime)
    print(f"stockfish: {'yes' if engine.engine else 'FALLBACK random moves'}")
    board = chess.Board()
    for text in args.moves:
        move = parse_move(board, text)
        if move is None:
            print(f"illegal: {text}")
            continue
        board.push(move)
    _, reply = play_reply(board, engine)
    if reply:
        print(f"position: {board.fen()}")
        print(f"engine plays: {describe(board, reply)} ({reply.uci()})")
    engine.close()


if __name__ == "__main__":
    main()
