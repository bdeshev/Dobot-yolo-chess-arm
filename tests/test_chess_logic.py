"""Unit tests that do not need a display: square mapping, move inference, geometry."""

import os
import sys

import chess
import pybullet as p
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))

import chess_game
import chess_scene as cs
import chess_vision as cv


@pytest.fixture(scope="module")
def scene():
    s = cs.build_chess_scene(gui=False, robot="simple")
    for _ in range(60):
        p.stepSimulation()
    yield s
    p.disconnect(s["cid"])


def test_square_roundtrip():
    for sq in chess.SQUARES:
        x, y = cs.square_to_xy(sq)
        assert cs.xy_to_square(x, y) == sq


def test_square_names():
    x, y = cs.square_to_xy(chess.A1)
    assert cs.xy_to_square(x, y) == chess.A1
    x, y = cs.square_to_xy(chess.H8)
    assert cs.xy_to_square(x, y) == chess.H8


def test_off_board_maps_to_none():
    assert cs.xy_to_square(-10.0, 0.0) is None
    assert cs.xy_to_square(0.0, 10.0) is None


def test_start_board_has_32_pieces(scene):
    assert len(scene["pieces"]) == 32
    colors = [piece["color"] for piece in scene["pieces"]]
    assert colors.count("w") == 16
    assert colors.count("b") == 16


def test_piece_heights_ordering(scene):
    h = scene["piece_heights"]
    assert h["pawn"] < h["knight"] < h["bishop"] < h["queen"] < h["king"]
    assert h["rook"] > h["pawn"]


def test_detect_move_simple():
    board = chess.Board()
    observed = board.copy()
    observed.push_uci("e2e4")
    move, diff = chess_game.detect_move(board, observed)
    assert move == chess.Move.from_uci("e2e4")
    assert diff == 0


def test_parse_move_san():
    board = chess.Board()
    assert chess_game.parse_move(board, "e4") == chess.Move.from_uci("e2e4")
    assert chess_game.parse_move(board, "Nf3") == chess.Move.from_uci("g1f3")
    assert chess_game.parse_move(board, "nf3") == chess.Move.from_uci("g1f3")
    assert chess_game.parse_move(board, "d4") == chess.Move.from_uci("d2d4")


def test_parse_move_uci_fallback():
    board = chess.Board()
    assert chess_game.parse_move(board, "e2e4") == chess.Move.from_uci("e2e4")


def test_parse_move_rejects_illegal():
    board = chess.Board()
    assert chess_game.parse_move(board, "Qd5") is None
    assert chess_game.parse_move(board, "") is None
    assert chess_game.parse_move(board, "xyz") is None


def test_parse_move_castling_and_promotion():
    castle = chess.Board(
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    )
    assert chess_game.parse_move(castle, "O-O") == chess.Move.from_uci("e1g1")
    assert chess_game.parse_move(castle, "O-O-O") is None
    promo = chess.Board("k7/4P3/8/8/8/8/8/4K3 w - - 0 1")
    assert chess_game.parse_move(promo, "e8=Q") == chess.Move.from_uci("e7e8q")
    assert chess_game.parse_move(promo, "e8=N") == chess.Move.from_uci("e7e8n")
    assert chess_game.parse_move(promo, "e8") == chess.Move.from_uci("e7e8q")
    assert chess_game.parse_move(promo, "e8Q") == chess.Move.from_uci("e7e8q")


def test_parse_move_en_passant_and_capture():
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
    assert chess_game.parse_move(board, "exd6") == chess.Move.from_uci("e5d6")
    board2 = chess.Board()
    for uci in ("e2e4", "d7d5"):
        board2.push_uci(uci)
    assert chess_game.parse_move(board2, "exd5") == chess.Move.from_uci("e4d5")


def test_detect_move_with_noise():
    board = chess.Board()
    observed = board.copy()
    observed.push_uci("g1f3")
    observed.remove_piece_at(chess.A1)  # camera missed a rook
    move, diff = chess_game.detect_move(board, observed)
    assert move == chess.Move.from_uci("g1f3")
    assert diff == 1


def test_detect_move_capture():
    board = chess.Board()
    for uci in ("e2e4", "d7d5"):
        board.push_uci(uci)
    observed = board.copy()
    observed.push_uci("e4d5")
    move, diff = chess_game.detect_move(board, observed)
    assert move == chess.Move.from_uci("e4d5")
    assert diff == 0


def test_detect_move_no_legal_match():
    board = chess.Board()
    observed = chess.Board(None)  # camera saw nothing
    move, _ = chess_game.detect_move(board, observed)
    assert move is None


def test_captured_square_en_passant():
    board = chess.Board()
    for uci in ("e2e4", "a7a6", "e4e5", "d7d5"):
        board.push_uci(uci)
    move = chess.Move.from_uci("e5d6")
    assert chess_game.captured_square(board, move) == chess.D5


def test_move_pairs_castling():
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"):
        board.push_uci(uci)
    move = chess.Move.from_uci("e1g1")
    pairs = chess_game.move_pairs(board, move)
    assert len(pairs) == 2
    assert pairs[0][0] == chess.E1 and pairs[0][1] == chess.G1
    assert pairs[1][0] == chess.H1 and pairs[1][1] == chess.F1


def test_fuse_boards_majority():
    a = chess.Board(None)
    a.set_piece_at(chess.E4, chess.Piece.from_symbol("P"))
    b = a.copy()
    b.set_piece_at(chess.A1, chess.Piece.from_symbol("R"))
    c = a.copy()
    fused = cv.fuse_boards([a, b, c])
    assert fused.piece_at(chess.E4).symbol() == "P"
    assert fused.piece_at(chess.A1) is None  # only 1/3 frames saw the rook


def test_board_is_sane_rejects_two_kings():
    board = chess.Board(None)
    board.set_piece_at(chess.E1, chess.Piece.from_symbol("K"))
    board.set_piece_at(chess.E8, chess.Piece.from_symbol("K"))
    board.set_piece_at(chess.A1, chess.Piece.from_symbol("k"))
    assert not cv.board_is_sane(board)


def test_board_is_sane_accepts_start():
    assert cv.board_is_sane(chess.Board())


def test_geometry_maps_ground_truth_boxes(scene):
    import chess_dataset as cd

    _rgb, _, _, view, proj = cs.render_camera()
    ok = 0
    for piece in scene["pieces"]:
        lo, hi = p.getAABB(piece["body"])
        corners = [
            [x, y, z]
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ]
        uv, _ = cd.project_points(corners, cs.CAM_W, cs.CAM_H, view, proj)
        box = [
            float(uv[:, 0].min()),
            float(uv[:, 1].min()),
            float(uv[:, 0].max()),
            float(uv[:, 1].max()),
        ]
        det = {
            "class_name": cd.class_name(piece["name"], piece["color"]),
            "confidence": 1.0,
            "box": box,
        }
        if (
            cv.detection_to_square(det, scene, cs.CAM_W, cs.CAM_H, view, proj)
            == piece["square"]
        ):
            ok += 1
    assert ok == len(scene["pieces"])
