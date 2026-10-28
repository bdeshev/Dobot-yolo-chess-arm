"""Vision: turn a rendered/sim board image into a python-chess position.

Pipeline per detection: YOLO box bottom-center -> camera raycast -> world (x, y)
-> nearest board square; box top-center -> height on the piece -> pick height.
Multiple frames are fused and sanity-checked against a legal position before
being trusted.

Camera intrinsics come from pybullet's projection matrix, so the same code
works for a real camera by replacing `make_ray` with an OpenCV camera model.
"""

import os
import sys

import chess
import numpy as np
import pybullet as p

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chess_scene as cs

CLASS_TO_PIECE = {
    "white-pawn": "P",
    "white-knight": "N",
    "white-bishop": "B",
    "white-rook": "R",
    "white-queen": "Q",
    "white-king": "K",
    "black-pawn": "p",
    "black-knight": "n",
    "black-bishop": "b",
    "black-rook": "r",
    "black-queen": "q",
    "black-king": "k",
}


def unproject(px, py, depth, width, height, view, proj):
    """Pixel + depth (0..1) to world point, using pybullet's column-major matrices."""
    proj_m = np.asarray(proj).reshape(4, 4).T
    view_m = np.asarray(view).reshape(4, 4).T
    inv = np.linalg.inv(proj_m @ view_m)
    ndc = np.array(
        [2.0 * px / width - 1.0, 1.0 - 2.0 * py / height, 2.0 * depth - 1.0, 1.0]
    )
    world = inv @ ndc
    return world[:3] / world[3]


def make_ray(px, py, width, height, view, proj):
    ray_from = unproject(px, py, 0.0, width, height, view, proj)
    ray_to = unproject(px, py, 1.0, width, height, view, proj)
    return ray_from, ray_to


def raycast_world(px, py, width, height, view, proj, exclude=()):
    """Intersect the camera ray through (px, py) with the scene.

    Returns (world_point, body_id, hit) where hit is "board"/"object"/"none".
    """
    ray_from, ray_to = make_ray(px, py, width, height, view, proj)
    hit = p.rayTest(ray_from, ray_to)[0]
    body_id, _, _frac, _, pos = hit
    if body_id >= 0 and body_id not in exclude:
        return list(pos), body_id, "object"
    direction = ray_to - ray_from
    if abs(direction[2]) < 1e-9:
        return None, -1, "none"
    t = (cs.TABLE_Z - ray_from[2]) / direction[2]
    if not 0 <= t <= 1:
        return None, -1, "none"
    return list(ray_from + t * direction), -1, "board"


def detection_to_square(det, scene, width, height, view, proj):
    """Map a detection to a board square.

    Raycasts through several points inside the box (bottom-center first): if a
    piece body is hit, its base x/y is used (exact in the sim); otherwise the
    ray's intersection with the board plane is used. This avoids the parallax
    error of using the box bottom alone.
    """
    x1, y1, x2, y2 = det["box"]
    cx = (x1 + x2) / 2.0
    probes = [
        (cx, y1 + 0.5 * (y2 - y1)),
        (cx, y1 + 0.35 * (y2 - y1)),
        (cx, y1 + 0.65 * (y2 - y1)),
        (cx, y2),
    ]
    exclude = set(scene.get("exclude", ()))
    piece_bodies = {piece["body"]: piece for piece in scene.get("pieces", [])}
    for px, py in probes:
        world, body, _hit = raycast_world(
            px, py, width, height, view, proj, exclude=exclude
        )
        if world is None:
            continue
        if body in piece_bodies:
            base = p.getBasePositionAndOrientation(body)[0]
            return cs.xy_to_square(
                base[0], base[1], scene["square_size"], scene["center"]
            )
        square = cs.xy_to_square(
            world[0], world[1], scene["square_size"], scene["center"]
        )
        if square is not None:
            return square
    return None


def detection_height(det, scene, width, height, view, proj):
    """Estimate the piece's top height from its box (AABB top when the body is hit)."""
    x1, _y1, x2, y2 = det["box"]
    cx = (x1 + x2) / 2.0
    exclude = set(scene.get("exclude", ()))
    piece_bodies = {piece["body"] for piece in scene.get("pieces", [])}
    for py in (y2, y2 - 3.0, y2 - 6.0):
        world, body, _hit = raycast_world(
            cx, max(0.0, py), width, height, view, proj, exclude=exclude
        )
        if world is None:
            continue
        if body in piece_bodies:
            return float(p.getAABB(body)[1][2])
        return float(world[2])
    return None


def board_from_detections(dets, scene, width, height, view, proj, min_conf=0.4):
    """Build a chess.Board from detections; last-seen wins per square."""
    board = chess.Board(None)
    by_square = {}
    for det in dets:
        if det["confidence"] < min_conf:
            continue
        piece_symbol = CLASS_TO_PIECE.get(det["class_name"])
        if piece_symbol is None:
            continue
        square = detection_to_square(det, scene, width, height, view, proj)
        if square is None:
            continue
        old = by_square.get(square)
        if old is None or det["confidence"] > old["confidence"]:
            by_square[square] = det
            board.set_piece_at(square, chess.Piece.from_symbol(piece_symbol))
    return board, by_square


def fuse_boards(board_list):
    """Majority vote per square across several observed boards."""
    counts = [{} for _ in range(64)]
    for board in board_list:
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            key = piece.symbol() if piece else None
            counts[sq][key] = counts[sq].get(key, 0) + 1
    fused = chess.Board(None)
    n_frames = len(board_list)
    for sq in chess.SQUARES:
        if not counts[sq]:
            continue
        symbol, votes = max(counts[sq].items(), key=lambda kv: kv[1])
        if symbol is not None and votes >= (n_frames + 1) // 2:
            fused.set_piece_at(sq, chess.Piece.from_symbol(symbol))
    return fused


def observe(scene, model, frames=3, conf=0.4, imgsz=1280, device="cpu", min_conf=0.4):
    """Render `frames` images, run YOLO, and fuse into one board.

    Returns (board, raw_boards, detections).
    """
    boards = []
    all_dets = []
    for _ in range(frames):
        rgb, _, _, view, proj = cs.render_camera()
        bgr = rgb[:, :, :3][:, :, ::-1].copy()
        results = model.predict(
            bgr, conf=conf, imgsz=imgsz, device=device, verbose=False
        )
        dets = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append(
                {
                    "class_name": results[0].names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "box": [x1, y1, x2, y2],
                }
            )
        boards.append(
            board_from_detections(
                dets, scene, cs.CAM_W, cs.CAM_H, view, proj, min_conf=min_conf
            )[0]
        )
        all_dets.append(dets)
        for _ in range(5):
            p.stepSimulation()
    return fuse_boards(boards), boards, all_dets


def board_is_sane(board):
    """Cheap legality screen for an observed position."""
    for color in (chess.WHITE, chess.BLACK):
        if len(board.pieces(chess.KING, color)) != 1:
            return False
        if len(board.pieces(chess.PAWN, color)) > 8:
            return False
        if board.pieces(chess.PAWN, chess.WHITE) & chess.BB_RANK_8:
            return False
        if board.pieces(chess.PAWN, chess.BLACK) & chess.BB_RANK_1:
            return False
        total = sum(
            1
            for sq in chess.SQUARES
            if (piece := board.piece_at(sq)) is not None and piece.color == color
        )
        if total > 16:
            return False
    return True


def square_error_stats(board, truth):
    """Compare an observed board against ground truth. Returns (correct, total, errors)."""
    errors = []
    correct = 0
    for sq in chess.SQUARES:
        got = board.piece_at(sq)
        want = truth.piece_at(sq)
        if got == want:
            correct += 1
        else:
            errors.append(
                (
                    chess.square_name(sq),
                    got.symbol() if got else None,
                    want.symbol() if want else None,
                )
            )
    return correct, 64, errors


def test_detection_accuracy(
    scene, model, truth, frames=3, conf=0.5, imgsz=1280, device="cpu"
):
    fused, raw_boards, dets = observe(
        scene, model, frames=frames, conf=conf, imgsz=imgsz, device=device
    )
    correct, total, errors = square_error_stats(fused, truth)
    raw_acc = []
    for b in raw_boards:
        c, _, _ = square_error_stats(b, truth)
        raw_acc.append(c)
    return {
        "fused_accuracy": correct / total,
        "raw_accuracies": [c / total for c in raw_acc],
        "errors": errors,
        "detections": sum(len(d) for d in dets) / len(dets),
        "sane": board_is_sane(fused),
        "board": fused,
    }
