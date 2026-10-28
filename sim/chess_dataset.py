"""Generate a synthetic YOLO dataset of chess pieces rendered in PyBullet.

Renders randomized boards from randomized camera poses and writes YOLO-format
labels computed in Python (mesh AABB projected to the image), so labeling is
exact and needs no human annotation.

Usage:
    python sim/chess_dataset.py --n 400 --out runs/chess_boards
    python sim/chess_dataset.py --n 400 --out runs/chess_boards --val 60 --gui
"""

import argparse
import os
import random
import shutil
import sys

import chess
import numpy as np
import pybullet as p
import pybullet_data

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chess_scene as cs
import dobot_vision_sim as dvs

CLASSES = [
    "black-bishop",
    "black-king",
    "black-knight",
    "black-pawn",
    "black-queen",
    "black-rook",
    "white-bishop",
    "white-king",
    "white-knight",
    "white-pawn",
    "white-queen",
    "white-rook",
]
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}

NAME_TO_CLASS = {
    "pawn": "pawn",
    "knight": "knight",
    "bishop": "bishop",
    "rook": "rook",
    "queen": "queen",
    "king": "king",
}


def class_name(piece_name, color):
    return f"{'white' if color == 'w' else 'black'}-{NAME_TO_CLASS[piece_name]}"


def random_board(rng):
    """Mostly start position with a few random legal moves played."""
    board = chess.Board()
    for _ in range(rng.randint(0, 6)):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
        if board.is_game_over():
            return chess.Board()
    return board


def random_camera(rng):
    """Jitter around the deployment camera (chess_scene.CAM_POS / CAM_TARGET)."""
    cx = cs.BOARD_CENTER[0] + rng.uniform(-0.02, 0.02)
    cy = cs.BOARD_CENTER[1] + rng.uniform(-0.02, 0.02)
    cam = [
        cx + rng.uniform(-0.03, 0.03),
        cy - rng.uniform(0.24, 0.36),
        rng.uniform(0.55, 0.70),
    ]
    target = [cx, cy + rng.uniform(-0.02, 0.02), 0.0]
    fov = rng.uniform(48, 60)
    return cam, target, fov


def project_points(points, width, height, view, proj):
    """Perspective-project world points into pixel coordinates with PyBullet matrices."""
    pts = np.asarray(points, dtype=np.float64)
    ones = np.ones((len(pts), 1))
    proj_m = np.asarray(proj).reshape(4, 4).T
    view_m = np.asarray(view).reshape(4, 4).T
    clip = (proj_m @ view_m @ np.hstack([pts, ones]).T).T
    ndc = clip[:, :3] / clip[:, 3:4]
    u = (ndc[:, 0] + 1) / 2 * width
    v = (1 - ndc[:, 1]) / 2 * height
    return np.stack([u, v], axis=1), ndc[:, 2]


def piece_bbox(body, width, height, view, proj):
    """Project the 8 corners of a body's AABB and return clipped pixel bbox."""
    lo, hi = p.getAABB(body)
    corners = [
        [x, y, z]
        for x in (lo[0], hi[0])
        for y in (lo[1], hi[1])
        for z in (lo[2], hi[2])
    ]
    uv, depth = project_points(corners, width, height, view, proj)
    if np.all(depth <= 0) or np.all(depth >= 1):
        return None
    u0, v0 = float(np.min(uv[:, 0])), float(np.min(uv[:, 1]))
    u1, v1 = float(np.max(uv[:, 0])), float(np.max(uv[:, 1]))
    u0, v0 = max(0.0, u0), max(0.0, v0)
    u1, v1 = min(width - 1.0, u1), min(height - 1.0, v1)
    if u1 - u0 < 6 or v1 - v0 < 6:
        return None
    return u0, v0, u1, v1


def write_label(lines, cls, bbox, width, height):
    u0, v0, u1, v1 = bbox
    cx = (u0 + u1) / 2 / width
    cy = (v0 + v1) / 2 / height
    w = (u1 - u0) / width
    h = (v1 - v0) / height
    lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")


def render_sample(rng, args, cam, target, fov, board, scene_pieces):
    proj = p.computeProjectionMatrixFOV(fov, args.width / args.height, 0.05, 5.0)
    view = p.computeViewMatrix(cam, target, [0, 0, 1])
    _, _, rgb, _, _ = p.getCameraImage(
        args.width,
        args.height,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
    )
    frame = np.reshape(rgb, (args.height, args.width, 4)).astype(np.uint8)

    lines = []
    for piece in scene_pieces:
        bbox = piece_bbox(piece["body"], args.width, args.height, view, proj)
        if bbox is None:
            continue
        cls = CLASS_INDEX[class_name(piece["name"], piece["color"])]
        write_label(lines, cls, bbox, args.width, args.height)

    return frame, lines


def spawn_board(board, scene, args, rng):
    """(Re)spawn all pieces for a given board with small position jitter."""
    for piece in scene["pieces"]:
        p.removeBody(piece["body"])
    scene["pieces"] = []
    pieces = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        name = {
            "p": "pawn",
            "n": "knight",
            "b": "bishop",
            "r": "rook",
            "q": "queen",
            "k": "king",
        }[piece.symbol().lower()]
        color = "w" if piece.color == chess.WHITE else "b"
        x, y = cs.square_to_xy(sq, scene["square_size"], scene["center"])
        x += rng.uniform(-0.004, 0.004)
        y += rng.uniform(-0.004, 0.004)
        body, height = cs.make_piece(name, color, (x, y), scene["piece_scale"])
        p.changeVisualShape(body, -1, rgbaColor=random_piece_color(rng, color))
        pieces.append(
            {
                "body": body,
                "name": name,
                "color": color,
                "square": sq,
                "height": height,
            }
        )
    scene["pieces"] = pieces
    return pieces


def random_arm_pose(rng, scene):
    """Move the arm around a bit so the detector sees varied occlusion."""
    arm = scene["arm"]
    base = scene["rest_q"]
    for jid, angle in base.items():
        jitter = rng.uniform(-0.5, 0.5)
        p.resetJointState(arm, jid, angle + jitter)
    if scene["robot"] == "ros":
        dvs.sync_mimic_joints(scene)


def random_piece_color(rng, color):
    if color == "w":
        base = rng.uniform(0.78, 0.95)
        return [base, base * rng.uniform(0.95, 1.0), base * rng.uniform(0.85, 1.0), 1.0]
    base = rng.uniform(0.05, 0.28)
    return [base, base * rng.uniform(0.9, 1.1), base * rng.uniform(0.9, 1.2), 1.0]


def random_board_colors(rng, scene):
    light = rng.uniform(0.78, 0.95)
    dark = rng.uniform(0.25, 0.5)
    for tile in scene["tiles"]:
        dark_tile = tile["dark"]
        if dark_tile:
            color = [
                dark * rng.uniform(0.8, 1.0),
                dark * rng.uniform(0.6, 0.9),
                dark * rng.uniform(0.4, 0.8),
                1.0,
            ]
        else:
            color = [
                light,
                light * rng.uniform(0.94, 1.0),
                light * rng.uniform(0.85, 0.98),
                1.0,
            ]
        p.changeVisualShape(tile["body"], -1, rgbaColor=color)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=400, help="number of training images")
    parser.add_argument(
        "--val", type=int, default=60, help="number of validation images"
    )
    parser.add_argument("--out", default="runs/chess_boards", help="dataset root")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gui", action="store_true", help="show the GUI while rendering"
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    root = os.path.abspath(args.out)
    for split in ("train", "val"):
        shutil.rmtree(os.path.join(root, "images", split), ignore_errors=True)
        shutil.rmtree(os.path.join(root, "labels", split), ignore_errors=True)
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)

    p.connect(p.GUI if args.gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    scene = dvs.build_scene(gui=args.gui, robot="simple")
    for obj in scene["objects"]:
        p.removeBody(obj["body"])
    scene["objects"] = []
    scene["square_size"] = cs.SQUARE
    scene["center"] = cs.BOARD_CENTER
    scene["piece_scale"] = cs.PIECE_SCALE
    scene["tiles"] = cs.make_board(scene["square_size"], scene["center"])
    cs.make_tray()
    scene["pieces"] = []

    total = args.n + args.val
    written = 0
    piece_counts = {}
    for idx in range(total):
        split = "val" if idx >= args.n else "train"
        board = random_board(rng)
        pieces = spawn_board(board, scene, args, rng)
        random_board_colors(rng, scene)
        random_arm_pose(rng, scene)
        for _ in range(20):
            p.stepSimulation()
        cam, target, fov = random_camera(rng)
        frame, lines = render_sample(rng, args, cam, target, fov, board, pieces)

        stem = f"{split}_{idx:05d}"
        import cv2

        cv2.imwrite(
            os.path.join(root, "images", split, f"{stem}.jpg"),
            frame[:, :, :3][:, :, ::-1],
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        with open(os.path.join(root, "labels", split, f"{stem}.txt"), "w") as f:
            f.write("\n".join(lines))
        written += 1
        for line in lines:
            cls = int(line.split()[0])
            piece_counts[CLASSES[cls]] = piece_counts.get(CLASSES[cls], 0) + 1
        if idx % 50 == 0:
            print(f"  {idx}/{total} images, {len(lines)} boxes", flush=True)

    with open(os.path.join(root, "data.yaml"), "w") as f:
        f.write(f"path: {root}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        f.writelines(f"  {i}: {name}\n" for i, name in enumerate(CLASSES))

    print(f"wrote {written} images to {root}")
    print("piece instances:", dict(sorted(piece_counts.items())))
    p.disconnect()


if __name__ == "__main__":
    main()
