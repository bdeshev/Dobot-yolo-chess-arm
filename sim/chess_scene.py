"""Chess board scene for the Dobot sim: 8x8 board, pieces, tray, square mapping.

Builds on dobot_vision_sim's scene (arm + table), replaces the cups with a
chess board and a standard start position. Piece meshes are CC0 chess pieces
from OpenGameArt, converted from the original GLB to OBJ under assets/chess.

Usage:
    python sim/chess_scene.py --gui                      # look at the scene
    python sim/chess_scene.py --detect                  # render + YOLO sanity check
    python sim/chess_scene.py --reach                   # IK reachability of all 64 squares
"""

import argparse
import math
import os
import sys

import chess
import numpy as np
import pybullet as p

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dobot_vision_sim as dvs

ASSETS = os.path.join(HERE, "assets", "chess")

TABLE_Z = 0.0
SQUARE = 0.031
BOARD_CENTER = (0.205, 0.0)
PIECE_SCALE = 0.55
TRAY_ORIGIN = (0.105, -0.160)
TRAY_STEP = 0.031
PIECE_MASS = 0.05

LIGHT_SQ = [0.92, 0.90, 0.84, 1.0]
DARK_SQ = [0.42, 0.30, 0.18, 1.0]
BOARD_EDGE = [0.50, 0.36, 0.22, 1.0]
TRAY_COLOR = [0.55, 0.55, 0.58, 1.0]
WHITE = [0.88, 0.86, 0.80, 1.0]
BLACK = [0.16, 0.15, 0.14, 1.0]

CAM_W, CAM_H = 960, 540
CAM_HFOV_DEG = 55.0
CAM_POS = [0.205, -0.30, 0.62]
CAM_TARGET = [0.205, 0.03, 0.0]

START_BOARD = chess.Board()


def square_to_xy(square, square_size=SQUARE, center=BOARD_CENTER):
    """Map a python-chess square index to world (x, y) at the square center."""
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    x = center[0] + (file_idx - 3.5) * square_size
    y = center[1] + (rank_idx - 3.5) * square_size
    return x, y


def xy_to_square(x, y, square_size=SQUARE, center=BOARD_CENTER):
    """Map world (x, y) to a python-chess square, or None if off the board."""
    file_idx = round((x - center[0]) / square_size + 3.5)
    rank_idx = round((y - center[1]) / square_size + 3.5)
    if not (0 <= file_idx < 8 and 0 <= rank_idx < 8):
        return None
    return chess.square(file_idx, rank_idx)


def make_board(square_size=SQUARE, center=BOARD_CENTER):
    """Draw the board slab and 64 tiles. a1 is dark, at (min x, min y)."""
    half = square_size * 4
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[half + 0.004, half + 0.004, 0.004]
        ),
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[half + 0.004, half + 0.004, 0.004],
            rgbaColor=BOARD_EDGE,
        ),
        basePosition=[center[0], center[1], TABLE_Z - 0.004],
    )
    tiles = []
    for rank_idx in range(8):
        for file_idx in range(8):
            dark = (file_idx + rank_idx) % 2 == 0
            x, y = square_to_xy(chess.square(file_idx, rank_idx), square_size, center)
            body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[
                        square_size / 2 - 0.0008,
                        square_size / 2 - 0.0008,
                        0.0004,
                    ],
                    rgbaColor=DARK_SQ if dark else LIGHT_SQ,
                ),
                basePosition=[x, y, TABLE_Z + 0.0016],
            )
            tiles.append({"body": body, "dark": dark})
    return tiles


PIECE_METRICS = {
    # (collision radius, height) at PIECE_SCALE=0.55, precomputed from mesh bounds
    "pawn": (0.012375, 0.027500),
    "knight": (0.012375, 0.033000),
    "bishop": (0.012372, 0.038500),
    "rook": (0.012375, 0.030250),
    "queen": (0.012375, 0.046750),
    "king": (0.012375, 0.052250),
}


def make_piece(name, color, pos, scale=PIECE_SCALE, mass=PIECE_MASS):
    """Load a chess piece mesh with a cylinder collision proxy, base at z=TABLE_Z."""
    radius, height = PIECE_METRICS[name]
    radius *= scale / PIECE_SCALE
    height *= scale / PIECE_SCALE
    mesh = os.path.abspath(os.path.join(ASSETS, f"{name}.obj"))
    visual = p.createVisualShape(
        p.GEOM_MESH,
        fileName=mesh,
        meshScale=[scale] * 3,
        rgbaColor=WHITE if color == "w" else BLACK,
        visualFramePosition=[0, 0, -height / 2],
    )
    collision = p.createCollisionShape(
        p.GEOM_CYLINDER, radius=radius * 0.85, height=height
    )
    body = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[pos[0], pos[1], TABLE_Z + height / 2],
    )
    p.changeDynamics(
        body,
        -1,
        lateralFriction=1.2,
        rollingFriction=0.4,
        spinningFriction=0.1,
        linearDamping=0.35,
        angularDamping=0.6,
    )
    aabb = p.getAABB(body)
    return body, aabb[1][2] - TABLE_Z


def make_tray():
    """Draw the capture tray slab and return its slot positions."""
    slots = []
    for row in range(2):
        for col in range(8):
            slots.append(
                (TRAY_ORIGIN[0] + col * TRAY_STEP, TRAY_ORIGIN[1] - row * TRAY_STEP)
            )
    half_x = 4 * TRAY_STEP
    half_y = TRAY_STEP
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[half_x + 0.003, half_y + 0.003, 0.0025]
        ),
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[half_x + 0.003, half_y + 0.003, 0.0025],
            rgbaColor=TRAY_COLOR,
        ),
        basePosition=[TRAY_ORIGIN[0], TRAY_ORIGIN[1] - TRAY_STEP / 2, TABLE_Z - 0.0025],
    )
    return slots


def build_chess_scene(
    gui=False,
    robot="simple",
    square_size=SQUARE,
    center=BOARD_CENTER,
    scale=PIECE_SCALE,
    board=None,
):
    """Build the Dobot scene, then swap the cups for a chess board.

    Returns a dobot_vision_sim scene dict extended with:
      board      : {"square", "center", "slots"}
      pieces     : list of {"body", "name", "color", "square", "height", "spawn"}
      piece_heights : {piece_name: height}
      tray, spares, square_size, center
    """
    scene = dvs.build_scene(gui=gui, robot=robot)
    for obj in scene["objects"]:
        p.removeBody(obj["body"])
    scene["objects"] = []

    board = board if board is not None else START_BOARD
    make_board(square_size, center)
    tray = make_tray()

    pieces = []
    heights = {}
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
        x, y = square_to_xy(sq, square_size, center)
        body, height = make_piece(name, color, (x, y), scale)
        heights[name] = height
        pieces.append(
            {
                "body": body,
                "name": name,
                "color": color,
                "square": sq,
                "height": height,
                "spawn": [x, y, TABLE_Z],
            }
        )

    spares = []
    for i, color in enumerate(("w", "b")):
        x, y = tray[8 + i * 1]
        body, height = make_piece("queen", color, (x, y), scale)
        spares.append(
            {
                "body": body,
                "name": "queen",
                "color": color,
                "square": None,
                "height": height,
                "xy": [x, y],
            }
        )

    scene.update(
        {
            "board": {"square": square_size, "center": center, "slots": tray},
            "pieces": pieces,
            "piece_heights": heights,
            "tray": tray,
            "spares": spares,
            "square_size": square_size,
            "center": center,
            "piece_scale": scale,
        }
    )
    return scene


def piece_at_square(scene, square):
    for piece in scene["pieces"]:
        if piece["square"] == square:
            return piece
    return None


def render_camera():
    aspect = CAM_W / CAM_H
    proj = p.computeProjectionMatrixFOV(CAM_HFOV_DEG, aspect, 0.05, 5.0)
    view = p.computeViewMatrix(CAM_POS, CAM_TARGET, [0, 0, 1])
    _, _, rgb, depth, seg = p.getCameraImage(
        CAM_W,
        CAM_H,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
    )
    rgb = np.reshape(rgb, (CAM_H, CAM_W, 4)).astype(np.uint8)
    depth = np.reshape(depth, (CAM_H, CAM_W)).astype(np.float32)
    seg = np.reshape(seg, (CAM_H, CAM_W)).astype(np.int32)
    return rgb, depth, seg, view, proj


def test_reachability(scene, clearance=0.02):
    """IK-check every occupied square at pick height; report per-square results."""
    results = {}
    for sq in chess.SQUARES:
        piece = piece_at_square(scene, sq)
        x, y = square_to_xy(sq, scene["square_size"], scene["center"])
        z = (piece["height"] if piece else 0.01) + clearance
        q = dvs.solve_ik(scene, x, y, z)
        ok = False
        if q is not None:
            arm = scene["arm"]
            in_limits = all(
                p.getJointInfo(arm, jid)[8] <= ang <= p.getJointInfo(arm, jid)[9]
                for jid, ang in q.items()
            )
            if in_limits:
                dvs.apply_joint_targets(scene, q, teleport=True)
                p.stepSimulation()
                ok = math.dist(dvs.tool_tip(scene), [x, y, z]) < 0.012
        results[sq] = ok
    return results


def print_reach(results):
    ok_count = sum(results.values())
    for rank_idx in range(7, -1, -1):
        row = []
        for file_idx in range(8):
            sq = chess.square(file_idx, rank_idx)
            row.append("." if results[sq] else "x")
        print(f"  {rank_idx + 1}  " + " ".join(row))
    print(f"  reachable: {ok_count}/64")
    return ok_count


def run_gui(args, scene):
    import cv2

    cv2.namedWindow("Chess scene", cv2.WINDOW_AUTOSIZE)
    detections = []
    if args.detect:
        from ultralytics import YOLO

        model = YOLO(args.model)
    else:
        model = None
    counter = 0
    while True:
        keys = p.getKeyboardEvents()
        if ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED:
            break
        if ord("d") in keys and keys[ord("d")] & p.KEY_WAS_TRIGGERED and model:
            detections = detect_pieces(model, args)
        counter += 1
        if counter % 10 == 0:
            rgb, _, _, _, _ = render_camera()
            bgr = rgb[:, :, :3][:, :, ::-1].copy()
            if model and not detections:
                detections = detect_pieces(model, args)
            frame = draw_detections(bgr, detections)
            cv2.imshow("Chess scene", frame)
        cv2.waitKey(1)
        dvs.hold_step(scene)
        p.stepSimulation()
    cv2.destroyAllWindows()
    p.disconnect()


def detect_pieces(model, args):
    rgb, _, _, _, _ = render_camera()
    bgr = rgb[:, :, :3][:, :, ::-1].copy()
    results = model.predict(
        bgr, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False
    )
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            {
                "class_name": results[0].names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "box": [x1, y1, x2, y2],
            }
        )
    return detections


def draw_detections(bgr, detections):
    import cv2

    frame = bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            frame,
            f"{det['class_name']} {det['confidence']:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            2,
        )
    return frame


def run_detect(args):
    from collections import Counter

    import cv2
    from ultralytics import YOLO

    scene = build_chess_scene(
        gui=args.gui, robot=args.robot, scale=args.scale, square_size=args.square
    )
    for _ in range(60):
        p.stepSimulation()
    rgb, _, _, _, _ = render_camera()
    bgr = rgb[:, :, :3][:, :, ::-1].copy()
    model = YOLO(args.model)
    detections = detect_pieces(model, args)
    expected = len(scene["pieces"])
    print(
        f"model={args.model} conf={args.conf} imgsz={args.imgsz} "
        f"scale={args.scale} detections={len(detections)}/{expected}"
    )
    for name, count in sorted(Counter(d["class_name"] for d in detections).items()):
        print(f"  {name}: {count}")
    stem = args.model.replace(".pt", "")
    cv2.imwrite(f"{args.out}_det_{stem}.png", draw_detections(bgr, detections))
    cv2.imwrite(f"{args.out}_raw.png", bgr)
    print(f"saved {args.out}_det_{stem}.png")
    p.disconnect()


def run_reach(args):
    scene = build_chess_scene(
        gui=False, robot=args.robot, scale=args.scale, square_size=args.square
    )
    for _ in range(60):
        p.stepSimulation()
    print(
        f"robot={args.robot} scale={args.scale} square={args.square} center={scene['center']}"
    )
    print("      a b c d e f g h")
    count = print_reach(test_reachability(scene))
    p.disconnect()
    return 0 if count == 64 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="show PyBullet GUI")
    parser.add_argument("--robot", choices=sorted(dvs.ROBOTS), default="simple")
    parser.add_argument("--scale", type=float, default=PIECE_SCALE, help="piece scale")
    parser.add_argument(
        "--square", type=float, default=SQUARE, help="square size in meters"
    )
    parser.add_argument(
        "--detect", action="store_true", help="run YOLO once and save annotated frame"
    )
    parser.add_argument("--reach", action="store_true", help="IK-check all 64 squares")
    parser.add_argument("--model", default="chess_pieces.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="/tmp/opencode/chess_scene")
    args = parser.parse_args()

    if args.reach:
        raise SystemExit(run_reach(args))
    if args.detect:
        raise SystemExit(run_detect(args))

    if args.gui:
        scene = build_chess_scene(
            gui=True, robot=args.robot, scale=args.scale, square_size=args.square
        )
        for _ in range(60):
            p.stepSimulation()
        run_gui(args, scene)
    else:
        scene = build_chess_scene(
            gui=False, robot=args.robot, scale=args.scale, square_size=args.square
        )
        print("use --gui, --detect or --reach")
        p.disconnect()


if __name__ == "__main__":
    main()
