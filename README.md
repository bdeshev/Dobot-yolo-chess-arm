# Chess Dobot

A simulated [Dobot Magician](https://www.dobot-robots.com/products/magician) that plays chess.
The arm lives in PyBullet, sees the board through a YOLO model, thinks with
Stockfish, and moves real meshes with IK — no hardware required.

## Features

- **Vision** — YOLO detections are unprojected through camera rays onto the
  board, fused over several frames, and checked against legal positions.
- **Gameplay** — Stockfish replies, with an optional ELO cap. A pure-Python
  random-move fallback keeps everything runnable without a Stockfish binary.
- **Motion** — jaw-gripper pick-and-place with grasp approach, transit height,
  captures to a tray, castling, en passant, and promotion.
- **Dataset generation** — render randomized board scenes and write exact
  YOLO labels (mesh bounding boxes projected to the image), no manual
  annotation needed.

## Quick start

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Play a game in the PyBullet sim (type moves as SAN, e.g. `e4`, `Nf3`):

```bash
uv run python sim/chess_robot_sim.py            # headless, type moves
uv run python sim/chess_robot_sim.py --gui      # with the PyBullet window
uv run python sim/chess_robot_sim.py --demo e4  # one human move + reply
```

Model weights (`*.pt`) are not committed. With no model file the sim falls back
to ground-truth piece positions, so the above runs out of the box.

Try individual pieces:

```bash
uv run python sim/chess_scene.py --gui       # board, pieces, tray
uv run python sim/chess_scene.py --reach     # IK reachability for all 64 squares
uv run python sim/dobot_vision_sim.py        # camera -> YOLO -> IK pick demo
```

## Vision model

Generate a synthetic dataset, then train YOLO on it:

```bash
uv run python sim/chess_dataset.py --n 400 --out runs/chess_boards --val 60
uv run yolo detect train data=runs/chess_boards/data.yaml model=yolo26n.pt \
  imgsz=1280 epochs=100 project=runs/detect/runs name=chess_detect_v2
```

`sim/chess_robot_sim.py` looks for
`runs/detect/runs/chess_detect_v2/weights/best.pt` by default; pass another
file with `--model path/to/weights.pt`.

## Engine

`sim/chess_game.py` uses a Stockfish binary at `local/stockfish-bin` when
present, otherwise it plays random legal moves. Download Stockfish into
`local/` (the directory is git-ignored) to get real replies:

```bash
mkdir -p local
# place the stockfish executable at local/stockfish-bin
chmod +x local/stockfish-bin
```

## Tests

```bash
uv run pytest -m "not slow"    # fast unit and simulator tests
uv run pytest -m slow          # end-to-end game (~minutes of arm motion)
uv run ruff check .
```

## Project layout

```text
sim/
  chess_robot_sim.py        Main loop: vision -> chess -> arm motion (PyBullet)
  chess_game.py             Move inference and Stockfish wrapper
  chess_vision.py           YOLO -> board position pipeline
  chess_motion.py           Chess move and grasp execution with the arm
  chess_scene.py            Board, pieces, tray, and square mapping
  chess_dataset.py          Synthetic YOLO dataset generator
  dobot_vision_sim.py       Core Dobot arm, IK, and pick demo
  assets/chess/             CC0 chess-piece OBJ meshes
  urdf/                     Dobot Magician URDF and meshes
tests/
  test_chess_logic.py       Fast logic and scene tests
  test_chess_sim_e2e.py     Slow end-to-end game test
```

## Credits

- Dobot Magician URDF and meshes from
  [jkaniuka/magician_ros2](https://github.com/jkaniuka/magician_ros2) (MIT,
  see `sim/urdf/LICENSE-magician_ros2`).
- Chess piece meshes are CC0 models from OpenGameArt, converted to OBJ.
- Vision powered by [Ultralytics YOLO](https://github.com/ultralytics/ultralytics);
  chess rules and engine protocol by [python-chess](https://python-chess.readthedocs.io/).
