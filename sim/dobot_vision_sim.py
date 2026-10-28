"""Dobot Magician vision pick demo in PyBullet.

Camera -> YOLO detection -> pixel unprojection/raycast -> IK arm motion.
No real robot or DobotLab needed. Everything runs in this venv.

Two robot models:
  ros    : authentic Dobot Magician URDF with real meshes and parallel-linkage
           mimic joints + prismatic jaw gripper, adapted from
           https://github.com/jkaniuka/magician_ros2 (MIT license). Default.
  simple : fast low-poly Dobot-style arm with a wrist joint (tool points down)
           and a prismatic jaw gripper

Gripping: set_gripper() drives the jaws open/closed; grip_object() pins a body
to the tool link with a fixed constraint (the physical equivalent of the jaws
holding a piece). The legacy suction helpers (attach_object/release_object)
remain for the original cup demo.

Usage:
    python sim/dobot_vision_sim.py                     # GUI demo, authentic Magician
    python sim/dobot_vision_sim.py --robot simple      # GUI demo, fast low-poly arm
    python sim/dobot_vision_sim.py --headless          # no windows, sanity test
    python sim/dobot_vision_sim.py --model chess_pieces.pt
"""

import argparse
import math
import os

import cv2
import numpy as np
import pybullet as p
import pybullet_data
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_Z = 0.0
CAM_W, CAM_H = 960, 540
CAM_HFOV_DEG = 55.0
CAM_POS = [0.22, -0.40, 0.75]
CAM_TARGET = [0.22, 0.12, 0.02]
MAX_JOINT_STEP = 0.006
MAX_JOINT_VELOCITY = 0.4

SCENE_CLASSES = {"cup"}

ROBOTS = {
    "simple": {
        "urdf": os.path.join(HERE, "urdf", "dobot_magician.urdf"),
        "rest_tcp": (0.22, 0.0, 0.30),
        "drop": (0.30, -0.20),
        "transit": [(0.17, 0.20)],
        "placements": [
            ([0.24, -0.06, 0.05], 2.0),
        ],
    },
    "ros": {
        "urdf": os.path.join(HERE, "urdf", "dobot_magician_ros.urdf"),
        "rest_q": {1: 0.55, 2: 1.2, 3: 1.0, 4: -1.2, 5: -1.0},
        "drop": (0.15, -0.16),
        "transit": [(0.10, 0.03), (0.17, 0.20)],
        "placements": [
            ([0.225, -0.045, 0.05], 1.3),
        ],
    },
}


def build_scene(gui: bool, robot: str = "ros") -> dict:
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 120.0)

    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=1.2,
            cameraYaw=35,
            cameraPitch=-35,
            cameraTargetPosition=[0.15, 0.1, 0.15],
        )

    p.loadURDF("plane.urdf", basePosition=[0, 0, -0.7])

    table = p.loadURDF("cube.urdf", basePosition=[0.15, 0.15, -0.4], globalScaling=0.8)
    p.changeVisualShape(table, -1, rgbaColor=[0.55, 0.5, 0.45, 1])
    p.changeDynamics(table, -1, mass=0.0)

    cfg = ROBOTS[robot]
    arm = p.loadURDF(cfg["urdf"], basePosition=[0, 0, TABLE_Z], useFixedBase=True)
    if robot == "simple":
        p.changeDynamics(arm, -1, mass=0.0)

    objects = []
    for pos, scale in cfg["placements"]:
        body = p.loadURDF("objects/mug.urdf", basePosition=pos, globalScaling=scale)
        p.changeDynamics(
            body,
            -1,
            lateralFriction=1.2,
            rollingFriction=0.5,
            spinningFriction=0.5,
            angularDamping=0.9,
            linearDamping=0.5,
        )
        rest_z = -p.getAABB(body)[0][2]
        p.resetBasePositionAndOrientation(body, [pos[0], pos[1], rest_z], [0, 0, 0, 1])
        objects.append({"body": body, "name": "cup", "spawn": [pos[0], pos[1], rest_z]})

    scene = {
        "cid": cid,
        "arm": arm,
        "robot": robot,
        "drop": cfg["drop"],
        "transit": cfg["transit"],
        "objects": objects,
        "exclude": {arm, table},
    }
    rest_q = cfg.get("rest_q")
    if rest_q is not None:
        rest_q = rest_q.copy()
    else:
        rest_q = solve_ik(scene, *cfg["rest_tcp"])
    if rest_q is None:
        raise ValueError(f"Could not solve IK for {robot!r} rest pose")
    scene["rest_q"] = rest_q
    scene["hold_q"] = rest_q.copy()
    apply_joint_targets(scene, scene["rest_q"], teleport=True)
    for _ in range(150):
        hold_step(scene)
        p.stepSimulation()
    return scene


def camera_matrices():
    aspect = CAM_W / CAM_H
    near, far = 0.05, 5.0
    proj = p.computeProjectionMatrixFOV(CAM_HFOV_DEG, aspect, near, far)
    view = p.computeViewMatrix(CAM_POS, CAM_TARGET, [0, 0, 1])
    return proj, view, near, far


def render_camera():
    proj, view, _, _ = camera_matrices()
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


def unproject(px, py, depth, view, proj):
    proj_m = np.reshape(proj, (4, 4)).T
    view_m = np.reshape(view, (4, 4)).T
    inv = np.linalg.inv(proj_m @ view_m)
    ndc = np.array(
        [2.0 * px / CAM_W - 1.0, 1.0 - 2.0 * py / CAM_H, 2.0 * depth - 1.0, 1.0]
    )
    world = inv @ ndc
    return world[:3] / world[3]


def pixel_to_target(scene, px, py, view, proj):
    """Raycast the camera pixel; aim at the hit object's top center or the table plane."""
    ray_from = unproject(px, py, 0.0, view, proj)
    ray_to = unproject(px, py, 1.0, view, proj)
    hits = p.rayTestBatch([ray_from], [ray_to])[0]
    body_id, link, frac, _, _ = hits
    if body_id >= 0 and body_id not in scene["exclude"] and frac < 1.0:
        aabb = p.getAABB(body_id, link)
        obj = next((o for o in scene["objects"] if o["body"] == body_id), None)
        center = (
            p.getBasePositionAndOrientation(body_id)[0]
            if obj
            else [(aabb[0][i] + aabb[1][i]) / 2.0 for i in range(3)]
        )
        top_z = aabb[1][2]
        return {
            "x": float(center[0]),
            "y": float(center[1]),
            "z_approach": top_z + 0.07,
            "z_top": float(top_z),
            "z_pick": top_z + 0.015,
            "body": body_id,
            "hit": "object",
        }
    direction = ray_to - ray_from
    if abs(direction[2]) < 1e-6:
        return None
    t = (TABLE_Z - ray_from[2]) / direction[2]
    if t < 0 or t > 1:
        return None
    point = ray_from + t * direction
    return {
        "x": float(point[0]),
        "y": float(point[1]),
        "z_approach": TABLE_Z + 0.12,
        "z_top": TABLE_Z,
        "z_pick": TABLE_Z + 0.03,
        "body": -1,
        "hit": "table",
    }


def detect(model, rgb, conf=0.45):
    bgr = rgb[:, :, :3][:, :, ::-1].copy()
    results = model.predict(bgr, conf=conf, verbose=False)
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            {
                "class_name": results[0].names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                "box": [x1, y1, x2, y2],
            }
        )
    return detections, bgr


SHOULDER_Z = 0.14
L1 = 0.22
L2 = 0.21
TOOL_LINK = 3
TOOL_TIP_OFFSET = 0.04

ROS_L1 = 0.135
ROS_L2 = 0.147
ROS_X0 = 0.06
ROS_Z0 = 0.131
ROS_CORE_LINK = 6
ROS_TCP_OFF = (-0.0128, 0.0212, -0.1078)
ROS_JOINT_IDS = [1, 2, 3, 4, 5]
ROS_MIMIC = {4: 2, 5: 3}
TOOL_GRAB_LINK = {"simple": 3, "ros": ROS_CORE_LINK}
TIP_IN_LINK = {"simple": (TOOL_TIP_OFFSET, 0, 0), "ros": ROS_TCP_OFF}


def solve_ik(scene, x, y, z):
    """Closed-form IK; returns {joint_id: angle} for the active robot."""
    yaw = math.atan2(y, x)
    if scene["robot"] == "simple":
        # The wrist keeps the tool vertical, so solve the 2-link chain for the
        # wrist (joint4) at 0.04 m above the tip, with L2 shortened by that tip.
        l2 = L2 - TOOL_TIP_OFFSET
        r = math.hypot(x, y)
        z_rel = (z + TOOL_TIP_OFFSET) - SHOULDER_Z
        d = min(math.hypot(r, z_rel), L1 + l2 - 1e-4)
        cos_e2 = (d * d - L1 * L1 - l2 * l2) / (2 * L1 * l2)
        e2 = -math.acos(max(-1.0, min(1.0, cos_e2)))
        e1 = math.atan2(z_rel, r) - math.atan2(
            l2 * math.sin(e2), L1 + l2 * math.cos(e2)
        )
        q = {0: yaw, 1: -e1, 2: -e2}
        q[3] = math.pi / 2 - (q[1] + q[2])
        return q

    ox, oy, oz = ROS_TCP_OFF
    tz = z - oz
    dist = math.hypot(x, y)
    phi = math.atan2(y, x)
    if dist > abs(oy):
        yaw = phi - math.asin(oy / dist)
        pr = math.sqrt(dist * dist - oy * oy) - ox - ROS_X0
    else:
        yaw = phi
        pr = dist - ox - ROS_X0
    h = tz - ROS_Z0
    d2 = pr * pr + h * h
    v = (ROS_L1 * ROS_L1 + ROS_L2 * ROS_L2 - d2) / (2 * ROS_L1 * ROS_L2)
    v = max(-1.0, min(1.0, v))
    margin = 0.15
    q2 = q3 = None
    for cand in (math.asin(v), math.pi - math.asin(v)):
        if not (-0.262 - margin <= cand <= 1.222 + margin):
            continue
        a = ROS_L1 - ROS_L2 * v
        b = ROS_L2 * math.cos(cand)
        q2_cand = math.atan2(a * pr - b * h, b * pr + a * h)
        if -0.087 - margin <= q2_cand <= 1.571 + margin:
            q2, q3 = q2_cand, cand
            break
    if q2 is None or q3 is None:
        return None  # out of workspace: caller must not command a sweep
    q2 = max(-0.087, min(1.571, q2))
    q3 = max(-0.262, min(1.222, q3))
    return {1: yaw, 2: q2, 3: q3, 4: -q2, 5: -q3}


def tool_tip(scene):
    if scene["robot"] == "simple":
        state = p.getLinkState(scene["arm"], TOOL_LINK, computeForwardKinematics=True)
        rot = np.array(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
        return np.array(state[4]) + rot @ np.array([TOOL_TIP_OFFSET, 0, 0])

    yaw = p.getJointState(scene["arm"], 1)[0]
    c, s = math.cos(yaw), math.sin(yaw)
    ox, oy, oz = ROS_TCP_OFF
    state = p.getLinkState(scene["arm"], ROS_CORE_LINK, computeForwardKinematics=True)
    core = np.array(state[4])
    return core + np.array([c * ox - s * oy, s * ox + c * oy, oz])


def apply_joint_targets(scene, targets, teleport=False):
    arm = scene["arm"]
    for jid, angle in targets.items():
        if scene["robot"] == "ros" and jid in ROS_MIMIC:
            continue  # owned by the mimic linkage, never motored directly
        if teleport:
            p.resetJointState(arm, jid, angle)
        else:
            p.setJointMotorControl2(
                arm,
                jid,
                p.POSITION_CONTROL,
                targetPosition=angle,
                force=200,
                maxVelocity=MAX_JOINT_VELOCITY,
            )
    if scene["robot"] == "ros":
        sync_mimic_joints(scene)


def sync_mimic_joints(scene):
    """PyBullet ignores URDF <mimic> tags; pin the linkage to its parents.
    Driven kinematically because the mesh collision + gravity stall the motors."""
    arm = scene["arm"]
    for child, parent in ROS_MIMIC.items():
        p.resetJointState(arm, child, -p.getJointState(arm, parent)[0])


def step_towards(scene, targets, steps=8):
    arm = scene["arm"]
    for _ in range(steps):
        done = True
        for jid, goal in targets.items():
            if scene["robot"] == "ros" and jid in ROS_MIMIC:
                continue
            cur = p.getJointState(arm, jid)[0]
            delta = goal - cur
            if abs(delta) > MAX_JOINT_STEP:
                done = False
            apply_joint_targets(
                scene, {jid: cur + max(-MAX_JOINT_STEP, min(MAX_JOINT_STEP, delta))}
            )
        p.stepSimulation()
        if scene["robot"] == "ros":
            sync_mimic_joints(scene)
        carry_step(scene)
        if done:
            return True
    return False


def hold_step(scene):
    """Keep the current pose while idle: hold parents, pin the mimic linkage."""
    arm = scene["arm"]
    for jid in scene["rest_q"]:
        if scene["robot"] == "ros" and jid in ROS_MIMIC:
            continue
        cur = p.getJointState(arm, jid)[0]
        p.setJointMotorControl2(
            arm,
            jid,
            p.POSITION_CONTROL,
            targetPosition=cur,
            force=200,
            maxVelocity=MAX_JOINT_VELOCITY,
        )
    if scene["robot"] == "ros":
        sync_mimic_joints(scene)


def draw_hud(img, detections, state, message):
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        color = (0, 220, 0) if det["class_name"] in SCENE_CLASSES else (0, 160, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"{det['class_name']} {det['confidence']:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    cv2.rectangle(img, (0, 0), (CAM_W, 34), (30, 30, 30), -1)
    cv2.putText(
        img,
        f"state={state}  {message}",
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return img


def run(args):
    gui = not args.headless
    scene = build_scene(gui, args.robot)
    model = YOLO(args.model)
    p.stepSimulation()

    if gui:
        return run_gui(args, scene, model)
    return run_headless(args, scene, model)


def run_headless(args, scene, model):
    rgb, _, _, view, proj = render_camera()
    detections, bgr = detect(model, rgb, conf=0.25)
    print(f"detections: {len(detections)}")
    scene_det = [d for d in detections if d["class_name"] in SCENE_CLASSES]
    if not scene_det:
        print("FAIL: no scene objects detected")
        return 1

    det = max(scene_det, key=lambda d: d["confidence"])
    cx, cy = det["center"]
    target = pixel_to_target(scene, cx, cy, view, proj)
    print(
        f"  {det['class_name']:12s} conf={det['confidence']:.2f} "
        f"center=({cx:.0f},{cy:.0f}) -> {target}"
    )
    cv2.imwrite(
        os.path.join(HERE, f"frame_{det['class_name'].replace(' ', '_')}.png"),
        draw_hud(bgr.copy(), [det], "headless", det["class_name"]),
    )
    if target is None:
        print("FAIL: raycast failed")
        return 1

    drop = scene["drop"]
    grabbed = go_pick(
        scene, (target["x"], target["y"], target["z_top"]), (drop[0], drop[1])
    )
    for _ in range(300):
        p.stepSimulation()

    if not grabbed:
        print("FAIL: suction missed the cup")
        return 1
    pos = p.getBasePositionAndOrientation(grabbed["body"])[0]
    err = math.dist(pos[:2], drop)
    print(
        f"  placed '{grabbed['name']}' at ({pos[0]:.3f}, {pos[1]:.3f}), drop err {err * 1000:.1f} mm"
    )
    ok = err < 0.05
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def live_frame(scene, model, conf=0.35):
    rgb, _, _, view, proj = render_camera()
    detections, bgr = detect(model, rgb, conf=conf)
    detections = [d for d in detections if d["class_name"] in SCENE_CLASSES]
    for det in detections:
        det["target"] = pixel_to_target(scene, *det["center"], view, proj)
    return bgr, detections


def move_to(scene, x, y, z, steps=2500):
    q = solve_ik(scene, x, y, z)
    if q is None:
        return False
    for _ in range(steps):
        if step_towards(scene, q, steps=1):
            return True
    return False


def move_to_safe(scene, x, y, z, steps=2500):
    """Move to a target, lowering z until the IK stays in the workspace."""
    for dz in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.09):
        if (
            move_to(scene, x, y, z - dz, steps)
            and math.dist(tool_tip(scene), [x, y, z - dz]) < 0.015
        ):
            return True
    return False


def attach_object(scene, obj):
    """Suction on: carry the mug upright under the tool tip like a vacuum cup."""
    tip = np.array(tool_tip(scene))
    mug_pos, _ = p.getBasePositionAndOrientation(obj["body"])
    scene["carry"] = obj
    scene["carry_off"] = np.array(mug_pos) - tip
    scene["carry_mug_orn"] = p.getBasePositionAndOrientation(obj["body"])[1]


def release_object(scene):
    """Suction off: drop whatever is being carried."""
    scene["carry"] = None


GRIPPER_JOINTS = {"simple": (4, 5), "ros": (7, 8)}
GRIPPER_OPEN = 0.0135  # half-stroke per jaw, meters


def gripper_directions(scene):
    """Sign of each jaw's opening stroke, read from the URDF limits.

    The opening bound is whichever limit has the larger magnitude; its sign
    says which way the jaw opens.
    """
    arm = scene["arm"]
    dirs = {}
    for jid in GRIPPER_JOINTS[scene["robot"]]:
        lo, hi = p.getJointInfo(arm, jid)[8:10]
        opening_bound = hi if abs(hi) > abs(lo) else lo
        dirs[jid] = 1.0 if opening_bound >= 0 else -1.0
    return dirs


def set_gripper(scene, opening=GRIPPER_OPEN, force=20.0):
    """Drive both jaws to their open position by `opening` meters (0 = closed)."""
    arm = scene["arm"]
    for jid, sign in gripper_directions(scene).items():
        if scene["robot"] == "ros":
            # The ROS jaw links have no inertia in the URDF; gravity makes the
            # motors unstable, so pose them kinematically like the mimic joints.
            p.resetJointState(arm, jid, sign * opening)
        else:
            p.setJointMotorControl2(
                arm,
                jid,
                p.POSITION_CONTROL,
                targetPosition=sign * opening,
                force=force,
                maxVelocity=0.5,
            )


def grip_object(scene, obj):
    """Gripper grasp: pin the object to the tool link (like a closing jaw grip)."""
    arm = scene["arm"]
    link = TOOL_GRAB_LINK[scene["robot"]]
    ls = p.getLinkState(arm, link, computeForwardKinematics=True)
    link_pos, link_orn = ls[4], ls[5]
    body_pos, body_orn = p.getBasePositionAndOrientation(obj["body"])
    inv_pos, inv_orn = p.invertTransform(link_pos, link_orn)
    local_pos, local_orn = p.multiplyTransforms(inv_pos, inv_orn, body_pos, body_orn)
    constraint = p.createConstraint(
        arm,
        link,
        obj["body"],
        -1,
        p.JOINT_FIXED,
        [0, 0, 0],
        parentFramePosition=local_pos,
        parentFrameOrientation=local_orn,
        childFramePosition=[0, 0, 0],
        childFrameOrientation=[0, 0, 0],
    )
    tip = np.array(tool_tip(scene))
    scene["carry_off"] = np.array(body_pos) - tip
    scene["grasp"] = {"obj": obj, "constraint": constraint}
    return constraint


def release_grip(scene):
    """Open the jaws: drop the constraint holding the object."""
    grasp = scene.get("grasp")
    if grasp:
        p.removeConstraint(grasp["constraint"])
        scene["grasp"] = None


def filter_gripper_piece(scene, body):
    """Disable jaw<->piece collisions; the grasp is constraint-based, so jaw
    contact would only push pieces around while closing."""
    arm = scene["arm"]
    for jid in GRIPPER_JOINTS[scene["robot"]]:
        p.setCollisionFilterPair(arm, body, jid, -1, 0)


def carry_step(scene):
    if not scene.get("carry"):
        return
    tip = np.array(tool_tip(scene))
    target = tip + scene["carry_off"]
    p.resetBasePositionAndOrientation(
        scene["carry"]["body"], target, scene["carry_mug_orn"]
    )
    p.resetBaseVelocity(scene["carry"]["body"], [0, 0, 0], [0, 0, 0])


def go_pick(scene, source, dest):
    """Suction-style pick and place, modeled on DoBotArm2.py goPick(source, dest).

    source is (x, y, z_top) where z_top is the top of the mug.
    """
    release_object(scene)  # suction off
    grip_z = source[2] - 0.015  # tip height that presses onto the rim
    yaw = math.atan2(source[1], source[0])
    radius = math.hypot(source[0], source[1])

    def at(rad, z):
        return move_to_safe(scene, rad * math.cos(yaw), rad * math.sin(yaw), z)

    transit = scene["transit"]
    for rad, z in transit:
        at(rad, z)
    at(radius, source[2] + 0.04)  # move above source along its radial
    move_to(scene, source[0], source[1], grip_z)  # descend onto rim
    grabbed = None  # suction on
    for obj in scene["objects"]:
        contact = p.getContactPoints(scene["arm"], obj["body"])
        if not contact:
            pts = p.getClosestPoints(scene["arm"], obj["body"], 0.03)
            contact = bool(pts)
        if contact:
            grabbed = obj
            attach_object(scene, obj)
            break
    if grabbed:
        off = scene["carry_off"]
        move_to_safe(scene, source[0], source[1], source[2] + 0.06)  # lift
        move_to_safe(
            scene, dest[0] - off[0], dest[1] - off[1], grip_z + 0.10
        )  # above dest
        move_to(scene, dest[0] - off[0], dest[1] - off[1], grip_z)  # descend to table
    release_object(scene)  # suction off
    for _ in range(120):
        p.stepSimulation()
    tip = tool_tip(scene)
    move_to(scene, tip[0], tip[1], tip[2] + 0.08)  # retreat straight up (never lower)
    at(*transit[-1])  # high transit pose, keeps off the table
    for _ in range(800):
        if step_towards(scene, scene["rest_q"], steps=1):
            break
    return grabbed


def pick(scene, model):
    """Detect a cup, then pick it and place it at the drop zone."""
    bgr, detections = live_frame(scene, model)
    candidates = [
        d for d in detections if d["target"] and d["target"]["hit"] == "object"
    ]
    if not candidates:
        return None, bgr, detections
    pick_det = max(candidates, key=lambda d: d["confidence"])
    target = pick_det["target"]
    grabbed = go_pick(
        scene,
        (target["x"], target["y"], target["z_top"]),
        (scene["drop"][0], scene["drop"][1]),
    )
    return grabbed, bgr, detections


def run_gui(args, scene, model):
    state = "IDLE"
    message = "keys: d=detect  p=pick  a=auto cycle  r=reset scene  q=quit"
    detections = []
    bgr = live_frame(scene, model)[0]
    auto = args.auto
    counter = 0

    cv2.namedWindow("Dobot YOLO view", cv2.WINDOW_AUTOSIZE)

    while True:
        keys = p.getKeyboardEvents()
        if ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED:
            break
        if ord("d") in keys and keys[ord("d")] & p.KEY_WAS_TRIGGERED:
            bgr, detections = live_frame(scene, model)
            message = f"{len(detections)} cup(s) detected"
        if ord("p") in keys and keys[ord("p")] & p.KEY_WAS_TRIGGERED:
            state = "PICKING"
        if ord("a") in keys and keys[ord("a")] & p.KEY_WAS_TRIGGERED:
            auto = not auto
            message = f"auto cycle {'ON' if auto else 'OFF'}"
        if ord("r") in keys and keys[ord("r")] & p.KEY_WAS_TRIGGERED:
            reset_scene(scene)
            bgr, detections = live_frame(scene, model)
            message = "scene reset"
        if ord("s") in keys and keys[ord("s")] & p.KEY_WAS_TRIGGERED:
            state, auto = "IDLE", False
            message = "stopped"

        if state == "PICKING" or (auto and state == "IDLE"):
            grabbed, bgr, detections = pick(scene, model)
            state = "IDLE"
            message = (
                f"picked '{detections[0]['class_name']}'"
                if grabbed
                else "no cup reachable"
            )

        counter += 1
        if counter % 10 == 0:
            bgr = live_frame(scene, model)[0]

        overlay = bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            color = (0, 220, 0) if det["target"] else (0, 160, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                overlay,
                f"{det['class_name']} {det['confidence']:.2f}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
        disp = draw_hud(overlay, [], state, message)
        cv2.imshow("Dobot YOLO view", disp)
        cv2.waitKey(1)
        hold_step(scene)
        p.stepSimulation()

    cv2.destroyAllWindows()
    p.disconnect()
    return 0


def reset_scene(scene):
    release_object(scene)
    for obj in scene["objects"]:
        p.resetBasePositionAndOrientation(obj["body"], obj["spawn"], [0, 0, 0, 1])
        p.resetBaseVelocity(obj["body"], [0, 0, 0], [0, 0, 0])
    apply_joint_targets(scene, scene["rest_q"], teleport=True)
    for _ in range(120):
        p.stepSimulation()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless", action="store_true", help="no windows, run sanity test"
    )
    parser.add_argument("--model", default="yolo26s.pt", help="YOLO weights")
    parser.add_argument(
        "--auto", action="store_true", help="auto pick cycle in GUI mode"
    )
    parser.add_argument(
        "--robot", choices=sorted(ROBOTS), default="ros", help="robot model"
    )
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
