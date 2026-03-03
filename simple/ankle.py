#!/usr/bin/env python3
import os
import math
import time
import argparse
import pybullet as p
import pybullet_data

KNEE_BASE = math.radians(20.0)

def simple_angles(side, swing_side, swing_t):
    q = max(0.0, min(1.0, swing_t))
    s = math.sin(math.pi * q)
    hip_pitch = 0.0
    knee_pitch = KNEE_BASE
    if side == swing_side:
        hip_pitch += math.radians(10.0) * s
        knee_pitch += math.radians(14.0) * s
    else:
        hip_pitch -= math.radians(3.0) * s
        knee_pitch += math.radians(4.0) * s
    return hip_pitch, knee_pitch

def joint_name_map(robot_id):
    m = {}
    n = p.getNumJoints(robot_id)
    for i in range(n):
        info = p.getJointInfo(robot_id, i)
        m[info[1].decode("utf-8")] = i
    return m

def build_pitch_map(name_to_idx):
    hardcoded_map = {
        "left_hip_pitch": "Revolute_4",
        "left_knee_pitch": "Revolute_3",
        "left_ankle_pitch": "Revolute_2",
        "right_hip_pitch": "Revolute_10",
        "right_knee_pitch": "Revolute_11",
        "right_ankle_pitch": "Revolute_12",
    }
    out = {}
    for req, urdf_name in hardcoded_map.items():
        if urdf_name not in name_to_idx:
            raise RuntimeError(f"Brak mapowania jointa: {urdf_name}")
        out[req] = name_to_idx[urdf_name]
    return out

def set_joint(robot_id, mp, key, val):
    p.resetJointState(robot_id, mp[key], val)

def zero_non_pitch(robot_id, name_to_idx):
    pitch_joints = {"Revolute_4", "Revolute_3", "Revolute_2", "Revolute_10", "Revolute_11", "Revolute_12"}
    for name, idx in name_to_idx.items():
        if name in pitch_joints:
            continue
        p.resetJointState(robot_id, idx, 0.0)

def draw_text(old_id, txt, pos, color):
    if old_id is not None:
        p.removeUserDebugItem(old_id)
    return p.addUserDebugText(txt, pos, textColorRGB=color, textSize=1.2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", type=str, default="beta.urdf")
    ap.add_argument("--mode", type=str, default="both", choices=["raw", "comp", "both"])
    ap.add_argument("--hz", type=float, default=60.0)
    ap.add_argument("--swing-time", type=float, default=0.56)
    ap.add_argument("--ankle-offset-deg", type=float, default=0.0)
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    urdf_path = args.urdf if os.path.isabs(args.urdf) else os.path.join(root, args.urdf)
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"Brak URDF: {urdf_path}")

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.resetDebugVisualizerCamera(cameraDistance=1.25, cameraYaw=35, cameraPitch=-18, cameraTargetPosition=[0, 0, 0.2])
    p.loadURDF("plane.urdf", [0, 0, -0.34])

    flags = p.URDF_USE_INERTIA_FROM_FILE

    if args.mode == "both":
        raw_id = p.loadURDF(urdf_path, [-0.20, 0.0, 0.0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True, flags=flags)
        comp_id = p.loadURDF(urdf_path, [0.20, 0.0, 0.0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True, flags=flags)

        raw_names = joint_name_map(raw_id)
        comp_names = joint_name_map(comp_id)

        raw_map = build_pitch_map(raw_names)
        comp_map = build_pitch_map(comp_names)
    else:
        rid = p.loadURDF(urdf_path, [0.0, 0.0, 0.0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True, flags=flags)
        one_names = joint_name_map(rid)
        one_map = build_pitch_map(one_names)

    dt = 1.0 / args.hz
    t0 = time.time()
    ankle_offset = math.radians(args.ankle_offset_deg)

    txt_a = None
    txt_b = None

    while p.isConnected():
        t = time.time() - t0
        phase = (t % (2.0 * args.swing_time)) / args.swing_time
        if phase < 1.0:
            swing_side = 0
            swing_t = phase
        else:
            swing_side = 1
            swing_t = phase - 1.0

        lh, lk = simple_angles(0, swing_side, swing_t)
        rh, rk = simple_angles(1, swing_side, swing_t)

        la_raw = -KNEE_BASE
        ra_raw = -KNEE_BASE
        la_comp = -lh + lk + ankle_offset
        ra_comp = rh - rk + ankle_offset

        if args.mode == "raw":
            zero_non_pitch(rid, one_names)
            set_joint(rid, one_map, "left_hip_pitch", lh)
            set_joint(rid, one_map, "left_knee_pitch", lk)
            set_joint(rid, one_map, "left_ankle_pitch", la_raw)
            set_joint(rid, one_map, "right_hip_pitch", rh)
            set_joint(rid, one_map, "right_knee_pitch", rk)
            set_joint(rid, one_map, "right_ankle_pitch", ra_raw)

        elif args.mode == "comp":
            zero_non_pitch(rid, one_names)
            set_joint(rid, one_map, "left_hip_pitch", lh)
            set_joint(rid, one_map, "left_knee_pitch", lk)
            set_joint(rid, one_map, "left_ankle_pitch", la_comp)
            set_joint(rid, one_map, "right_hip_pitch", rh)
            set_joint(rid, one_map, "right_knee_pitch", rk)
            set_joint(rid, one_map, "right_ankle_pitch", ra_comp)

        else:
            zero_non_pitch(raw_id, raw_names)
            zero_non_pitch(comp_id, comp_names)

            set_joint(raw_id, raw_map, "left_hip_pitch", lh)
            set_joint(raw_id, raw_map, "left_knee_pitch", lk)
            set_joint(raw_id, raw_map, "left_ankle_pitch", la_raw)
            set_joint(raw_id, raw_map, "right_hip_pitch", rh)
            set_joint(raw_id, raw_map, "right_knee_pitch", rk)
            set_joint(raw_id, raw_map, "right_ankle_pitch", ra_raw)

            set_joint(comp_id, comp_map, "left_hip_pitch", lh)
            set_joint(comp_id, comp_map, "left_knee_pitch", lk)
            set_joint(comp_id, comp_map, "left_ankle_pitch", la_comp)
            set_joint(comp_id, comp_map, "right_hip_pitch", rh)
            set_joint(comp_id, comp_map, "right_knee_pitch", rk)
            set_joint(comp_id, comp_map, "right_ankle_pitch", ra_comp)

        txt_a = draw_text(txt_a, f"mode={args.mode}  offset={args.ankle_offset_deg:.2f} deg", [-0.35, -0.35, 0.52], [1, 1, 1])
        txt_b = draw_text(txt_b, f"swing={'LEFT' if swing_side==0 else 'RIGHT'}  p={swing_t:.2f}", [-0.35, -0.35, 0.46], [0.5, 1, 0.5])

        p.stepSimulation()
        time.sleep(dt)

if __name__ == "__main__":
    main()
