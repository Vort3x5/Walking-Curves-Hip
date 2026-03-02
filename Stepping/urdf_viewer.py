#!/usr/bin/env python3
import argparse
import os
import time
import pybullet as p
import pybullet_data
import yarp

PORT_LEFT_ANGLES = "/gait/left/angles"
PORT_RIGHT_ANGLES = "/gait/right/angles"

LEFT_ORDER = [
    "left_hip_roll", "left_hip_pitch", "left_knee_pitch",
    "left_ankle_pitch", "left_ankle_roll", "left_ankle_yaw"
]
RIGHT_ORDER = [
    "right_hip_roll", "right_hip_pitch", "right_knee_pitch",
    "right_ankle_pitch", "right_ankle_roll", "right_ankle_yaw"
]

def bottle_add_f64(b, v):
    if hasattr(b, "addFloat64"):
        b.addFloat64(float(v))
    elif hasattr(b, "addDouble"):
        b.addDouble(float(v))
    else:
        b.add(float(v))

def open_yarp_out(name):
    p_out = yarp.Port()
    ok = p_out.open(name)
    if not ok:
        raise RuntimeError(f"cannot open yarp port: {name}")
    p_out.enableBackgroundWrite(True)
    return p_out

def write_angles(port, vals):
    b = yarp.Bottle()
    for v in vals:
        bottle_add_f64(b, v)
    port.write(b)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", default="../beta.urdf")
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--dt", type=float, default=1.0 / 240.0)
    parser.add_argument("--base-z", type=float, default=0.35)
    parser.add_argument("--yarp", action="store_true")
    parser.add_argument("--left-port", default="/urdf_viewer/left_cmd:o")
    parser.add_argument("--right-port", default="/urdf_viewer/right_cmd:o")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.abspath(os.path.join(base_dir, args.urdf))
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(urdf_path)

    if p.connect(p.GUI) < 0:
        raise RuntimeError("PyBullet GUI connection failed")

    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_path = os.path.join(pybullet_data.getDataPath(), "plane.urdf")
    if os.path.exists(plane_path):
        p.loadURDF("plane.urdf")
    p.setAdditionalSearchPath(os.path.dirname(urdf_path))

    robot = p.loadURDF(
        urdf_path,
        [0, 0, args.base_z],
        p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=args.fixed,
        flags=p.URDF_USE_INERTIA_FROM_FILE
    )

    for j in range(p.getNumJoints(robot)):
        p.changeDynamics(robot, j, linearDamping=0.04, angularDamping=0.04)
        p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL, force=0)

    sliders = []
    for j in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, j)
        joint_name = info[1].decode("utf-8", errors="ignore")
        joint_type = info[2]
        lower = info[8]
        upper = info[9]
        if joint_type not in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            continue
        if lower > upper:
            lower, upper = -3.14159, 3.14159
        st = p.getJointState(robot, j)[0]
        sid = p.addUserDebugParameter(joint_name, lower, upper, st)
        sliders.append((j, sid, joint_name))

    yarp_left = None
    yarp_right = None
    if args.yarp:
        yarp.Network.init()
        if not yarp.Network.checkNetwork(3.0):
            raise RuntimeError("yarpserver not reachable")
        yarp_left = open_yarp_out(args.left_port)
        yarp_right = open_yarp_out(args.right_port)
        yarp.Network.connect(args.left_port, PORT_LEFT_ANGLES)
        yarp.Network.connect(args.right_port, PORT_RIGHT_ANGLES)

    print(f"[viewer] loaded: {urdf_path}")
    print(f"[viewer] sliders: {len(sliders)}")
    if args.yarp:
        print(f"[viewer] yarp out: {args.left_port} -> {PORT_LEFT_ANGLES}")
        print(f"[viewer] yarp out: {args.right_port} -> {PORT_RIGHT_ANGLES}")
    print("[viewer] Ctrl+C to quit")

    try:
        while True:
            current = {}
            for j, sid, jname in sliders:
                target = p.readUserDebugParameter(sid)
                current[jname] = target
                p.setJointMotorControl2(
                    bodyUniqueId=robot,
                    jointIndex=j,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target,
                    force=200.0,
                    positionGain=0.2,
                    velocityGain=1.0
                )

            if args.yarp:
                left_vals = [current.get(n, 0.0) for n in LEFT_ORDER]
                right_vals = [current.get(n, 0.0) for n in RIGHT_ORDER]
                write_angles(yarp_left, left_vals)
                write_angles(yarp_right, right_vals)

            p.stepSimulation()
            time.sleep(args.dt)
    finally:
        if yarp_left is not None:
            yarp_left.close()
        if yarp_right is not None:
            yarp_right.close()
        if args.yarp:
            yarp.Network.fini()

if __name__ == "__main__":
    main()
