#!/usr/bin/env python3
import time, sys, math, threading, os
import numpy as np
import yarp

try:
    from ik_solver import LegIK
    _IK_AVAILABLE = True
except ImportError as _e:
    print(f"[gait] IK unavailable: {_e}")
    _IK_AVAILABLE = False

try:
    import yaml as _yaml
    def _load_yaml(path):
        with open(path) as f:
            return _yaml.safe_load(f)
except ImportError:
    def _load_yaml(path):
        raise RuntimeError("PyYAML not installed")

HIP_WIDTH = 0.08
THIGH_LEN = 0.15
SHANK_LEN = 0.15
STEP_LEN = 0.03
STEP_HEIGHT = 0.02
STAND_HEIGHT = -(THIGH_LEN + SHANK_LEN) * 0.87

RATE_HZ = 50
DT = 1.0 / RATE_HZ
SWING_TIME = 0.56
TURN_STEP_ANGLE = math.radians(10)

PORT_CMD = "/robot/cmd"
PORT_LEFT = "/gait/left/foot"
PORT_RIGHT = "/gait/right/foot"
PORT_LEFT_ANGLES = "/gait/left/angles"
PORT_RIGHT_ANGLES = "/gait/right/angles"
PORT_VIZ = "/gait/viz"

LEFT_JOINT_ORDER = ["left_hip_roll", "left_hip_pitch", "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll", "left_ankle_yaw"]
RIGHT_JOINT_ORDER = ["right_hip_roll", "right_hip_pitch", "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll", "right_ankle_yaw"]

_DEFAULT_CURVE_PROFILE = [
    [0.00, 0.00, 0.00],
    [0.20, 0.00, 0.25],
    [0.45, 0.00, 0.95],
    [0.60, 0.00, 1.00],
    [0.82, 0.00, 0.55],
    [1.00, 0.00, 0.00],
]

HIP_ROLL_MAX = math.radians(7.0)
HIP_PITCH_MAX = math.radians(5.0)
HIP_YAW_MAX = math.radians(8.0)
ANKLE_ROLL_COMP = 0.65
ANKLE_PITCH_COMP = 0.60
KNEE_BASE = math.radians(20.0)

COM_SHIFT_BASE = 0.018
COM_SHIFT_GAIN_PHASE = 0.014
COM_SHIFT_FILTER = 0.22

ROLL_FILTER = 0.20
PITCH_FILTER = 0.20
YAW_FILTER = 0.22

TURN_YAW_GAIN = 1.15
TURN_ROLL_GAIN = 0.35
TURN_PITCH_GAIN = 0.25

def add_f64(b, v):
    if hasattr(b, "addFloat64"):
        b.addFloat64(float(v))
    elif hasattr(b, "addDouble"):
        b.addDouble(float(v))
    else:
        b.add(float(v))

def add_i32(b, v):
    if hasattr(b, "addInt32"):
        b.addInt32(int(v))
    elif hasattr(b, "addInt"):
        b.addInt(int(v))
    else:
        b.add(int(v))

def load_swing_profile():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(here, "swing_curves.yaml"), os.path.join(here, "swing curves.yaml")]:
        try:
            data = _load_yaml(p)
            a = data.get("active_curve", "normal")
            c = data.get("curves", {})
            if a in c and "points" in c[a]:
                print(f"[gait] loaded curve '{a}' from {p}")
                return c[a]["points"]
        except Exception:
            pass
    print("[gait] curve file not usable, using default")
    return _DEFAULT_CURVE_PROFILE

def make_swing_curve(start, end, height, profile):
    s = start.copy()
    e = end.copy()
    horiz = np.array([e[0]-s[0], e[1]-s[1], 0.0])
    hl = np.linalg.norm(horiz[:2])
    perp = np.array([-horiz[1], horiz[0], 0.0]) / hl if hl > 1e-6 else np.array([0., 1., 0.])
    pts = []
    for xf, yf, zf in profile:
        pts.append(np.array([s[0] + xf*horiz[0] + yf*perp[0], s[1] + xf*horiz[1] + yf*perp[1], s[2] + (e[2]-s[2])*xf + zf*height]))
    pts[0] = s.copy()
    pts[-1] = e.copy()
    return np.array(pts, dtype=float)

def bezier(pts, t):
    p = pts.copy()
    n = len(p)
    for r in range(1, n):
        p[:n-r] = (1-t)*p[:n-r] + t*p[1:n-r+1]
    return p[0]

class Side:
    LEFT = 0
    RIGHT = 1
    OTHER = {0: 1, 1: 0}

class GaitState:
    IDLE = "IDLE"
    STEPPING = "STEPPING"
    STOPPING = "STOPPING"
    TURNING = "TURNING"

class GaitCoordinator:
    def __init__(self, swing_profile):
        self.swing_profile = swing_profile
        self.heading = 0.0
        self.target_heading = 0.0
        self.body_pos = np.array([0.0, 0.0])
        self.foot_world = [np.array([-HIP_WIDTH, 0.0, STAND_HEIGHT]), np.array([HIP_WIDTH, 0.0, STAND_HEIGHT])]
        self.state = GaitState.IDLE
        self.swing_side = Side.LEFT
        self.swing_t = 0.0
        self.swing_curve = None
        self.swing_end = None
        self.turn_queue = []
        self.turn_heading_end = 0.0
        self.turn_resume = GaitState.IDLE
        self.cmd_lock = threading.Lock()
        self.cmd = "stop"
        self.hip_roll_bias = 0.0
        self.hip_pitch_bias = 0.0
        self.hip_yaw_bias = 0.0
        self.com_lateral_shift = 0.0
        self.turn_cmd_sign = 0.0

    def set_command(self, c):
        with self.cmd_lock:
            if c in ("left", "right"):
                self.turn_queue.append(c)
            else:
                self.cmd = c

    def _get_state(self):
        with self.cmd_lock:
            return self.cmd, list(self.turn_queue)

    def _pop_turn(self):
        with self.cmd_lock:
            return self.turn_queue.pop(0) if self.turn_queue else None

    def _hip_world(self, side):
        sign = -1 if side == Side.LEFT else 1
        perp = np.array([-math.sin(self.heading), math.cos(self.heading)])
        hip_xy = self.body_pos + sign * HIP_WIDTH * perp
        return np.array([hip_xy[0], hip_xy[1], 0.0])

    def _next_foot_pos(self, side):
        fwd = np.array([math.cos(self.heading), math.sin(self.heading)])
        perp = np.array([-math.sin(self.heading), math.cos(self.heading)])
        sign = -1 if side == Side.LEFT else 1
        xy = self.body_pos + fwd * STEP_LEN + sign * HIP_WIDTH * perp
        return np.array([xy[0], xy[1], STAND_HEIGHT])

    def _turn_foot_pos(self, swing_side, new_heading):
        pivot = self.foot_world[Side.OTHER[swing_side]].copy()
        sign = -1 if swing_side == Side.LEFT else 1
        perp = np.array([-math.sin(new_heading), math.cos(new_heading)])
        xy = pivot[:2] + sign * 2 * HIP_WIDTH * perp
        return np.array([xy[0], xy[1], STAND_HEIGHT])

    def _advance_body(self):
        fwd = np.array([math.cos(self.heading), math.sin(self.heading)])
        self.body_pos = self.body_pos + fwd * STEP_LEN * 0.35

    def _start_swing(self, side):
        self.swing_side = side
        self.swing_t = 0.0
        start = self.foot_world[side].copy()
        end = self._next_foot_pos(side)
        self.swing_end = end
        self.swing_curve = make_swing_curve(start, end, STEP_HEIGHT, self.swing_profile)

    def _begin_turn(self, direction, resume_state):
        d = +1 if direction == "left" else -1
        self.turn_cmd_sign = float(d)
        swing_side = Side.RIGHT if direction == "left" else Side.LEFT
        self.turn_resume = resume_state
        self.turn_heading_end = self.heading + d * TURN_STEP_ANGLE
        start = self.foot_world[swing_side].copy()
        end = self._turn_foot_pos(swing_side, self.turn_heading_end)
        self.swing_side = swing_side
        self.swing_t = 0.0
        self.swing_end = end
        self.swing_curve = make_swing_curve(start, end, STEP_HEIGHT, self.swing_profile)
        self.state = GaitState.TURNING

    def _update_balance(self):
        support = Side.OTHER[self.swing_side] if self.state != GaitState.IDLE else Side.LEFT
        p = max(0.0, min(1.0, self.swing_t if self.state != GaitState.IDLE else 0.0))
        phase = math.sin(math.pi * p)

        roll_target = (-HIP_ROLL_MAX if support == Side.LEFT else HIP_ROLL_MAX) * (0.55 + 0.45 * phase)
        pitch_target = HIP_PITCH_MAX * (0.20 + 0.80 * phase) if self.state in (GaitState.STEPPING, GaitState.TURNING, GaitState.STOPPING) else 0.0

        if self.state == GaitState.TURNING:
            yaw_target = self.turn_cmd_sign * HIP_YAW_MAX * (0.35 + 0.65 * phase) * TURN_YAW_GAIN
            roll_target += self.turn_cmd_sign * HIP_ROLL_MAX * TURN_ROLL_GAIN * (0.4 + 0.6 * phase)
            pitch_target += HIP_PITCH_MAX * TURN_PITCH_GAIN * (0.3 + 0.7 * phase)
        else:
            yaw_target = 0.0

        lat_shift_target = (-1.0 if support == Side.LEFT else 1.0) * (COM_SHIFT_BASE + COM_SHIFT_GAIN_PHASE * phase)

        self.com_lateral_shift += COM_SHIFT_FILTER * (lat_shift_target - self.com_lateral_shift)
        self.hip_roll_bias += ROLL_FILTER * (roll_target - self.hip_roll_bias)
        self.hip_pitch_bias += PITCH_FILTER * (pitch_target - self.hip_pitch_bias)
        self.hip_yaw_bias += YAW_FILTER * (yaw_target - self.hip_yaw_bias)

    def tick(self):
        cmd, q = self._get_state()
        if q and self.state != GaitState.TURNING:
            d = self._pop_turn()
            self._begin_turn(d, GaitState.STEPPING if cmd == "forward" else GaitState.IDLE)

        if self.state == GaitState.IDLE:
            if cmd == "forward":
                self._start_swing(Side.LEFT)
                self.state = GaitState.STEPPING
        elif self.state == GaitState.STEPPING:
            if cmd == "stop":
                self.state = GaitState.STOPPING
            self.swing_t += DT / SWING_TIME
            if self.swing_t >= 1.0:
                self.foot_world[self.swing_side] = self.swing_end.copy()
                self._advance_body()
                self._start_swing(Side.OTHER[self.swing_side])
        elif self.state == GaitState.STOPPING:
            self.swing_t += DT / SWING_TIME
            if self.swing_t >= 1.0:
                self.foot_world[self.swing_side] = self.swing_end.copy()
                self._advance_body()
                lf, rf = self.foot_world[Side.LEFT], self.foot_world[Side.RIGHT]
                if abs(lf[0] - rf[0]) < STEP_LEN * 0.5:
                    self.state = GaitState.IDLE
                else:
                    self._start_swing(Side.OTHER[self.swing_side])
            if cmd == "forward":
                self.state = GaitState.STEPPING
        elif self.state == GaitState.TURNING:
            self.swing_t += DT / SWING_TIME
            hdiff = (self.turn_heading_end - self.heading + math.pi) % (2*math.pi) - math.pi
            self.heading += hdiff * min(1.0, 4.0 * DT)
            if self.swing_t >= 1.0:
                self.foot_world[self.swing_side] = self.swing_end.copy()
                self.heading = self.turn_heading_end
                self.target_heading = self.heading
                lf, rf = self.foot_world[Side.LEFT], self.foot_world[Side.RIGHT]
                self.body_pos = np.array([(lf[0]+rf[0])*0.5, (lf[1]+rf[1])*0.5])
                nxt = self._pop_turn()
                if nxt is not None:
                    self._begin_turn(nxt, GaitState.STEPPING if cmd == "forward" else GaitState.IDLE)
                else:
                    self.state = self.turn_resume
                    self.turn_cmd_sign = 0.0
                    if self.state == GaitState.STEPPING:
                        self._start_swing(Side.OTHER[self.swing_side])

        if self.state != GaitState.TURNING:
            hdiff = (self.target_heading - self.heading + math.pi) % (2*math.pi) - math.pi
            self.heading += hdiff * min(1.0, 6.0 * DT)

        if self.state != GaitState.IDLE and self.swing_curve is not None:
            self.foot_world[self.swing_side] = bezier(self.swing_curve, max(0.0, min(1.0, self.swing_t)))

        self._update_balance()

        feet_hip_rel = [self.foot_world[s] - self._hip_world(s) for s in (Side.LEFT, Side.RIGHT)]
        feet_hip_rel[0][1] -= self.com_lateral_shift
        feet_hip_rel[1][1] -= self.com_lateral_shift

        return {
            "state": self.state,
            "heading": self.heading,
            "body_pos": self.body_pos.copy(),
            "foot_world": [f.copy() for f in self.foot_world],
            "feet_hip_rel": feet_hip_rel,
            "swing_side": self.swing_side,
            "swing_t": self.swing_t,
            "swing_curve": self.swing_curve,
            "hip_roll_bias": self.hip_roll_bias,
            "hip_pitch_bias": self.hip_pitch_bias,
            "hip_yaw_bias": self.hip_yaw_bias,
            "com_lateral_shift": self.com_lateral_shift,
        }

class CmdListener(threading.Thread):
    def __init__(self, gait):
        super().__init__(daemon=True)
        self.gait = gait
        self.port = yarp.Port()
        self.stop_evt = threading.Event()

    def run(self):
        self.port.open(PORT_CMD)
        while not self.stop_evt.is_set():
            b = yarp.Bottle()
            if self.port.read(b, False):
                if b.size() > 0:
                    self.gait.set_command(b.get(0).asString())
            else:
                time.sleep(0.01)

    def stop(self):
        self.stop_evt.set()
        self.port.interrupt()
        self.port.close()

def open_port_unique(base_name):
    p = yarp.Port()
    if p.open(base_name):
        p.enableBackgroundWrite(True)
        return p, base_name
    alt = f"{base_name}_{int(time.time()*1000)}"
    if p.open(alt):
        p.enableBackgroundWrite(True)
        print(f"[gait] port conflict on {base_name}, using {alt}")
        return p, alt
    raise RuntimeError(f"cannot open yarp port: {base_name}")

def simple_angles(side, swing_side, swing_t, roll_bias, pitch_bias, yaw_bias):
    p = max(0.0, min(1.0, swing_t))
    s = math.sin(math.pi * p)
    hip_pitch = 0.0
    knee = KNEE_BASE
    ankle_pitch = -0.5 * KNEE_BASE
    hip_yaw = 0.0
    if side == swing_side:
        hip_pitch += math.radians(10.0) * s
        knee += math.radians(14.0) * s
        ankle_pitch -= math.radians(6.0) * s
        hip_yaw += yaw_bias * (0.55 + 0.45 * s)
    else:
        hip_pitch -= math.radians(3.0) * s
        knee += math.radians(4.0) * s
        hip_yaw -= yaw_bias * 0.35
    if side == Side.LEFT:
        return {
            "left_hip_roll": roll_bias,
            "left_hip_pitch": hip_pitch + pitch_bias,
            "left_knee_pitch": knee,
            "left_ankle_pitch": ankle_pitch - ANKLE_PITCH_COMP * pitch_bias,
            "left_ankle_roll": -ANKLE_ROLL_COMP * roll_bias,
            "left_ankle_yaw": hip_yaw
        }
    return {
        "right_hip_roll": roll_bias,
        "right_hip_pitch": hip_pitch + pitch_bias,
        "right_knee_pitch": knee,
        "right_ankle_pitch": ankle_pitch - ANKLE_PITCH_COMP * pitch_bias,
        "right_ankle_roll": -ANKLE_ROLL_COMP * roll_bias,
        "right_ankle_yaw": hip_yaw
    }

def gait_loop(gait, ports, ik):
    port_left, port_right, port_viz, port_left_angles, port_right_angles = ports
    while True:
        t0 = time.monotonic()
        snap = gait.tick()

        for port, rel in zip((port_left, port_right), snap["feet_hip_rel"]):
            b = yarp.Bottle()
            for v in rel:
                add_f64(b, v)
            port.write(b)

        support = Side.OTHER[snap["swing_side"]] if snap["state"] != GaitState.IDLE else Side.LEFT
        if support == Side.LEFT:
            left_roll = snap["hip_roll_bias"]
            right_roll = 0.0
        else:
            left_roll = 0.0
            right_roll = snap["hip_roll_bias"]

        if ik is not None:
            foot_l, foot_r = snap["foot_world"][0], snap["foot_world"][1]
            bp, hdg = snap["body_pos"], snap["heading"]
            cos_h, sin_h = math.cos(-hdg), math.sin(-hdg)

            def hip_world(side):
                sign = -1 if side == Side.LEFT else 1
                px, py = -math.sin(hdg), math.cos(hdg)
                return np.array([bp[0] + sign*HIP_WIDTH*px, bp[1] + sign*HIP_WIDTH*py, 0.0])

            def to_hip_rel(foot, hip):
                dx, dy, dz = foot[0]-hip[0], foot[1]-hip[1], foot[2]-hip[2]
                return np.array([cos_h*dx - sin_h*dy, sin_h*dx + cos_h*dy, dz])

            left_rel = to_hip_rel(foot_l, hip_world(Side.LEFT))
            right_rel = to_hip_rel(foot_r, hip_world(Side.RIGHT))

            left_rel[1] -= snap["com_lateral_shift"]
            right_rel[1] -= snap["com_lateral_shift"]

            left_angles = ik.solve_left(left_rel)
            right_angles = ik.solve_right(right_rel)

            if support == Side.LEFT:
                left_angles["left_hip_roll"] = left_angles.get("left_hip_roll", 0.0) + snap["hip_roll_bias"]
                left_angles["left_hip_pitch"] = left_angles.get("left_hip_pitch", 0.0) + snap["hip_pitch_bias"]
                left_angles["left_ankle_roll"] = left_angles.get("left_ankle_roll", 0.0) - ANKLE_ROLL_COMP * snap["hip_roll_bias"]
                left_angles["left_ankle_pitch"] = left_angles.get("left_ankle_pitch", 0.0) - ANKLE_PITCH_COMP * snap["hip_pitch_bias"]
                left_angles["left_ankle_yaw"] = left_angles.get("left_ankle_yaw", 0.0) + snap["hip_yaw_bias"] * (0.35 if snap["state"] == GaitState.TURNING else 0.18)
                right_angles["right_ankle_yaw"] = right_angles.get("right_ankle_yaw", 0.0) - snap["hip_yaw_bias"] * 0.10
            else:
                right_angles["right_hip_roll"] = right_angles.get("right_hip_roll", 0.0) + snap["hip_roll_bias"]
                right_angles["right_hip_pitch"] = right_angles.get("right_hip_pitch", 0.0) + snap["hip_pitch_bias"]
                right_angles["right_ankle_roll"] = right_angles.get("right_ankle_roll", 0.0) - ANKLE_ROLL_COMP * snap["hip_roll_bias"]
                right_angles["right_ankle_pitch"] = right_angles.get("right_ankle_pitch", 0.0) - ANKLE_PITCH_COMP * snap["hip_pitch_bias"]
                right_angles["right_ankle_yaw"] = right_angles.get("right_ankle_yaw", 0.0) + snap["hip_yaw_bias"] * (0.35 if snap["state"] == GaitState.TURNING else 0.18)
                left_angles["left_ankle_yaw"] = left_angles.get("left_ankle_yaw", 0.0) - snap["hip_yaw_bias"] * 0.10
        else:
            left_angles = simple_angles(Side.LEFT, snap["swing_side"], snap["swing_t"], left_roll, snap["hip_pitch_bias"] if support == Side.LEFT else 0.0, snap["hip_yaw_bias"])
            right_angles = simple_angles(Side.RIGHT, snap["swing_side"], snap["swing_t"], right_roll, snap["hip_pitch_bias"] if support == Side.RIGHT else 0.0, snap["hip_yaw_bias"])

        for port_a, angles, order in [
            (port_left_angles, left_angles, LEFT_JOINT_ORDER),
            (port_right_angles, right_angles, RIGHT_JOINT_ORDER),
        ]:
            ab = yarp.Bottle()
            for j in order:
                add_f64(ab, angles.get(j, 0.0))
            port_a.write(ab)

        vb = yarp.Bottle()
        vb.addString(snap["state"])
        add_f64(vb, snap["heading"])
        add_f64(vb, snap["body_pos"][0]); add_f64(vb, snap["body_pos"][1])
        for x in snap["foot_world"][0]: add_f64(vb, x)
        for x in snap["foot_world"][1]: add_f64(vb, x)
        add_i32(vb, snap["swing_side"])
        add_f64(vb, snap["swing_t"])
        add_f64(vb, 1.0)
        if snap["swing_curve"] is not None:
            for pt in snap["swing_curve"]:
                for x in pt: add_f64(vb, x)
        port_viz.write(vb)

        wait = DT - (time.monotonic() - t0)
        if wait > 0:
            time.sleep(wait)

def main():
    yarp.Network.init()
    if not yarp.Network.checkNetwork(3.0):
        print("[gait] ERROR: yarpserver not reachable")
        sys.exit(1)

    gait = GaitCoordinator(load_swing_profile())
    listener = CmdListener(gait)
    listener.start()

    port_left, left_name = open_port_unique(PORT_LEFT)
    port_right, right_name = open_port_unique(PORT_RIGHT)
    port_viz, viz_name = open_port_unique(PORT_VIZ)
    port_left_angles, la_name = open_port_unique(PORT_LEFT_ANGLES)
    port_right_angles, ra_name = open_port_unique(PORT_RIGHT_ANGLES)

    print(f"[gait] out: {left_name}")
    print(f"[gait] out: {right_name}")
    print(f"[gait] out: {viz_name}")
    print(f"[gait] out: {la_name}")
    print(f"[gait] out: {ra_name}")

    ik = None
    if _IK_AVAILABLE:
        try:
            ik = LegIK()
        except Exception as e:
            print(f"[gait] IK load failed, using fallback angles: {e}")

    threading.Thread(target=gait_loop, args=(gait, (port_left, port_right, port_viz, port_left_angles, port_right_angles), ik), daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        for p in (port_left, port_right, port_viz, port_left_angles, port_right_angles):
            p.close()
        yarp.Network.fini()

if __name__ == "__main__":
    main()
