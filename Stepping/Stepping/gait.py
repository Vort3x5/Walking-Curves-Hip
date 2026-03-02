#!/usr/bin/env python3

import time, sys, math, threading, os, subprocess, queue
import numpy as np

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

HIP_WIDTH    = 0.08
THIGH_LEN    = 0.15
SHANK_LEN    = 0.15
STEP_LEN     = 0.06
STEP_HEIGHT  = 0.05
STAND_HEIGHT = -(THIGH_LEN + SHANK_LEN) * 0.85

RATE_HZ         = 50
DT              = 1.0 / RATE_HZ
SWING_TIME      = 0.40
TURN_STEP_ANGLE = math.radians(20)

PORT_CMD          = "/robot/cmd"
PORT_LEFT         = "/gait/left/foot"
PORT_RIGHT        = "/gait/right/foot"
PORT_LEFT_ANGLES  = "/gait/left/angles"
PORT_RIGHT_ANGLES = "/gait/right/angles"
PORT_VIZ          = "/gait/viz"

LEFT_JOINT_ORDER  = ["left_hip_roll",  "left_hip_pitch",  "left_knee_pitch",
                     "left_ankle_pitch",  "left_ankle_roll",  "left_ankle_yaw"]
RIGHT_JOINT_ORDER = ["right_hip_roll", "right_hip_pitch", "right_knee_pitch",
                     "right_ankle_pitch", "right_ankle_roll", "right_ankle_yaw"]

_DEFAULT_CURVE_PROFILE = [
    [0.00, 0.00, 0.00],
    [0.15, 0.00, 0.20],
    [0.35, 0.00, 0.90],
    [0.50, 0.00, 1.00],
    [0.70, 0.00, 0.60],
    [1.00, 0.00, 0.00],
]


def load_swing_profile(yaml_path=None):
    if yaml_path is None:
        here      = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(here, "swing curves.yaml")
        print(f"[gait] swing curves: {yaml_path}")
    try:
        data        = _load_yaml(yaml_path)
        active_name = data.get("active_curve", "normal")
        curves      = data.get("curves", {})
        if active_name not in curves:
            print(f"[gait] curve '{active_name}' not found, using default")
            return _DEFAULT_CURVE_PROFILE
        print(f"[gait] loaded curve '{active_name}'")
        return curves[active_name]["points"]
    except FileNotFoundError:
        print(f"[gait] swing_curves.yaml not found, using default")
        return _DEFAULT_CURVE_PROFILE
    except Exception as e:
        print(f"[gait] curve load error ({e}), using default")
        return _DEFAULT_CURVE_PROFILE


def make_swing_curve(start, end, height, profile=None):
    if profile is None:
        profile = _DEFAULT_CURVE_PROFILE
    s     = start.copy()
    e     = end.copy()
    horiz = np.array([e[0]-s[0], e[1]-s[1], 0.0])
    hl    = np.linalg.norm(horiz[:2])
    perp  = np.array([-horiz[1], horiz[0], 0.0]) / hl if hl > 1e-6 else np.array([0., 1., 0.])
    pts   = []
    for xf, yf, zf in profile:
        pts.append(np.array([
            s[0] + xf*horiz[0] + yf*perp[0],
            s[1] + xf*horiz[1] + yf*perp[1],
            s[2] + (e[2]-s[2])*xf + zf*height,
        ]))
    pts[0]  = s.copy()
    pts[-1] = e.copy()
    return np.array(pts, dtype=float)


def bezier(pts, t):
    p = pts.copy()
    n = len(p)
    for r in range(1, n):
        p[:n-r] = (1-t)*p[:n-r] + t*p[1:n-r+1]
    return p[0]


class Side:
    LEFT  = 0
    RIGHT = 1
    OTHER = {0: 1, 1: 0}


class GaitState:
    IDLE     = "IDLE"
    STEPPING = "STEPPING"
    STOPPING = "STOPPING"
    TURNING  = "TURNING"


class GaitCoordinator:
    def __init__(self, swing_profile=None):
        self.swing_profile    = swing_profile if swing_profile is not None else _DEFAULT_CURVE_PROFILE
        self.heading          = 0.0
        self.target_heading   = 0.0
        self.body_pos         = np.array([0.0, 0.0])
        self.foot_world       = [
            np.array([-HIP_WIDTH, 0.0, STAND_HEIGHT]),
            np.array([ HIP_WIDTH, 0.0, STAND_HEIGHT]),
        ]
        self.state            = GaitState.IDLE
        self.swing_side       = Side.LEFT
        self.swing_t          = 0.0
        self.swing_curve      = None
        self.swing_end        = None
        self.speed_scale      = 1.0
        self.turn_queue       = []
        self.turn_dir         = 0
        self.turn_resume      = GaitState.IDLE
        self.turn_pivot_side  = Side.LEFT
        self.turn_heading_end = 0.0
        self.cmd_lock         = threading.Lock()
        self.cmd              = "stop"

    def set_command(self, c):
        with self.cmd_lock:
            if c in ("left", "right"):
                self.turn_queue.append(c)
                print(f"[gait] turn queued: {c} (depth={len(self.turn_queue)})")
            else:
                self.cmd = c

    def _get_state(self):
        with self.cmd_lock:
            return self.cmd, list(self.turn_queue)

    def _pop_turn(self):
        with self.cmd_lock:
            return self.turn_queue.pop(0) if self.turn_queue else None

    def _hip_world(self, side):
        sign   = -1 if side == Side.LEFT else 1
        perp   = np.array([-math.sin(self.heading), math.cos(self.heading)])
        hip_xy = self.body_pos + sign * HIP_WIDTH * perp
        return np.array([hip_xy[0], hip_xy[1], 0.0])

    def _next_foot_pos(self, side):
        fwd  = np.array([math.cos(self.heading), math.sin(self.heading)])
        perp = np.array([-math.sin(self.heading), math.cos(self.heading)])
        sign = -1 if side == Side.LEFT else 1
        xy   = self.body_pos + fwd * STEP_LEN * self.speed_scale + sign * HIP_WIDTH * perp
        return np.array([xy[0], xy[1], STAND_HEIGHT])

    def _turn_foot_pos(self, swing_side, new_heading):
        pivot = self.foot_world[Side.OTHER[swing_side]].copy()
        sign  = -1 if swing_side == Side.LEFT else 1
        perp  = np.array([-math.sin(new_heading), math.cos(new_heading)])
        xy    = pivot[:2] + sign * 2 * HIP_WIDTH * perp
        return np.array([xy[0], xy[1], STAND_HEIGHT])

    def _advance_body(self):
        fwd = np.array([math.cos(self.heading), math.sin(self.heading)])
        self.body_pos = self.body_pos + fwd * STEP_LEN * self.speed_scale * 0.5

    def _start_swing(self, side):
        self.swing_side  = side
        self.swing_t     = 0.0
        start            = self.foot_world[side].copy()
        end              = self._next_foot_pos(side)
        self.swing_end   = end
        self.swing_curve = make_swing_curve(start, end, STEP_HEIGHT, self.swing_profile)

    def _begin_turn(self, direction, resume_state):
        self.turn_dir         = +1 if direction == "left" else -1
        self.turn_resume      = resume_state
        self.turn_pivot_side  = Side.LEFT  if direction == "left" else Side.RIGHT
        swing_side            = Side.RIGHT if direction == "left" else Side.LEFT
        self.turn_heading_end = self.heading + self.turn_dir * TURN_STEP_ANGLE
        start                 = self.foot_world[swing_side].copy()
        end                   = self._turn_foot_pos(swing_side, self.turn_heading_end)
        self.swing_side       = swing_side
        self.swing_t          = 0.0
        self.swing_end        = end
        self.swing_curve      = make_swing_curve(start, end, STEP_HEIGHT, self.swing_profile)
        self.state            = GaitState.TURNING

    def tick(self):
        cmd, turn_queue = self._get_state()

        if turn_queue and self.state != GaitState.TURNING:
            direction = self._pop_turn()
            resume    = GaitState.STEPPING if cmd == "forward" else GaitState.IDLE
            self._begin_turn(direction, resume_state=resume)

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
                if abs(lf[0] - rf[0]) < STEP_LEN * 0.6:
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
                self.heading        = self.turn_heading_end
                self.target_heading = self.heading
                lf = self.foot_world[Side.LEFT]
                rf = self.foot_world[Side.RIGHT]
                self.body_pos = np.array([(lf[0]+rf[0])*0.5, (lf[1]+rf[1])*0.5])

                next_turn = self._pop_turn()
                if next_turn is not None:
                    resume = GaitState.STEPPING if cmd == "forward" else GaitState.IDLE
                    self._begin_turn(next_turn, resume_state=resume)
                else:
                    self.state = self.turn_resume
                    if self.state == GaitState.STEPPING:
                        self._start_swing(Side.OTHER[self.swing_side])
                    elif self.state == GaitState.IDLE and cmd == "forward":
                        self._start_swing(Side.LEFT)
                        self.state = GaitState.STEPPING

        if self.state != GaitState.TURNING:
            hdiff = (self.target_heading - self.heading + math.pi) % (2*math.pi) - math.pi
            self.heading += hdiff * min(1.0, 6.0 * DT)

        if self.state != GaitState.IDLE and self.swing_curve is not None:
            self.foot_world[self.swing_side] = bezier(self.swing_curve, max(0., min(1., self.swing_t)))

        feet_hip_rel = [self.foot_world[s] - self._hip_world(s) for s in (Side.LEFT, Side.RIGHT)]

        return {
            "state":        self.state,
            "heading":      self.heading,
            "body_pos":     self.body_pos.copy(),
            "foot_world":   [f.copy() for f in self.foot_world],
            "feet_hip_rel": feet_hip_rel,
            "swing_side":   self.swing_side,
            "swing_t":      self.swing_t,
            "swing_curve":  self.swing_curve,
            "speed_scale":  self.speed_scale,
        }


def yarp_write_port(port_name):
    p = subprocess.Popen(
        ["yarp", "write", port_name],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    return p


def send_bottle(proc, values):
    line = " ".join(str(float(v)) for v in values) + "\n"
    try:
        proc.stdin.write(line)
        proc.stdin.flush()
    except BrokenPipeError:
        pass


def cmd_reader_thread(gait, cmd_queue):
    proc = subprocess.Popen(
        ["yarp", "read", PORT_CMD],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    print(f"[gait] cmd reader started on {PORT_CMD}")
    for line in proc.stdout:
        line = line.strip()
        if not line or line.startswith("yarp:"):
            continue
        print(f"[gait] << '{line}'  state={gait.state}")
        gait.set_command(line)


def gait_loop(gait, ik):
    p_left         = yarp_write_port(PORT_LEFT)
    p_right        = yarp_write_port(PORT_RIGHT)
    p_left_angles  = yarp_write_port(PORT_LEFT_ANGLES)
    p_right_angles = yarp_write_port(PORT_RIGHT_ANGLES)
    p_viz          = yarp_write_port(PORT_VIZ)

    print(f"[gait] {RATE_HZ}Hz  IK={'on' if ik else 'off'}")

    while True:
        t0   = time.monotonic()
        snap = gait.tick()

        for proc, rel in zip((p_left, p_right), snap["feet_hip_rel"]):
            send_bottle(proc, rel)

        if ik is not None:
            foot_l = snap["foot_world"][0]
            foot_r = snap["foot_world"][1]
            bp     = snap["body_pos"]
            hdg    = snap["heading"]
            cos_h  = math.cos(-hdg)
            sin_h  = math.sin(-hdg)

            def hip_world(side):
                sign = -1 if side == Side.LEFT else 1
                px   = -math.sin(hdg)
                py   =  math.cos(hdg)
                return np.array([bp[0] + sign*HIP_WIDTH*px, bp[1] + sign*HIP_WIDTH*py, 0.0])

            def to_hip_rel(foot, hip):
                dx, dy, dz = foot[0]-hip[0], foot[1]-hip[1], foot[2]-hip[2]
                return np.array([cos_h*dx - sin_h*dy, sin_h*dx + cos_h*dy, dz])

            left_angles  = ik.solve_left( to_hip_rel(foot_l, hip_world(Side.LEFT)))
            right_angles = ik.solve_right(to_hip_rel(foot_r, hip_world(Side.RIGHT)))

            send_bottle(p_left_angles,  [left_angles.get(j, 0.0)  for j in LEFT_JOINT_ORDER])
            send_bottle(p_right_angles, [right_angles.get(j, 0.0) for j in RIGHT_JOINT_ORDER])

        viz_vals = (
            [0.0, snap["heading"], snap["body_pos"][0], snap["body_pos"][1]]
            + list(snap["foot_world"][0])
            + list(snap["foot_world"][1])
            + [float(snap["swing_side"]), snap["swing_t"], snap["speed_scale"]]
        )
        if snap["swing_curve"] is not None:
            for pt in snap["swing_curve"]:
                viz_vals.extend(pt)
        send_bottle(p_viz, viz_vals)

        wait = DT - (time.monotonic() - t0)
        if wait > 0:
            time.sleep(wait)


def main():
    gait = GaitCoordinator(swing_profile=load_swing_profile())

    threading.Thread(
        target=cmd_reader_thread,
        args=(gait, None),
        daemon=True,
    ).start()

    ik = None
    if _IK_AVAILABLE:
        try:
            ik = LegIK()
        except Exception as e:
            print(f"[gait] IK load failed: {e}")

    try:
        gait_loop(gait, ik)
    except KeyboardInterrupt:
        print("[gait] bye")


if __name__ == "__main__":
    main()
