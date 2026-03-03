#!/usr/bin/env python3
import os
import sys
import argparse
import warnings
import math
import numpy as np
import xml.etree.ElementTree as ET
from collections import deque

try:
    from ikpy.chain import Chain
except ImportError:
    raise ImportError("ikpy not installed. Run: pip install ikpy")

RIGHT_BASE_LINK = "GE_27_2"
LEFT_BASE_LINK = "GE_27_1"
RIGHT_TIP_LINK = "Part_1_4_1"
LEFT_TIP_LINK = "StopaLewa_1"

GAIT_RIGHT_ORDER = ["right_hip_roll", "right_hip_pitch", "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll", "right_ankle_yaw"]
GAIT_LEFT_ORDER = ["left_hip_roll", "left_hip_pitch", "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll", "left_ankle_yaw"]

STAND_TARGET = [0.0, 0.0, -0.197]


def _local_name(tag):
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _joint_score(name):
    n = name.lower()
    s = 0
    if "segment1" in n:
        s += 50
    if "segment3-4" in n:
        s += 40
    if "wj-wk00-0018_45" in n:
        s += 30
    if "part_2_" in n:
        s += 15
    if "stopa" in n or "part_1_4_1" in n:
        s -= 20
    if "ge_27" in n or "gacie" in n or "mocowanie" in n:
        s -= 100
    return s


class _LegChain:
    def __init__(self, urdf_path, base_link):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.chain = Chain.from_urdf_file(
                urdf_path,
                base_elements=[base_link],
                last_link_vector=[0, 0, 0],
            )

        self.link_names = [l.name for l in self.chain.links]
        self._last = [0.0] * len(self.chain.links)

        mask = getattr(self.chain, "active_links_mask", [False] * len(self.chain.links))
        self.active_names = [self.chain.links[i].name for i, m in enumerate(mask) if bool(m)]

        self.limits = {}
        for i, l in enumerate(self.chain.links):
            if l.name in self.active_names and hasattr(l, "bounds") and l.bounds is not None:
                lo, hi = l.bounds
                if lo is not None and hi is not None:
                    self.limits[l.name] = (float(lo), float(hi))

    def solve(self, target_xyz):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            angles = self.chain.inverse_kinematics(
                target_position=list(target_xyz),
                initial_position=self._last,
                orientation_mode=None,
            )
        self._last = list(angles)

        out = {}
        mask = getattr(self.chain, "active_links_mask", [False] * len(self.chain.links))
        for i, l in enumerate(self.chain.links):
            if bool(mask[i]):
                a = float(angles[i])
                if l.name in self.limits:
                    lo, hi = self.limits[l.name]
                    a = max(lo, min(hi, a))
                out[l.name] = a
        return out

    def reset(self):
        self._last = [0.0] * len(self.chain.links)


class LegIK:
    def __init__(self, urdf_dir=None):
        if urdf_dir is None:
            urdf_dir = os.path.dirname(os.path.abspath(__file__))

        self.r_path = os.path.join(urdf_dir, "leg_right.urdf")
        self.l_path = os.path.join(urdf_dir, "leg_left.urdf")

        for p in (self.r_path, self.l_path):
            if not os.path.exists(p) or os.path.getsize(p) < 64:
                raise FileNotFoundError(f"Missing or empty: {p}")

        self._right = _LegChain(self.r_path, RIGHT_BASE_LINK)
        self._left = _LegChain(self.l_path, LEFT_BASE_LINK)

        self._right_map = self._build_name_map(self._right, GAIT_RIGHT_ORDER, "right")
        self._left_map = self._build_name_map(self._left, GAIT_LEFT_ORDER, "left")

        self.solve_right(np.array(STAND_TARGET))
        self.solve_left(np.array(STAND_TARGET))

    @staticmethod
    def _pick_six(active_names):
        if len(active_names) == 6:
            return active_names
        ranked = sorted(active_names, key=lambda n: (_joint_score(n), n), reverse=True)
        chosen = set(ranked[:6])
        ordered = [n for n in active_names if n in chosen]
        if len(ordered) < 6:
            for n in ranked:
                if n not in ordered:
                    ordered.append(n)
                if len(ordered) == 6:
                    break
        return ordered[:6]

    @staticmethod
    def _build_name_map(chain_obj, gait_order, leg_name):
        names = chain_obj.active_names
        if len(names) < 6:
            dbg = ", ".join(chain_obj.link_names)
            raise RuntimeError(
                f"[ik] {leg_name}: not enough active joints ({len(names)}). "
                f"active={names}; chain_links=[{dbg}]"
            )
        selected = names[-6:]
        print(f"[ik] {leg_name} active joints: {names}")
        print(f"[ik] {leg_name} selected 6: {selected}")
        return dict(zip(gait_order, selected))

    @staticmethod
    def _remap(raw, mp):
        return {g: raw.get(i, 0.0) for g, i in mp.items()}

    def solve_right(self, foot_xyz_hip_rel):
        raw = self._right.solve(np.asarray(foot_xyz_hip_rel, dtype=float))
        return self._remap(raw, self._right_map)

    def solve_left(self, foot_xyz_hip_rel):
        raw = self._left.solve(np.asarray(foot_xyz_hip_rel, dtype=float))
        return self._remap(raw, self._left_map)

    def solve_both(self, left_xyz, right_xyz):
        out = {}
        out.update(self.solve_left(np.asarray(left_xyz, dtype=float)))
        out.update(self.solve_right(np.asarray(right_xyz, dtype=float)))
        return out

    def reset(self):
        self._right.reset()
        self._left.reset()


    def generate_leg_urdfs(src_urdf, out_dir, right_base, right_tip, left_base, left_tip):
        tree = ET.parse(src_urdf)
        root = tree.getroot()

        links = {}
        joints = {}

        for e in root.iter():
            ln = _local_name(e.tag)
            if ln == "link":
                n = e.get("name")
                if n:
                    links[n] = e
            elif ln == "joint":
                n = e.get("name")
                if n:
                    joints[n] = e

        children_of = {}
        edge_joint = {}

        for jn, j in joints.items():
            p = None
            c = None
            for child in list(j):
                ln = _local_name(child.tag)
                if ln == "parent":
                    p = child
                elif ln == "child":
                    c = child
            if p is None or c is None:
                continue
            pl = p.get("link")
            cl = c.get("link")
            if not pl or not cl:
                continue
            children_of.setdefault(pl, []).append(cl)
            edge_joint[(pl, cl)] = jn

    def run_self_test(src_urdf, out_dir, right_base, right_tip, left_base, left_tip):
        print("[ik:test] regenerating leg URDFs...")
        generate_leg_urdfs(
            src_urdf=src_urdf,
            out_dir=out_dir,
            right_base=right_base,
            right_tip=right_tip,
            left_base=left_base,
            left_tip=left_tip,
        )

        print("[ik:test] loading LegIK...")
        ik = LegIK(out_dir)

        print("[ik:test] checking output keys...")
        r = ik.solve_right(np.array([0.02, 0.0, -0.20], dtype=float))
        l = ik.solve_left(np.array([0.02, 0.0, -0.20], dtype=float))

        r_expected = set(GAIT_RIGHT_ORDER)
        l_expected = set(GAIT_LEFT_ORDER)

        if set(r.keys()) != r_expected:
            raise RuntimeError(f"[ik:test] right keys mismatch: got={sorted(r.keys())}, expected={sorted(r_expected)}")
        if set(l.keys()) != l_expected:
            raise RuntimeError(f"[ik:test] left keys mismatch: got={sorted(l.keys())}, expected={sorted(l_expected)}")

        print("[ik:test] probing knee response over z sweep...")
        z_vals = np.linspace(-0.16, -0.24, 7)
        rk = []
        lk = []
        for z in z_vals:
            rr = ik.solve_right(np.array([0.02, 0.0, z], dtype=float))
            ll = ik.solve_left(np.array([0.02, 0.0, z], dtype=float))
            rk.append(rr["right_knee_pitch"])
            lk.append(ll["left_knee_pitch"])

        r_span = max(rk) - min(rk)
        l_span = max(lk) - min(lk)

        print(f"[ik:test] right knee span: {r_span:.6f} rad")
        print(f"[ik:test] left  knee span: {l_span:.6f} rad")
        print(f"[ik:test] right knee seq: {[round(x, 4) for x in rk]}")
        print(f"[ik:test] left  knee seq: {[round(x, 4) for x in lk]}")

        if r_span < math.radians(2.0):
            raise RuntimeError("[ik:test] right knee response too small")
        if l_span < math.radians(2.0):
            raise RuntimeError("[ik:test] left knee response too small")

        for i in range(1, len(rk)):
            if abs(rk[i] - rk[i - 1]) > math.radians(35):
                raise RuntimeError("[ik:test] right knee has unnatural jump")
        for i in range(1, len(lk)):
            if abs(lk[i] - lk[i - 1]) > math.radians(35):
                raise RuntimeError("[ik:test] left knee has unnatural jump")

        print("[ik:test] OK")

    def directed_path(base_link, tip_link):
        q = deque([base_link])
        prev = {base_link: None}
        while q:
            u = q.popleft()
            if u == tip_link:
                break
            for v in children_of.get(u, []):
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        if tip_link not in prev:
            return None
        path = []
        cur = tip_link
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    def emit_leg(path_links, out_name):
        used_links = set(path_links)
        used_joints = [edge_joint[(path_links[i], path_links[i + 1])] for i in range(len(path_links) - 1)]

        lines = [f'<robot name="{out_name.replace(".urdf","")}">']
        for ln in sorted(used_links):
            lines.append(f'  <link name="{ln}"/>')

        for jn in used_joints:
            j = joints[jn]
            jtype = j.get("type", "fixed")
            origin = None
            axis_el = None
            limit_el = None
            parent_link = None
            child_link = None

            for ch in list(j):
                ln = _local_name(ch.tag)
                if ln == "origin":
                    origin = ch
                elif ln == "axis":
                    axis_el = ch
                elif ln == "limit":
                    limit_el = ch
                elif ln == "parent":
                    parent_link = ch.get("link")
                elif ln == "child":
                    child_link = ch.get("link")

            xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
            rpy = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
            axis = axis_el.get("xyz", "0 0 1") if axis_el is not None else "0 0 1"

            lines.append(f'  <joint name="{jn}" type="{jtype}">')
            lines.append(f'    <parent link="{parent_link}"/>')
            lines.append(f'    <child link="{child_link}"/>')
            lines.append(f'    <origin xyz="{xyz}" rpy="{rpy}"/>')
            if jtype != "fixed":
                lines.append(f'    <axis xyz="{axis}"/>')
            if jtype in ("revolute", "prismatic"):
                lo = "-3.14159265359"
                hi = "3.14159265359"
                effort = "100"
                velocity = "10"
                if limit_el is not None:
                    lo = limit_el.get("lower", lo)
                    hi = limit_el.get("upper", hi)
                    effort = limit_el.get("effort", effort)
                    velocity = limit_el.get("velocity", velocity)
                lines.append(f'    <limit lower="{lo}" upper="{hi}" effort="{effort}" velocity="{velocity}"/>')
            lines.append("  </joint>")

        lines.append("</robot>")
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        non_fixed = sum(1 for jn in used_joints if joints[jn].get("type", "fixed") != "fixed")
        print(f"[ik] generated {out_path} with {len(used_joints)} joints ({non_fixed} non-fixed)")

    for n in [right_base, right_tip, left_base, left_tip]:
        if n not in links:
            raise RuntimeError(f"Missing link in URDF: {n}")

    rp = directed_path(right_base, right_tip)
    lp = directed_path(left_base, left_tip)
    if rp is None:
        raise RuntimeError(f"No directed path: {right_base} -> {right_tip}")
    if lp is None:
        raise RuntimeError(f"No directed path: {left_base} -> {left_tip}")

    os.makedirs(out_dir, exist_ok=True)
    emit_leg(rp, "leg_right.urdf")
    print(f"[ik] path {right_base} -> {right_tip}: {' -> '.join(rp)}")
    emit_leg(lp, "leg_left.urdf")
    print(f"[ik] path {left_base} -> {left_tip}: {' -> '.join(lp)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-urdfs", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--right-base", type=str, default=RIGHT_BASE_LINK)
    ap.add_argument("--right-tip", type=str, default=RIGHT_TIP_LINK)
    ap.add_argument("--left-base", type=str, default=LEFT_BASE_LINK)
    ap.add_argument("--left-tip", type=str, default=LEFT_TIP_LINK)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        if not args.gen_urdfs:
            raise RuntimeError("--self-test requires --gen-urdfs <file>")
        out = args.out if args.out else os.path.dirname(os.path.abspath(__file__))
        run_self_test(
            src_urdf=args.gen_urdfs,
            out_dir=out,
            right_base=args.right_base,
            right_tip=args.right_tip,
            left_base=args.left_base,
            left_tip=args.left_tip,
        )
        sys.exit(0)

    if args.gen_urdfs:
        out = args.out if args.out else os.path.dirname(os.path.abspath(__file__))
        generate_leg_urdfs(
            src_urdf=args.gen_urdfs,
            out_dir=out,
            right_base=args.right_base,
            right_tip=args.right_tip,
            left_base=args.left_base,
            left_tip=args.left_tip,
        )
        sys.exit(0)

    ik = LegIK()
    print("[ik] right:", ik.solve_right(np.array([0.02, 0.0, -0.18])))
    print("[ik] left: ", ik.solve_left(np.array([0.02, 0.0, -0.18])))
