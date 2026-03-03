#!/usr/bin/env python3
import os
import sys
import argparse
import warnings
import numpy as np
import xml.etree.ElementTree as ET
from collections import deque

try:
    from ikpy.chain import Chain
except ImportError:
    raise ImportError("ikpy not installed. Run: pip install ikpy")

RIGHT_BASE_LINK = "GE_27_1"
LEFT_BASE_LINK = "GE_27_2"
RIGHT_TIP_LINK = "StopaPrawa_1"
LEFT_TIP_LINK = "StopaLewa_1"

GAIT_RIGHT_ORDER = [
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee_pitch",
    "right_ankle_pitch",
    "right_ankle_roll",
    "right_ankle_yaw",
]

GAIT_LEFT_ORDER = [
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee_pitch",
    "left_ankle_pitch",
    "left_ankle_roll",
    "left_ankle_yaw",
]

STAND_TARGET = [0.0, 0.0, -0.197]


def _ns_prefix(tag):
    if tag.startswith("{") and "}" in tag:
        return tag.split("}", 1)[0] + "}"
    return ""


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

        mask = getattr(self.chain, "active_links_mask", None)
        if mask is None:
            mask = [False] * len(self.chain.links)
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
    def _build_name_map(chain_obj, gait_order, leg_name):
        names = chain_obj.active_names
        if len(names) < 6:
            dbg = ", ".join(chain_obj.link_names)
            raise RuntimeError(
                f"[ik] {leg_name}: not enough active joints ({len(names)}). "
                f"active={names}; chain_links=[{dbg}]"
            )
        selected = names[-6:]
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
    ns = _ns_prefix(root.tag)

    def all_e(name):
        return root.findall(f"{ns}{name}")

    links = {l.get("name"): l for l in all_e("link") if l.get("name")}
    joints = {j.get("name"): j for j in all_e("joint") if j.get("name")}

    graph = {}
    edge_info = {}
    for jn, j in joints.items():
        p = j.find(f"{ns}parent")
        c = j.find(f"{ns}child")
        if p is None or c is None:
            continue
        pl = p.get("link")
        cl = c.get("link")
        if not pl or not cl:
            continue
        graph.setdefault(pl, []).append(cl)
        graph.setdefault(cl, []).append(pl)
        edge_info[(pl, cl)] = (jn, pl, cl)
        edge_info[(cl, pl)] = (jn, pl, cl)

    def find_link_path(start, end):
        q = deque([start])
        prev = {start: None}
        while q:
            u = q.popleft()
            if u == end:
                break
            for v in graph.get(u, []):
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        if end not in prev:
            return None
        path = []
        cur = end
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    def make_leg(base_link, tip_link, out_name):
        link_path = find_link_path(base_link, tip_link)
        if link_path is None:
            raise RuntimeError(f"[ik] no path found: {base_link} -> {tip_link}")

        edges = []
        for i in range(len(link_path) - 1):
            u, v = link_path[i], link_path[i + 1]
            edges.append(edge_info[(u, v)])

        lines = [f'<robot name="leg_{out_name.replace(".urdf","")}">']
        for ln in sorted(set(link_path)):
            lines.append(f'  <link name="{ln}"/>')

        for (jn, parent_orig, child_orig) in edges:
            j = joints[jn]
            jtype = j.get("type", "fixed")
            origin = j.find(f"{ns}origin")
            axis_el = j.find(f"{ns}axis")
            xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
            rpy = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
            axis = axis_el.get("xyz", "0 0 1") if axis_el is not None else "0 0 1"

            lines.append(f'  <joint name="{jn}" type="{jtype}">')
            lines.append(f'    <parent link="{parent_orig}"/>')
            lines.append(f'    <child link="{child_orig}"/>')
            lines.append(f'    <origin xyz="{xyz}" rpy="{rpy}"/>')
            if jtype != "fixed":
                lines.append(f'    <axis xyz="{axis}"/>')
            if jtype == "revolute":
                lines.append('    <limit lower="-3.1415926535" upper="3.1415926535" effort="1000" velocity="1000"/>')
            lines.append("  </joint>")

        lines.append("</robot>")
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[ik] generated {out_path} with {len(edges)} joints")

    if right_base not in links or right_tip not in links or left_base not in links or left_tip not in links:
        raise RuntimeError("One of base/tip links does not exist in URDF")

    os.makedirs(out_dir, exist_ok=True)
    make_leg(right_base, right_tip, "leg_right.urdf")
    make_leg(left_base, left_tip, "leg_left.urdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-urdfs", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--right-base", type=str, default=RIGHT_BASE_LINK)
    ap.add_argument("--right-tip", type=str, default=RIGHT_TIP_LINK)
    ap.add_argument("--left-base", type=str, default=LEFT_BASE_LINK)
    ap.add_argument("--left-tip", type=str, default=LEFT_TIP_LINK)
    args = ap.parse_args()

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
