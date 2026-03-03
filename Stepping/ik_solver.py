#!/usr/bin/env python3
import os
import sys
import argparse
import warnings
import difflib
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


def _local_name(tag):
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


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
    parent_of = {}
    edge_joint = {}
    edge_joint_any = {}

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
        parent_of[cl] = pl
        edge_joint[(pl, cl)] = jn
        edge_joint_any[(pl, cl)] = jn
        edge_joint_any[(cl, pl)] = jn

    def suggest(name, universe, n=8):
        out = difflib.get_close_matches(name, list(universe), n=n, cutoff=0.35)
        if not out:
            lo = name.lower()
            out = [x for x in universe if lo in x.lower() or x.lower() in lo][:n]
        return out

    def directed_path(base_link, tip_link, forbidden_link):
        q = deque([base_link])
        prev = {base_link: None}
        while q:
            u = q.popleft()
            if u == tip_link:
                break
            for v in children_of.get(u, []):
                if v == forbidden_link:
                    continue
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

    def descendants(root_link, forbidden):
        out = set()
        q = deque([root_link])
        out.add(root_link)
        while q:
            u = q.popleft()
            for v in children_of.get(u, []):
                if v == forbidden:
                    continue
                if v not in out:
                    out.add(v)
                    q.append(v)
        return out

    def choose_tip(base_link, other_base, wanted_tip):
        if wanted_tip in links:
            p = directed_path(base_link, wanted_tip, other_base)
            if p is not None:
                return wanted_tip, p

        cand_set = descendants(base_link, other_base)
        if other_base in cand_set:
            cand_set.remove(other_base)

        foot_like = [n for n in cand_set if ("stopa" in n.lower() or "foot" in n.lower())]
        ordered = foot_like + sorted(cand_set - set(foot_like))

        best = None
        best_path = None
        best_score = -10**9

        for n in ordered:
            p = directed_path(base_link, n, other_base)
            if p is None:
                continue
            score = 0
            ln = n.lower()
            if "stopa" in ln or "foot" in ln:
                score += 200
            score += len(p)
            if score > best_score:
                best_score = score
                best = n
                best_path = p

        return best, best_path

    def emit_leg_from_path(path_links, out_name):
        used_links = set(path_links)
        used_joints = []
        for i in range(len(path_links) - 1):
            u, v = path_links[i], path_links[i + 1]
            jn = edge_joint.get((u, v))
            if jn is None:
                jn = edge_joint_any[(u, v)]
            used_joints.append(jn)

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
            for child in list(j):
                ln = _local_name(child.tag)
                if ln == "origin":
                    origin = child
                elif ln == "axis":
                    axis_el = child
                elif ln == "limit":
                    limit_el = child
                elif ln == "parent":
                    parent_link = child.get("link")
                elif ln == "child":
                    child_link = child.get("link")

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

        nonfixed = 0
        for jn in used_joints:
            jt = joints[jn].get("type", "fixed")
            if jt != "fixed":
                nonfixed += 1

        return out_path, len(used_joints), nonfixed

    required = [right_base, left_base]
    missing_base = [n for n in required if n not in links]
    if missing_base:
        print(f"[ik] URDF parsed: links={len(links)}, joints={len(joints)}")
        for m in missing_base:
            print(f"[ik] missing base link: {m}")
            cand = suggest(m, links.keys(), n=10)
            if cand:
                print(f"[ik] close matches for {m}: {cand}")
        raise RuntimeError("Base link does not exist in URDF")

    r_tip, r_path = choose_tip(right_base, left_base, right_tip)
    l_tip, l_path = choose_tip(left_base, right_base, left_tip)

    if r_path is None or l_path is None:
        print(f"[ik] could not build directed path(s)")
        print(f"[ik] right: base={right_base} wanted_tip={right_tip} chosen_tip={r_tip}")
        print(f"[ik] left : base={left_base} wanted_tip={left_tip} chosen_tip={l_tip}")
        raise RuntimeError("Directed base->tip path not found")

    os.makedirs(out_dir, exist_ok=True)

    rp, rj, rnf = emit_leg_from_path(r_path, "leg_right.urdf")
    lp, lj, lnf = emit_leg_from_path(l_path, "leg_left.urdf")

    print(f"[ik] generated {rp} with {rj} joints ({rnf} non-fixed)")
    print(f"[ik] path {right_base} -> {r_tip}: {' -> '.join(r_path)}")
    print(f"[ik] generated {lp} with {lj} joints ({lnf} non-fixed)")
    print(f"[ik] path {left_base} -> {l_tip}: {' -> '.join(l_path)}")

    if right_tip != r_tip:
        print(f"[ik] right tip adjusted: requested={right_tip} chosen={r_tip}")
    if left_tip != l_tip:
        print(f"[ik] left tip adjusted: requested={left_tip} chosen={l_tip}")


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
