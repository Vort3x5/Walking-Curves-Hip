#!/usr/bin/env python3
"""
Humanoid URDF Walk Animation
Renders humanoid_beta.urdf with STL meshes in matplotlib.
No pybullet - pure numpy/matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import xml.etree.ElementTree as ET
import struct, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_FILE = os.path.join(SCRIPT_DIR, '../beta.urdf')
MAX_TRIS_PER_MESH = 100  # Aggressive decimation for smooth animation


# ====================================================================
# 1. STL Loader (binary)
# ====================================================================
def load_stl(filepath, max_tris=MAX_TRIS_PER_MESH):
    """Load binary STL and decimate to max_tris triangles."""
    with open(filepath, 'rb') as f:
        f.read(80)  # skip header
        n = struct.unpack('<I', f.read(4))[0]
        if n == 0:
            return np.zeros((0, 3, 3))
        dt = np.dtype([
            ('normal', '<3f'), ('v0', '<3f'), ('v1', '<3f'), ('v2', '<3f'), ('attr', '<u2')
        ])
        raw = f.read(n * 50)
        actual_n = min(n, len(raw) // 50)
        data = np.frombuffer(raw[:actual_n * 50], dtype=dt)
    tris = np.stack([data['v0'], data['v1'], data['v2']], axis=1).astype(np.float64)
    if len(tris) > max_tris:
        tris = tris[np.linspace(0, len(tris) - 1, max_tris, dtype=int)]
    return tris


# ====================================================================
# 2. Transform Utilities
# ====================================================================
def rpy_to_mat(r, p, y):
    """URDF RPY (extrinsic XYZ = intrinsic ZYX) -> 4x4 homogeneous."""
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    T = np.eye(4)
    T[:3, :3] = [
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [  -sp,            cp*sr,            cp*cr],
    ]
    return T

def make_tf(xyz, rpy):
    T = rpy_to_mat(*rpy)
    T[:3, 3] = xyz
    return T

def aa_rot(axis, angle):
    """Axis-angle rotation (Rodrigues) -> 4x4."""
    a = np.asarray(axis, float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(4)
    a = a / n
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    T = np.eye(4)
    T[:3, :3] = R
    return T


# ====================================================================
# 3. URDF Parser & FK Engine
# ====================================================================
class URDFModel:
    def __init__(self, path, max_tris=MAX_TRIS_PER_MESH):
        self.basedir = os.path.dirname(os.path.abspath(path))
        self.link_data = {}    # name -> {mesh_file, vis_origin, color, tris}
        self.joint_data = {}   # name -> {type, parent, child, origin, axis}
        self.child_map = {}    # parent_link -> [(joint_name, child_link)]
        self.root = None
        self._parse(path)
        self._load_meshes(max_tris)

    def _parse(self, path):
        tree = ET.parse(path)
        r = tree.getroot()
        for el in r.findall('link'):
            name = el.get('name')
            mf, vo, col = None, np.eye(4), [0.6, 0.6, 0.6, 1.0]
            v = el.find('visual')
            if v is not None:
                o = v.find('origin')
                if o is not None:
                    vo = make_tf(
                        list(map(float, o.get('xyz', '0 0 0').split())),
                        list(map(float, o.get('rpy', '0 0 0').split()))
                    )
                g = v.find('geometry')
                if g is not None:
                    m = g.find('mesh')
                    if m is not None:
                        mf = m.get('filename')
                mt = v.find('material')
                if mt is not None:
                    c = mt.find('color')
                    if c is not None:
                        col = list(map(float, c.get('rgba', '0.6 0.6 0.6 1').split()))
            self.link_data[name] = {'mesh_file': mf, 'vis_origin': vo, 'color': col, 'tris': None}

        cset = set()
        for el in r.findall('joint'):
            nm = el.get('name')
            jt = el.get('type')
            par = el.find('parent').get('link')
            ch = el.find('child').get('link')
            o = el.find('origin')
            xyz = list(map(float, o.get('xyz', '0 0 0').split()))
            rpy = list(map(float, o.get('rpy', '0 0 0').split()))
            ax_el = el.find('axis')
            ax = list(map(float, ax_el.get('xyz').split())) if ax_el is not None else [0, 0, 1]
            self.joint_data[nm] = {
                'type': jt, 'parent': par, 'child': ch,
                'origin': make_tf(xyz, rpy), 'axis': np.array(ax)
            }
            self.child_map.setdefault(par, []).append((nm, ch))
            cset.add(ch)

        pset = set(self.child_map.keys())
        self.root = (pset - cset).pop() if (pset - cset) else list(self.link_data.keys())[0]

    def _load_meshes(self, max_tris):
        loaded = 0
        for name, lk in self.link_data.items():
            if lk['mesh_file']:
                fp = os.path.join(self.basedir, lk['mesh_file'])
                if os.path.exists(fp):
                    try:
                        lk['tris'] = load_stl(fp, max_tris)
                        loaded += 1
                    except Exception as e:
                        print(f"  Warning: could not load {fp}: {e}")
        print(f"  Loaded {loaded} meshes")

    def fk(self, angles=None):
        """Forward kinematics. angles = {joint_name: angle_rad}.
        Returns {link_name: 4x4 world transform}."""
        if angles is None:
            angles = {}
        result = {self.root: np.eye(4)}
        stack = [self.root]
        while stack:
            p = stack.pop()
            Tp = result[p]
            for jn, cn in self.child_map.get(p, []):
                j = self.joint_data[jn]
                T = Tp @ j['origin']
                if j['type'] == 'revolute':
                    T = T @ aa_rot(j['axis'], angles.get(jn, 0.0))
                result[cn] = T
                stack.append(cn)
        return result

    def get_mesh_polys(self, fk_result):
        """Return [(Nx3x3 transformed triangles, rgba color)] for rendering."""
        out = []
        for name, lk in self.link_data.items():
            tris = lk['tris']
            if tris is not None and len(tris) > 0 and name in fk_result:
                T = fk_result[name] @ lk['vis_origin']
                R, t = T[:3, :3], T[:3, 3]
                tr = np.einsum('ij,ntj->nti', R, tris) + t
                out.append((tr, lk['color']))
        return out

    def foot_pos(self, fk_result, side='left'):
        """Get foot position from FK result."""
        link = 'StopaLewa_1' if side == 'left' else 'StopaPrawa_1'
        return fk_result[link][:3, 3]


# ====================================================================
# 4. Walk Cycle (Inverse Kinematics) - PRONOUNCED HIPS & SWAY
# ====================================================================

L_THIGH = 0.115     
L_CALF  = 0.115     
Z_REST  = 0.21      

STEP_LENGTH = 0.12  
STEP_HEIGHT = 0.04  
PELVIS_YAW_AMP = 0.25   # INCREASED: ~14 degrees of twist for a much looser, natural stride
SWAY_AMP       = 0.06   # NEW: Amplitude for shifting the whole body side-to-side

def solve_leg_ik(x, z):
    """2D Analytic IK for a 2-bone leg.
    x: forward distance from hip to ankle
    z: downward distance from hip to ankle
    Returns (hip_pitch, knee_pitch) in standard radians (+ is forward/bent).
    """
    target_dist = np.sqrt(x**2 + z**2)
    # Prevent math crashes if the step target is out of reach
    max_reach = L_THIGH + L_CALF - 0.001
    target_dist = np.clip(target_dist, 0.01, max_reach)
    
    # 1. Knee Angle via Law of Cosines
    cos_knee_int = (L_THIGH**2 + L_CALF**2 - target_dist**2) / (2 * L_THIGH * L_CALF)
    internal_knee = np.arccos(np.clip(cos_knee_int, -1.0, 1.0))
    knee_angle = np.pi - internal_knee  # 0 = straight leg, positive = bent
    
    # 2. Hip Angle 
    alpha = np.arctan2(x, z)  # Angle from straight down to target
    cos_thigh = (L_THIGH**2 + target_dist**2 - L_CALF**2) / (2 * L_THIGH * target_dist)
    beta = np.arccos(np.clip(cos_thigh, -1.0, 1.0))
    
    hip_angle = alpha + beta 
    return hip_angle, knee_angle

def get_foot_target(phase_offset):
    """Calculate the Cartesian (x, z) foot trajectory."""
    phase_offset = phase_offset % 1.0
    
    if phase_offset < 0.5:
        # STANCE PHASE: Foot moves backward at a CONSTANT speed (no sine waves!)
        progress = phase_offset / 0.5
        x = (STEP_LENGTH / 2.0) - (progress * STEP_LENGTH)
        z = Z_REST
    else:
        # SWING PHASE: Foot swings forward and lifts up via a sine arc
        progress = (phase_offset - 0.5) / 0.5
        x = -(STEP_LENGTH / 2.0) + (progress * STEP_LENGTH)
        z = Z_REST - (STEP_HEIGHT * np.sin(progress * np.pi))
        
    return x, z

def walk_cycle(t_cycle):
    phase = t_cycle % 1.0
    rad_phase = phase * 2.0 * np.pi
    
    # 1. Foot IK
    x_L, z_L = get_foot_target(phase)
    x_R, z_R = get_foot_target(phase + 0.5)
    
    hip_L_ik, knee_L_ik = solve_leg_ik(x_L, z_L)
    hip_R_ik, knee_R_ik = solve_leg_ik(x_R, z_R)
    
    # --- PITCH CONSTRAINTS ---
    left_hip   = -hip_L_ik
    left_knee  = -knee_L_ik
    left_ankle = left_hip - left_knee  
    
    right_hip   = -hip_R_ik
    right_knee  = -knee_R_ik
    # *** KEEP YOUR RIGHT ANKLE FIX HERE ***
    right_ankle = right_knee - right_hip  
    
    # 2. PRONOUNCED PELVIS TWIST (Yaw)
    twist = -PELVIS_YAW_AMP * np.sin(rad_phase)
    
    # Counter-rotate hips so feet stay pointing straight forward
    left_hip_yaw = -twist
    right_hip_yaw = twist
    
    # 3. CONTINUOUS LATERAL SWAY (Roll)
    # This continuously shifts the body weight side to side over the stance leg
    sway = SWAY_AMP * np.sin(rad_phase)
    
    left_hip_roll  = sway
    right_hip_roll = sway
    
    # Counter-rotate ankle rolls to keep feet flat on the floor during the sway
    # (You may need to flip one of these signs if your URDF roll axes are mirrored!)
    left_ankle_roll  = -sway
    right_ankle_roll = sway 

    return {
        'Revolute_14': twist,
        'Revolute_6':  left_hip_yaw,
        'Revolute_7':  right_hip_yaw,
        'Revolute_5':  left_hip_roll,
        'Revolute_9':  right_hip_roll,
        'Revolute_4':  left_hip,
        'Revolute_3':  left_knee,
        'Revolute_2':  left_ankle,
        'Revolute_1':  left_ankle_roll,
        'Revolute_10': right_hip,
        'Revolute_11': right_knee,
        'Revolute_12': right_ankle,
        'Revolute_13': right_ankle_roll,
    }

# ====================================================================
# 5. Animation
# ====================================================================
def run_animation():
    print("Loading URDF model...")
    model = URDFModel(URDF_FILE, max_tris=MAX_TRIS_PER_MESH)

    # Rest pose for centering
    fk0 = model.fk()
    pelvis = fk0['gacieOgarniete_1'][:3, 3]
    foot_L = fk0['StopaLewa_1'][:3, 3]
    foot_R = fk0['StopaPrawa_1'][:3, 3]
    center = pelvis.copy()
    print(f"  Pelvis: {pelvis}")
    print(f"  Foot L: {foot_L},  Foot R: {foot_R}")
    print(f"  Leg length: {np.linalg.norm(foot_L - pelvis):.4f} m")

    # Display transform: model (X=lateral, Y≈fwd, Z=down) → chart (X=fwd, Y=lat, Z=up)
    def to_display(pts):
        c = pts - center
        out = np.empty_like(c)
        out[..., 0] = c[..., 1]   # chart X = model Y (forward)
        out[..., 1] = c[..., 0]   # chart Y = model X (lateral)
        out[..., 2] = -c[..., 2]  # chart Z = -model Z (up)
        return out

    # Setup figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    span = 0.25
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_zlim(-span, span)
    ax.set_xlabel('Forward')
    ax.set_ylabel('Lateral')
    ax.set_zlabel('Up')
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=15, azim=135)

    collections = []

    def update(frame):
        nonlocal collections
        for c in collections:
            c.remove()
        collections.clear()

        t = frame / 100.0
        angles = walk_cycle(t)
        fk_result = model.fk(angles)
        mesh_data = model.get_mesh_polys(fk_result)

        for tris, color in mesh_data:
            dtris = to_display(tris)
            poly = Poly3DCollection(dtris, alpha=0.85)
            poly.set_facecolor(list(color[:3]) + [0.75])
            poly.set_edgecolor([0.15, 0.15, 0.15, 0.25])
            ax.add_collection3d(poly)
            collections.append(poly)

        ax.set_title(f'Humanoid URDF Walk Cycle  (t={t:.2f})')
        return collections

    print("Starting animation...")
    ani = FuncAnimation(fig, update, frames=np.arange(0, 100),
                        interval=80, blit=False, repeat=True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_animation()
