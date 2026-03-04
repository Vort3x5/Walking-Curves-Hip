#!/usr/bin/env python3
import os
import sys
import math
import time
import numpy as np
import pybullet as p
import pybullet_data

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

L_THIGH = 0.115     
L_CALF  = 0.115     
Z_REST  = 0.21      

STEP_LENGTH = 0.12  
STEP_HEIGHT = 0.04  
PELVIS_YAW_AMP = 0.25
SWAY_AMP       = 0.06

def solve_leg_ik(x, z):
    target_dist = np.sqrt(x**2 + z**2)
    max_reach = L_THIGH + L_CALF - 0.001
    target_dist = np.clip(target_dist, 0.01, max_reach)
    
    cos_knee_int = (L_THIGH**2 + L_CALF**2 - target_dist**2) / (2 * L_THIGH * L_CALF)
    internal_knee = np.arccos(np.clip(cos_knee_int, -1.0, 1.0))
    knee_angle = np.pi - internal_knee
    
    alpha = np.arctan2(x, z)
    cos_thigh = (L_THIGH**2 + target_dist**2 - L_CALF**2) / (2 * L_THIGH * target_dist)
    beta = np.arccos(np.clip(cos_thigh, -1.0, 1.0))
    
    hip_angle = alpha + beta 
    return hip_angle, knee_angle

def get_foot_target(phase_offset):
    phase_offset = phase_offset % 1.0
    if phase_offset < 0.5:
        progress = phase_offset / 0.5
        x = (STEP_LENGTH / 2.0) - (progress * STEP_LENGTH)
        z = Z_REST
    else:
        progress = (phase_offset - 0.5) / 0.5
        x = -(STEP_LENGTH / 2.0) + (progress * STEP_LENGTH)
        z = Z_REST - (STEP_HEIGHT * np.sin(progress * np.pi))
    return x, z

def walk_cycle(t_cycle):
    phase = t_cycle % 1.0
    rad_phase = phase * 2.0 * np.pi
    
    x_L, z_L = get_foot_target(phase)
    x_R, z_R = get_foot_target(phase + 0.5)
    
    hip_L_ik, knee_L_ik = solve_leg_ik(x_L, z_L)
    hip_R_ik, knee_R_ik = solve_leg_ik(x_R, z_R)
    
    left_hip   = -hip_L_ik
    left_knee  = -knee_L_ik
    left_ankle = left_hip - left_knee  
    
    right_hip   = -hip_R_ik
    right_knee  = -knee_R_ik
    right_ankle = right_knee - right_hip  
    
    twist = -PELVIS_YAW_AMP * np.sin(rad_phase)
    
    left_hip_yaw = -twist
    right_hip_yaw = twist
    
    sway = SWAY_AMP * np.sin(rad_phase)
    
    left_hip_roll  = sway
    right_hip_roll = sway
    
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

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf", [0, 0, 0.0])

    urdf_path = os.path.join(root_dir, "beta.urdf")
    
    flags = p.URDF_USE_INERTIA_FROM_FILE
    start_pos = [0, 0, 0.45]
    start_ori = p.getQuaternionFromEuler([math.pi, 0, math.pi/2])
    
    robot_id = p.loadURDF(urdf_path, start_pos, start_ori, useFixedBase=True, flags=flags)
    
    name_to_idx = {}
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode("utf-8")
        name_to_idx[joint_name] = i

    hz = 50.0
    dt = 1.0 / hz
    p.setTimeStep(dt)

    t0 = time.time()
    walk_speed = 1.2 

    while p.isConnected():
        t_current = time.time()
        
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=45, cameraPitch=-20, cameraTargetPosition=pos)

        t_cycle = (t_current - t0) * walk_speed
        
        angles = walk_cycle(t_cycle)

        for joint_name, target_angle in angles.items():
            if joint_name in name_to_idx:
                p.setJointMotorControl2(
                    bodyIndex=robot_id,
                    jointIndex=name_to_idx[joint_name],
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_angle,
                    force=100.0,
                    maxVelocity=10.0
                )

        p.stepSimulation()
        
        t1 = time.time()
        time.sleep(max(0, dt - (t1 - t_current)))

if __name__ == "__main__":
    main()
