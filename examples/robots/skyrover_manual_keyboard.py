import mujoco
import glfw
import numpy as np
import torch
from discoverse.utils import BaseConfig
from pid_control import PIDControl
import control_util as cu

from skyrover_manual_base import SkyRoverSoloBase

# 初始坐标
SCENE_CENTER = [0.014, -0.053, -3]

class SkyRoverSoloCfg(BaseConfig):
    mjcf_file_path = "mjcf/skyrover_floor.xml"
    
    timestep       = 1/240 
    decimation     = 10 
    sync           = True
    headless       = False
    render_set     = {
        "fps"    : 60,
        "width"  : 640,
        "height" : 480,
        "window_title": "SkyRover Solo Flight"
    }
    
    obs_rgb_cam_id   = [-1] # 相机id列表，-1代表自由相机
    obs_depth_cam_id = None
    
    rb_link_list   = [
        "skyrover", "skyrover_base_link", "skyrover_folder1_link",
        "skyrover_folder2_link", "skyrover_wheel1_link", "skyrover_wheel2_link",
        "skyrover_wheel3_link", "skyrover_wheel4_link", "skyrover_rotor1_link",
        "skyrover_rotor2_link", "skyrover_rotor3_link", "skyrover_rotor4_link",
    ]
    obj_list       = []

    use_gaussian_renderer = True
    gs_model_dict  = {
        "background"            :   "uav/feicuiwan.ply",
        
        "skyrover"              :   "skyrover/stretch_link.ply",
        "skyrover_base_link"    :   "skyrover/skyrover_base.ply",
        "skyrover_folder1_link" :   "skyrover/folder1_link.ply",
        "skyrover_folder2_link" :   "skyrover/folder2_link.ply",
        "skyrover_wheel2_link"  :   "skyrover/wheel1_2.ply",
        "skyrover_wheel1_link"  :   "skyrover/wheel1_2.ply",
        "skyrover_wheel3_link"  :   "skyrover/wheel3_4.ply",
        "skyrover_wheel4_link"  :   "skyrover/wheel3_4.ply",
        "skyrover_rotor2_link"  :   "skyrover/rotor2_4.ply",
        "skyrover_rotor1_link"  :   "skyrover/rotor1_3.ply",
        "skyrover_rotor3_link"  :   "skyrover/rotor1_3.ply",
        "skyrover_rotor4_link"  :   "skyrover/rotor2_4.ply",    
    }

def fix_scene_orientation(sim):
    """
    修复场景方向：绕 X 轴旋转 180 度。
    解决场景倒置的问题。
    """
    print("Applying 180-degree rotation to 3DGS scene...")
    try:
        renderer = sim.gs_renderer.renderer
        with torch.no_grad():
            # 1. 翻转位置 (X, Y, Z) -> (X, -Y, -Z)
            renderer.gaussians.xyz[:, 1] *= -1
            renderer.gaussians.xyz[:, 2] *= -1
            
            # 2. 翻转旋转四元数
            qs = renderer.gaussians.rot
            w, x, y, z = qs[:, 0].clone(), qs[:, 1].clone(), qs[:, 2].clone(), qs[:, 3].clone()
            
            # 绕X轴旋转180度的四元数乘法结果: [-x, w, -z, y]
            renderer.gaussians.rot[:, 0] = -x
            renderer.gaussians.rot[:, 1] = w
            renderer.gaussians.rot[:, 2] = -z
            renderer.gaussians.rot[:, 3] = y
            
            renderer.need_rerender = True
        print("Scene rotation applied successfully.")
    except Exception as e:
        print(f"[Error] Failed to rotate scene: {e}")

if __name__ == "__main__":

    cfg = SkyRoverSoloCfg()
    exec_node = SkyRoverSoloBase(cfg)
    obs = exec_node.reset()

    # # 将物理地面移到 Z = -10.0 米处
    # floor_id = mujoco.mj_name2id(exec_node.mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    # if floor_id != -1:
    #     exec_node.mj_model.geom_pos[floor_id][2] = -10.0
    #     print("Fixed: Physical floor moved to Z = -10.0m")
    # else:
    #     print("Warning: Could not find floor geom to move.")

    # 翻转场景
    fix_scene_orientation(exec_node)

    # 将无人机瞬移到指定的中心位置
    print(f"Relocating drone to: {SCENE_CENTER}")
    exec_node.reset_pose(SCENE_CENTER)

    flight_ctrl = PIDControl()
    move_speed = 0.1
    yaw_speed = 0.05    # 转向速度
    
    # 获取初始状态并设置控制目标
    init_pos, _, _, _, _ = exec_node.get_skyrover_state()
    target_pos = np.array(init_pos) # SCENE_CENTER
    target_yaw = 0.0

    print("Simulation Started. Control with W/S/A/D/Q/E.")

    while exec_node.running:
        # --- 键盘输入更新目标位置 ---
        if exec_node.key_state[glfw.KEY_W]: 
            target_pos[0] += move_speed # 前
            print("按下W键")
        if exec_node.key_state[glfw.KEY_S]: 
            target_pos[0] -= move_speed # 后
            print("按下S键")
        if exec_node.key_state[glfw.KEY_A]: 
            target_pos[1] += move_speed # 左
            print("按下A键")
        if exec_node.key_state[glfw.KEY_D]: 
            target_pos[1] -= move_speed # 右
            print("按下D键")
        if exec_node.key_state[glfw.KEY_Q]: 
            target_pos[2] += move_speed # 上
            print("按下Q键")
        if exec_node.key_state[glfw.KEY_E]: 
            target_pos[2] -= move_speed # 下
            print("按下E键")
        if exec_node.key_state[glfw.KEY_J]:
            target_yaw += yaw_speed
            print("按下J键 (左转)")
        if exec_node.key_state[glfw.KEY_L]:
            target_yaw -= yaw_speed
            print("按下L键 (右转)")
        # 最低高度限制，防止穿地太深 (放宽限制)
        if target_pos[2] < -2.0: target_pos[2] = -2.0 

        # --- 获取当前状态 ---
        pos, vel, quat, pqr, eul = exec_node.get_skyrover_state()
        state = np.concatenate((pos, quat, vel, pqr))

        # --- PID 计算 ---
        rpm, _, _ = flight_ctrl.compute_control_from_state(
            control_timestep=1/24,  # 原先是exec_node.dt
            state=state,
            target_pos=target_pos,
            target_rpy=np.array([0., 0., target_yaw]),
            target_vel=np.zeros(3),
            target_rpy_rates=np.zeros(3)
        )
        
        scaled_rpm = cu.scale_rpm_array(rpm)

        # print(f"当前高度: {pos[2]:.2f} | 目标高度: {target_pos[2]:.2f} | 推力RPM: {scaled_rpm[0]:.1f}")
        roll_deg = np.rad2deg(eul[0])
        pitch_deg = np.rad2deg(eul[1])
        yaw_deg = np.rad2deg(eul[2])

        print(f"[Pos]: {pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f} | [Target H]: {target_pos[2]:.3f} | "
              f"RPY: [{roll_deg:.1f}, {pitch_deg:.1f}, {yaw_deg:.1f}]")

        exec_node.skyrover_fly(scaled_rpm)

        # --- 仿真步进 ---
        obs, _, _, _, _ = exec_node.step(exec_node.mj_data.ctrl)

    print("\nExiting...")