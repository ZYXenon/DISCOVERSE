import glfw
import numpy as np
from discoverse.utils import BaseConfig
from pid_control import PIDControl
import control_util as cu

from skyrover_manual_base import SkyRoverSoloBase

class SkyRoverSoloCfg(BaseConfig):
    # 仅加载 SkyRover 无人机和地板场景

    mjcf_file_path = "mjcf/skyrover_floor.xml"
    
    timestep       = 1/240 
    decimation     = 4
    sync           = True
    headless       = False
    render_set     = {
        "fps"    : 60,
        "width"  : 1920,
        "height" : 1080,
        "window_title": "SkyRover Solo Flight (WASD=Move, QE=Up/Down)"
    }
    
    obs_rgb_cam_id   = [-1] # 相机id列表，-1代表自由相机
    obs_depth_cam_id = None
    
    rb_link_list   = [
        "skyrover", 
        "skyrover_base_link", 
        "skyrover_folder1_link",
        "skyrover_folder2_link", 
        "skyrover_wheel1_link", 
        "skyrover_wheel2_link",
        "skyrover_wheel3_link", 
        "skyrover_wheel4_link", 
        "skyrover_rotor1_link",
        "skyrover_rotor2_link", 
        "skyrover_rotor3_link", 
        "skyrover_rotor4_link",
    ]
    obj_list       = []
    use_gaussian_renderer = False
    gs_model_dict  = {
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

if __name__ == "__main__":

    cfg = SkyRoverSoloCfg()
    cfg.use_gaussian_renderer = False
    
    # cfg.gs_model_dict["background"] = "scene/riverside1/point_cloud.ply"
    # cfg.gs_model_dict["background_env"] = "scene/riverside1/environment.ply"

    exec_node = SkyRoverSoloBase(cfg)
    obs = exec_node.reset()

    flight_ctrl = PIDControl()

    move_speed = 0.02
    
    # 获取初始状态
    init_pos, _, _, _, _ = exec_node.get_skyrover_state()
    target_pos = np.array(init_pos)
    target_pos[2] = 1.0  # 初始起飞高度 1米
    target_yaw = 0.0

    print("Simulation Started. Control with W/S/A/D/Q/E.")

    while exec_node.running:
        # --- 键盘输入更新目标位置 ---
        if exec_node.key_state[glfw.KEY_W]: target_pos[0] += move_speed # 前
        if exec_node.key_state[glfw.KEY_S]: target_pos[0] -= move_speed # 后
        if exec_node.key_state[glfw.KEY_A]: target_pos[1] += move_speed # 左
        if exec_node.key_state[glfw.KEY_D]: target_pos[1] -= move_speed # 右
        if exec_node.key_state[glfw.KEY_Q]: target_pos[2] += move_speed # 上
        if exec_node.key_state[glfw.KEY_E]: target_pos[2] -= move_speed # 下
        if target_pos[2] < 0.1: target_pos[2] = 0.1 # 地面限制

        # --- 获取当前状态 ---
        pos, vel, quat, pqr, eul = exec_node.get_skyrover_state()
        state = np.concatenate((pos, quat, vel, pqr))

        # --- PID 计算 ---
        rpm, _, _ = flight_ctrl.compute_control_from_state(
            control_timestep=exec_node.dt,
            state=state,
            target_pos=target_pos,
            target_rpy=np.array([0., 0., target_yaw]),
            target_vel=np.zeros(3),
            target_rpy_rates=np.zeros(3)
        )
        
        # --- 缩放 RPM 并执行 ---
        scaled_rpm = cu.scale_rpm_array(rpm)
        exec_node.skyrover_fly(scaled_rpm)

        # --- 仿真步进 ---
        obs, _, _, _, _ = exec_node.step(exec_node.mj_data.ctrl)

    print("Exiting...")