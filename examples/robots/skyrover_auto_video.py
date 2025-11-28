import cv2
import numpy as np
import os
import time
from discoverse.utils import BaseConfig
from pid_control import PIDControl
import control_util as cu
from skyrover_manual_base import SkyRoverSoloBase

class SkyRoverVideoCfg(BaseConfig):
    mjcf_file_path = "mjcf/skyrover_floor.xml"
    
    timestep       = 1/240 
    decimation     = 4
    sync           = False 
    headless       = True 
    enable_render  = True 

    use_gaussian_renderer = False
    
    render_set     = {
        "fps"    : 30,
        "width"  : 1280,   
        "height" : 720,  
    }
    
    # 指定相机ID: 0 对应 skyrover_track_1 (自动跟踪视角)
    obs_rgb_cam_id   = [0] 
    obs_depth_cam_id = []

if __name__ == "__main__":
    cfg = SkyRoverVideoCfg()
    output_video_path = "skyrover_demo.mp4"
    
    exec_node = SkyRoverSoloBase(cfg)
    obs = exec_node.reset()
    
    flight_ctrl = PIDControl()
    
    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        output_video_path, 
        fourcc, 
        cfg.render_set["fps"], 
        (cfg.render_set["width"], cfg.render_set["height"])
    )
    print(f"Start recording video to {output_video_path}...")

    # 起飞 -> 前飞 -> 右飞 -> 降落
    waypoints = [
        np.array([0.0, 0.0, 1.0]),  # 起飞到 1m
        np.array([2.0, 0.0, 1.0]),  # 向前飞 2m
        np.array([2.0, -2.0, 1.0]), # 向右飞 2m
        np.array([0.0, 0.0, 0.2]),  # 返回并降落
    ]
    current_wp_idx = 0
    target_pos = waypoints[0]
    target_yaw = 0.0
    
    sim_duration = 5.0 # 总模拟时长(秒)
    sim_steps = int(sim_duration / exec_node.dt)
    error_threshold = 0.2 # 到达航点的判定距离
    
    print("Simulation Auto-Pilot Started.")

    try:
        for i in range(sim_steps):
            # --- 状态机逻辑 ---
            # 获取当前状态
            pos, vel, quat, pqr, eul = exec_node.get_skyrover_state()
            
            # 判断是否到达当前航点
            dist = np.linalg.norm(pos - target_pos)
            if dist < error_threshold:
                print(f"Reached waypoint {current_wp_idx}: {target_pos}")
                current_wp_idx += 1
                if current_wp_idx < len(waypoints):
                    target_pos = waypoints[current_wp_idx]
                else:
                    print("All waypoints completed. Hovering...")
            
            # --- PID 控制 ---
            state = np.concatenate((pos, quat, vel, pqr))
            rpm, _, _ = flight_ctrl.compute_control_from_state(
                control_timestep=exec_node.dt,
                state=state,
                target_pos=target_pos,
                target_rpy=np.array([0., 0., target_yaw]),
                target_vel=np.zeros(3),
                target_rpy_rates=np.zeros(3)
            )
            
            # 执行控制
            scaled_rpm = cu.scale_rpm_array(rpm)
            exec_node.skyrover_fly(scaled_rpm)
            
            # --- 仿真步进 ---
            # 注意：step函数内部会根据 fps 设置决定是否进行渲染(render)
            obs, _, _, _, _ = exec_node.step(exec_node.mj_data.ctrl)
            
            # --- 视频录制 ---
            # 只有当渲染发生时（img数据更新）才写入视频帧
            # simulator.py 中逻辑是 step -> post_physics_step -> render(如果时间到了)
            # 可以检查 obs['img'] 是否有数据
            
            if 0 in obs["img"] and obs["img"][0] is not None:
                # 获取图像 (RGB)
                img_rgb = obs["img"][0]
                
                # OpenCV 需要 BGR 格式
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                
                # 写入视频
                video_writer.write(img_bgr)
            
            if i % 100 == 0:
                print(f"Step {i}/{sim_steps}, Pos: {pos[:2]}, Alt: {pos[2]:.2f}")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # 资源释放
        video_writer.release()
        print(f"\nVideo saved successfully: {os.path.abspath(output_video_path)}")