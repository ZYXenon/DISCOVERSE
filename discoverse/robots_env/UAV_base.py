# only upload static 3dgs model of feicuiwan

from discoverse.envs import SimulatorBase          
from discoverse.utils.base_config import BaseConfig 
import numpy as np
import torch

class UAVCfg(BaseConfig):
    mjcf_file_path = "mjcf/camera_env.xml"   
    
    use_gaussian_renderer = True
    gs_model_dict = {
        "background": "uav/feicuiwan.ply",        
    }

    obj_list = []
    rb_link_list = []

    decimation = 10
    timestep   = 1/240
    sync       = True
    headless   = False
    render_set = {
        "fps":    60,
        "width":  640,
        "height": 480,
        "window_title": "feicuiwan (Rotated)"
    }

def fix_scene_orientation(sim):
    """
    修复场景方向：绕 X 轴旋转 180 度。
    这会将 (x, y, z) 变为 (x, -y, -z)，解决场景倒置的问题。
    """
    print("Applying 180-degree rotation to 3DGS scene...")
    try:
        # 获取渲染器中的高斯模型数据
        renderer = sim.gs_renderer.renderer
        
        with torch.no_grad():
            # 1. 翻转位置 (XYZ) -> (X, -Y, -Z)
            renderer.gaussians.xyz[:, 1] *= -1
            renderer.gaussians.xyz[:, 2] *= -1
            
            # 2. 翻转旋转四元数 (Rot)
            # 对应 180 度 X 轴旋转的四元数乘法
            # 假设四元数格式为 [w, x, y, z]
            qs = renderer.gaussians.rot
            w, x, y, z = qs[:, 0].clone(), qs[:, 1].clone(), qs[:, 2].clone(), qs[:, 3].clone()
            
            # q_new = q_rot(1,0,0,0) * q_old
            # 结果推导为: [-x, w, -z, y]
            renderer.gaussians.rot[:, 0] = -x
            renderer.gaussians.rot[:, 1] = w
            renderer.gaussians.rot[:, 2] = -z
            renderer.gaussians.rot[:, 3] = y
            
            # 强制刷新渲染缓冲
            renderer.need_rerender = True
            
        print("Scene rotation applied successfully.")
        
    except Exception as e:
        print(f"[Error] Failed to rotate scene: {e}")
        print("Ensure 'torch' is installed and the renderer is initialized correctly.")

class UAVBase(SimulatorBase):
    def post_physics_step(self):
        pass
    def getChangedObjectPose(self):
        return {}
    def checkTerminated(self):
        return False
    def getObservation(self):
        return {
            "rgb": getattr(self, "img_rgb_obs_s", {}),
            "depth": getattr(self, "img_depth_obs_s", {})
        }
    def getPrivilegedObservation(self):
        return self.getObservation()
    def getReward(self):
        return None

if __name__ == "__main__":
    cfg = UAVCfg()
    sim = UAVBase(cfg)
    
    # 在启动循环前修复场景方向
    fix_scene_orientation(sim)

    print("Viewer started. Move with Mouse.")
    
    while sim.running:
        sim.view()
        pos, quat = sim.getCameraPose(sim.cam_id)
        print(f"[Camera Pos]: X={pos[0]:.3f}  Y={pos[1]:.3f}  Z={pos[2]:.3f}")

    print("\nExiting...")