# only upload static 3dgs model of feicuiwan

from discoverse.envs import SimulatorBase          
from discoverse.utils.base_config import BaseConfig 

class UAVCfg(BaseConfig):
    """camera_env.xml只提供相机与“空世界”，3DGS 负责背景可视化"""
    mjcf_file_path = "mjcf/camera_env.xml"   
    
    use_gaussian_renderer = True
    gs_model_dict = {
        "background": "uav/feicuiwan.ply",        
    }

    obj_list = []
    rb_link_list = []

    decimation = 2
    timestep   = 0.005
    sync       = True
    headless   = False
    render_set = {
        "fps":    30,
        "width":  1280,
        "height": 720,
        "window_title": "feicuiwan"
    }

class UAVBase(SimulatorBase):
    def post_physics_step(self):
        pass

    def getChangedObjectPose(self):
        return {}

    def checkTerminated(self):
        # 持续显示，直到关掉窗口
        return False

    def getObservation(self):
        # 返回当下缓存的图像
        return {
            "rgb":   getattr(self, "img_rgb_obs_s", {}),
            "depth": getattr(self, "img_depth_obs_s", {})
        }

    def getPrivilegedObservation(self):
        return self.getObservation()

    def getReward(self):
        return None

if __name__ == "__main__":
    cfg = UAVCfg()

    # 创建并进入查看循环
    # 窗口内默认按 ESC 切自由视角
    # Ctrl+G 可开/关 3DGS 显示
    sim = UAVBase(cfg)
    while sim.running:
        sim.view()  # 仅推进时间并渲染一帧（不做物理步进）
