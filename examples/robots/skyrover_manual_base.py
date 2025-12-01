import mujoco
import xml.etree.ElementTree as ET
import glfw
from discoverse.envs import SimulatorBase
import os
from discoverse.utils import BaseConfig

class SkyRoverSoloBase(SimulatorBase):
    def __init__(self, config: BaseConfig):
        super().__init__(config)

        self.robot_names = ["skyrover"] # 仅包含无人机
        self.robots = {name: {"joints": {}, "bodies": {}, "sites": {}} for name in self.robot_names}    
        
        # 解析 XML 获取电机映射
        # self.actuator_mapping = self._parse_actuators_from_xml(self.mjcf_file)  # robot attributes
        control_xml_path = os.path.join(os.path.dirname(self.mjcf_file), "skyrover/skyrover_control.xml")
        self.actuator_mapping = self._parse_actuators_from_xml(control_xml_path)

        # 初始化机器人映射
        for robot_name in self.robot_names:
            self._initialize_robot(robot_name)

        self.dt = self.mj_model.opt.timestep

        print("="*100)
        print("Robots initialized with the following mappings:", self.robots)
        print("="*100)

        # 默认设置为 Drone 模式 (展开姿态)
        # 设置关节位置，模拟展开状态
        self.set_joint_position("skyrover", "skyrover_stretch_joint", -5.04e-2)
        self.set_joint_position("skyrover", "skyrover_folder1_joint", 2.5e-4)
        self.set_joint_position("skyrover", "skyrover_folder2_joint", 2.5e-4)
        
        self.key_state = {
            glfw.KEY_W: False, glfw.KEY_S: False,
            glfw.KEY_A: False, glfw.KEY_D: False,
            glfw.KEY_Q: False, glfw.KEY_E: False,
            glfw.KEY_J: False, glfw.KEY_L: False
        }

    # 重置无人机位姿的方法
    def reset_pose(self, pos, quat=None):
        """
        强制设置无人机(skyrover)的位置和姿态
        pos: [x, y, z]
        quat: [w, x, y, z] (可选)
        """
        if "skyrover" in self.free_body_qpos_ids:
            q_idx = self.free_body_qpos_ids["skyrover"]
            
            # 设置位置
            self.mj_data.qpos[q_idx:q_idx+3] = pos
            
            # 设置姿态 (默认为单位四元数)
            if quat is not None:
                self.mj_data.qpos[q_idx+3:q_idx+7] = quat
            else:
                self.mj_data.qpos[q_idx+3:q_idx+7] = [1.0, 0.0, 0.0, 0.0]
            
            # 重置速度为0
            v_idx = self.mj_model.jnt_dofadr[self.mj_model.jnt_qposadr[q_idx]]
            self.mj_data.qvel[v_idx:v_idx+6] = 0.0
            
            # 刷新物理状态
            mujoco.mj_forward(self.mj_model, self.mj_data)
        else:
            print("[ERROR] Cannot find 'skyrover' free joint to reset pose.")

    def on_key(self, window, key, scancode, action, mods):
        # 键盘控制
        # 调用父类方法以保留基础功能（如ESC退出，S截图等）
        super().on_key(window, key, scancode, action, mods)
        if key in self.key_state:
            if action == glfw.PRESS:    # 按下
                self.key_state[key] = True
            elif action == glfw.RELEASE:    # 松开
                self.key_state[key] = False

    def updateControl(self, action):
        self.mj_data.ctrl[:] = action[:]

    def checkTerminated(self):
        return False

    def getObservation(self):
        self.obs = {
            "jq"  : self.mj_data.qpos.tolist(),
            "jv"  : self.mj_data.qvel.tolist(),
            "img" : self.img_rgb_obs_s
        }
        return self.obs
    
    def getPrivilegedObservation(self): # 特权观测(Privileged Observation)训练时有而测试时没有，RL会用
        return self.obs
    
    def getReward(self):
        return None

    # --- 核心功能接口 ---

    def skyrover_fly(self, rotor_thrust):
        """ 应用推力到四个旋翼 """
        thrust_fl = "skyrover_thrust_fl"
        thrust_bl = "skyrover_thrust_bl"
        thrust_br = "skyrover_thrust_br"
        thrust_fr = "skyrover_thrust_fr"
        self.set_site_control_value("skyrover", thrust_fl, rotor_thrust[0])
        self.set_site_control_value("skyrover", thrust_bl, rotor_thrust[1])
        self.set_site_control_value("skyrover", thrust_br, rotor_thrust[2])
        self.set_site_control_value("skyrover", thrust_fr, rotor_thrust[3])

    def get_skyrover_state(self):
        """ 获取无人机状态 (Pos, Vel, Quat, PQR, Eul) """
        pos, quat = self.get_root_body_pose("skyrover", "skyrover")
        vel, pqr = self.get_root_body_velocity("skyrover", "skyrover")
        
        from scipy.spatial.transform import Rotation as R
        rotation = R.from_quat(quat[[1, 2, 3, 0]])
        eul = rotation.as_euler('ZYX', degrees=False)[::-1]
        return pos, vel, quat, pqr, eul

    def set_joint_position(self, robot_name, joint_name, position):
        """
        设置机器人特定关节的位置。
        
        注意：由于切换了 XML 文件 (skyrover_floor.xml)，ctrl 的索引顺序发生了变化。
        此处的索引是根据 skyrover_control.xml 的 motor 定义顺序重新映射的。
        """
        if self._sanity_check(robot_name=robot_name, attribute_name=joint_name, type="joints"):
            if robot_name == "skyrover":
                if joint_name == "skyrover_stretch_joint":
                    self.mj_data.ctrl[0] = position
                elif joint_name == "skyrover_folder1_joint":
                    self.mj_data.ctrl[1] = position
                elif joint_name == "skyrover_folder2_joint":
                    self.mj_data.ctrl[6] = position
                elif joint_name == "skyrover_rotor1_joint":
                    self.mj_data.ctrl[10] = position
                elif joint_name == "skyrover_rotor2_joint":
                    self.mj_data.ctrl[4] = position
                elif joint_name == "skyrover_rotor3_joint":
                    self.mj_data.ctrl[9] = position
                elif joint_name == "skyrover_rotor4_joint":
                    self.mj_data.ctrl[5] = position
        else:
            print(f"[ERROR] Sanity check failed for robot '{robot_name}' and attribute '{joint_name}'. Check that the attribute exists and the robot is initialized properly.")

    def set_site_control_value(self, robot_name, site_name, control_value):
        actuator_id = self.actuator_mapping.get(site_name)
        if actuator_id is not None:
            self.mj_data.ctrl[actuator_id] = control_value

    def get_root_body_pose(self, robot_name, root_body_name):
        body_id = self.robots[robot_name]["bodies"][root_body_name]
        joint_id = self._get_joint_id_from_body_id(body_id, "qpos")
        if joint_id is not None:
            pos = self.mj_data.qpos[joint_id:joint_id+3]
            quat = self.mj_data.qpos[joint_id+3:joint_id+7]
            return pos, quat
        return None, None

    def get_root_body_velocity(self, robot_name, root_body_name):
        body_id = self.robots[robot_name]["bodies"][root_body_name]
        joint_id = self._get_joint_id_from_body_id(body_id, "qvel")
        if joint_id is not None:
            vel = self.mj_data.qvel[joint_id:joint_id+3]
            pqr = self.mj_data.qvel[joint_id+3:joint_id+6]
            return vel, pqr
        return None, None

    def _initialize_robot(self, robot_name):
        """
        Set up joint, body, site mappings for each robot.
        """
        # Initialize joints
        for i in range(self.mj_model.njnt):
            joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name and joint_name.startswith(robot_name):
                self.robots[robot_name]["joints"][joint_name] = i

        # Initialize bodies
        for i in range(self.mj_model.nbody):
            body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and body_name.startswith(robot_name):
                self.robots[robot_name]["bodies"][body_name] = i

        # Initialize sites (for control, we need actuator ID not site ID)
        for i in range(self.mj_model.nsite):
            site_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, i)
            if site_name and site_name.startswith(robot_name):
                self.robots[robot_name]["sites"][site_name] = i

        # Error handling if no bodies or joints are found
        if not self.robots[robot_name]["bodies"] or not self.robots[robot_name]["joints"]:
            print("[ERROR] To properly manage joints and other properties for multiple robots using this script,",
            "ensure that the bodies and joints elements in the XML file are prefixed with the robot name.")

    def _sanity_check(self, robot_name, attribute_name, type):
        """ 
        Sanity check for joints and sites in robot's body. 
        """
        # Check if the robot exists
        if robot_name not in self.robots:
            print(f"[ERROR] Robot {robot_name} not found.")
            return False

        # Check if the attribute exists in the robot
        if type not in ["joints", "sites"]:
            raise TypeError("[ERROR] Such attribute type does not exist!")
        
        else:
            if attribute_name not in self.robots[robot_name][type]:
                print(f"[ERROR] Site {attribute_name} not found for robot {robot_name}.")
                return False
            else:
                return True

    def _parse_actuators_from_xml(self, xml_file):
        """
        Parse the XML to manually map actuator indices to sites.
        This is a workaround that circumvents the need for `mj_id2name` for actuators with sites. 
        
        MuJoCo doesn't find the names because actuators that use site attributes (rather than joint attributes) are not directly linked to body IDs in a way that `mj_id2name` can access. MuJoCo expects actuators to be associated with a joint attribute to create a clear mapping for `mj_id2name` calls, but the site parameter does not directly map to an ID within MuJoCo's internal structure.
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()

        actuator_mapping = {}
        
        # Iterate over motor actuators in XML
        for i, motor in enumerate(root.findall("motor")): # get the site linked to this actuator
            site_name = motor.get("site")
            
            # Check if the site name is valid
            if site_name: actuator_mapping[site_name] = i   # Map site name to index
            
        return actuator_mapping

    def _get_joint_id_from_body_id(self, body_id, qpos_qvel_flag):
        joint_addr_map = {"qpos": self.mj_model.jnt_qposadr, "qvel": self.mj_model.jnt_dofadr}
        joint_addr = joint_addr_map[qpos_qvel_flag]
        for j_id, b_id in enumerate(self.mj_model.jnt_bodyid):
            if b_id == body_id: return joint_addr[j_id]
        return None