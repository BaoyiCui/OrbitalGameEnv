from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from gymnasium import spaces
from collections import deque
import torch
import datetime
import ctypes
import os

# 导入父类
from .mpe_env import MPEEnv, MPEEnvCfg

# ================= Ctypes Interface Start =================
# 尝试加载库，失败则回退到 ECI
try:
    # 请根据实际路径修改
    so_path = "/home/star/Downloads/gemini-cli-main/OrbitalGameEnv/demos/OrbitLib/so/X86/libOrbit.so"
    if not os.path.exists(so_path):
        # 尝试相对路径
        so_path = os.path.join(os.path.dirname(__file__), "..", "OrbitLib", "so", "X86", "libOrbit.so")
    
    if os.path.exists(so_path):
        orbit_lib_c = ctypes.CDLL(so_path)
    else:
        raise FileNotFoundError("libOrbit.so not found")
        
except Exception as e:
    print(f"[93mWarning: Failed to load libOrbit.so ({e}). LVLH transformation disabled.")
    orbit_lib_c = None

if orbit_lib_c:
    orbit_lib_c.DCM_J2000_to_LVLH.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER((ctypes.c_double * 3) * 3)]
    orbit_lib_c.DCM_J2000_to_LVLH.restype = None

def get_lvlh_dcm(state_j2000: np.ndarray) -> np.ndarray | None:
    if not orbit_lib_c: return None
    rv_in = state_j2000.astype(np.float64)
    dcm_out = ((ctypes.c_double * 3) * 3)()
    orbit_lib_c.DCM_J2000_to_LVLH(rv_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), dcm_out)
    return np.array([[dcm_out[i][j] for j in range(3)] for i in range(3)])

def eci_to_lvlh_relative(observer_state, target_pos, target_vel=None):
    """ 计算目标相对于观察者的 LVLH 坐标 """
    dcm = get_lvlh_dcm(observer_state)
    observer_pos = observer_state[:3]
    rel_pos_eci = target_pos - observer_pos
    
    if dcm is None: # 回退模式
        rel_vel_eci = (target_vel - observer_state[3:]) if target_vel is not None else None
        return rel_pos_eci, rel_vel_eci

    rel_pos_lvlh = dcm @ rel_pos_eci
    rel_vel_lvlh = None
    if target_vel is not None:
        rel_vel_eci = target_vel - observer_state[3:]
        rel_vel_lvlh = dcm @ rel_vel_eci
        
    return rel_pos_lvlh, rel_vel_lvlh
# ================= Ctypes Interface End =================


@dataclass
class MPE_POMDP_EnvCfg(MPEEnvCfg):
    use_partial_obs: bool = True
    obs_interval: int = 2
    history_len: int = 20

    # === 新增课程学习参数 (Defaults Updated) ===
    init_distance_m: float = 2000.0   
    ring_width_delta: float = 5000.0  
    
    # [修改] 更新物理参数的默认值以匹配更宽松的约束
    p_init_dv: float = 500.0
    dist_cap: float = 30000.0 

    use_lambert_reward: bool = True
    lambert_reward_weight: float = 0.05
    GEO_ORBIT_RADIUS: float = 42164000.0

class MPE_POMDP_Env(MPEEnv):
    @staticmethod
    def _symlog(x):
        # 保持之前的归一化改进：除以10
        return (np.sign(x) * np.log(np.abs(x) + 1.0)) / 10.0

    def __init__(self, config: MPE_POMDP_EnvCfg = MPE_POMDP_EnvCfg()):
        super().__init__(config)
        self._config: MPE_POMDP_EnvCfg = config
        self.agent_memory_orb_feat = {} # 为轨道特征新增记忆模块

        # 注册 RV2COE 函数 (如果 OrbitLib 没封装，我们在这里手动设置一下以防万一)
        if hasattr(self, '_orbit_lib') and hasattr(self._orbit_lib, 'orbit_lib_c') and self._orbit_lib.orbit_lib_c is not None:
             self._orbit_lib.orbit_lib_c.RV2COE.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
             self._orbit_lib.orbit_lib_c.RV2COE.restype = None

        if self._config.use_partial_obs:
            self.evader_history_buffers = {f'e_{i}': deque(maxlen=self._config.history_len) for i in range(self._config.num_e)}
            self.obs_counters = {f'e_{i}': 0 for i in range(self._config.num_e)}
            self.evader_history_mask_buffers = {f'e_{i}': deque(maxlen=self._config.history_len) for i in range(self._config.num_e)}

            self.observation_spaces = {}
            for a in self.possible_agents:
                if a.startswith('p_'):
                    # V5.1 观测空间 (增加相位特征):
                    # 1. 自身 (8维)
                    # 2. 目标上次已知位置 (3维)
                    # 3. 队友 (7 * (Num_P - 1))
                    # 4. [新增] 相对轨道要素 (3维): [Symlog(Delta_SMA), Sin(Delta_TA), Cos(Delta_TA)]
                    
                    self_obs_dim = 8
                    target_obs_dim = 3 * self._config.num_e 
                    teammates_obs_dim = 7 * (self._config.num_p - 1)
                    orbital_feat_dim = 3 # <--- 新增维度
                    
                    obs_shape = (self_obs_dim + target_obs_dim + teammates_obs_dim + orbital_feat_dim,)
                    self.observation_spaces[a] = spaces.Box(-np.inf, np.inf, shape=obs_shape)
                else:
                    self.observation_spaces[a] = spaces.Box(-np.inf, np.inf, shape=(6,))

    def _get_orbital_features(self, state_p, state_e):
        """
        计算追击者相对于逃逸者的轨道要素差异。
        输入: J2000 状态向量 [x, y, z, vx, vy, vz] (米, 米/秒)
        输出: np.array([symlog(delta_a), sin(delta_ta), cos(delta_ta)])
        """
        # 准备 Ctypes 数据
        rv_p = np.ascontiguousarray(state_p, dtype=np.float64)
        rv_e = np.ascontiguousarray(state_e, dtype=np.float64)
        coe_p = np.zeros(6, dtype=np.float64)
        coe_e = np.zeros(6, dtype=np.float64)

        # 调用 C++ RV2COE
        # coe 结构: [sma, ecc, inc, raan, argp, ta] (单位: 米, 弧度)
        if self._orbit_lib and hasattr(self._orbit_lib, 'orbit_lib_c') and self._orbit_lib.orbit_lib_c is not None:
            self._orbit_lib.orbit_lib_c.RV2COE(
                rv_p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                coe_p.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            )
            self._orbit_lib.orbit_lib_c.RV2COE(
                rv_e.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                coe_e.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            )
        else:
            return np.zeros(3) # 如果库加载失败，返回0

        # --- 1. 半长轴差异 (Delta SMA) ---
        # 物理意义：决定追击速度（相对漂移率）。
        # Delta_a < 0 -> 轨道低 -> 周期短 -> 相位增加 -> 追赶前方目标
        # 数值处理：SMA 差异范围极大(10m - 100km)，使用 Symlog 完美压缩
        delta_sma = coe_p[0] - coe_e[0]
        feat_sma = self._symlog(np.array([delta_sma]))

        # --- 2. 相位差异 (Delta True Anomaly) ---
        # 物理意义：决定两者在圆周上的相对位置。
        # 数值处理：使用 Sin/Cos 解决 0/2pi 周期突变问题，范围固定在 [-1, 1]
        delta_ta = coe_p[5] - coe_e[5]
        feat_sin_ta = np.sin(delta_ta)
        feat_cos_ta = np.cos(delta_ta)

        return np.concatenate([feat_sma, [feat_sin_ta], [feat_cos_ta]])

    def _get_privileged_state(self):
        """
        辅助函数：计算基于 Anchor LVLH 的特权观测。
        同步增强：为 Teacher 加入轨道要素特征，确保 Teacher 的信息量 >= Student。
        """
        privileged_components = []
        anchor_id = self.evader_ids[0]
        
        anchor_state = np.zeros(6)
        anchor_dcm = None

        # 1. Anchor (参考系中心) 的绝对状态
        if anchor_id in self.states:
            anchor_state = self.states[anchor_id]
            anchor_dcm = get_lvlh_dcm(anchor_state)
            privileged_components.append(self._symlog(anchor_state))
        else:
            privileged_components.append(np.zeros(6))

        # 2. 遍历其他所有 Agent (追击者 + 其他逃逸者)
        # 顺序：所有追击者 -> 剩余逃逸者
        agent_order = self.pursuer_ids + [eid for eid in self.evader_ids if eid != anchor_id]
        
        for agent_id in agent_order:
            if agent_id in self.states:
                target_state = self.states[agent_id]
                
                # A. 计算相对状态向量 (LVLH)
                if anchor_dcm is not None:
                    diff_pos = target_state[:3] - anchor_state[:3]
                    diff_vel = target_state[3:] - anchor_state[3:]
                    rel_pos = anchor_dcm @ diff_pos
                    rel_vel = anchor_dcm @ diff_vel
                    rel_state = np.concatenate([rel_pos, rel_vel])
                else:
                    rel_state = target_state - anchor_state
                
                # B. [新增] 计算相对轨道要素特征
                # 这一步至关重要，让 Teacher 也能显式看到相位和能量差异
                orb_feat = self._get_orbital_features(target_state, anchor_state)
                
                # 拼接: [相对位置速度(6), 相对轨道要素(3)]
                privileged_components.append(np.concatenate([self._symlog(rel_state), orb_feat]))
            else:
                # 补零: 6维状态 + 3维轨道特征 = 9维
                privileged_components.append(np.zeros(9))
        
        return np.concatenate(privileged_components)

    def reset(self, seed=None, options=None):
        observations, infos = super().reset(seed, options)

        if self._config.use_partial_obs:
            self.obs_counters = {f'e_{i}': 0 for i in range(self._config.num_e)}
            # 重置 Last Known
            if hasattr(self, 'agent_memory_evader_lvlh'):
                self.agent_memory_evader_lvlh.clear()
            if hasattr(self, 'agent_memory_orb_feat'):
                self.agent_memory_orb_feat.clear()

            for evader_id in self.evader_ids:
                self.evader_history_buffers[evader_id].clear()
                self.evader_history_mask_buffers[evader_id].clear()
                if evader_id in self.states:
                    initial_state = self.states[evader_id]
                    for _ in range(self._config.history_len):
                        self.evader_history_buffers[evader_id].append(initial_state)
                        self.evader_history_mask_buffers[evader_id].append(1.0)

            # 这里必须先 update history，算出初始的 last_known，再 get observations
            self._update_history_and_prepare_data() 
            observations = self._get_observations() # 重新获取包含 target 的 obs

            # 使用新的 LVLH 特权观测
            privileged_state = self._get_privileged_state()
            for agent in self.pursuer_ids:
                if agent not in infos: infos[agent] = {}
                infos[agent]['privileged_state'] = privileged_state

        return observations, infos

    def set_difficulty_parameters(self, m_distance=None, ring_width_delta=None, p_init_dv=None, dist_cap=None):
        """
        课程学习接口修改
        m_distance: 对应 min_offset
        """
        if m_distance is not None:
            self._config.init_distance_m = m_distance
        
        if ring_width_delta is not None:
            self._config.ring_width_delta = ring_width_delta

        # 映射到父类参数，确保父类 reset 逻辑生成的圆环正确
        # min_offset = m
        # max_offset = m + delta
        self._config.e_init_dist_min_offset = self._config.init_distance_m
        self._config.e_init_dist_max_offset = self._config.init_distance_m + self._config.ring_width_delta
            
        if p_init_dv is not None:
            self._config.p_init_dv = p_init_dv
            
        if dist_cap is not None:
            self._config.dist_cap = dist_cap

    def step(self, actions: dict[str, np.ndarray]):
        self.step_count += 1
        # 在step执行前，基于上一步的状态准备好给agent的输入
        if self._config.use_partial_obs:
            self._update_history_and_prepare_data()
        
        # (与父类MPEEnv相同的动力学和奖励计算)
        for a in self.agents:
            # 1. 获取该智能体的物理限制参数 (dv_step)
            dv_step = self._config.p_dv_step if a.startswith('p_') else self._config.e_dv_step

            # 2. 获取网络输出的“标准化动作”
            norm_action = actions.get(a, np.zeros(3))
            
            physical_action = np.zeros(3)

            # === 3. 维度处理与反归一化 (核心逻辑) ===
            if self._config.dim_mode == 2:
                # [2D 模式逻辑]
                action_xy_norm = norm_action[:2]
                action_xy_phys = action_xy_norm * dv_step 
                xy_mag = np.linalg.norm(action_xy_phys)
                if xy_mag > dv_step:
                    action_xy_phys = action_xy_phys / xy_mag * dv_step
                physical_action = np.array([action_xy_phys[0], action_xy_phys[1], 0.0])
            else: 
                # [3D 模式逻辑]
                action_3d_phys = norm_action * dv_step
                xyz_mag = np.linalg.norm(action_3d_phys)
                if xyz_mag > dv_step:
                    action_3d_phys = action_3d_phys / xyz_mag * dv_step
                physical_action = action_3d_phys

            # === 4. 燃料限制 (物理硬约束) ===
            current_fuel = self.remain_Dvs.get(a, 0.0)
            p_norm = np.linalg.norm(physical_action)
            
            if p_norm > current_fuel:
                if p_norm > 1e-8:
                    physical_action = physical_action / p_norm * current_fuel
                else:
                    physical_action = np.zeros(3)

            # === 5. 执行物理更新 (仅速度) ===
            self.states[a][3:] += physical_action
            
            used_fuel = np.linalg.norm(physical_action)
            self.remain_Dvs[a] = max(0.0, self.remain_Dvs.get(a, 0.0) - used_fuel)
            
            # [重要] 更新 actions 字典中的值为真实的物理动作
            actions[a] = physical_action

        # 轨道积分
        for a in self.agents:
            _, new_state = self._orbit_lib.orbit_hpop(self._time, self.states[a], self._config.dt, self._config.hpop_in)
            self.states[a] = new_state
        self._time = self._time + datetime.timedelta(seconds=self._config.dt)

        # 观测和奖励计算
        observations = self._get_observations()
        rewards, debug_reward_info = self._get_rewards(actions) # 使用更新后的 actions
        terminations, termination_reasons = self._get_terminations()
        truncations = self._get_truncations()
        self.terminations, self.truncations = terminations, truncations

        # 准备Infos
        current_infos = {a: self.infos.get(a, {}) for a in self.possible_agents if a in self.agents}
        
        if any(terminations.values()) or any(truncations.values()):
            self.episode_statistics['total_episodes'] += 1
            reason = list(termination_reasons.values())[0] if termination_reasons else 'unknown'
            
            if reason == 'capture_success': self.episode_statistics['success_count'] += 1
            elif reason == 'timeout': self.episode_statistics['timeout_count'] += 1
            elif reason == 'fuel_out': self.episode_statistics['fuelout_count'] += 1
            
            if self.episode_statistics['total_episodes'] > 0:
                self.episode_statistics['success_rate'] = self.episode_statistics['success_count'] / self.episode_statistics['total_episodes']

        # 更新特权信息和最终观测
        privileged_state = self._get_privileged_state()
        for agent in self.possible_agents:
            if agent not in current_infos: current_infos[agent] = {}
            current_infos[agent]['termination_reason'] = termination_reasons.get(agent, None)
            current_infos[agent]['episode_statistics'] = self.episode_statistics.copy()
            if self._config.debug_rewards and agent in debug_reward_info:
                current_infos[agent]['reward_components'] = debug_reward_info[agent]
            if agent.startswith('p_'):
                current_infos[agent]['privileged_state'] = privileged_state

        for agent in list(self.agents):
            if terminations.get(agent, False) or truncations.get(agent, False):
                if agent in observations: current_infos[agent]['final_observation'] = observations[agent]
                self.agents.remove(agent)

        return observations, rewards, self.terminations, self.truncations, current_infos


    def _get_observations(self):
        if not self._config.use_partial_obs:
            return super()._get_observations()

        observations = {}
        all_states = {aid: self.states[aid] for aid in self.possible_agents if aid in self.states}
        
        primary_evader_id = self.evader_ids[0] if self.evader_ids else None

        for agent_id in self.pursuer_ids:
            if agent_id not in all_states: continue
            
            my_state = all_states[agent_id]
            
            # 1. 自身信息 (Symlog + 归一化)
            my_pos = my_state[:3]
            my_vel = my_state[3:]
            my_dist = np.linalg.norm(my_pos)
            alt_dev = self._symlog(np.array([my_dist - self._config.GEO_ORBIT_RADIUS]))
            pos_dir = my_pos / (my_dist + 1e-6)
            vel_sym = self._symlog(my_vel)
            fuel = np.array([self.remain_Dvs.get(agent_id, 0.0) / self._config.p_init_dv])
            
            # 2. 目标位置信息 (Last Known, LVLH, Symlog)
            target_feat = []
            for eid in self.evader_ids:
                is_visible = (self.obs_counters[eid] % self._config.obs_interval == 0)
                
                mem_key = (agent_id, eid)
                if not hasattr(self, 'agent_memory_evader_lvlh'):
                     self.agent_memory_evader_lvlh = {}
                
                if eid in all_states:
                    if is_visible or mem_key not in self.agent_memory_evader_lvlh:
                        e_state = all_states[eid]
                        rel_pos, _ = eci_to_lvlh_relative(my_state, e_state[:3], None)
                        if rel_pos is not None:
                            self.agent_memory_evader_lvlh[mem_key] = rel_pos
                    
                    if mem_key in self.agent_memory_evader_lvlh:
                        target_feat.append(self._symlog(self.agent_memory_evader_lvlh[mem_key]))
                    else:
                        target_feat.append(np.zeros(3))
                else:
                    target_feat.append(np.zeros(3))

            # 3. 队友信息 (LVLH)
            teammate_data = []
            for i in range(self._config.num_p):
                tid = f'p_{i}'
                if tid != agent_id:
                    if tid in all_states:
                        t_state = all_states[tid]
                        rel_pos, rel_vel = eci_to_lvlh_relative(my_state, t_state[:3], t_state[3:])
                        t_fuel = np.array([self.remain_Dvs.get(tid, 0.0) / self._config.p_init_dv])
                        if rel_pos is not None and rel_vel is not None:
                            teammate_data.append(np.concatenate([self._symlog(rel_pos), self._symlog(rel_vel), t_fuel]))
                        else:
                            teammate_data.append(np.zeros(7))
                    else:
                        teammate_data.append(np.zeros(7))
            
            # 4. 轨道相位特征 (使用 Last Known State)
            current_orbital_feat = np.zeros(3)
            if primary_evader_id and primary_evader_id in all_states:
                is_primary_visible = (self.obs_counters[primary_evader_id] % self._config.obs_interval == 0)
                orb_mem_key = (agent_id, 'orb_feat')

                if is_primary_visible:
                    # 可见：计算真实特征并更新记忆
                    real_orb_feat = self._get_orbital_features(my_state, all_states[primary_evader_id])
                    self.agent_memory_orb_feat[orb_mem_key] = real_orb_feat
                    current_orbital_feat = real_orb_feat
                else:
                    # 不可见：使用记忆中的特征
                    current_orbital_feat = self.agent_memory_orb_feat.get(orb_mem_key, np.zeros(3))

            # 5. 拼接所有观测
            obs_list = [alt_dev, pos_dir, vel_sym, fuel] + target_feat
            if teammate_data:
                obs_list.append(np.concatenate(teammate_data))
            obs_list.append(current_orbital_feat)
            
            observations[agent_id] = np.concatenate(obs_list)

        for eid in self.evader_ids:
            if eid in all_states:
                observations[eid] = all_states[eid]
                
        return observations

    def _update_history_and_prepare_data(self):
        # 这里的逻辑与你之前的代码一致，通过 eci_to_lvlh_relative 处理历史数据
        # 确保 history buffer 更新逻辑正确
        for evader_id in self.evader_ids:
            if evader_id not in self.states: continue

            self.obs_counters[evader_id] += 1
            h_buf = self.evader_history_buffers[evader_id]
            m_buf = self.evader_history_mask_buffers[evader_id]
            
            # 更新 Buffer (存绝对 ECI)
            if self.obs_counters[evader_id] % self._config.obs_interval == 0:
                h_buf.append(self.states[evader_id])
                m_buf.append(1.0)
            else:
                if h_buf:
                    h_buf.append(h_buf[-1])
                    m_buf.append(0.0)
                else:
                    h_buf.append(self.states[evader_id])
                    m_buf.append(1.0)

            # 准备 Transformer 输入 (转为相对 LVLH)
            hist_eci = np.array(list(h_buf))
            hist_mask = np.array(list(m_buf))

            for pid in self.pursuer_ids:
                if pid in self.states:
                    my_state = self.states[pid]
                    
                    processed_hist = []
                    for h_state in hist_eci:
                        rp, rv = eci_to_lvlh_relative(my_state, h_state[:3], h_state[3:])
                        processed_hist.append(np.concatenate([rp, rv]))
                    
                    rel_hist_symlog = self._symlog(np.array(processed_hist))
                    
                    # 存入 self.infos 供 step 合并
                    if pid not in self.infos: self.infos[pid] = {}
                    self.infos[pid].update({
                        f'history_input_{evader_id}': rel_hist_symlog,
                        f'history_mask_{evader_id}': hist_mask
                    })
    
    