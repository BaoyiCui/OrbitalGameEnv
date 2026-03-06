import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

# ==========================================
# 1. 设置出版级绘图参数 (Elsevier Standard)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 7.5
plt.rcParams['axes.labelsize'] = 7.5
plt.rcParams['axes.titlesize'] = 9.0
plt.rcParams['xtick.labelsize'] = 7.5
plt.rcParams['ytick.labelsize'] = 7.5
plt.rcParams['legend.fontsize'] = 7.5
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['mathtext.fontset'] = 'stix'

# ==========================================
# 2. 路径配置
# ==========================================
# 将项目根目录添加到Python路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from env.mpe_pomdp_env import MPE_POMDP_Env, MPE_POMDP_EnvCfg
from demos.pe_env.train_pomdp import HRG_ActorCritic

def load_checkpoint(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    print(f"Loading checkpoint from {path}...")
    return torch.load(path, map_location=device)

def plot_2x2_relative_trajectories(all_traj_data, pursuer_ids, evader_id, capture_radius, output_path):
    """
    绘制 2x2 的相对轨迹组合图
    """
    print(f"Plotting figure to {output_path}...")
    
    # 定义单位转换
    cm = 1 / 2.54
    # Elsevier 全栏宽度通常为 15cm
    fig, axes = plt.subplots(2, 2, figsize=(15*cm, 14*cm), constrained_layout=True)
    
    formation_order = ['ring', 'cluster', 'string', 'pincer']
    titles = {
        'ring': 'Square',
        'cluster': 'Random',
        'string': 'String',
        'pincer': 'Pincer'
    }
    
    # 自定义颜色: P0: Navy, P1: Cyan, P2: Gold, P3: DarkRed
    custom_colors = ['#000080', '#00BFFF', '#FFD700', '#800000'] 
    
    scale = 1000.0  # m to km
    cap_rad_km = capture_radius / scale

    ax_flat = axes.flatten()

    for idx, form_key in enumerate(formation_order):
        ax = ax_flat[idx]
        traj_data = all_traj_data.get(form_key)

        ax.set_title(titles[form_key], pad=3)
        ax.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
        
        # 绘制捕获边界 (绿色虚线)
        circle = plt.Circle((0, 0), cap_rad_km, color='green', linestyle='--', fill=False, linewidth=1.0, zorder=5)
        ax.add_artist(circle)
        
        # 绘制逃逸者 (中心红色五角星)
        ax.scatter(0, 0, marker='*', color='red', s=60, zorder=10)

        if traj_data is None:
            ax.text(0, 0, "No Success Data", ha='center', va='center')
            continue

        evader_traj = np.array(traj_data[evader_id])
        
        all_x, all_y = [], []
        
        # 绘制追击者
        for p_idx, pid in enumerate(pursuer_ids):
            if pid in traj_data:
                color = custom_colors[p_idx % len(custom_colors)]
                
                p_traj = np.array(traj_data[pid])
                min_len = min(len(evader_traj), len(p_traj))
                
                # 计算相对位置
                rel_traj = (p_traj[:min_len] - evader_traj[:min_len]) / scale
                
                # 绘制轨迹线
                ax.plot(rel_traj[:, 0], rel_traj[:, 1], color=color, linewidth=1.2, alpha=0.9)
                
                # 绘制起点
                ax.scatter(rel_traj[0, 0], rel_traj[0, 1], marker='o', color=color, s=15, zorder=6)
                
                # 绘制终点
                ax.scatter(rel_traj[-1, 0], rel_traj[-1, 1], marker='x', color=color, s=40, linewidth=1.5, zorder=7)

                all_x.extend(rel_traj[:, 0])
                all_y.extend(rel_traj[:, 1])

        # 设置坐标轴范围
        if all_x and all_y:
            max_val = max(np.max(np.abs(all_x)), np.max(np.abs(all_y)))
            limit = max_val * 1.15
            limit = max(limit, cap_rad_km * 1.2)
            
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
        
        ax.set_xlabel('Rel. X (km)', labelpad=1)
        ax.set_ylabel('Rel. Y (km)', labelpad=1)
        ax.set_aspect('equal')

    # ==========================================
    # 创建统一图例
    # ==========================================
    legend_elements = []
    for i in range(len(pursuer_ids)):
        c = custom_colors[i % len(custom_colors)]
        legend_elements.append(plt.Line2D([0], [0], color=c, lw=1.5, label=f'Pursuer {i}'))
    
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=6, label='Start'))
    legend_elements.append(plt.Line2D([0], [0], marker='x', color='gray', linestyle='None', markersize=6, markeredgewidth=1.5, label='End'))
    legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=8, label='Evader (Center)'))
    legend_elements.append(plt.Line2D([0], [0], color='green', linestyle='--', lw=1.2, label='Capture Boundary'))

    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
               ncol=4, frameon=True, fancybox=True, shadow=False)
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Generate specific 2x2 Relative Trajectory Plot")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--student_model_type", type=str, default="hafn")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    # 1. 加载环境配置
    device = torch.device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    
    env_cfg = MPE_POMDP_EnvCfg()
    if 'env_cfg' in ckpt:
        for k, v in ckpt['env_cfg'].items():
            if hasattr(env_cfg, k):
                setattr(env_cfg, k, v)
    
    # -------------------------------------------------------------
    # [PATCH START] 修复 AttributeError: 'MPE_POMDP_EnvCfg' object has no attribute 'dim_mode'
    # -------------------------------------------------------------
    if not hasattr(env_cfg, 'dim_mode'):
        print("Warning: 'dim_mode' missing in checkpoint (Old Model). Setting default to 2 (2D).")
        setattr(env_cfg, 'dim_mode', 2)
    # -------------------------------------------------------------
    # [PATCH END]
    # -------------------------------------------------------------

    # 强制覆盖部分参数以匹配评估
    env_cfg.num_e = 1 
    if env_cfg.num_p != 4:
        print(f"Warning: Checkpoint has {env_cfg.num_p} pursuers, but plot style is optimized for 4.")
    
    # 2. 加载模型
    student_obs_dim = 8 + 3*env_cfg.num_e + 7*(env_cfg.num_p-1) + 3
    priv_obs_dim = 6 + (env_cfg.num_p + env_cfg.num_e - 1) * 9
    act_dim = 3
    
    agent = HRG_ActorCritic(env_cfg, act_dim, priv_obs_dim, student_obs_dim, student_model_type=args.student_model_type).to(device)
    agent.load_state_dict(ckpt['agent_state_dict'])
    agent.eval()

    # 3. 收集数据
    # 注意：此时传入的 env_cfg 已经具备 dim_mode 属性，不会报错
    env = MPE_POMDP_Env(env_cfg)
    formations = ['ring', 'cluster', 'string', 'pincer']
    
    all_traj_data = {}
    
    pursuer_ids = [f'p_{i}' for i in range(env_cfg.num_p)]
    evader_id = 'e_0'

    print("Collecting trajectories for each formation...")
    
    for form in formations:
        success = False
        attempts = 0
        max_attempts = 50 
        
        current_seed = args.seed
        
        while not success and attempts < max_attempts:
            obs, infos = env.reset(seed=current_seed, options={'force_formation': form})
            
            traj_data = defaultdict(list)
            
            # 使用 getattr 安全获取 states，防止接口变化
            if hasattr(env, 'states'):
                # 兼容旧代码
                for aid in env.states:
                    traj_data[aid].append(env.states[aid][:3].copy())
            else:
                # 尝试从 world.agents 获取 (MPE 标准)
                for agent_obj in env.world.agents:
                    aid = agent_obj.name
                    traj_data[aid].append(agent_obj.state.p_pos.copy())

            done = False
            
            while not done:
                p_obs = torch.stack([torch.Tensor(obs[pid]) for pid in pursuer_ids]).to(device)
                p_priv = torch.stack([torch.Tensor(infos[pid]['privileged_state']) for pid in pursuer_ids]).to(device)
                p_hist = torch.stack([torch.from_numpy(infos[pid][f'history_input_{evader_id}']).float() for pid in pursuer_ids]).to(device)
                p_mask = torch.stack([torch.from_numpy(infos[pid][f'history_mask_{evader_id}']).float() for pid in pursuer_ids]).to(device)

                with torch.no_grad():
                    actions, _, _, _, _, _ = agent.get_action_and_value(p_obs, p_priv, p_hist, p_mask, deterministic=True)
                
                action_dict = {pid: actions[i].cpu().numpy() for i, pid in enumerate(pursuer_ids)}
                action_dict.update(env.get_evader_actions())
                
                next_obs, _, term, trunc, infos = env.step(action_dict)
                
                # 记录步进后的位置
                if hasattr(env, 'states'):
                    for aid in env.states:
                        traj_data[aid].append(env.states[aid][:3].copy())
                else:
                    for agent_obj in env.world.agents:
                        aid = agent_obj.name
                        traj_data[aid].append(agent_obj.state.p_pos.copy())
                
                obs = next_obs
                if any(term.values()) or any(trunc.values()):
                    done = True
                    reason = infos[pursuer_ids[0]].get('termination_reason')
                    if reason == 'capture_success':
                        success = True
                        all_traj_data[form] = traj_data
                        print(f"  - {form}: Success (Seed {current_seed})")
            
            current_seed += 1
            attempts += 1
            
        if not success:
            print(f"  - {form}: Failed to find success trajectory after {max_attempts} attempts.")

    env.close()

    # 4. 绘图
    ckpt_path = Path(args.checkpoint)
    output_dir = ckpt_path.parent.parent / "paper_figures"
    output_dir.mkdir(exist_ok=True, parents=True) # parents=True 防止父目录不存在报错
    
    output_path = output_dir / "Figure_Relative_Trajectories_2x2.png"
    output_path_tiff = output_dir / "Figure_Relative_Trajectories_2x2.tiff"
    
    # 绘制 PNG
    plot_2x2_relative_trajectories(all_traj_data, pursuer_ids, evader_id, env_cfg.dist_cap, output_path)
    
    # 绘制 TIFF (独立调用，防止 figure 状态干扰)
    print("Saving TIFF version...")
    plot_2x2_relative_trajectories(all_traj_data, pursuer_ids, evader_id, env_cfg.dist_cap, output_path_tiff)
    
    print("Done.")

if __name__ == "__main__":
    main()