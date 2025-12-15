import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import io
import imageio
from pathlib import Path

# 将项目根目录添加到Python路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from env.mpe_pomdp_env import MPE_POMDP_Env, MPE_POMDP_EnvCfg
from demos.pe_env.train_pomdp import HRG_ActorCritic, TrainConfig

def load_checkpoint(path, device):
    """加载模型检查点"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    print(f"Loading checkpoint from {path}...")
    return torch.load(path, map_location=device)

def set_axes_equal(ax):
    """让3D Matplotlib的坐标轴比例尺相等"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def save_static_plot(traj_data, pursuer_ids, evader_id, filename):
    """生成并保存在绝对坐标系下的静态轨迹图"""
    print(f"Generating static trajectory plot...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 地球
    R_earth = 6378.137 # km
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    x = R_earth * np.cos(u) * np.sin(v)
    y = R_earth * np.sin(u) * np.sin(v)
    z = R_earth * np.cos(v)
    ax.plot_surface(x, y, z, color='blue', alpha=0.1, edgecolor='none')

    scale = 1000.0 # m to km
    
    # 追击者轨迹
    colors = plt.cm.jet(np.linspace(0, 1, len(pursuer_ids)))
    for i, pid in enumerate(pursuer_ids):
        if pid in traj_data:
            pos = np.array(traj_data[pid]) / scale
            ax.plot(pos[:,0], pos[:,1], pos[:,2], label=f'Pursuer {i}', color=colors[i], linewidth=1.5, alpha=0.8)
            ax.scatter(pos[0,0], pos[0,1], pos[0,2], marker='o', color=colors[i], s=30, alpha=0.8, label=f'P{i} Start')
            ax.scatter(pos[-1,0], pos[-1,1], pos[-1,2], marker='x', color=colors[i], s=60, linewidth=2, label=f'P{i} End')

    # 逃逸者轨迹
    if evader_id in traj_data:
        pos = np.array(traj_data[evader_id]) / scale
        ax.plot(pos[:,0], pos[:,1], pos[:,2], label='Evader', color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.scatter(pos[0,0], pos[0,1], pos[0,2], marker='o', color='red', s=30, alpha=0.8, label='Evader Start')
        ax.scatter(pos[-1,0], pos[-1,1], pos[-1,2], marker='*', color='red', s=100, label='Evader End')

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Absolute Trajectory')
    ax.legend()
    set_axes_equal(ax)

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Static plot saved to {filename}")

def save_top_down_plot(traj_data, pursuer_ids, evader_id, filename):
    """生成并保存在Z轴俯视视角下的2D静态轨迹图"""
    print(f"Generating top-down (Z-axis view) static trajectory plot...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    # 地球 (2D视图)
    R_earth = 6378.137 # km
    earth_circle = plt.Circle((0, 0), R_earth, color='blue', alpha=0.1)
    ax.add_artist(earth_circle)

    scale = 1000.0 # m to km
    
    # 追击者轨迹
    colors = plt.cm.jet(np.linspace(0, 1, len(pursuer_ids)))
    for i, pid in enumerate(pursuer_ids):
        if pid in traj_data:
            pos = np.array(traj_data[pid]) / scale
            ax.plot(pos[:,0], pos[:,1], label=f'Pursuer {i}', color=colors[i], linewidth=1.5, alpha=0.8)
            ax.scatter(pos[0,0], pos[0,1], marker='o', color=colors[i], s=30, alpha=0.8, label=f'P{i} Start')
            ax.scatter(pos[-1,0], pos[-1,1], marker='x', color=colors[i], s=60, linewidth=2, label=f'P{i} End')

    # 逃逸者轨迹
    if evader_id in traj_data:
        pos = np.array(traj_data[evader_id]) / scale
        ax.plot(pos[:,0], pos[:,1], label='Evader', color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.scatter(pos[0,0], pos[0,1], marker='o', color='red', s=30, alpha=0.8, label='Evader Start')
        ax.scatter(pos[-1,0], pos[-1,1], marker='*', color='red', s=100, label='Evader End')

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title('Top-Down Trajectory (Z-axis view)')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Top-down plot saved to {filename}")

def save_relative_plot(traj_data, pursuer_ids, evader_id, filename):
    """生成并保存以逃逸者为中心的2D相对轨迹图"""
    print(f"Generating relative trajectory plot...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    scale = 1000.0 # m to km
    
    if evader_id not in traj_data:
        print("Evader trajectory not found, skipping relative plot.")
        plt.close(fig)
        return

    evader_traj = np.array(traj_data[evader_id])
    
    # 逃逸者在中心
    ax.scatter(0, 0, marker='*', color='red', s=150, label='Evader (Reference)', zorder=10)

    colors = plt.cm.jet(np.linspace(0, 1, len(pursuer_ids)))
    for i, pid in enumerate(pursuer_ids):
        if pid in traj_data:
            pursuer_traj = np.array(traj_data[pid])
            
            # 确保轨迹长度一致
            min_len = min(len(evader_traj), len(pursuer_traj))
            relative_traj = (pursuer_traj[:min_len] - evader_traj[:min_len]) / scale
            
            ax.plot(relative_traj[:,0], relative_traj[:,1], label=f'Pursuer {i}', color=colors[i], linewidth=1.5, alpha=0.8)
            ax.scatter(relative_traj[0,0], relative_traj[0,1], marker='o', color=colors[i], s=40, label=f'P{i} Start')
            ax.scatter(relative_traj[-1,0], relative_traj[-1,1], marker='x', color=colors[i], s=80, linewidth=2, label=f'P{i} End')

    ax.set_xlabel('Relative X (km)')
    ax.set_ylabel('Relative Y (km)')
    ax.set_title('Relative Trajectory to Evader (Top-Down View)')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Relative plot saved to {filename}")

def create_gif(traj_data, pursuer_ids, evader_id, filename, frame_skip=5):
    """创建并保存以逃逸者为中心的相对位置GIF"""
    print(f"Generating relative trajectory GIF...")
    
    if evader_id not in traj_data:
        print("Evader trajectory not found, skipping GIF generation.")
        return

    evader_traj = np.array(traj_data[evader_id])
    
    # 找到所有轨迹中的最大长度
    max_len = 0
    if evader_id in traj_data:
        max_len = len(traj_data[evader_id])
    for pid in pursuer_ids:
        if pid in traj_data:
            max_len = max(max_len, len(traj_data[pid]))

    images = []
    colors = plt.cm.jet(np.linspace(0, 1, len(pursuer_ids)))
    scale = 1000.0 # m to km

    for t in tqdm(range(0, max_len, frame_skip), desc="Creating GIF frames"):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        
        # 逃逸者在中心
        ax.scatter(0, 0, marker='*', color='red', s=150, label='Evader', zorder=10)

        all_x = [0]
        all_y = [0]

        for i, pid in enumerate(pursuer_ids):
            if pid in traj_data and t < len(traj_data[pid]) and t < len(evader_traj):
                pursuer_pos = np.array(traj_data[pid][t])
                evader_pos = np.array(evader_traj[t])
                relative_pos = (pursuer_pos - evader_pos) / scale
                
                ax.scatter(relative_pos[0], relative_pos[1], marker='o', color=colors[i], s=100, label=f'Pursuer {i}')
                all_x.append(relative_pos[0])
                all_y.append(relative_pos[1])

        # 动态调整坐标轴范围
        max_range = max(np.abs(all_x).max(), np.abs(all_y).max()) * 1.2 + 5 # 加上一点余量
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        
        ax.set_xlabel('Relative X (km)')
        ax.set_ylabel('Relative Y (km)')
        ax.set_title(f'Relative Positions at Step {t}')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')

        # 将图像保存到内存
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        images.append(imageio.imread(buf))
        plt.close(fig)

    imageio.mimsave(filename, images, duration=0.1, loop=0)
    print(f"GIF saved to {filename}")

def run_eval(args):
    """主评估函数"""
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    checkpoint = load_checkpoint(args.checkpoint, device)
    
    # --- 1. 创建结果文件夹 ---
    run_name = Path(args.checkpoint).parent.parent.name
    checkpoint_name = Path(args.checkpoint).stem
    output_dir_name = f"{run_name}_{checkpoint_name}_eval"
    results_base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    output_dir = os.path.join(results_base_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")

    # --- 2. 配置和加载模型 ---
    env_cfg = MPE_POMDP_EnvCfg()
    if 'env_cfg' in checkpoint:
        env_cfg_ckpt = checkpoint['env_cfg']
        for key, value in env_cfg_ckpt.items():
            if hasattr(env_cfg, key):
                setattr(env_cfg, key, value)
    
    env_cfg.num_p = args.num_p if args.num_p is not None else env_cfg.num_p
    env_cfg.num_e = args.num_e
    env_cfg.history_len = args.history_len
    env_cfg.evader_policy_level = args.evader_policy_level
    env_cfg.dim_mode = args.dim_mode

    print(f"Evaluating with: Pursuers={env_cfg.num_p}, Evaders={env_cfg.num_e}, Evader Policy={env_cfg.evader_policy_level}, Dim Mode={env_cfg.dim_mode}")

    env = MPE_POMDP_Env(env_cfg)
    
    pursuer_ids = [f'p_{i}' for i in range(env_cfg.num_p)]
    evader_ids = [f'e_{i}' for i in range(env_cfg.num_e)]
    
    student_obs_dim = env.observation_spaces[pursuer_ids[0]].shape[0]
    num_others = env_cfg.num_p + (env_cfg.num_e - 1)
    privileged_obs_dim = 6 + num_others * 9 
    act_dim = env.action_spaces[pursuer_ids[0]].shape[0]

    model = HRG_ActorCritic(env_cfg, act_dim, privileged_obs_dim, student_obs_dim).to(device)
    model.load_state_dict(checkpoint['agent_state_dict'])
    model.eval()
    print("Model loaded successfully.")

    # --- 3. 开始评估循环 ---
    formations_to_test = ['ring', 'cluster', 'string', 'pincer']
    success_counts = defaultdict(int)
    total_counts = defaultdict(int)
    media_saved_trajs = {}

    print(f"Running evaluation for {args.test_episodes} episodes, testing all 4 formations...")
    for ep in tqdm(range(args.test_episodes), desc="Evaluating Episodes"):
        
        formation_this_ep = formations_to_test[ep % len(formations_to_test)]
        total_counts[formation_this_ep] += 1

        # 使用 'options' 参数强制指定阵型
        obs, infos = env.reset(seed=args.seed + ep, options={'force_formation': formation_this_ep})
        
        traj_data = defaultdict(list)
        done = False
        
        while not done:
            pursuer_obs_list = [torch.Tensor(obs[name]).to(device) for name in pursuer_ids]
            pursuer_obs_tensor = torch.stack(pursuer_obs_list)
            pursuer_priv_obs_list = [torch.Tensor(infos[name]['privileged_state']).to(device) for name in pursuer_ids]
            pursuer_priv_obs_tensor = torch.stack(pursuer_priv_obs_list)
            history_list = [torch.from_numpy(infos[name][f'history_input_{evader_ids[0]}']).float().to(device) for name in pursuer_ids]
            history_tensor = torch.stack(history_list)
            mask_list = [torch.from_numpy(infos[name][f'history_mask_{evader_ids[0]}']).float().to(device) for name in pursuer_ids]
            mask_tensor = torch.stack(mask_list)

            with torch.no_grad():
                actions_tensor, _, _, _, _, _ = model.get_action_and_value(
                    pursuer_obs_tensor, pursuer_priv_obs_tensor, history_tensor, mask_tensor, deterministic=True
                )
            
            evader_actions = env.get_evader_actions()
            actions_to_step = {name: actions_tensor[i].cpu().numpy() for i, name in enumerate(pursuer_ids)}
            actions_to_step.update(evader_actions)

            next_obs, _, terminations, truncations, infos = env.step(actions_to_step)
            
            for agent_id in env.states:
                traj_data[agent_id].append(env.states[agent_id][:3].copy())

            obs = next_obs
            if any(terminations.values()) or any(truncations.values()):
                done = True
                reason = infos.get(pursuer_ids[0], {}).get('termination_reason', 'unknown')
                if reason == "capture_success":
                    success_counts[formation_this_ep] += 1
                    # 如果该阵型还没有保存过媒体，则保存
                    if formation_this_ep not in media_saved_trajs and args.save_media:
                        media_saved_trajs[formation_this_ep] = traj_data
                        print(f"\nFirst success for '{formation_this_ep}' formation in Ep {ep+1}. Trajectory saved for media generation.")

    env.close()

    # --- 4. 最终结果 ---
    total_successes = sum(success_counts.values())
    total_episodes = args.test_episodes
    overall_success_rate = (total_successes / total_episodes) * 100 if total_episodes > 0 else 0

    print("\n" + "="*50)
    print("             EVALUATION REPORT             ")
    print("="*50)
    print(f"Overall Success Rate: {overall_success_rate:.2f}% ({total_successes}/{total_episodes})")
    print("-" * 50)
    print("Success Rate per Formation:")
    for form_name in formations_to_test:
        s_count = success_counts[form_name]
        t_count = total_counts[form_name]
        rate = (s_count / t_count) * 100 if t_count > 0 else 0
        print(f"  - {form_name.capitalize():<10}: {rate:>6.2f}% ({s_count}/{t_count})")
    print("="*50 + "\n")

    # --- 5. 生成媒体文件 ---
    if media_saved_trajs and args.save_media:
        print("\n--- Generating Media for First Successful Run of Each Formation ---")
        for form_name, traj_data in media_saved_trajs.items():
            print(f"\n--- Generating for '{form_name}' formation ---")
            static_plot_path = os.path.join(output_dir, f"trajectory_static_3D_{form_name}.png")
            top_down_plot_path = os.path.join(output_dir, f"trajectory_top_down_2D_{form_name}.png")
            relative_plot_path = os.path.join(output_dir, f"trajectory_relative_2D_{form_name}.png")
            gif_path = os.path.join(output_dir, f"trajectory_relative_{form_name}.gif")
            
            save_static_plot(traj_data, pursuer_ids, evader_ids[0], filename=static_plot_path)
            save_top_down_plot(traj_data, pursuer_ids, evader_ids[0], filename=top_down_plot_path)
            save_relative_plot(traj_data, pursuer_ids, evader_ids[0], filename=relative_plot_path)
            create_gif(traj_data, pursuer_ids, evader_ids[0], filename=gif_path)
    elif args.save_media:
        print("No successful episodes were recorded for some formations, so no media will be generated for them.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MPE POMDP Agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint file")
    parser.add_argument("--num_p", type=int, default=None, help="Number of pursuers (loads from checkpoint if not set)")
    parser.add_argument("--num_e", type=int, default=1, help="Number of evaders")
    parser.add_argument("--use_encoder", type=lambda x: (str(x).lower() == 'true'), default=None, help="Use Attention Encoder (loads from checkpoint if not set)")
    parser.add_argument("--history_len", type=int, default=20, help="History length for Transformer")
    parser.add_argument("--dim_mode", type=int, default=2, choices=[2, 3], help="Environment dimension mode: 2=2D (default), 3=3D")
    parser.add_argument("--evader_policy_level", type=int, default=0, choices=[0, 1, 2], help="Evader policy: 0=Drift, 1=Random, 2=APF")
    
    parser.add_argument("--test_episodes", type=int, default=40, help="Number of episodes to test for success rate")
    parser.add_argument("--save_media", action="store_true", help="Save GIF and static plot for the first successful run of each formation")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for inference")
    
    args = parser.parse_args()
    
    run_eval(args)