# 无绘图测试成功率:
# python ./demos/pe_env/eval4train_pomdp.py --maneuver_strategy drift --save_animation --checkpoint ./runs/Dec\ 4th:new\ lstm\ 2/checkpoints/update_2400.pt --test_success_rate True --test_episodes 10

# 单次运行并绘制图:
# python ./demos/pe_env/eval4train_pomdp.py --maneuver_strategy drift --checkpoint ./runs/Dec\ 4th:new\ lstm\ 2/checkpoints/update_2400.pt

# 单次运行并保存为双视图动图:
# python ./demos/pe_env/eval4train_pomdp.py --maneuver_strategy drift --save_animation --checkpoint ./runs/Dec\ 4th:new\ lstm\ 2/checkpoints/update_2400.pt

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from env.mpe_pomdp_env import MPE_POMDP_Env, MPE_POMDP_EnvCfg
from demos.pe_env.train_pomdp import ActorCritic, TrainConfig

def load_checkpoint(path, device):
    print(f"从{path}导入检查点...")
    return torch.load(path, map_location=device)

def load_params_from_txt(checkpoint_path):
    run_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    params_path = os.path.join(run_dir, "all_params.txt")
    params = {}
    if not os.path.exists(params_path):
        print("无法加载训练参数，使用默认配置。\n")
        return params
    print(f"加载参数中...")
    with open(params_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ": " in line:
                key, value_str = line.split(": ", 1)
                if value_str.lower() == 'true': params[key] = True
                elif value_str.lower() == 'false': params[key] = False
                else:
                    try: params[key] = int(value_str)
                    except ValueError:
                        try: params[key] = float(value_str)
                        except ValueError: params[key] = value_str
    return params

def run_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")
    file_params = load_params_from_txt(args.checkpoint)
    env_cfg = MPE_POMDP_EnvCfg()
    # 从txt里取参数
    if 'num_p' in file_params: env_cfg.num_p = file_params['num_p']
    if 'num_e' in file_params: env_cfg.num_e = file_params['num_e']
    if 'use_partial_obs' in file_params: env_cfg.use_partial_obs = file_params['use_partial_obs']
    if 'obs_interval' in file_params: env_cfg.obs_interval = file_params['obs_interval']
    if 'lstm_history_len' in file_params: env_cfg.lstm_history_len = file_params['lstm_history_len']
    if 'lstm_future_len' in file_params: env_cfg.lstm_future_len = file_params['lstm_future_len']
    if 'lstm_scheme' in file_params: env_cfg.lstm_scheme = file_params['lstm_scheme']
    
    env_cfg.episode_length = 3600 * 10#时间s
    strategy_map = {'drift': 0, 'random': 1, 'apf': 2}
    env_cfg.evader_policy_level = strategy_map.get(args.maneuver_strategy, 0)
    
    print("-" * 30)
    print(f"Eval Config:")
    print(f"  Agents        : {env_cfg.num_p} Pursuers vs {env_cfg.num_e} Evader")
    print(f"  Evader Policy : {args.maneuver_strategy} ")
    print(f"  LSTM Scheme   : {env_cfg.lstm_scheme}")
    print("-" * 30)

    checkpoint = load_checkpoint(args.checkpoint, device)
    if 'current_dist_cap' in checkpoint:
        env_cfg.dist_cap = checkpoint['current_dist_cap']
        print(f"Loaded dist_cap from checkpoint: {env_cfg.dist_cap:.2f} m")
    if 'current_p_init_dv' in checkpoint:
        env_cfg.p_init_dv = checkpoint['current_p_init_dv']
        print(f"Loaded p_init_dv from checkpoint: {env_cfg.p_init_dv:.2f} m/s")
    
    env = MPE_POMDP_Env(env_cfg)
    env.reset(seed=args.seed)
    #网络准备
    pursuer_ids = [f'p_{i}' for i in range(env_cfg.num_p)]
    actor_obs_dim = env.observation_spaces[pursuer_ids[0]].shape[0]
    critic_obs_dim = sum(env.observation_spaces[p_id].shape[0] for p_id in pursuer_ids)
    act_dim = env.action_spaces[pursuer_ids[0]].shape[0]

    train_cfg = TrainConfig()
    if 'use_encoder' in file_params:
        train_cfg.use_encoder = file_params['use_encoder']
    
    print(f"Model Config - Encoder: {train_cfg.use_encoder}")
    model = ActorCritic(actor_obs_dim, critic_obs_dim, act_dim, env, env_cfg, train_cfg).to(device)
    model.load_state_dict(checkpoint['agent_state_dict'])
    env.set_policy_lstm(model.lstm)
    model.eval()
    print("Model loaded")

    def run_episode(env_instance, model_instance):
        obs, infos = env_instance.reset()
        traj_data = defaultdict(list)
        done = False
        step_cnt = 0
        total_reward = 0.0
        reason = "unknown"
        while not done:
            pursuer_obs_list = [torch.Tensor(obs[name]).to(device) for name in pursuer_ids]
            pursuer_obs_tensor = torch.stack(pursuer_obs_list)
            central_obs_tensor = torch.cat(pursuer_obs_list, dim=-1).unsqueeze(0)
            with torch.no_grad():
                actions_tensor, _, _, _ = model_instance.get_action_and_value(
                    pursuer_obs_tensor, central_obs_tensor, deterministic=True
                )
            evader_actions = env_instance.get_evader_actions() 
            actions_to_step = {name: actions_tensor[i].cpu().numpy() for i, name in enumerate(pursuer_ids)}
            actions_to_step.update(evader_actions)
            next_obs, rewards, terminations, truncations, infos = env_instance.step(actions_to_step)
            for agent_id in env_instance.states:
                traj_data[agent_id].append(env_instance.states[agent_id][:3].copy())
            total_reward += sum(rewards.get(pid, 0) for pid in pursuer_ids)
            obs = next_obs
            step_cnt += 1
            if any(terminations.values()) or any(truncations.values()):
                done = True
                if pursuer_ids[0] in infos:
                    reason = infos[pursuer_ids[0]].get('termination_reason', 'unknown')
        return traj_data, step_cnt, total_reward, reason, reason == "capture_success"

    print("\n开始仿真...")
    if args.test_success_rate:# 测试成功率模式
        success_count = 0
        print(f"Testing success rate over {args.test_episodes} episodes...")
        for ep in range(args.test_episodes):
            env_test = MPE_POMDP_Env(env_cfg)
            env_test.reset(seed=args.seed + ep) 
            env_test.set_policy_lstm(model.lstm)
            traj, steps, reward, reason, success = run_episode(env_test, model)
            if success: success_count += 1
            if (ep + 1) % 10 == 0:
                print(f"  Episode {ep+1}/{args.test_episodes} | Current SR: {success_count/(ep+1)*100:.1f}%")
            env_test.close()
            if ep == args.test_episodes - 1:
                analyze_and_plot(traj, env_cfg, steps, reward, reason, args.output_plot, True, args.test_episodes)
        final_sr = success_count / args.test_episodes * 100
        print(f"Final Success Rate over {args.test_episodes} episodes: {final_sr:.2f}%")
    else:# 单次运行模式
        traj, steps, reward, reason, success = run_episode(env, model)
        env.close()
        print(f"Episode finished at step {steps}. Reason: {reason}")
        if args.save_animation:#如果要动图
            anim_file = args.output_plot
            if anim_file.endswith('.png'): anim_file = anim_file.replace('.png', '.gif')
            animate_trajectory(traj, env_cfg, steps, reward, reason, anim_file)
            analyze_and_plot(traj, env_cfg, steps, reward, reason, args.output_plot, False, 1)
        else:
            analyze_and_plot(traj, env_cfg, steps, reward, reason, args.output_plot, False, 1)

def animate_trajectory(traj_data, cfg, steps, reward, reason, output_file):
    # 双视图动图生成
    print(f"准备生成双视图动图，共 {steps} 帧...")
    for k in traj_data: traj_data[k] = np.array(traj_data[k])

    # 创建画布
    fig = plt.figure(figsize=(20, 10))
    scale = 1000.0
    colors = plt.cm.jet(np.linspace(0, 1, cfg.num_p))
    # 全局视图
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Global View')
    ax1.set_xlabel('X (km)'); ax1.set_ylabel('Y (km)'); ax1.set_zlabel('Z (km)')
    
    # 背景地球和轨道
    ax1.scatter(0, 0, 0, color='blue', s=200, label='Earth Center', alpha=0.2)
    geo_radius = 42164.0 
    theta = np.linspace(0, 2 * np.pi, 400)
    x_geo = geo_radius * np.cos(theta); y_geo = geo_radius * np.sin(theta); z_geo = np.zeros_like(theta)
    ax1.plot(x_geo, y_geo, z_geo, color='green', linestyle='--', alpha=0.3, linewidth=0.5, label='GEO Orbit')

    # 范围锁定
    all_gx, all_gy, all_gz = [], [], []
    for pid in [f'p_{i}' for i in range(cfg.num_p)]:
        if pid in traj_data:
            pos = traj_data[pid] / scale
            all_gx.extend(pos[:, 0]); all_gy.extend(pos[:, 1]); all_gz.extend(pos[:, 2])
    if 'e_0' in traj_data:
        pos = traj_data['e_0'] / scale
        all_gx.extend(pos[:, 0]); all_gy.extend(pos[:, 1]); all_gz.extend(pos[:, 2])

    if all_gx:
        mid_x = (np.max(all_gx) + np.min(all_gx)) / 2; mid_y = (np.max(all_gy) + np.min(all_gy)) / 2
        range_x = np.max(all_gx) - np.min(all_gx); range_y = np.max(all_gy) - np.min(all_gy)
        max_range_xy = max(range_x, range_y, 10.0) * 1.2
        ax1.set_xlim(mid_x - max_range_xy/2, mid_x + max_range_xy/2)
        ax1.set_ylim(mid_y - max_range_xy/2, mid_y + max_range_xy/2)
        
        z_min, z_max = np.min(all_gz), np.max(all_gz)
        z_range = z_max - z_min
        z_mid = (z_max + z_min) / 2
        z_pad = max(z_range * 0.1, 2.0)
        ax1.set_zlim(z_mid - z_range/2 - z_pad, z_mid + z_range/2 + z_pad)

    #相对视图 
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Relative View (Evader Centered)')
    ax2.set_xlabel('Rel X (km)'); ax2.set_ylabel('Rel Y (km)'); ax2.set_zlabel('Rel Z (km)')

    # 静态evader和捕获范围
    ax2.scatter(0, 0, 0, marker='*', color='red', s=150, label='Evader (Fixed)', zorder=10)
    cap_dist_km = cfg.dist_cap / 1000.0
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_cap = cap_dist_km * np.cos(u) * np.sin(v)
    y_cap = cap_dist_km * np.sin(u) * np.sin(v)
    z_cap = cap_dist_km * np.cos(v)
    ax2.plot_wireframe(x_cap, y_cap, z_cap, color='gray', alpha=0.2, linewidth=0.5)

    # 预计算相对轨迹并锁定范围
    rel_trajs = {}
    if 'e_0' in traj_data:
        e_traj = traj_data['e_0'] / scale
        all_rx, all_ry, all_rz = [], [], []
        # 加入捕获球范围
        all_rx.extend([-cap_dist_km, cap_dist_km])
        all_ry.extend([-cap_dist_km, cap_dist_km])
        all_rz.extend([-cap_dist_km, cap_dist_km])
        
        min_len = len(e_traj)
        for i in range(cfg.num_p):
            pid = f'p_{i}'
            if pid in traj_data:
                p_len = len(traj_data[pid])
                l = min(p_len, min_len)
                rel_pos = (traj_data[pid][:l] / scale) - e_traj[:l]
                rel_trajs[pid] = rel_pos
                all_rx.extend(rel_pos[:, 0]); all_ry.extend(rel_pos[:, 1]); all_rz.extend(rel_pos[:, 2])
        
        max_range_r = np.array([np.ptp(all_rx), np.ptp(all_ry), np.ptp(all_rz)]).max() / 2.0 * 1.1
        ax2.set_xlim(-max_range_r, max_range_r)
        ax2.set_ylim(-max_range_r, max_range_r)
        ax2.set_zlim(-max_range_r, max_range_r)
    else:
        min_len = steps # Fallback

    # 初始化动态元素
    # Global的Artists
    g_heads = []; g_trails = []
    # Relative的Artists
    r_heads = []; r_trails = []
    #P
    for i in range(cfg.num_p):
        # Global
        gh = ax1.scatter([], [], [], marker='o', color=colors[i], s=80, edgecolor='k')
        gt, = ax1.plot([], [], [], color=colors[i], linewidth=2, alpha=0.8)
        g_heads.append(gh); g_trails.append(gt)
        # Relative
        rh = ax2.scatter([], [], [], marker='o', color=colors[i], s=80, edgecolor='k')
        rt, = ax2.plot([], [], [], color=colors[i], linewidth=2, alpha=0.8)
        r_heads.append(rh); r_trails.append(rt)

    # E
    g_e_head = ax1.scatter([], [], [], marker='*', color='red', s=150, edgecolor='k')
    g_e_trail, = ax1.plot([], [], [], color='red', linestyle='--', linewidth=2, alpha=0.9)

    title_text = plt.suptitle('Trajectory Animation (Step 0)', fontsize=16)

    # ---------------------------
    # 更新函数
    # ---------------------------
    def update(frame):
        current_time_min = frame * cfg.dt / 60.0
        title_text.set_text(f'Time: {current_time_min:.1f} min (Step {frame})')
        artists = [title_text, g_e_head, g_e_trail] + g_heads + g_trails + r_heads + r_trails

        # 更新位置
        if 'e_0' in traj_data:
            pos_e = traj_data['e_0'] / scale
            if frame < len(pos_e):
                g_e_head._offsets3d = (pos_e[frame:frame+1, 0], pos_e[frame:frame+1, 1], pos_e[frame:frame+1, 2])
                g_e_trail.set_data(pos_e[:frame+1, 0], pos_e[:frame+1, 1])
                g_e_trail.set_3d_properties(pos_e[:frame+1, 2])

        for i in range(cfg.num_p):
            pid = f'p_{i}'
            if pid in traj_data and frame < len(traj_data[pid]):
                pos = traj_data[pid] / scale
                # Global update
                g_heads[i]._offsets3d = (pos[frame:frame+1, 0], pos[frame:frame+1, 1], pos[frame:frame+1, 2])
                g_trails[i].set_data(pos[:frame+1, 0], pos[:frame+1, 1])
                g_trails[i].set_3d_properties(pos[:frame+1, 2])
                
                # Relative update
                if pid in rel_trajs and frame < len(rel_trajs[pid]):
                    r_pos = rel_trajs[pid]
                    r_heads[i]._offsets3d = (r_pos[frame:frame+1, 0], r_pos[frame:frame+1, 1], r_pos[frame:frame+1, 2])
                    r_trails[i].set_data(r_pos[:frame+1, 0], r_pos[:frame+1, 1])
                    r_trails[i].set_3d_properties(r_pos[:frame+1, 2])

        return artists

    ani = animation.FuncAnimation(fig, update, frames=min_len, interval=30, blit=False)
    
    print(f"正在保存动图到 {output_file}...")
    ani.save(output_file, writer='pillow', fps=30)
    print("动图保存成功！")
    plt.close(fig)

def analyze_and_plot(traj_data, cfg, steps, reward, reason, output_file, test_success_rate, test_episodes):
    for k in traj_data: traj_data[k] = np.array(traj_data[k])
    result_str = f"{reason}"
    print("\n" + "=" * 40)
    print(f"       EVALUATION REPORT       ")
    print("=" * 40)
    print(f"  Strategy      : {['Drift', 'Random', 'APF'][cfg.evader_policy_level]}")
    print(f"  Time          : {steps * cfg.dt / 3600:.2f} hours ({steps} steps)")
    print(f"  Total Reward  : {reward:.4f}")
    print(f"  Result        : {result_str}")
    print("=" * 40 + "\n")

    if not test_success_rate:
        fig = plt.figure(figsize=(24, 16))
        scale = 1000.0 
        colors = plt.cm.jet(np.linspace(0, 1, cfg.num_p))
        
        # 坐标轴缩放
        all_x, all_y, all_z = [], [], []
        for i in range(cfg.num_p):
            pid = f'p_{i}'
            if pid in traj_data:
                pos = traj_data[pid] / scale
                all_x.extend(pos[:, 0]); all_y.extend(pos[:, 1]); all_z.extend(pos[:, 2])
        if 'e_0' in traj_data:
            pos = traj_data['e_0'] / scale
            all_x.extend(pos[:, 0]); all_y.extend(pos[:, 1]); all_z.extend(pos[:, 2])

        focus_xlim, focus_ylim, focus_zlim = None, None, None
        if all_x and all_y:
            mid_x = (np.max(all_x) + np.min(all_x)) / 2; mid_y = (np.max(all_y) + np.min(all_y)) / 2
            range_x = np.max(all_x) - np.min(all_x); range_y = np.max(all_y) - np.min(all_y)
            max_range = max(range_x, range_y, 10.0) * 1.2
            focus_xlim = (mid_x - max_range/2, mid_x + max_range/2)
            focus_ylim = (mid_y - max_range/2, mid_y + max_range/2)
            z_min, z_max = np.min(all_z), np.max(all_z); z_range = z_max - z_min
            z_mid = (z_max + z_min) / 2; z_pad = max(z_range * 0.1, 2.0)
            focus_zlim = (z_mid - z_range/2 - z_pad, z_mid + z_range/2 + z_pad)

        # Global (Left) & Relative (Right)
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.scatter(0, 0, 0, color='blue', s=200, label='Earth Center', alpha=0.2)
        geo_radius = 42164.0 
        theta = np.linspace(0, 2 * np.pi, 400)
        x_geo = geo_radius * np.cos(theta); y_geo = geo_radius * np.sin(theta); z_geo = np.zeros_like(theta)
        ax1.plot(x_geo, y_geo, z_geo, color='green', linestyle='--', alpha=0.3, linewidth=0.5, label='GEO Orbit')#轨道圆
        #画轨迹
        for i in range(cfg.num_p):
            pid = f'p_{i}'
            if pid in traj_data:
                pos = traj_data[pid] / scale
                ax1.plot(pos[:,0], pos[:,1], pos[:,2], color=colors[i], linewidth=1.5, alpha=0.8)
                ax1.scatter(pos[-1,0], pos[-1,1], pos[-1,2], marker='x', color=colors[i], s=50)
        if 'e_0' in traj_data:
            pos = traj_data['e_0'] / scale
            ax1.plot(pos[:,0], pos[:,1], pos[:,2], color='red', linestyle='--', linewidth=1.5, alpha=0.9)
            ax1.scatter(pos[-1,0], pos[-1,1], pos[-1,2], marker='*', color='red', s=100)
        ax1.set_xlabel('X (km)'); ax1.set_ylabel('Y (km)'); ax1.set_zlabel('Z (km)')
        ax1.set_title('1. Global View (Focused)')
        if focus_xlim: ax1.set_xlim(focus_xlim); ax1.set_ylim(focus_ylim)
        if focus_zlim: ax1.set_zlim(focus_zlim)

        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        if 'e_0' in traj_data:
            e_traj = traj_data['e_0'] 
            ax2.scatter(0, 0, 0, marker='*', color='red', s=100, label='Evader (Centered)')
            for i in range(cfg.num_p):
                pid = f'p_{i}'
                if pid in traj_data:
                    p_traj = traj_data[pid]
                    rel_pos_km = (p_traj - e_traj) / 1000.0
                    ax2.plot(rel_pos_km[:,0], rel_pos_km[:,1], rel_pos_km[:,2], color=colors[i], linewidth=1.5, alpha=0.9)
                    ax2.scatter(rel_pos_km[0,0], rel_pos_km[0,1], rel_pos_km[0,2], marker='o', color=colors[i], s=20, alpha=0.5)
                    ax2.scatter(rel_pos_km[-1,0], rel_pos_km[-1,1], rel_pos_km[-1,2], marker='x', color=colors[i], s=60, linewidth=2)
            cap_dist_km = cfg.dist_cap / 1000.0
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x_cap = cap_dist_km * np.cos(u) * np.sin(v)
            y_cap = cap_dist_km * np.sin(u) * np.sin(v)
            z_cap = cap_dist_km * np.cos(v)
            ax2.plot_wireframe(x_cap, y_cap, z_cap, color='gray', alpha=0.2, linewidth=0.5, label='Capture Range')
            ax2.set_title(f'2. Relative View'); set_axes_equal(ax2)

        # Row 2: X, Y, Z vs Time (3 separate plots)
        ax_x = fig.add_subplot(2, 3, 4)
        ax_y = fig.add_subplot(2, 3, 5)
        ax_z = fig.add_subplot(2, 3, 6)
        
        # x轴时间
        time_axis = np.arange(len(traj_data['p_0'])) if 'p_0' in traj_data else np.arange(steps)
        time_min = time_axis * cfg.dt / 60.0 # Time in minutes

        # 绘制函数
        def plot_component(ax, comp_idx, title, ylabel):
            ax.grid(True, linestyle='--', alpha=0.6)
            for i in range(cfg.num_p):
                pid = f'p_{i}'
                if pid in traj_data:
                    pos = traj_data[pid] / scale
                    ax.plot(time_min, pos[:, comp_idx], color=colors[i], label=f'P{i}', linewidth=1.5)
            if 'e_0' in traj_data:
                pos = traj_data['e_0'] / scale
                ax.plot(time_min, pos[:, comp_idx], color='red', linestyle='--', label='Evader', linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel('Time (min)')
            ax.set_ylabel(ylabel)
            if comp_idx == 0: 
                ax.legend(loc='upper right', fontsize='small')

        plot_component(ax_x, 0, '3. X vs Time', 'Global X (km)')
        plot_component(ax_y, 1, '4. Y vs Time', 'Global Y (km)')
        plot_component(ax_z, 2, '5. Z vs Time', 'Global Z (km)')

        # 共享x轴
        ax_y.sharex(ax_x)
        ax_z.sharex(ax_x)

        plt.suptitle(f'Result: {result_str}', fontsize=16)
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"轨迹保存在 {output_file}")
            plt.show()

def set_axes_equal(ax):
    #为了保持3D图的比例一致
    x_limits = ax.get_xlim3d(); y_limits = ax.get_ylim3d(); z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0]); x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0]); y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0]); z_middle = np.mean(z_limits)
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MPE POMDP Agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="路径到 .pt 检查点文件")
    parser.add_argument("--output_plot", type=str, default="trajectory_eval.png", help="保存轨迹图文件名")
    parser.add_argument("--seed", type=int, default=42, help="评估时的随机种子")
    parser.add_argument("--device", type=str, default="cpu", help="推理设备")
    parser.add_argument("--maneuver_strategy", type=str, default="drift", choices=["drift", "random", "apf"], help="逃逸者机动策略")
    parser.add_argument("--test_success_rate", type=bool, default=False, help="是否测试成功率")
    parser.add_argument("--test_episodes", type=int, default=100, help="测试成功率时的评估集数量")
    parser.add_argument("--save_animation", action="store_true", help="是否保存为动图 (仅在单次运行模式下有效)")
    args = parser.parse_args()
    run_eval(args)