#无绘图python ./demos/pe_env/eval4train_pomdp.py --checkpoint ./runs/newest\ with\ new\ scale/checkpoints/update_350.pt --num_p 4 --use_encoder --lstm_scheme 2 --test_success_rate True --test_episodes 10
#有绘图python ./demos/pe_env/eval4train_pomdp.py --checkpoint ./runs/newest\ with\ new\ scale/checkpoints/update_350.pt --num_p 4 --use_encoder --lstm_scheme 2
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from env.mpe_pomdp_env import MPE_POMDP_Env, MPE_POMDP_EnvCfg
from demos.pe_env.train_pomdp import ActorCritic, TrainConfig

def load_checkpoint(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    print(f"Loading checkpoint from {path}...")
    return torch.load(path, map_location=device)

def run_eval(args):
    # 配置设置
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # 加载检查点
    checkpoint = load_checkpoint(args.checkpoint, device)
    
    # 初始化环境配置
    env_cfg = MPE_POMDP_EnvCfg()
    env_cfg.num_p = args.num_p
    env_cfg.num_e = args.num_e
    env_cfg.use_partial_obs = True 
    env_cfg.obs_interval = args.obs_interval
    env_cfg.lstm_history_len = args.lstm_history_len
    env_cfg.lstm_future_len = args.lstm_future_len
    env_cfg.lstm_scheme = args.lstm_scheme
    env_cfg.episode_length = 3600*10 # 10 hours for eval
    # 从 checkpoint 恢复难度参数
    if 'current_dist_cap' in checkpoint:
        env_cfg.dist_cap = checkpoint['current_dist_cap']
        print(f"Loaded dist_cap from checkpoint: {env_cfg.dist_cap:.2f} m")
    if 'current_p_init_dv' in checkpoint:
        env_cfg.p_init_dv = checkpoint['current_p_init_dv']
        print(f"Loaded p_init_dv from checkpoint: {env_cfg.p_init_dv:.2f} m/s")
    
    # 创建环境
    env = MPE_POMDP_Env(env_cfg)
    env.reset(seed=args.seed)

    # 初始化模型
    pursuer_ids = [f'p_{i}' for i in range(env_cfg.num_p)]
    evader_ids = [f'e_{i}' for i in range(env_cfg.num_e)]
    
    actor_obs_dim = env.observation_spaces[pursuer_ids[0]].shape[0]
    critic_obs_dim = sum(env.observation_spaces[p_id].shape[0] for p_id in pursuer_ids)
    act_dim = env.action_spaces[pursuer_ids[0]].shape[0]

    train_cfg = TrainConfig()
    train_cfg.use_encoder = args.use_encoder 

    model = ActorCritic(actor_obs_dim, critic_obs_dim, act_dim, env, env_cfg, train_cfg).to(device)
    
    # 加载权重
    model.load_state_dict(checkpoint['agent_state_dict'])
    env.set_policy_lstm(model.lstm) #注入 LSTM
    model.eval()
    
    print("Model loaded and linked to Environment.")

    # 循环运行
    obs, infos = env.reset()
    
    traj_data = defaultdict(list)
    done = False
    step_cnt = 0
    total_reward = 0.0
    reason = "unknown" # 初始化默认原因

    print("Starting simulation...")
    if args.test_success_rate:
        success_count = 0
        for ep in range(args.test_episodes):
            # 每次初始化都重新创建环境和模型，确保独立性
            # 创建环境
            env = MPE_POMDP_Env(env_cfg)
            env.reset(seed=args.seed)

            # 初始化模型
            pursuer_ids = [f'p_{i}' for i in range(env_cfg.num_p)]
            evader_ids = [f'e_{i}' for i in range(env_cfg.num_e)]
            
            actor_obs_dim = env.observation_spaces[pursuer_ids[0]].shape[0]
            critic_obs_dim = sum(env.observation_spaces[p_id].shape[0] for p_id in pursuer_ids)
            act_dim = env.action_spaces[pursuer_ids[0]].shape[0]

            train_cfg = TrainConfig()
            train_cfg.use_encoder = args.use_encoder 

            # 循环运行
            obs, infos = env.reset()
            
            traj_data = defaultdict(list)
            done = False
            step_cnt = 0
            total_reward = 0.0
            reason = "unknown" # 初始化默认原因


            while not done:
                pursuer_obs_list = [torch.Tensor(obs[name]).to(device) for name in pursuer_ids]
                pursuer_obs_tensor = torch.stack(pursuer_obs_list)
                central_obs_tensor = torch.cat(pursuer_obs_list, dim=-1).unsqueeze(0)

                with torch.no_grad():
                    actions_tensor, _, _, _ = model.get_action_and_value(
                        pursuer_obs_tensor, central_obs_tensor, deterministic=True
                    )
                
                #逃逸者保持默认漂移
                evader_actions = {name: np.zeros(3) for name in evader_ids} 

                actions_to_step = {name: actions_tensor[i].cpu().numpy() for i, name in enumerate(pursuer_ids)}
                actions_to_step.update(evader_actions)

                next_obs, rewards, terminations, truncations, infos = env.step(actions_to_step)
                
                # 记录轨迹
                for agent_id in env.states:
                    traj_data[agent_id].append(env.states[agent_id][:3].copy())

                total_reward += sum(rewards.get(pid, 0) for pid in pursuer_ids)
                obs = next_obs
                step_cnt += 1

                if any(terminations.values()) or any(truncations.values()):
                    done = True
                    # 获取终止原因
                    if pursuer_ids[0] in infos:
                        reason = infos[pursuer_ids[0]].get('termination_reason', 'unknown')
                        if reason == "capture_success":
                            success_count += 1
                    print(f"Episode finished at step {step_cnt}. Reason: {reason}")
                
                if args.render:
                    env.render()

            env.close()

            # 4. 结果统计与绘图
            analyze_and_plot(traj_data, env_cfg, step_cnt, total_reward, reason, args.output_plot,args.test_success_rate,args.test_episodes)
        print(f"Success Rate over {args.test_episodes} episodes: {success_count/args.test_episodes*100:.2f}%")
    else:
        while not done:
            pursuer_obs_list = [torch.Tensor(obs[name]).to(device) for name in pursuer_ids]
            pursuer_obs_tensor = torch.stack(pursuer_obs_list)
            central_obs_tensor = torch.cat(pursuer_obs_list, dim=-1).unsqueeze(0)

            with torch.no_grad():
                actions_tensor, _, _, _ = model.get_action_and_value(
                    pursuer_obs_tensor, central_obs_tensor, deterministic=True
                )
            
            #逃逸者保持默认漂移
            evader_actions = {name: np.zeros(3) for name in evader_ids} 

            actions_to_step = {name: actions_tensor[i].cpu().numpy() for i, name in enumerate(pursuer_ids)}
            actions_to_step.update(evader_actions)

            next_obs, rewards, terminations, truncations, infos = env.step(actions_to_step)
            
            # 记录轨迹
            for agent_id in env.states:
                traj_data[agent_id].append(env.states[agent_id][:3].copy())

            total_reward += sum(rewards.get(pid, 0) for pid in pursuer_ids)
            obs = next_obs
            step_cnt += 1

            if any(terminations.values()) or any(truncations.values()):
                done = True
                # 获取终止原因
                if pursuer_ids[0] in infos:
                    reason = infos[pursuer_ids[0]].get('termination_reason', 'unknown')
                print(f"Episode finished at step {step_cnt}. Reason: {reason}")
            
            if args.render:
                env.render()

        env.close()

        # 4. 结果统计与绘图
        analyze_and_plot(traj_data, env_cfg, step_cnt, total_reward, reason, args.output_plot,args.test_success_rate,args.test_episodes)
def analyze_and_plot(traj_data, cfg, steps, reward, reason, output_file,test_success_rate,test_episodes):
    # 转换为 numpy 数组
    for k in traj_data:
        traj_data[k] = np.array(traj_data[k])

    # 计算最终统计
    p0_pos = traj_data['p_0'][-1]
    e0_pos = traj_data['e_0'][-1]
    final_dist = np.linalg.norm(p0_pos - e0_pos)
    
    # --- 打印详细结果 ---
    result_str = f"{reason}"
    
    print("\n" + "=" * 40)
    print(f"       EVALUATION REPORT       ")
    print("=" * 40)
    print(f"  Time      : {steps * cfg.dt / 3600:.2f} hours ({steps} steps)")
    print(f"  Final Dist    : {final_dist/1000:.2f} km")
    print(f"  Capture Dist  : {cfg.dist_cap/1000:.2f} km")
    print(f"  Total Reward  : {reward:.4f}")
    print(f"  Result        : {result_str}") # 这里会打印具体的失败原因
    print("=" * 40 + "\n")

    if not test_success_rate:

    # --- 3D 绘图 ---
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        #地球
        R_earth = 6378.137 # km
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
        x = R_earth * np.cos(u) * np.sin(v)
        y = R_earth * np.sin(u) * np.sin(v)
        z = R_earth * np.cos(v)
        ax.plot_surface(x, y, z, color='blue', alpha=0.1, edgecolor='none')

        #画轨迹
        scale = 1000.0
        
        #追击者
        colors = plt.cm.jet(np.linspace(0, 1, cfg.num_p))
        for i in range(cfg.num_p):
            pid = f'p_{i}'
            if pid in traj_data:
                pos = traj_data[pid] / scale
                ax.plot(pos[:,0], pos[:,1], pos[:,2], label=f'Pursuer {i}', color=colors[i], linewidth=1, alpha=0.8)
                ax.scatter(pos[0,0], pos[0,1], pos[0,2], marker='o', color=colors[i], s=25, alpha=0.6)
                ax.scatter(pos[-1,0], pos[-1,1], pos[-1,2], marker='x', color=colors[i], s=50, linewidth=2)

        #逃逸者
        eid = 'e_0'
        if eid in traj_data:
            pos = traj_data[eid] / scale
            ax.plot(pos[:,0], pos[:,1], pos[:,2], label='Evader', color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax.scatter(pos[0,0], pos[0,1], pos[0,2], marker='o', color='red', s=25, alpha=0.6)
            ax.scatter(pos[-1,0], pos[-1,1], pos[-1,2], marker='*', color='red', s=80, label='Evader End')

        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        
        # 标题显示具体原因
        title_str = f'Orbital Pursuit Evasion\nResult: {result_str} | Dist: {final_dist/1000:.1f}km'
        ax.set_title(title_str)
        ax.legend()
        
        set_axes_equal(ax)

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Trajectory plot saved to {output_file}")
            plt.show()

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MPE POMDP Agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="路径到 .pt 检查点文件")
    parser.add_argument("--num_p", type=int, default=4, help="追击者数量")
    parser.add_argument("--num_e", type=int, default=1, help="逃逸者数量")
    parser.add_argument("--use_encoder", action="store_true", help="是否使用Encoder")
    parser.add_argument("--render", action="store_true", help="是否开启实时显示")
    parser.add_argument("--output_plot", type=str, default="trajectory_eval.png", help="保存轨迹图文件名")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cpu", help="推理设备")
    
    
    parser.add_argument("--obs_interval", type=int, default=2)
    parser.add_argument("--lstm_history_len", type=int, default=20)
    parser.add_argument("--lstm_future_len", type=int, default=10)
    parser.add_argument("--lstm_scheme", type=int, default=2)

    parser.add_argument("--test_success_rate",type=bool,default=False,help="是否测试成功率")
    parser.add_argument("--test_episodes",type=int,default=100,help="测试成功率时的评估集数量")
    args = parser.parse_args()
    
    run_eval(args)


