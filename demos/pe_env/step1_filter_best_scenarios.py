import os
import sys
import torch
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# 路径设置 (请根据你的实际目录结构调整)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from env.mpe_pomdp_env import MPE_POMDP_Env, MPE_POMDP_EnvCfg
from demos.pe_env.train_pomdp import HRG_ActorCritic

def load_model(checkpoint_path, device):
    """加载 HAFN 模型用于筛选"""
    print(f"Loading HAFN model from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # 恢复环境配置
    env_cfg = MPE_POMDP_EnvCfg()
    if 'env_cfg' in ckpt:
        for k, v in ckpt['env_cfg'].items():
            setattr(env_cfg, k, v)
    
    # 初始化模型
    pursuer_ids = [f'p_{i}' for i in range(env_cfg.num_p)]
    evader_ids = [f'e_{i}' for i in range(env_cfg.num_e)]
    # This dimension calculation is based on the structure in mpe_pomdp_env.py
    student_obs_dim = 8 + 3*env_cfg.num_e + 7*(env_cfg.num_p-1) + 3
    priv_obs_dim = 6 + (env_cfg.num_p + env_cfg.num_e - 1) * 9
    act_dim = 3
    
    # 强制使用 hafn
    agent = HRG_ActorCritic(env_cfg, act_dim, priv_obs_dim, student_obs_dim, student_model_type='hafn').to(device)
    agent.load_state_dict(ckpt['agent_state_dict'])
    agent.eval()
    
    return agent, env_cfg, pursuer_ids, evader_ids

def run_simulation(agent, env_cfg, formation, seed, device, pursuer_ids, evader_ids):
    """运行单个 Episode 并返回性能指标"""
    env = MPE_POMDP_Env(env_cfg)
    
    # --- 核心：这里固定了参数 ---
    # 通过 seed + force_formation，保证了位置、速度初始化的一致性
    obs, infos = env.reset(seed=seed, options={'force_formation': formation})
    
    done = False
    total_steps = 0
    success = False
    
    while not done:
        # 准备数据
        p_obs = torch.stack([torch.Tensor(obs[pid]) for pid in pursuer_ids]).to(device)
        p_priv = torch.stack([torch.Tensor(infos[pid]['privileged_state']) for pid in pursuer_ids]).to(device)
        p_hist = torch.stack([torch.from_numpy(infos[pid][f'history_input_{evader_ids[0]}']).float() for pid in pursuer_ids]).to(device)
        p_mask = torch.stack([torch.from_numpy(infos[pid][f'history_mask_{evader_ids[0]}']).float() for pid in pursuer_ids]).to(device)

        with torch.no_grad():
            actions, _, _, _, _, _ = agent.get_action_and_value(p_obs, p_priv, p_hist, p_mask, deterministic=True)
            
        action_dict = {pid: actions[i].cpu().numpy() for i, pid in enumerate(pursuer_ids)}
        action_dict.update(env.get_evader_actions()) # 逃逸者策略
        
        obs, _, term, trunc, infos = env.step(action_dict)
        total_steps += 1
        
        if any(term.values()) or any(trunc.values()):
            done = True
            if list(term.values())[0]: # 如果是因为 termination (抓捕/没油)
                reason = infos[pursuer_ids[0]].get('termination_reason', 'unknown')
                if reason == 'capture_success':
                    success = True
    
    # 计算剩余总燃料作为次要指标
    remaining_fuel = sum([env.remain_Dvs[pid] for pid in pursuer_ids])
    env.close()
    
    return {
        'seed': seed,
        'formation': formation,
        'success': success,
        'steps': total_steps,
        'fuel': remaining_fuel
    }

def main(args):
    device = torch.device(args.device)
    agent, env_cfg, p_ids, e_ids = load_model(args.checkpoint, device)
    
    formations = ['ring', 'cluster', 'string', 'pincer']
    candidates = defaultdict(list)
    
    print(f"Starting Monte Carlo Search: {args.samples_per_form} samples per formation...")
    
    # 1. 蒙特卡洛生成与评估
    # 这里的 seed 我们使用简单的整数偏移，保证可复现
    base_seed = 20240101 
    
    total_iter = len(formations) * args.samples_per_form
    pbar = tqdm(total=total_iter, desc="Scanning Scenarios")
    
    counter = 0
    for form in formations:
        for i in range(args.samples_per_form):
            seed = base_seed + counter
            counter += 1
            
            result = run_simulation(agent, env_cfg, form, seed, device, p_ids, e_ids)
            candidates[form].append(result)
            pbar.update(1)
            
    pbar.close()
    
    # 2. 筛选最优 Top-K
    final_benchmark_set = []
    
    # [新增] 计算并打印总体成功率
    total_runs = len(formations) * args.samples_per_form
    total_successes = sum(1 for form_results in candidates.values() for r in form_results if r['success'])
    success_rate = (total_successes / total_runs) * 100 if total_runs > 0 else 0
    print("\n--- HAFN Monte Carlo Run Summary ---")
    print(f"Total Scenarios Simulated: {total_runs}")
    print(f"Overall Success Rate: {success_rate:.2f}% ({total_successes}/{total_runs})")
    
    print("\n--- Filtering for Best and Worst Scenarios ---")
    for form in formations:
        scenarios = candidates[form]
        
        # 排序逻辑: 1. 成功, 2. 步数(少), 3. 燃料(多)
        scenarios.sort(key=lambda x: (x['success'], -x['steps'], x['fuel']), reverse=True)
        
        # [修改] 选取最好的9个和最差的1个
        num_best = 9
        num_worst = 1
        
        if len(scenarios) < num_best + num_worst:
            print(f"Warning: Not enough scenarios for formation '{form}' to select {num_best}+{num_worst}. Taking all {len(scenarios)}.")
            top_scenarios = scenarios
        else:
            best_9 = scenarios[:num_best]
            worst_1 = scenarios[-num_worst:]
            top_scenarios = best_9 + worst_1
            print(f"Formation [{form}]: Selected {len(best_9)} best and {len(worst_1)} worst scenarios.")

        final_benchmark_set.extend(top_scenarios)
        
        best = scenarios[0]
        print(f"    -> Best of '{form}': Steps={best['steps']}, Fuel={best['fuel']:.1f}, Seed={best['seed']}")

    # 3. 保存文件
    output_dir = "benchmarks"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "hafn_best_benchmark.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(final_benchmark_set, f)
        
    print(f"\nSaved {len(final_benchmark_set)} scenarios to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="HAFN model path")
    parser.add_argument("--samples_per_form", type=int, default=100, help="Monte Carlo samples per formation")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
