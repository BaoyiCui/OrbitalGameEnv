import os
import sys
import torch
import argparse
import pickle
import numpy as np
from tqdm import tqdm
try:
    from prettytable import PrettyTable
except ImportError:
    print("Warning: 'prettytable' not found. Please install it using 'pip install prettytable' for a formatted report.")
    PrettyTable = None

# 路径设置
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from env.mpe_pomdp_env import MPE_POMDP_Env, MPE_POMDP_EnvCfg
from demos.pe_env.train_pomdp import HRG_ActorCritic

def run_fixed_benchmark(args):
    device = torch.device(args.device)
    
    # 1. 加载基准测试集
    if not os.path.exists(args.benchmark_file):
        print(f"Error: Benchmark file '{args.benchmark_file}' not found. Run Step 1 first.")
        return
    with open(args.benchmark_file, 'rb') as f:
        scenarios = pickle.load(f)
    print(f"Loaded {len(scenarios)} scenarios from {args.benchmark_file}")
    
    # 2. 加载待测模型
    print(f"Loading candidate model ({args.model_type}) from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    env_cfg = MPE_POMDP_EnvCfg()
    if 'env_cfg' in ckpt:
        for k, v in ckpt['env_cfg'].items():
            setattr(env_cfg, k, v)
    
    # 初始化环境和Agent
    env = MPE_POMDP_Env(env_cfg)
    p_ids = [f'p_{i}' for i in range(env_cfg.num_p)]
    e_ids = [f'e_{i}' for i in range(env_cfg.num_e)]
    
    student_obs_dim = 8 + 3*env_cfg.num_e + 7*(env_cfg.num_p-1) + 3
    priv_obs_dim = 6 + (env_cfg.num_p + env_cfg.num_e - 1) * 9
    act_dim = 3
    
    agent = HRG_ActorCritic(env_cfg, act_dim, priv_obs_dim, student_obs_dim, student_model_type=args.model_type).to(device)
    agent.load_state_dict(ckpt['agent_state_dict'])
    agent.eval()
    
    # 3. 开始考试
    results = []
    
    for sc in tqdm(scenarios, desc=f"Testing {args.model_type}"):
        seed = sc['seed']
        formation = sc['formation']
        
        # --- 强制复现 ---
        obs, infos = env.reset(seed=seed, options={'force_formation': formation})
        
        done = False
        steps = 0
        success = False
        
        while not done:
            p_obs = torch.stack([torch.Tensor(obs[pid]) for pid in p_ids]).to(device)
            p_priv = torch.stack([torch.Tensor(infos[pid]['privileged_state']) for pid in p_ids]).to(device)
            p_hist = torch.stack([torch.from_numpy(infos[pid][f'history_input_{e_ids[0]}']).float() for pid in p_ids]).to(device)
            p_mask = torch.stack([torch.from_numpy(infos[pid][f'history_mask_{e_ids[0]}']).float() for pid in p_ids]).to(device)
            
            with torch.no_grad():
                actions, _, _, _, _, _ = agent.get_action_and_value(p_obs, p_priv, p_hist, p_mask, deterministic=True)
            
            action_dict = {pid: actions[i].cpu().numpy() for i, pid in enumerate(p_ids)}
            action_dict.update(env.get_evader_actions())
            
            obs, _, term, trunc, infos = env.step(action_dict)
            steps += 1
            
            if any(term.values()) or any(trunc.values()):
                done = True
                if list(term.values())[0]:
                    reason = infos[p_ids[0]].get('termination_reason', 'unknown')
                    if reason == 'capture_success':
                        success = True
        
        results.append({
            'formation': formation,
            'success': success,
            'steps': steps,
            'hafn_steps': sc['steps'] # 这是一个非常有用的对比基准
        })

    env.close()
    
    # 4. 生成对比报告
    print_report(results, args.model_type)

def print_report(results, model_name):
    if PrettyTable is None:
        print("\n--- Raw Report (prettytable not installed) ---")
        print(results)
        return

    table = PrettyTable()
    table.field_names = ["Formation", "Count", "Success Rate", "Avg Steps", "vs HAFN Baseline"]
    
    formations = sorted(list(set([r['formation'] for r in results])))
    
    total_success = 0
    
    for form in formations:
        subset = [r for r in results if r['formation'] == form]
        n = len(subset)
        succ = sum([1 for r in subset if r['success']])
        avg_steps = np.mean([r['steps'] for r in subset if r['success']]) if succ > 0 else 0
        
        # 计算与HAFN基准的差距 (只有成功的才比步数)
        step_diffs = [r['steps'] - r['hafn_steps'] for r in subset if r['success']]
        avg_diff = np.mean(step_diffs) if step_diffs else 0
        diff_str = f"{avg_diff:+.1f}" if succ > 0 else "N/A"
        
        table.add_row([form, n, f"{succ/n*100:.0f}%", f"{avg_steps:.1f}", diff_str])
        
        total_success += succ

    print(f"\n=== Evaluation Report: {model_name.upper()} on HAFN-Favored Set ===")
    print(table)
    print(f"Total Success: {total_success}/{len(results)} ({total_success/len(results)*100:.1f}%)")
    if total_success > 0:
        print(f"Note: 'vs HAFN Baseline' positive means {model_name} is slower than HAFN.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint to test")
    parser.add_argument("--model_type", type=str, required=True, choices=['hafn', 'lstm', 'mlp'], help="Model architecture")
    parser.add_argument("--benchmark_file", type=str, default="benchmarks/hafn_best_benchmark.pkl", help="Generated benchmark file")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    run_fixed_benchmark(args)
