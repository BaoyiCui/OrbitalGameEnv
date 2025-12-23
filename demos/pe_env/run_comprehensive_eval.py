import os
import sys
import torch
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
try:
    from prettytable import PrettyTable
except ImportError:
    print("Warning: 'prettytable' not found. Please install it using 'pip install prettytable' for a formatted report.")
    PrettyTable = None

# 路径设置
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from env.mpe_pomdp_env import MPE_POMDP_Env, MPE_POMDP_EnvCfg
from demos.pe_env.train_pomdp import HRG_ActorCritic

def load_model(checkpoint_path, model_type, device):
    """通用模型加载函数"""
    print(f"Loading {model_type.upper()} model from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    env_cfg = MPE_POMDP_EnvCfg()
    if 'env_cfg' in ckpt:
        for k, v in ckpt['env_cfg'].items():
            setattr(env_cfg, k, v)
    
    student_obs_dim = 8 + 3*env_cfg.num_e + 7*(env_cfg.num_p-1) + 3
    priv_obs_dim = 6 + (env_cfg.num_p + env_cfg.num_e - 1) * 9
    act_dim = 3
    
    agent = HRG_ActorCritic(env_cfg, act_dim, priv_obs_dim, student_obs_dim, student_model_type=model_type).to(device)
    agent.load_state_dict(ckpt['agent_state_dict'])
    agent.eval()
    
    return agent, env_cfg

def run_simulation(agent, env_cfg, formation, seed, device):
    """运行单个 Episode 并返回详细性能指标"""
    env = MPE_POMDP_Env(env_cfg)
    p_ids = [f'p_{i}' for i in range(env_cfg.num_p)]
    e_ids = [f'e_{i}' for i in range(env_cfg.num_e)]
    
    obs, infos = env.reset(seed=seed, options={'force_formation': formation})
    
    done = False
    steps = 0
    success = False
    failure_reason = 'in_progress'
    
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
            reason = infos[p_ids[0]].get('termination_reason', 'unknown')
            if reason == 'capture_success':
                success = True
                failure_reason = 'none'
            else:
                failure_reason = reason

    remaining_fuel = sum([env.remain_Dvs[pid] for pid in p_ids])
    env.close()
    
    return {
        'seed': seed,
        'formation': formation,
        'success': success,
        'steps': steps,
        'fuel': remaining_fuel,
        'failure_reason': failure_reason
    }

def print_and_save_report(all_results, output_dir):
    """生成并打印详细报告，同时保存原始数据到CSV"""
    df = pd.DataFrame(all_results)
    
    # --- 1. 保存原始数据到 CSV ---
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "comprehensive_eval_raw_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nRaw results saved to {csv_path}")

    # --- 2. 创建 HAFN 基准查找表 ---
    hafn_df = df[df['model'] == 'hafn']
    hafn_baseline = {}
    for _, row in hafn_df.iterrows():
        hafn_baseline[(row['formation'], row['seed'])] = {'steps': row['steps'], 'fuel': row['fuel']}

    # --- 3. 生成并打印 PrettyTable ---
    if PrettyTable is None:
        print("\nInstall 'prettytable' for a formatted report. Showing raw summary instead.")
        print(df.groupby(['model', 'formation']).agg(
            success_rate=('success', lambda x: x.mean() * 100),
            avg_steps=('steps', 'mean')
        ))
        return

    table = PrettyTable()
    table.field_names = ["Model", "Formation", "Success Rate", "Avg Steps", "Avg Fuel", "vs HAFN Steps"]
    
    model_types = ['hafn', 'lstm', 'mlp']
    formations = ['ring', 'cluster', 'string', 'pincer']

    for model in model_types:
        for i, form in enumerate(formations):
            subset = df[(df['model'] == model) & (df['formation'] == form)]
            if subset.empty: continue

            n = len(subset)
            succ_subset = subset[subset['success']]
            succ_rate = len(succ_subset) / n * 100
            
            avg_steps = succ_subset['steps'].mean() if not succ_subset.empty else 0
            avg_fuel = succ_subset['fuel'].mean() if not succ_subset.empty else 0
            
            # 计算与 HAFN 的差距
            vs_hafn_steps = 0
            if model != 'hafn' and not succ_subset.empty:
                step_diffs = []
                for _, row in succ_subset.iterrows():
                    baseline = hafn_baseline.get((row['formation'], row['seed']))
                    if baseline and baseline['steps'] > 0:
                        step_diffs.append(row['steps'] - baseline['steps'])
                avg_diff = np.mean(step_diffs) if step_diffs else 0
                vs_hafn_steps = f"{avg_diff:+.1f}"
            elif model == 'hafn':
                vs_hafn_steps = "Baseline"
            else:
                vs_hafn_steps = "N/A"

            # 只在每个模型的第一行打印模型名称
            model_name = model.upper() if i == 0 else ""
            table.add_row([model_name, form, f"{succ_rate:.1f}%", f"{avg_steps:.1f}", f"{avg_fuel:.1f}", vs_hafn_steps])
        
        if model != model_types[-1]:
            table.add_row(["---", "---", "---", "---", "---", "---"], divider=True)

    print("\n" + "="*80)
    print(" " * 25 + "COMPREHENSIVE EVALUATION REPORT")
    print("="*80)
    print(table)
    print(f"Note: 'vs HAFN Steps' positive means the model is slower than HAFN.")
    print("="*80)


def main(args):
    device = torch.device(args.device)
    
    # 1. 加载所有模型
    models = {}
    model_paths = {
        'hafn': args.hafn_ckpt,
        'lstm': args.lstm_ckpt,
        'mlp': args.mlp_ckpt
    }
    
    # 使用 HAFN 的 env_cfg 作为通用配置
    _, env_cfg = load_model(model_paths['hafn'], 'hafn', device)
    
    for name, path in model_paths.items():
        agent, _ = load_model(path, name, device)
        models[name] = agent
        
    # 2. 生成测试场景
    formations = ['ring', 'cluster', 'string', 'pincer']
    scenarios = []
    base_seed = 20240101
    for i in range(args.samples_per_form):
        for form in formations:
            scenarios.append({'formation': form, 'seed': base_seed + len(scenarios)})

    # 3. 运行所有模拟
    all_results = []
    pbar = tqdm(total=len(scenarios) * len(models), desc="Running All Evaluations")
    
    for scenario in scenarios:
        for model_name, agent in models.items():
            result = run_simulation(agent, env_cfg, scenario['formation'], scenario['seed'], device)
            result['model'] = model_name
            all_results.append(result)
            pbar.set_postfix_str(f"Model: {model_name}, Form: {scenario['formation']}, Seed: {scenario['seed']}")
            pbar.update(1)
    pbar.close()

    # 4. 生成报告并保存
    print_and_save_report(all_results, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive Evaluation for HAFN, LSTM, and MLP models.")
    parser.add_argument("--hafn_ckpt", type=str, required=True, help="Path to HAFN model checkpoint")
    parser.add_argument("--lstm_ckpt", type=str, required=True, help="Path to LSTM model checkpoint")
    parser.add_argument("--mlp_ckpt", type=str, required=True, help="Path to MLP model checkpoint")
    
    parser.add_argument("--samples_per_form", type=int, default=100, help="Monte Carlo samples per formation")
    parser.add_argument("--output_dir", type=str, default="paper_results", help="Directory to save CSV report and figures")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    
    args = parser.parse_args()
    main(args)
