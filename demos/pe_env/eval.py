
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# --- 添加项目根目录到sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from OrbitalGameEnv.env.mpe_env import MPEEnv, MPEEnvCfg
# 从 train.py 中导入 Actor 网络结构
from train import Actor

# --- 评估配置 ---

class EvalConfig:
    # **重要**: 请将此路径修改为您要评估的模型文件的实际路径
    MODEL_PATH = "runs/mappo_curriculum_2025-10-18_12-00-00/final_model.pth" # 这是一个示例路径
    
    NUM_EVAL_EPISODES = 100  # 评估的总局数
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 评估时环境的难度应与模型最终训练阶段的难度相匹配
    DIST_CAP_EVAL = 30.0e3 # 最终阶段的抓捕半径

# --- 主评估函数 ---

def main():
    config = EvalConfig()
    device = torch.device(config.DEVICE)

    # 检查模型文件是否存在
    if not os.path.exists(config.MODEL_PATH):
        print(f"Error: Model file not found at '{config.MODEL_PATH}'")
        print("Please update the MODEL_PATH in the EvalConfig class.")
        sys.exit(1)

    # --- 初始化环境 ---
    env_cfg = MPEEnvCfg()
    env_cfg.dist_cap = config.DIST_CAP_EVAL # 设置为最难的评估难度
    env = MPEEnv(config=env_cfg)

    pursuer_ids = [a for a in env.possible_agents if a.startswith('p_')]

    # --- 初始化 Actor 网络 ---
    actors = {
        agent_id: Actor(
            env.observation_space(agent_id).shape[0],
            env.action_space(agent_id).shape[0],
            [256, 256] # 确保这里的隐藏层尺寸与训练时一致
        ).to(device)
        for agent_id in pursuer_ids
    }

    # --- 加载模型权重 ---
    print(f"Loading model from {config.MODEL_PATH}...")
    state_dict = torch.load(config.MODEL_PATH, map_location=device)
    
    for agent_id in pursuer_ids:
        actor_key = f"actor_{agent_id}"
        if actor_key in state_dict:
            actors[agent_id].load_state_dict(state_dict[actor_key])
            actors[agent_id].eval() # 设置为评估模式
        else:
            print(f"Error: Actor key '{actor_key}' not found in the model file.")
            sys.exit(1)
    
    print("Model loaded successfully.")

    # --- 运行评估 ---
    success_count = 0
    total_steps = 0

    print(f"Running evaluation for {config.NUM_EVAL_EPISODES} episodes...")
    for episode in tqdm(range(config.NUM_EVAL_EPISODES), desc="Evaluating"):
        obs, _ = env.reset()
        done = False
        episode_steps = 0

        while not done:
            actions = {}
            with torch.no_grad():
                for agent_id, agent_obs in obs.items():
                    if agent_id.startswith('p_'):
                        agent_obs_tensor = torch.Tensor(agent_obs).to(device)
                        # 在评估时，我们通常取概率分布的均值作为确定性动作
                        dist = actors[agent_id](agent_obs_tensor)
                        action = dist.mean.cpu().numpy()
                        actions[agent_id] = action
                    elif agent_id.startswith('e_'):
                        # 为逃逸者生成随机机动
                        e_dv_step = env.unwrapped._config.e_dv_step
                        action_vec = np.random.randn(3)
                        norm = np.linalg.norm(action_vec)
                        if norm > 1e-6:
                            action_vec = action_vec / norm
                        magnitude = np.random.uniform(0, e_dv_step)
                        actions[agent_id] = action_vec * magnitude
            
            obs, _, terminations, truncations, _ = env.step(actions)
            episode_steps += 1

            # 检查是否终止
            if any(terminations.values()) or any(truncations.values()):
                done = True
                # 如果是因为抓捕（terminations）而不是超时（truncations），则算作成功
                if any(terminations.values()) and not any(truncations.values()):
                    success_count += 1
        
        total_steps += episode_steps

    env.close()

    # --- 打印评估结果 ---
    success_rate = success_count / config.NUM_EVAL_EPISODES
    avg_steps = total_steps / config.NUM_EVAL_EPISODES

    print("\n--- Evaluation Results ---")
    print(f"Total Episodes: {config.NUM_EVAL_EPISODES}")
    print(f"Successful Captures: {success_count}")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Average Steps per Episode: {avg_steps:.2f}")
    print("--------------------------\n")

if __name__ == "__main__":
    main()
