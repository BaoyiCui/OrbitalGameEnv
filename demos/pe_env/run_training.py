import argparse
import torch
import sys
import os

# 将项目根目录添加到Python路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from env.mpe_pomdp_env import MPE_POMDP_EnvCfg
from env.mpe_env import MPEEnvCfg # 导入基础配置
from demos.pe_env.train_pomdp import train, TrainConfig

# ==================================================================
# ==================== 在此处设置新的默认奖励系数 ====================
#
# 可以在下面的字典中为奖励相关参数设置新的默认值。
# 取消注释并修改数值，即可让该值成为新的默认值。
#
# 优先级顺序: 命令行参数 > 此处的默认值 > env.py 的默认值
#
# ==================================================================
NEW_REWARD_DEFAULTS = {
    # --- 奖励 ---
    "reward_phase_dist_weight": 1.0, # 新的相位距离奖励的权重
    "reward_time_weight": 0.02,
    "reward_formation_weight": 0.04,
    "reward_fuel_weight": 0.02,     # 单步燃料惩罚 
    "reward_advantage_weight": 0.12,
    "capture_reward": 20.0,         # 适当提高成功奖励，保持正负激励平衡
    "reward_timeout_penalty": -9,   # 超时惩罚，迫使它在省油和快速之间权衡
    "reward_fuelout_penalty": -11, # 加大燃料耗尽惩罚 
}

# ==================================================================
# ==================== 在此处设置新的环境默认参数 ====================
# ==================================================================
NEW_ENV_DEFAULTS = {
    # --- 环境与观测 ---
    "obs_interval": 3, # 在POMDP中，每隔多少步进行一次真实观测
}

# ==================================================================
# =================== 在此处设置新的课程学习默认参数 =================
# ==================================================================
NEW_CURRICULUM_DEFAULTS = {
    # --- 课程学习 ---
    "success_rate_threshold": 0.8, # 提升难度的成功率阈值
}

def main():
    """
    主函数，用于解析命令行参数并启动训练。
    """
    parser = argparse.ArgumentParser(description="启动 PPO 训练，可以通过命令行调整超参数")

    # --- 创建参数组 ---
    env_parser = parser.add_argument_group("环境与观测配置 (Environment & Observation)")
    model_parser = parser.add_argument_group("模型结构配置 (Model Architecture)")
    reward_parser = parser.add_argument_group("奖励权重配置 (Reward Weights)")
    algo_parser = parser.add_argument_group("算法超参数 (Algorithm Hyperparameters)")
    train_parser = parser.add_argument_group("训练过程超参数 (Training Process)")
    curriculum_parser = parser.add_argument_group("课程学习参数 (Curriculum Learning)")
    misc_parser = parser.add_argument_group("通用与杂项 (General & Miscellaneous)")

    # --- 环境与观测配置 ---
    env_parser.add_argument("--num_p", type=int, default=MPE_POMDP_EnvCfg.num_p, help="追捕者(pursuer)的数量")
    env_parser.add_argument("--num_e", type=int, default=MPE_POMDP_EnvCfg.num_e, help="逃逸者(evader)的数量")
    env_parser.add_argument("--dim_mode", type=int, default=2, choices=[2, 3], help="环境维度模式: 2=2D (默认), 3=3D")
    env_parser.add_argument("--use_partial_obs", type=lambda x: (str(x).lower() == 'true'), default=MPE_POMDP_EnvCfg.use_partial_obs, help="是否使用部分可观测环境")
    env_parser.add_argument("--obs_interval", type=int, default=NEW_ENV_DEFAULTS.get('obs_interval', MPE_POMDP_EnvCfg.obs_interval), help="在POMDP中，每隔多少步进行一次真实观测")
    env_parser.add_argument("--evader_policy_level", type=int, default=MPEEnvCfg.evader_policy_level, choices=[0, 1, 2], 
                            help="逃逸者策略等级: 0=无机动, 1=随机, 2=势场法(APF)")
    env_parser.add_argument("--p_dv_step", type=float, default=MPEEnvCfg.p_dv_step, help="追击方每次机动的最大速度增量 (m/s)")
    env_parser.add_argument("--e_dv_step", type=float, default=MPEEnvCfg.e_dv_step, help="逃逸方每次机动的最大速度增量 (m/s)")

    # --- 模型结构配置 ---
    model_parser.add_argument("--history_len", type=int, default=MPE_POMDP_EnvCfg.history_len, help="Transformer输入序列的历史长度")
    model_parser.add_argument("--student_model_type", type=str, default='hafn', choices=['hafn', 'mlp', 'lstm'], help="选择Student Encoder的架构: hafn (默认), mlp, lstm")

    # --- 奖励权重配置 ---
    reward_parser.add_argument("--reward_phase_dist_weight", type=float, default=NEW_REWARD_DEFAULTS.get('reward_phase_dist_weight', MPEEnvCfg.reward_phase_dist_weight), help="[新] 相位/距离混合奖励的权重")
    reward_parser.add_argument("--reward_time_weight", type=float, default=NEW_REWARD_DEFAULTS.get('reward_time_weight', MPEEnvCfg.reward_time_weight), help="基础奖励：时间惩罚权重")
    reward_parser.add_argument("--reward_formation_weight", type=float, default=NEW_REWARD_DEFAULTS.get('reward_formation_weight', MPEEnvCfg.reward_formation_weight), help="基础奖励：群体形成奖励权重")
    reward_parser.add_argument("--reward_fuel_weight", type=float, default=NEW_REWARD_DEFAULTS.get('reward_fuel_weight', MPEEnvCfg.reward_fuel_weight), help="基础奖励：燃料消耗惩罚权重")
    reward_parser.add_argument("--reward_advantage_weight", type=float, default=NEW_REWARD_DEFAULTS.get('reward_advantage_weight', MPEEnvCfg.reward_advantage_weight), help="过程优势奖励的权重")
    reward_parser.add_argument("--capture_reward", type=float, default=NEW_REWARD_DEFAULTS.get('capture_reward', MPEEnvCfg.capture_reward), help="终端奖励：抓捕成功奖励")
    reward_parser.add_argument("--reward_timeout_penalty", type=float, default=NEW_REWARD_DEFAULTS.get('reward_timeout_penalty', MPEEnvCfg.reward_timeout_penalty), help="终端奖励：超时失败的惩罚")
    reward_parser.add_argument("--reward_fuelout_penalty", type=float, default=NEW_REWARD_DEFAULTS.get('reward_fuelout_penalty', MPEEnvCfg.reward_fuelout_penalty), help="终端奖励：燃料耗尽的惩罚")
    # 新增防碰撞参数
    reward_parser.add_argument("--dist_collision", type=float, default=MPEEnvCfg.dist_collision, help="防碰撞：安全半径 (米)")
    reward_parser.add_argument("--reward_collision_weight", type=float, default=MPEEnvCfg.reward_collision_weight, help="防碰撞：惩罚权重")
    reward_parser.add_argument("--max_collision_penalty", type=float, default=MPEEnvCfg.max_collision_penalty, help="防碰撞：最大单步惩罚值")

    # --- 算法超参数 ---
    algo_parser.add_argument("--lr", type=float, default=TrainConfig.lr, help="Actor-Critic网络的学习率")
    algo_parser.add_argument("--gamma", type=float, default=TrainConfig.gamma, help="折扣因子")
    algo_parser.add_argument("--gae_lambda", type=float, default=TrainConfig.gae_lambda, help="GAE的lambda参数")
    algo_parser.add_argument("--clip_coef", type=float, default=TrainConfig.clip_coef, help="PPO的裁剪系数")
    algo_parser.add_argument("--ent_coef", type=float, default=TrainConfig.ent_coef, help="熵损失的系数")
    algo_parser.add_argument("--vf_coef", type=float, default=TrainConfig.vf_coef, help="值函数损失的系数")
    algo_parser.add_argument("--distil_coef", type=float, default=TrainConfig.distil_coef, help="蒸馏损失的权重")
    algo_parser.add_argument("--anneal_ent", type=lambda x: (str(x).lower() == 'true'), default=TrainConfig.anneal_ent, help="是否对熵系数进行退火")
    algo_parser.add_argument("--ent_anneal_start_frac", type=float, default=TrainConfig.ent_anneal_start_frac, help="熵系数退火起始点 (占总训练步数的百分比)")
    algo_parser.add_argument("--final_ent_coef", type=float, default=TrainConfig.final_ent_coef, help="熵系数退火的最终目标值")
    
    # --- 训练过程超参数 ---
    train_parser.add_argument("--total_timesteps", type=int, default=TrainConfig.total_timesteps, help="总训练步数")
    train_parser.add_argument("--num_steps", type=int, default=TrainConfig.num_steps, help="每个更新周期采集的步数 (rollout buffer size)")
    train_parser.add_argument("--num_envs", type=int, default=TrainConfig.num_envs, help="并行环境的数量 (默认为 1)")
    train_parser.add_argument("--num_mini_batches", type=int, default=TrainConfig.num_mini_batches, help="每个epoch中mini-batch的数量")
    train_parser.add_argument("--update_epochs", type=int, default=TrainConfig.update_epochs, help="每个更新周期训练的epoch数")

    # --- 课程学习参数 ---
    curriculum_parser.add_argument("--initial_episode_length", type=int, default=TrainConfig.initial_episode_length, help="初始任务时长")
    curriculum_parser.add_argument("--success_rate_threshold", type=float, default=NEW_CURRICULUM_DEFAULTS.get('success_rate_threshold', TrainConfig.success_rate_threshold), help="提升难度的成功率阈值")
    # 新的距离课程参数
    curriculum_parser.add_argument("--initial_m", type=float, default=TrainConfig.initial_m, help="课程学习：初始距离m (米)")
    curriculum_parser.add_argument("--target_m", type=float, default=TrainConfig.target_m, help="课程学习：目标距离m (米)")
    curriculum_parser.add_argument("--m_increment", type=float, default=TrainConfig.m_increment, help="课程学习：每次提升的距离增量 (米)")
    curriculum_parser.add_argument("--ring_width_delta", type=float, default=TrainConfig.ring_width_delta, help="课程学习：初始生成圆环的宽度 (米)")
    # 其他课程参数
    curriculum_parser.add_argument("--initial_p_init_dv", type=float, default=TrainConfig.initial_p_init_dv, help="初始燃料")
    curriculum_parser.add_argument("--p_init_dv_decrement", type=float, default=TrainConfig.p_init_dv_decrement, help="燃料的缩减量")
    curriculum_parser.add_argument("--min_p_init_dv", type=float, default=TrainConfig.min_p_init_dv, help="最小燃料")
    # SMA Perturbation Curriculum
    curriculum_parser.add_argument("--sma_perturb_start_update", type=int, default=MPEEnvCfg.sma_perturb_start_update, help="The update count to start SMA perturbation curriculum.")
    curriculum_parser.add_argument("--sma_perturb_end_update", type=int, default=MPEEnvCfg.sma_perturb_end_update, help="The update count to end SMA perturbation curriculum.")
    curriculum_parser.add_argument("--sma_perturb_km_max", type=float, default=MPEEnvCfg.sma_perturb_km_max, help="The maximum SMA perturbation in kilometers.")

    # --- 通用与杂项 ---
    misc_parser.add_argument("--seed", type=int, default=TrainConfig.seed, help="随机种子")
    misc_parser.add_argument("--device", type=str, default=TrainConfig.device, help="计算设备 (cuda or cpu)")
    misc_parser.add_argument("--run_name", type=str, default=None, help="实验名称，用于TensorBoard，默认为自动生成。")
    misc_parser.add_argument("--debug_rewards", type=lambda x: (str(x).lower() == 'true'), default=False, help="是否打印每一步详细的奖励构成")
    misc_parser.add_argument("--debug_critic", type=lambda x: (str(x).lower() == 'true'), default=False, help="是否打印Critic诊断信息")
    misc_parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从指定的检查点文件路径恢复训练")
    misc_parser.add_argument("--use-fixed-reset", dest="use_fixed_seed_for_reset", type=lambda x: (str(x).lower() == 'true'), default=MPEEnvCfg.use_fixed_seed_for_reset, help="[调试] 若为True, 则每回合都从固定的初始位置开始")
    misc_parser.add_argument("--debug_observation", type=lambda x: (str(x).lower() == 'true'), default=False, help="每隔200步打印p0的详细观测和网络输入")

    args = parser.parse_args()

    # --- 1. 创建并填充环境配置 ---
    env_cfg = MPE_POMDP_EnvCfg()
    env_cfg.num_p = args.num_p
    env_cfg.num_e = args.num_e
    env_cfg.use_partial_obs = args.use_partial_obs
    env_cfg.obs_interval = args.obs_interval
    env_cfg.dim_mode = args.dim_mode
    env_cfg.history_len = args.history_len # 使用新的参数
    env_cfg.use_fixed_seed_for_reset = args.use_fixed_seed_for_reset
    env_cfg.evader_policy_level = args.evader_policy_level
    env_cfg.p_dv_step = args.p_dv_step
    env_cfg.e_dv_step = args.e_dv_step
    # Curriculum
    env_cfg.sma_perturb_start_update = args.sma_perturb_start_update
    env_cfg.sma_perturb_end_update = args.sma_perturb_end_update
    env_cfg.sma_perturb_km_max = args.sma_perturb_km_max
    
    # 填充奖励参数
    env_cfg.reward_phase_dist_weight = args.reward_phase_dist_weight
    env_cfg.reward_time_weight = args.reward_time_weight
    env_cfg.reward_formation_weight = args.reward_formation_weight
    env_cfg.reward_fuel_weight = args.reward_fuel_weight
    env_cfg.capture_reward = args.capture_reward
    env_cfg.reward_timeout_penalty = args.reward_timeout_penalty
    env_cfg.reward_fuelout_penalty = args.reward_fuelout_penalty
    env_cfg.reward_advantage_weight = args.reward_advantage_weight

    # --- 2. 创建并填充训练配置 ---
    train_cfg = TrainConfig()
    train_cfg.student_model_type = args.student_model_type
    
    train_cfg.num_envs = args.num_envs
            
    train_cfg.gamma = args.gamma
    train_cfg.gae_lambda = args.gae_lambda
    train_cfg.clip_coef = args.clip_coef
    train_cfg.ent_coef = args.ent_coef
    train_cfg.vf_coef = args.vf_coef
    train_cfg.distil_coef = args.distil_coef
    train_cfg.anneal_ent = args.anneal_ent
    train_cfg.ent_anneal_start_frac = args.ent_anneal_start_frac
    train_cfg.final_ent_coef = args.final_ent_coef
    train_cfg.lr = args.lr
    train_cfg.num_mini_batches = args.num_mini_batches
    train_cfg.update_epochs = args.update_epochs
    train_cfg.total_timesteps = args.total_timesteps
    train_cfg.num_steps = args.num_steps
    train_cfg.initial_episode_length = args.initial_episode_length
    train_cfg.curriculum_check_episodes = TrainConfig.curriculum_check_episodes
    train_cfg.success_rate_threshold = args.success_rate_threshold
    # 填充新的课程学习参数
    train_cfg.initial_m = args.initial_m
    train_cfg.target_m = args.target_m
    train_cfg.m_increment = args.m_increment
    train_cfg.ring_width_delta = args.ring_width_delta
    # 填充其他课程参数
    train_cfg.initial_p_init_dv = args.initial_p_init_dv
    train_cfg.p_init_dv_decrement = args.p_init_dv_decrement
    train_cfg.min_p_init_dv = args.min_p_init_dv
    train_cfg.seed = args.seed
    train_cfg.device = args.device
    train_cfg.debug_critic = args.debug_critic
    train_cfg.debug_observation = args.debug_observation
    train_cfg.resume_from_checkpoint = args.resume_from_checkpoint
    if args.run_name:
        train_cfg.run_name = args.run_name
    
    env_cfg.debug_rewards = args.debug_rewards

    # --- 3. 启动训练 ---
    if args.debug_rewards:
        print("\n" + "-" * 30)
        print("--- Initial Reward Weights ---")
        print(f"  reward_phase_dist_weight: {env_cfg.reward_phase_dist_weight}")
        print(f"  reward_time_weight: {env_cfg.reward_time_weight}")
        print(f"  reward_formation_weight: {env_cfg.reward_formation_weight}")
        print(f"  reward_fuel_weight: {env_cfg.reward_fuel_weight}")
        print(f"  reward_advantage_weight: {env_cfg.reward_advantage_weight}")
        print(f"  capture_reward: {env_cfg.capture_reward}")
        print(f"  reward_timeout_penalty: {env_cfg.reward_timeout_penalty}")
        print(f"  reward_fuelout_penalty: {env_cfg.reward_fuelout_penalty}")
        print("-" * 30)

    print("--- 使用命令行配置启动训练 ---")
    train(train_cfg, env_cfg, vars(args))
    print("--- 训练结束 ---")


if __name__ == "__main__":
    main()
    print(f"  reward_fuelout_penalty: {env_cfg.reward_fuelout_penalty}")
    print("-" * 30)

    print("--- 使用命令行配置启动训练 ---")
    train(train_cfg, env_cfg, vars(args))
    print("--- 训练结束 ---")

if __name__ == "__main__":
    main()