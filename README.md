# ERNIE鲁棒性训练集成指南(训练效果待测试)

本文档说明如何在OrbitalGameEnv项目中集成ERNIE鲁棒性训练方法。

## 背景

在轨道追逃环境中，追方对逃方的观测可能存在噪声。ERNIE方法通过在训练时对观测添加噪声，增强策略的鲁棒性。核心思想是训练时对目标相对位置添加均匀噪声，计算加噪前后策略输出的差异，将差异作为正则项加入损失函数，使策略对观测噪声不敏感。

## 修改文件

只需修改2个文件：demos/pe_env/run_training.py(添加命令行参数) 和 demos/pe_env/train_pomdp.py(添加ERNIE训练逻辑)

## 详细修改步骤

### 修改 run_training.py

在 misc_parser 定义之后，main() 函数之前添加ERNIE参数：

    # --- ERNIE 鲁棒性训练参数 ---
    ernie_parser = parser.add_argument_group("ERNIE鲁棒性训练 (ERNIE Robustness)")
    ernie_parser.add_argument("--use_ernie", type=lambda x: (str(x).lower() == 'true'), default=False)
    ernie_parser.add_argument("--ernie_epsilon", type=float, default=0.1, help="ERNIE噪声范围 [-ε, +ε]")
    ernie_parser.add_argument("--ernie_lambda", type=float, default=0.05, help="ERNIE鲁棒性损失权重")

在 train_cfg.debug_observation = args.debug_observation 之后添加：

    # --- ERNIE 鲁棒性训练配置 --- 
    train_cfg.use_ernie = args.use_ernie
    train_cfg.ernie_epsilon = args.ernie_epsilon
    train_cfg.ernie_lambda = args.ernie_lambda
    
### 修改 train_pomdp.py

TrainConfig类定义中，PPO 核心参数之后，学习率与优化器之前，添加：

    # --- 新增：ERNIE 鲁棒性训练 --- 
    use_ernie: bool = False
    ernie_epsilon: float = 0.1      # 噪声范围 [-ε, +ε]
    ernie_lambda: float = 0.05      # 鲁棒性损失权重


在 layer_init 函数定义之后，HRG_ActorCritic 类定义之前，添加辅助函数：

    def add_uniform_noise(obs: torch.Tensor, epsilon: float, noise_dims: list) -> torch.Tensor:
        """
        对观测添加均匀分布噪声 U(-ε, +ε)
        
        Args:
            obs: [batch, obs_dim] 观测张量
            epsilon: 噪声范围上界
            noise_dims: 要加噪声的维度索引列表 (例如 [6, 7, 8])
        
        Returns:
            noisy_obs: 加噪后的观测
        """
        if epsilon == 0:
            return obs
        
        # 生成均匀噪声 U(-ε, +ε)
        noise = torch.rand_like(obs) * 2 * epsilon - epsilon
        
        # 创建掩码，只对指定维度加噪声
        noise_mask = torch.zeros_like(obs)
        noise_mask[:, noise_dims] = 1.0
        noise = noise * noise_mask
        
        return obs + noise
    
    
    def compute_ernie_robustness_loss(
        obs: torch.Tensor,
        hist: torch.Tensor,
        mask: torch.Tensor,
        agent: nn.Module,
        epsilon: float,
        device: str
    ) -> torch.Tensor:
        """
        计算ERNIE鲁棒性损失
        
        Args:
            obs: [batch, obs_dim] 干净观测
            hist: [batch, hist_len, 6] 历史序列
            mask: [batch, hist_len] 历史掩码
            agent: HRG_ActorCritic网络
            epsilon: 噪声强度
            device: 计算设备
        
        Returns:
            loss: 鲁棒性损失（标量）
        """
        # 对目标位置（第6-8维）添加噪声
        target_pos_dims = [6, 7, 8]
        noisy_obs = add_uniform_noise(obs, epsilon, target_pos_dims)
        
        # 计算干净观测的策略输出（不需要梯度）
        with torch.no_grad():
            clean_feat, _ = agent.get_student_features(obs, hist, mask)
            clean_action = agent.actor_head(clean_feat)
        
        # 计算加噪观测的策略输出（需要梯度）
        noisy_feat, _ = agent.get_student_features(noisy_obs, hist, mask)
        noisy_action = agent.actor_head(noisy_feat)
        
        # 计算动作差异（L2距离）
        action_diff = torch.norm(clean_action - noisy_action, dim=-1).mean()
        
        # ERNIE目标：最大化差异 → 损失取负号
        # 这样网络会学习对噪声不敏感（策略输出差异小）
        return -action_diff

  损失计算集成ERNIE，训练循环中的损失计算部分，找到这段代码：
  
                if cfg.use_distillation:
                    distil_loss = F.mse_loss(s_feat, distil_target)
                else:
                    distil_loss = torch.tensor(0.0).to(cfg.device)
                
                loss = pg_loss + cfg.vf_coef * v_loss - current_ent_coef * entropy_loss + cfg.distil_coef * distil_loss

替换为：

                if cfg.use_distillation:
                    distil_loss = F.mse_loss(s_feat, distil_target)
                else:
                    distil_loss = torch.tensor(0.0).to(cfg.device)
                
                # --- ERNIE 鲁棒性损失 --- 
                if cfg.use_ernie:
                    ernie_loss = compute_ernie_robustness_loss(
                        b_obs, b_hist, b_mask, agent, 
                        cfg.ernie_epsilon, cfg.device
                    )
                else:
                    ernie_loss = torch.tensor(0.0).to(cfg.device)
                
                # 总损失（添加ERNIE项）
                loss = (pg_loss 
                       + cfg.vf_coef * v_loss 
                       - current_ent_coef * entropy_loss 
                       + cfg.distil_coef * distil_loss 
                       + cfg.ernie_lambda * ernie_loss)
                       
损失日志记录部分，找到这段代码：

        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy_loss", (-current_ent_coef * entropy_loss).item(), global_step)
        writer.add_scalar("losses/distillation_loss", distil_loss.item(), global_step)
        writer.add_scalar("losses/total_loss", loss.item(), global_step)

在其后添加：
        
        # --- ERNIE 鲁棒性训练日志 --- 
        if cfg.use_ernie:
            writer.add_scalar("losses/ernie_robustness_loss", ernie_loss.item(), global_step)
            writer.add_scalar("ernie/epsilon", cfg.ernie_epsilon, global_step)

### 使用方法

  python demos/pe_env/run_training.py \
    --evader_policy_level 2 \
    --use_ernie true \
    --ernie_epsilon 0.1 \
    --ernie_lambda 0.05 \
    --run_name "ernie_eps01_lam005"
