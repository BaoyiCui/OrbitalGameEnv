# 轨道多智能体追逃强化学习框架 (Orbital MPE RL Framework)

## 1. 项目概述

本项目是一个专为空间态势感知（Space Situational Awareness, SSA）研究而设计的强化学习框架，专注于解决地球同步轨道（GEO）背景下的多智能体卫星追逃（Multi-agent Pursuit-Evasion, MPE）问题。

在现实场景中，由于地面站观测能力、通信延迟和对手的隐蔽机动，我方通常无法实时、精确地掌握所有空间目标的状态。为了更好地模拟这一挑战，本项目的核心环境被建模为一个**部分可观测马尔可夫决策过程（POMDP）**。在此设定下，追击智能体（Pursuers）必须在信息不完整的情况下，通过历史观测进行推断，并与其他队友协同，以最高效的方式捕获机动的逃逸目标（Evader）。

该框架采用了一系列前沿的强化学习技术，旨在训练出能够应对复杂、动态、不完整信息环境的鲁棒决策策略。

## 2. 核心技术与特性

- **多智能体强化学习 (MARL):** 支持 `N vs M` 的追逃场景，允许多个追击智能体协同作战。
- **2D/3D 可切换仿真模式:** 环境支持在二维平面（赤道面）和三维空间中进行模拟，可通过 `--dim_mode` 参数轻松切换，便于从简单到复杂的场景迁移和分析。
- **高保真轨道动力学:** 
    - 底层调用C++编译的 `libOrbit.so` 动态库进行高精度轨道外推（HPOP），确保了环境的物理真实性。
    - 支持J2摄动等复杂动力学模型。
- **部分可观测环境 (POMDP):** 
    - **不完全观测:** 追击方只能在特定时间间隔（`obs_interval`）获得目标的精确位置，模拟了地面雷达的非连续观测特性。
    - **历史依赖:** 智能体必须依赖历史观测序列来推断目标的当前状态和意图。
- **先进的智能体架构 (HRG - Hierarchical Recurrent Graph Network):** 
    - **记忆模块 (Transformer):** 每个智能体都配备了一个Transformer编码器，作为其记忆核心。它负责处理历史观测序列，提取关于目标运动模式的关键时序特征。
    - **态势感知模块 (Student Encoder):** 这是一个基于注意力机制的编码器，负责在每一时刻动态地融合多种信息源：
        1.  **自身状态:** 包括轨道六根数、剩余燃料等。
        2.  **即时观测:** 当前时刻获取到的（可能不精确的）目标信息。
        3.  **队友信息:** 队友的相对位置、速度和状态。
        4.  **历史记忆:** 由Transformer处理后输出的关于目标轨迹的记忆特征。
    - **标准化动作空间:** 为了提升训练稳定性和收敛速度，策略网络的输出被限制在 `[-1, 1]` 的标准化立方体空间内。环境的 `step` 函数负责将此标准化动作反归一化为真实的物理指令（如 `m/s` 的速度增量），并进行精确的球形/圆形边界截断，这是处理连续动作物理控制问题的业界标准实践。
    - **非对称训练范式 (Asymmetric Actor-Critic):** 
        - **演员 (Actor):** 最终部署的决策网络，严格遵守POMDP设定，仅依赖自身的部分观测和历史记忆进行决策。
        - **评论家 (Critic):** 在训练阶段，Critic可以获取包含所有智能体真实状态的“上帝视角”特权信息。这使其能对局势做出更准确的价值判断，从而更稳定、高效地指导Actor的学习。
    - **知识蒸馏 (Knowledge Distillation):** 引入一个“教师”网络（`Aligned_Teacher`），它直接从特权信息中学习一个“理想”的态势感知表征，并通过MSE损失来“指导”学生网络（Actor的编码器）的特征学习，加速收敛。
- **课程学习 (Curriculum Learning):** 
    - 为了解决稀疏奖励和任务难度过大的问题，训练采用课程学习策略，从简单任务开始，随智能体能力提升自动增加难度。
    - **可调难度参数:** 
        - `m` (初始距离): 从近距离开始，逐步增大追逃的初始分离距离。
        - `p_init_dv` (初始燃料): 从充足的燃料开始，逐步减少追击方的可用燃料。
        - `dist_cap` (捕获半径): 从一个较大的捕获半径开始，逐步缩小，要求更精确的拦截。
- **可配置的对手策略:** 
    - 逃逸智能体可以配置多种等级的脚本策略，用于评估和增强追击方的鲁棒性：
        - **Level 0:** 静止不动 (Drift)
        - **Level 1:** 随机机动 (Random)
        - **Level 2:** 基于人工势场法（APF）的逃逸策略

## 3. 项目结构

```
HLS_LLS_2D/
├── demos/pe_env/
│   ├── run_training.py   # 训练主入口，负责解析参数和启动训练
│   ├── train_pomdp.py    # 核心训练脚本，包含PPO和课程学习的完整循环
│   └── eval4train_pomdp.py # 评估脚本，用于测试模型和生成可视化结果
├── env/
│   ├── pe_env.py         # 基础的1v1追逃环境
│   ├── mpe_env.py        # 继承PEEnv，扩展为多智能体环境
│   ├── mpe_pomdp_env.py  # 继承MPEEnv，加入POMDP特性，是当前的核心环境
│   └── hrg_models.py     # 定义HRG学生网络和教师网络等模型结构
├── runs/                   # (训练后生成) 存放TensorBoard日志和模型检查点
└── results/                # (评估后生成) 存放评估结果，如轨迹图和GIF
```

## 4. 使用指南

### 4.1 环境与依赖

请确保您的系统中已安装 `conda`。首先，创建一个新的conda环境并安装必要的依赖。

```bash
# 创建conda环境
conda create -n orbit python=3.11 -y
conda activate orbit

# 安装核心依赖 (请根据需要调整版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gymnasium numpy matplotlib imageio tqdm

# 编译和配置 libOrbit.so
# (请根据您本地的C++编译环境完成 libOrbit.so 的编译)
# 确保最终的 .so 文件路径与 mpe_pomdp_env.py 中指定的路径一致。
```

### 4.2 模型训练

通过 `demos/pe_env/run_training.py` 脚本启动训练。所有超参数和配置均可通过命令行进行调整。

**训练示例:**
```bash
# 启动一次3D模式下的训练，命名为 "3D_Run_01"，使用4个追击者和2级逃逸策略
conda run -n orbit python demos/pe_env/run_training.py \
    --run_name "3D_Run_01" \
    --num_p 4 \
    --evader_policy_level 2 \
    --dim_mode 3 \
    --total_timesteps 10000000
```

**常用训练参数:**
- `--run_name`: 本次训练的名称，用于区分不同的实验。
- `--dim_mode`: 环境维度模式 (2: 2D平面, 3: 3D空间)。**默认为2D模式**。
- `--num_p`: 追击者的数量。
- `--evader_policy_level`: 逃逸者策略等级 (0: 静止, 1: 随机, 2: APF)。
- `--total_timesteps`: 总训练步数。
- `--lr`: 学习率。
- `--target_m`: 课程学习的最终目标距离（米）。
- `--min_p_init_dv`: 课程学习的最小目标燃料。
- `--debug_rewards true`: 在控制台打印详细的奖励构成，便于调试。

### 4.3 训练监控

训练过程中的各项指标（如奖励、损失、成功率等）会通过TensorBoard记录。

```bash
# 启动TensorBoard
tensorboard --logdir=runs
```
然后在浏览器中打开 `http://localhost:6006` 查看。

### 4.4 模型评估

训练过程中生成的模型检查点（`.pth`文件）会保存在 `runs/<run_name>/checkpoints/` 目录下。使用 `demos/pe_env/eval4train_pomdp.py` 脚本来评估模型性能。

**评估示例:**
```bash
# 评估某个最终保存的检查点，并生成可视化媒体文件
conda run -n orbit python demos/pe_env/eval4train_pomdp.py \
    --checkpoint "runs/3D_Run_01/checkpoints/ckpt_final_XXX.pth" \
    --save_media \
    --test_episodes 100
```
- `--checkpoint`: 指定要评估的模型文件路径。
- `--dim_mode`: 可在评估时指定2D或3D模式，以测试模型在不同维度下的表现。
- `--save_media`: 为第一个成功的评估案例生成轨迹图（.png）和GIF动画。
- `--test_episodes`: 指定评估的总回合数，以计算成功率。
- `--evader_policy_level`: 可在评估时指定与训练时不同的对手策略，以测试模型的泛化能力。