  # Gate-Adv 机制重设计（面向“单模态稀有细胞”）

  ## Summary

  目标是在你现有 VAE/Diffusion + adversarial + anchor 框架里，保留 gate 创新点，同时解决三类问题：

  1. 早期特征差导致置信度失真
  2. 稀有细胞只存在单模态，易被错罚或被忽略
  3. NMI/ARI 上升但 iLISI 大幅下滑

  核心策略：把现在“单路 gate 控全部 adv”改为“双路 adv + 分层 gate + 预算约束 + 稀有保护”。

  ———

  ## 1) 机制设计（决策已定）

  ### 1.1 双路 adversarial（防 iLISI 崩）

  定义每个样本的对抗项 a_i = -CE(D(z_i), m_i)，总对抗损失改为：

  [
  L_{adv} = \lambda_{base}\cdot \frac{1}{B}\sum_i a_i + \lambda_{gate}\cdot \frac{1}{B}\sum_i w_i a_i
  ]

  - base 路永远开启，保证全局混合压力不消失（保护 iLISI）
  - gate 路做精细重加权，服务稀有/困难样本鲁棒性
  - 默认：lambda_base = 0.35 * lambda_adv, lambda_gate = 0.65 * lambda_adv

  ### 1.2 分层 gate（替换现有单分数）

  每个样本权重 w_i 由三部分构成：

  [
  w_i = clip(w_{floor} + (1-w_{floor})\cdot g_i \cdot t_i + b_i,; w_{floor},; 1)
  ]

  - g_i（可靠性 gate）
    g_i = sigmoid(( (1-H_i) - tau_H ) / T_H )
    用 1-entropy，不再直接用高熵高权重
  - t_i（可迁移性 gate）
    用跨模态 MNN 的 top1 相似度与 margin 组合（都来自 stop-grad 特征）
  - b_i（稀有保护 boost）
    对稀有模态/稀有簇样本加小幅正偏置（防被系统性降权）

  ### 1.3 稀有单模态样本（orphan）策略

  定义 orphan：跨模态相似度长期低于阈值且 margin 不稳定。
  对 orphan：

  - 不把权重打到极低，设 w_orphan_min >= 0.45
  - 仅在 gate 路减弱，不影响 base 路
  - 禁止 orphan 参与“高置信跨模态配对统计”，避免污染队列阈值

  ### 1.4 早期不稳定问题（硬性时间表）

  训练分三段：

  1. Warmup-A：仅重构/KL/isometric（可保留现有 warmup）
  2. Warmup-B：训练判别器 + base-adv，不启用 gate 打分，只填队列
  3. Gate-On：启用完整 gate + 双路 adv，lambda_gate 再线性 ramp

  默认 epoch 比例：20% / 20% / 60%

  ### 1.5 预算约束（防某模态被“饿死”）

  把你现在的 per-modality quantile gate 改为“目标平均权重约束”：

  - 每模态目标 E[w|m]=rho_m，默认 rho_m=0.65
  - 稀有模态 rho_m 自动上调：rho_m += rarity_factor
  - 通过每模态可学习阈值 tau_m 的 EMA 调整实现

  ———

  ## 2) 与现有代码的接口改动（public API）

  ### 2.1 Integration.setup(...) 新增

  - gate_mode: str = "robust_adv" (none|legacy|robust_adv)
  - gate_start_epoch: int
  - gate_ramp_epochs: int
  - lambda_adv_base_ratio: float = 0.35
  - w_floor: float = 0.15
  - w_orphan_min: float = 0.45
  - rho_target: float = 0.65
  - rarity_boost: float = 0.10

  ### 2.2 train_anchor.py 新增 CLI

  - --confidence-weighted（真正暴露开关）
  - --gate-mode
  - --gate-start-epoch
  - --gate-ramp-epochs
  - --lambda-adv-base-ratio
  - --w-floor
  - --w-orphan-min
  - --rho-target
  - --rarity-boost

  ### 2.3 loss_dict / 日志新增

  - adv_base_loss, adv_gate_loss
  - mean_weight_by_modality, orphan_ratio_by_modality
  - gate_tau_H, gate_tau_nn, rho_realized_by_modality

  ———

  ## 3) 代码实现路径（文件级）

  1. src/scMRDR/confidence.py

  - 新增 RobustAdvGate（保留旧 ConfidenceWeighter 兼容）
  - 实现三分量权重、orphan 掩码、每模态预算阈值 EMA

  2. src/scMRDR/train.py

  - 改 cw is not None 分支：从单路 w_i * adv_i 改为双路组合
  - 增加三阶段调度
  - 记录新增诊断指标

  3. src/scMRDR/module.py

  - setup/train 参数透传与默认值
  - gate_mode 分流：none/legacy/robust_adv

  4. scripts/train_anchor.py

  - 增加上述 CLI，并传递到 model.setup()/train()

  5. scripts/grid_search_anchor.py

  - 首轮仅搜稳态关键项：lambda_adv, lambda_anchor, lambda_adv_base_ratio, rho_target
  - 其余使用稳健默认，避免组合爆炸

  ———

  ## 4) 为什么这套能覆盖你提的“所有问题”

  1. 早期特征差

  - Gate 延迟启用 + 独立 ramp + EMA 阈值，避免冷启动误判

  2. 单模态稀有细胞

  - orphan 下限 + 稀有 boost + 模态预算，避免被全局门控压死

  3. NMI/ARI 与 iLISI 冲突

  - base-adv 保底全局混合，gate 只做精修，不再“一刀切”

  4. isometric loss 的定位

  - 保留其“类内结构稳定器”角色，但不把它当跨模态稀有问题主解法

  ———

  ## 5) 验证方案（必须执行）

  ### 5.1 Ablation（最小集）

  1. baseline adv（无 gate）
  2. legacy gate（你当前）
  3. robust gate（新方案）
  4. robust gate + anchor
  5. robust gate + anchor + diffusion（可选）

  ### 5.2 验收阈值（相对 baseline）

  - 稀有群召回/邻域纯度不下降（自定义 rare-cell 指标）

  ### 5.3 失败模式与回退

  - 若 iLISI 仍明显下滑：提高 lambda_adv_base_ratio 到 0.5
  - 若 rare 群仍被压：提高 w_orphan_min 到 0.55
  - 若 NMI/ARI 回落过大：增 rarity_boost 并收紧 tau_H

  ———

  ## 6) 默认假设（已选定）

  - 你继续用当前 adv + anchor 主训练路径，不引入 unbalanced OT 主干
  - 你接受先做“鲁棒 gate”再考虑更重的 OT 方案
  - 主目标是“保持 iLISI 不崩的前提下提升 rare-cell 对齐质量”