# Gate-Adv 机制完整说明（实现对齐版）

本文档对应当前分支 `feat/robust-gate-adv` 的实现，覆盖：
- 置信度（gate weight）如何计算
- 如何处理稀有/单模态（orphan）样本
- 如何接入 adversarial loss（双分支）
- 训练时序与关键超参数

对应代码：
- `src/scMRDR/confidence.py` (`ConfidenceWeighter`, `RobustAdvGate`)
- `src/scMRDR/train.py`（双分支 adversarial 训练）
- `src/scMRDR/module.py`（参数透传）
- `scripts/train_anchor.py` / `scripts/grid_search_anchor.py`（CLI）

---

## 1. 记号定义

- 共享表征：`z_i`（第 `i` 个样本）
- 模态标签：`m_i \in \{1,\dots,M\}`
- 判别器输出 logits：`\ell_i = D(z_i)`
- 判别器概率：`p_i = softmax(\ell_i)`
- 对抗样本项：

$$
a_i = -\mathrm{CE}(D(z_i), m_i)
$$

这里 `a_i` 越大，代表“越难被判别器正确识别模态”，对抗目标越满足。

---

## 2. 置信度权重 w_i 的计算（RobustAdvGate）

最终权重由三层组成：
1. 可靠性（reliability）
2. 可迁移性（transferability）
3. 模态预算 + 稀有保护（budget + rare protection）

### 2.1 可靠性分量（来自判别器熵）

先算归一化熵：

$$
H_i = -\frac{1}{\log M}\sum_{k=1}^{M} p_{ik}\log(p_{ik}+\epsilon)
$$

可靠性原分数：

$$
r_i = 1 - H_i
$$

然后使用分位数阈值 + EMA：

$$
\tau_H^{(batch)} = Q_{1-\rho_{target}}(\{r_i\})
$$

$$
\tau_H \leftarrow \eta\,\tau_H + (1-\eta)\,\tau_H^{(batch)}
$$

可靠性门：

$$
g_i^{(rel)} = \sigma\!\left(\frac{r_i - \tau_H}{\tau_w}\right)
$$

> 与旧版不同：这里使用 `1-entropy` 而不是直接使用 entropy，避免“高不确定样本总被加权”。

### 2.2 可迁移性分量（跨模态 NN）

维护每个模态的 L2-normalized 队列（FIFO）。
对样本 `i`，只在“其它模态队列”里找最近邻余弦相似度。

- `s_i^{top1}`: top-1 cosine similarity
- `s_i^{top2}`: top-2 cosine similarity（若不足2个邻居则退化为 top1）
- 距离：

$$
d_i = 1 - s_i^{top1}
$$

扩散温度同样使用中位数 + EMA：

$$
\tau_{nn}^{(batch)} = clip\left(c_\tau\cdot median(\{d_i\}),\,\tau_{min},\tau_{max}\right)
$$

$$
\tau_{nn} \leftarrow \eta\,\tau_{nn} + (1-\eta)\,\tau_{nn}^{(batch)}
$$

相似性得分：

$$
s_i^{(sim)} = \exp\left(-\frac{d_i}{\tau_{nn}}\right)
$$

margin 得分（用 top1-top2）：

$$
\Delta_i = max(s_i^{top1} - s_i^{top2}, 0)
$$

$$
s_i^{(margin)} = \sigma\!\left(\frac{\Delta_i - \delta_{margin}}{\tau_w}\right)
$$

可迁移性融合：

$$
t_i = \alpha\, s_i^{(sim)} + (1-\alpha)\, s_i^{(margin)}
$$

### 2.3 预评分（reliability × transferability）

$$
u_i = g_i^{(rel)} \cdot t_i
$$

### 2.4 模态预算阈值（防止某模态被“饿死”）

对每个模态 `m`，记 batch 内样本数为 `n_m`，最大模态样本数为 `n_{max}`。
稀有度：

$$
rare_m = 1 - \frac{n_m}{max(1,n_{max})}
$$

该模态的目标保留率：

$$
\rho_m = clip(\rho_{target} + rarity\_boost\cdot rare_m,\,0.05,\,0.95)
$$

阈值规则：
- 若 `n_m < min_count`，使用全局阈值
- 否则使用该模态分位数阈值 + EMA

$$
\tau_m^{(batch)} = Q_{1-\rho_m}(\{u_i\mid m_i=m\})
$$

$$
\tau_m \leftarrow \eta\,\tau_m + (1-\eta)\,\tau_m^{(batch)}
$$

模态内 gate：

$$
g_i = \sigma\!\left(\frac{u_i - \tau_{m_i}}{\tau_w}\right)
$$

### 2.5 基础权重 + 稀有加成

$$
\tilde w_i = w_{floor} + (1-w_{floor})\,g_i + rarity\_boost\cdot rare_{m_i}
$$

然后裁剪到 `[w_floor,1]`：

$$
\tilde w_i \leftarrow clip(\tilde w_i, w_{floor}, 1)
$$

### 2.6 orphan 判定与保护

当样本存在跨模态候选时（`has_cross=True`），若满足：

$$
s_i^{top1} < \delta_{sim} \quad \text{or} \quad \Delta_i < \delta_{margin}
$$

则判定为 orphan。对 orphan 施加下限保护：

$$
w_i = max(\tilde w_i, w_{orphan\_min})
$$

非 orphan 则：

$$
w_i = \tilde w_i
$$

最终 `w_i \in [w_floor,1]`。

---

## 3. Gate 如何作用到对抗器（双分支）

## 3.1 总体思想

把 adv 拆成两条路：
- `base` 路：不加权，保证全局混合压力（保护 iLISI）
- `gate` 路：加权，做细粒度控制（提升稀有/难样本鲁棒性）

### 3.2 对抗系数分解

设当前 epoch 的对抗强度（含 warmup 后 ramp）为 `\lambda_{adv}^{cur}`：

$$
\lambda_{base} = \lambda_{adv}^{cur}\cdot r_{base}
$$

$$
\lambda_{gate}^{full} = \lambda_{adv}^{cur}\cdot (1-r_{base})
$$

其中 `r_base = lambda_adv_base_ratio`。

gate 分支再有独立启用/爬坡：

$$
r_{gate}(e)=
\begin{cases}
0, & e < e_{gate\_start}\\
min\left(1, \frac{e-e_{gate\_start}}{E_{gate\_ramp}}\right), & e\ge e_{gate\_start}
\end{cases}
$$

$$
\lambda_{gate} = \lambda_{gate}^{full} \cdot r_{gate}(e)
$$

### 3.3 双分支对抗损失

$$
L_{adv}^{base} = \frac{1}{B}\sum_i a_i
$$

$$
L_{adv}^{gate} = \frac{1}{B}\sum_i w_i a_i
$$

$$
L_{adv}^{total} = \lambda_{base}L_{adv}^{base} + \lambda_{gate}L_{adv}^{gate}
$$

训练中与主干损失相加：

$$
L = L_{base\_model} + L_{adv}^{total} + L_{anchor}
$$

其中 `L_base_model` 为模型返回的 `base_loss`（重构、KL、保结构、diff prior 等）。

---

## 4. 训练时序

- `epoch < num_warmup`：warmup（无判别器对抗）
- `epoch >= num_warmup`：进入判别器-生成器交替训练
  - 若 `confidence_weighted=False`：旧式 adv
  - 若 `confidence_weighted=True` 且 `gate_mode=legacy`：旧 confidence weighter
  - 若 `confidence_weighted=True` 且 `gate_mode=robust_adv`：使用本文双分支 gate-adv

额外注意：
- `lambda_adv_current` 本身会在 warmup 后按 `cw_adv_ramp_epochs` 线性拉升
- 即使 gate 分支关闭，base 分支仍可工作（只要 `r_base>0`）

---

## 5. 关键超参数解释

- `lambda_adv_base_ratio`：base 路占比，越大越保混合（iLISI 更稳）
- `rho_target`：目标保留率，越大整体权重越高
- `w_floor`：最小权重下限
- `w_orphan_min`：orphan 样本下限保护
- `rarity_boost`：稀有模态加权强度（同时影响 `rho_m` 与 `w_i`）
- `gate_start_epoch`：gate 分支开始生效时刻
- `gate_ramp_epochs`：gate 分支爬坡时长
- `orphan_sim_threshold` / `orphan_margin_threshold`：orphan 判定阈值

---

## 6. 日志与诊断（TensorBoard）

robust gate 模式新增监控：
- `tau_h/train`, `tau_nn/train`
- `adv_base_unscaled/train`, `adv_gate_unscaled/train`
- `lambda_adv_gate/train`
- `mean_adv_weight_by_modality/train/m{idx}`
- `orphan_ratio_by_modality/train/m{idx}`
- `gate_threshold_by_modality/train/m{idx}`

建议重点看：
1. orphan ratio 是否长期过高
2. 各模态 mean weight 是否长期失衡
3. iLISI 下滑时 `lambda_adv_gate` 是否过快拉满

---

## 7. CLI 使用示例

单次训练（robust gate）：

```bash
python scripts/train_anchor.py \
  --input-h5ad <data.h5ad> \
  --output-h5ad <out.h5ad> \
  --confidence-weighted \
  --gate-mode robust_adv \
  --lambda-adv 10 \
  --lambda-adv-base-ratio 0.35 \
  --rho-target 0.65 \
  --w-floor 0.15 \
  --w-orphan-min 0.45 \
  --rarity-boost 0.10 \
  --gate-start-epoch 20 \
  --gate-ramp-epochs 10
```

网格搜索（含 gate 关键项）：

```bash
python scripts/grid_search_anchor.py \
  --input-h5ad <data.h5ad> \
  --search-outdir <outdir> \
  --confidence-weighted \
  --gate-mode robust_adv \
  --lambda-adv-grid 5,10,15 \
  --lambda-adv-base-ratio-grid 0.3,0.4 \
  --rho-target-grid 0.6,0.7
```

---

## 8. 与 legacy gate 的关系

- `legacy`：保持旧行为，便于对照实验
- `robust_adv`：新增双分支 + 分层 gate + orphan 保护 + 模态预算

建议在论文/实验中至少做以下对照：
1. no-gate adv
2. legacy gate
3. robust gate（本方案）
4. robust gate + anchor
