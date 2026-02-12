# scMRDR Adversarial 部分改进方案（Confidence-aware Gating / Anchor-follow）

面向：在 **scMRDR** 的对抗学习（adversarial alignment）里引入 **置信度细胞筛选（confidence gating）**，只对“可对齐的共享细胞”施加对抗对齐压力；对低置信（潜在模态特异）细胞，用 **锚点几何保持（anchor-follow geometry）** 保护其拓扑，避免假对齐与信号压缩。

---

## 1. 背景与目标

### 1.1 scMRDR 当前对抗对齐的风险
scMRDR 的 adversarial alignment 强制不同模态的共享嵌入 `z_u` 分布不可区分。现实数据存在 **modality-unique cell states**（某些亚群只在 ATAC 或只在 RNA），全局对齐会导致：
- **假对齐**：unique cells 被拉到错误的共享簇附近；
- **信号压缩/泄漏**：unique 生物差异被挤进 `z_s`，或扭曲 `z_u` 局部几何；
- **对抗梯度主导**：即使有 preserve loss，unique 细胞仍可能被强拉。

### 1.2 改进目标
- **Selective alignment**：仅对高置信（共享）细胞施加 adversarial 对齐；
- **Topology preservation for low-confidence**：低置信细胞不参与对齐，但保持与高置信锚点的几何关系；
- **训练稳定**：避免 hard mask 抖动；避免置信度被模型“钻空子”学习成低置信以逃避对齐。

---

## 2. 总体设计（核心思想）

### 2.1 disentangle 不变
对每个模态 `m`：
- 编码器输出：`z_u^m`（shared）与 `z_s^m`（specific）
- 维持原 scMRDR 的 reconstruction、KL、preserve（isometric）等项不变或仅作轻微调权。

### 2.2 引入置信度 `c_i` 与锚点集合 `A`
对非参考模态的每个细胞 `i`（可扩展到所有模态成对/多对）：
- 计算置信度 `c_i ∈ (0,1)`：越大表示越可能是跨模态共享、可对齐的状态；
- 定义锚点（高置信集合）：`A = { i | c_i > δ }`（阈值 δ 可用 GMM / EMA 方案）
- 低置信集合：`N = { i | c_i ≤ δ }`

### 2.3 对抗对齐“加权/门控”
- **对抗对齐仅由锚点驱动**：对抗损失只对 `A` 生效（或用 `c_i` 作为软权重）
- 低置信细胞 `N` 不直接参与对抗对齐，避免被拉扯产生假对齐。

### 2.4 低置信“跟随”几何约束（anchor-follow）
- 低置信细胞保持其在 **完整潜变量** `z=(z_u,z_s)` 的相对几何结构，并“跟随”本模态锚点在 `z_u` 空间中的位置，避免拓扑崩塌。
- 建议优先使用 **anchor-based** 几何保持（比原始表达空间距离更稳、更符合 scMRDR 思路）。

---

## 3. 公式（可直接落地实现）

> 下面给出**推荐默认版本**（软权重更稳）。如果你坚持 hard mask，可把 `c_i` 换为 `M_i∈{0,1}`。

### 3.1 置信度计算（每个细胞 i）

#### 3.1.1 within-modality distinctness（局部熵 / 不确定性）
在同模态共享空间 `z_u^m` 中取 KNN：
- 相似度权重（例如用高斯核+softmax）：  
`π_ij = softmax_j( -||z_u,i - z_u,j||^2 / σ_w^2 )`
- 归一化熵（越“尖锐”越可信）：  
` s_within(i) = - (1 / log K) * Σ_j π_ij log π_ij `

#### 3.1.2 cross-modality affinity（跨模态接近度）
在参考模态 `r` 的共享空间 `z_u^r` 中取 `Kc` 个最近邻：  
` d_cross(i) = (1/Kc) * Σ_{j in KNN_r(i)} || z_u,i^m - z_u,j^r || `  
为让“越小越可信”转成“越大越可信”，可取负号或后续做鲁棒标准化：  
` s_cross(i) = - d_cross(i) `

#### 3.1.3 融合与 sigmoid
- 融合分数：  
` s(i) = 0.5 * ( s_within(i) + s_cross(i) )`
- 鲁棒标准化（推荐）：  
` s_norm(i) = ( s(i) - median(s) ) / MAD(s) `
- 置信度：  
` c_i = sigmoid( s_norm(i) / τ )`

> **关键实现细节**：`c_i` 必须 `detach()`（stop-gradient），否则模型可能通过“故意降低置信度”来逃避对齐损失。

---

### 3.2 Gated / Weighted Adversarial Alignment

设 discriminator `D` 为多分类（预测模态标签），对一个 batch 内细胞 `i` 的模态标签为 `y_i`。

#### 3.2.1 更新判别器（D step）
` L_D = Σ_i c_i * CE( D(z_u,i), y_i )`

#### 3.2.2 更新编码器（E step, 通过 GRL 或 min-max）
编码器希望“骗过”判别器，使模态不可辨别：  
` L_adv = Σ_i c_i * CE( D(z_u,i), y_i )`  
对编码器参数使用 **Gradient Reversal Layer**（GRL）或等价 min-max 实现：
- D step：最小化 `L_D`
- E step：最小化 `-L_adv`（或 GRL 直接最小化 `L_adv` 但梯度反转）

> 若用 hard mask：把 `c_i` 替换为 `M_i`（高置信=1，低置信=0）。

---

### 3.3 Anchor-follow Geometry Loss（低置信保护项）

#### 方案 B（推荐默认，稳定且便于实现）：KNN-anchor follow（在全潜变量 z 上找邻居）
对低置信细胞 `i`，在 **完整潜变量** `z=(z_u,z_s)` 内找同模态邻居 `Kf`，只保留其中高置信锚点 `j`：  
- `N_A(i) = { j in KNN_z(i) | c_j > δ }`

定义：  
` L_follow = Σ_{i} (1 - c_i) * (1/|N_A(i)|) * Σ_{j in N_A(i)} ( ||z_u,i - z_u,j|| - ||z_i - z_j|| )^2`

解释：
- `||z_i - z_j||` 是“更可信”的结构参照（scMRDR 已用 z 来保留模态内结构）；
- `||z_u,i - z_u,j||` 约束低置信细胞在 shared 空间里“跟着锚点”保持相对几何；
- 权重 `(1-c_i)` 强化低置信细胞的几何保持。

> 若你更想贴 scCotag，可实现更重的 all-anchors 版本，但计算更贵；KNN-anchor 是更现实的折中。

---

### 3.4 最终损失（在 scMRDR 的基础上加两项）

保持 scMRDR 原目标：  
- `L_recon + β L_KL + γ L_preserve`

替换/改造 adversarial：  
- `λ L_adv`（加权/门控版）

新增：  
- `η L_follow`

最终：  
` L_total = L_recon + β L_KL + γ L_preserve + λ L_adv + η L_follow`

---

## 4. 训练计划（可执行步骤）

### 4.1 阶段化训练（强烈建议）
**Stage 0: Warm-up（不启用 gating）**
- epoch 0 ~ E_warm
- 仅训练：`L_recon + β L_KL + γ L_preserve`
- 目的：先让 `z_u` 与 `z` 有基本可用的几何结构，避免早期 cross-distance 全噪声。

**Stage 1: 启用 gating + 逐步拉高对抗权重**
- epoch E_warm ~ E_mid
- 开始计算 `c_i`（detach）
- 启用：`λ(t)` 从小到大（线性或 cosine）
- 目的：让共享细胞逐步对齐，避免对抗梯度瞬间压垮结构项。

**Stage 2: 启用/增强 follow（低置信保护）**
- epoch E_mid ~ end
- 启用：`η(t)` 从 0 拉升到目标值
- 目的：当锚点集合稳定后，让低置信拓扑“跟随”锚点，减少假对齐。

### 4.2 置信度阈值 δ 的两种实现（选一种即可）
**Option A（更贴论文、稳定）：两分量 GMM**
- 对每个 epoch（或每 N 个 step）收集一批 `c_i`，拟合 2-GMM
- 用 Bayes decision point 得到 `δ`
- 适合：细胞组成波动大、unique cells 比例未知

**Option B（更简单）：EMA 阈值**
- `δ_t = α δ_{t-1} + (1-α) P_p(c_batch)`（取某个分位点的滑动平均）
- 适合：实现简洁、无需外部 GMM 依赖

---

## 5. 实现细节（工程要点）

### 5.1 计算复杂度控制（避免 O(N^2)）
- `s_within`：在每个模态 batch 内做 KNN（K=20~50）
- `s_cross`：对非参考模态 batch，使用 reference batch 的 KNN（Kc=20~50）
- 建议用 FAISS / torch.cdist + topk（batch 不大时可行）

### 5.2 stop-gradient（非常重要）
- `c_i = c_i.detach()`，同时 `M_i`（若用 hard mask）也必须基于 detach 的 `c_i`。

### 5.3 判别器采样策略
- 对抗训练时，只用高置信细胞更新 D 会使 D 的样本变少；
- 建议每步保证每个模态至少抽到一定数量锚点（不足则降阈值或回退到软权重）。

### 5.4 参考模态选择
- 默认 RNA 为 reference；
- 可扩展为“每个模态对其余模态”或“选中心模态”，但先做 RNA anchor 最稳。

---

## 6. 超参数建议（起步默认）

- `K`（within KNN）：30
- `Kc`（cross KNN）：30
- `Kf`（follow KNN）：30
- `τ`：1.0（若用 robust 标准化，τ 不敏感）
- `E_warm`：总 epoch 的 10%~20%
- `λ`：与原 scMRDR alignment 权重同量级起步，但用 schedule 拉升
- `η`：通常比 `γ` 小一档起步（例如 γ=1，η=0.1），再看 distortion 调

---

## 7. 评估与消融（必须做，不然难判断是真提升还是“指标偏置”）

### 7.1 评估指标
- **Shared cell alignment quality**：只在高置信/锚点集合评估（例如 label transfer / FOSCTTM / mixing）
- **Unique cell preservation**：低置信集合的
  - within-modality clustering stability
  - local neighborhood preservation（kNN overlap）
  - modality-specific marker enrichment（是否被错误混入共享簇）
- **整体结构**：UMAP/邻域图 + 批次混合度 + 生物分离度

### 7.2 消融实验
1) baseline scMRDR
2) + soft-weighted adversarial（只加 `c_i` 加权）
3) + hard mask（只对齐锚点）
4) + follow（在 2/3 基础上加 `L_follow`）
5) δ 取 GMM vs EMA vs percentile

---

## 8. 风险清单与对策

- **风险 A：训练早期 c_i 全噪声**  
  对策：Warm-up + λ schedule；或只在 epoch>某阈值才启用 c_i。

- **风险 B：模型学会让 c_i 变低逃避对齐**  
  对策：`detach(c_i)`（必做）。

- **风险 C：锚点太少导致 D 学不到东西**  
  对策：软权重优先；或阈值 δ 设上限/下限；或保证每模态最少锚点数。

- **风险 D：follow 过强导致低置信细胞被“过度绑架”**  
  对策：η 用 schedule；Kf 不要太大；只用 anchor neighbors。

---

## 9. 交付物（你可以按此拆任务）

### 9.1 代码改动点（最小侵入）
1) 在训练循环中新增：`compute_confidence(batch)`  
   - 输入：`z_u`（必要），可选 `z`（用于 follow KNN）
   - 输出：`c`（detach），可选 `mask`（hard）
2) 替换 adversarial loss：`weighted_CE` 或 `masked_CE`
3) 新增 follow loss：`L_follow`
4) 加入 schedule：`λ(t), η(t)` 和 warm-up

### 9.2 实验计划（最短路径）
- 先做 **soft-weighted adversarial**（最稳，最容易看到趋势）
- 再加 follow（看 unique preservation 是否明显改善）
- 最后再尝试 hard mask / GMM（锦上添花）

---
