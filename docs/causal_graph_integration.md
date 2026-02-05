# scMRDR 因果图集成文档

## 概述

本文档描述了在 scMRDR 的 `DisentanglementEncoder` 模块中集成因果图（Causal DAG）的改动。该改动参考了 CausCell 项目的因果推理方法，使得模型能够：

1. 建模概念因子之间的因果关系
2. 通过结构因果模型（SCM）约束潜在表示
3. 支持因果干预和反事实生成

## 改动文件

- `src/scmrdr/model.py` - `DisentanglementEncoder` 类

## 技术原理

### 结构因果模型 (SCM)

因果图的核心是结构因果模型，其数学形式为：

```
z = Az + u
```

其中：
- `z` 是概念嵌入向量 (concept embeddings)
- `A` 是因果DAG邻接矩阵
- `u` 是外生变量 (exogenous variables)

通过求解，得到：

```
z = (I - A)^(-1) * u
```

### 因果DAG邻接矩阵

邻接矩阵 `A` 的定义：
- `A[i,j] = 1` 表示 `factor_j → factor_i`（j 是 i 的父节点/原因）
- `A[i,j] = 0` 表示 j 和 i 之间没有直接因果关系

**示例**：假设有4个因子，因果关系为 `factor_0 → factor_2` 和 `factor_1 → factor_2`

```python
causal_dag = [
    [0, 0, 0, 0],  # factor_0: 无父节点
    [0, 0, 0, 0],  # factor_1: 无父节点
    [1, 1, 0, 0],  # factor_2: 受 factor_0 和 factor_1 影响
    [0, 0, 0, 0],  # factor_3: 无父节点 (通常是 unexplained factor)
]
```

对应的因果图：
```
factor_0 ──┐
           ├──→ factor_2
factor_1 ──┘

factor_3 (独立)
```

### 因果一致性损失

为了确保学到的表示符合因果结构，引入因果一致性约束损失：

```python
m_concept_embs = A @ concept_embs + exogenous_embs
mask_recon_loss = ((concept_embs - m_concept_embs) ** 2).mean()
```

这个损失确保 `z = Az + u` 关系成立。

---

## API 参考

### DisentanglementEncoder

#### 构造函数

```python
DisentanglementEncoder(
    profile_size,      # int: 输入特征维度
    out_dim,           # int: 每个因子的嵌入维度
    num_factor,        # int: 因子数量
    label_categories,  # List[int]: 每个因子的类别数
    causal_dag=None,   # Optional[torch.Tensor]: 因果DAG邻接矩阵 (num_factor, num_factor)
    bias=False,        # bool: 是否使用偏置
    out_act="gelu",    # str: 输出激活函数
    gamma=35           # float: 损失权重
)
```

#### 新增方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `mask_z(x)` | `x: (B, num_factor, out_dim)` | `(B, num_factor, out_dim)` | 计算 `A @ x`，用于因果掩码 |
| `extract_concept_embs(x)` | `x: (B, profile_size)` | `(B, num_factor, out_dim)` | 提取因果转换后的概念嵌入 |
| `causality_based_transform(u)` | `u: (B, num_factor, out_dim)` | `(B, num_factor, out_dim)` | 对外生嵌入应用因果转换 `(I-A)^(-1) @ u` |
| `intervene_and_transform(u, idx, val)` | 见下文 | `(B, num_factor, out_dim)` | 执行因果干预 do(factor=value) |

**`intervene_and_transform` 参数**：
- `exogenous_embs`: `(B, num_factor, out_dim)` 原始外生嵌入
- `target_factor_idx`: `int` 要干预的因子索引
- `target_embs`: `(B, out_dim)` 干预后的目标嵌入值

#### forward 返回值

```python
concept_embs, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl = encoder(x, o)
```

- `mask_recon_loss`: 当启用因果图时，返回因果一致性损失；否则返回 0

---

## 使用示例

### 1. 基础使用：创建带因果图的编码器

```python
import torch
from scmrdr.model import DisentanglementEncoder

# 定义因果DAG
# 场景：细胞类型(0) → 基因表达模式(2)，批次效应(1) 独立，unexplained(3) 独立
causal_dag = torch.tensor([
    [0, 0, 0, 0],  # cell_type: 根节点
    [0, 0, 0, 0],  # batch: 根节点
    [1, 0, 0, 0],  # expression_pattern: 受 cell_type 影响
    [0, 0, 0, 0],  # unexplained: 独立
], dtype=torch.float32)

# 创建编码器
encoder = DisentanglementEncoder(
    profile_size=3000,
    out_dim=64,
    num_factor=4,
    label_categories=[10, 3, 5, 1],  # 各因子类别数
    causal_dag=causal_dag
)

# 前向传播
x = torch.randn(32, 3000)  # 基因表达数据
o = torch.randint(0, 10, (32, 4))  # 标签

concept_embs, mask_loss, pred_loss, disc_loss, kl = encoder(x, o)
print(f"因果一致性损失: {mask_loss.item():.4f}")
```

### 2. 反事实生成：细胞类型转换

```python
import torch

# 假设已经训练好模型
encoder.eval()

# 获取参考细胞的外生嵌入
ref_cells = torch.randn(100, 3000)  # 参考细胞 (如: T细胞)
exo_embs = encoder.extract_exogenous_embs(ref_cells)

# 构建目标嵌入池 (从目标细胞类型中采样)
target_cells = torch.randn(500, 3000)  # 目标细胞 (如: B细胞)
target_exo = encoder.extract_exogenous_embs(target_cells)
target_factor_embs = target_exo[:, 0, :]  # 提取细胞类型因子

# 随机采样目标嵌入
sampled_idx = torch.randint(0, 500, (100,))
sampled_target = target_factor_embs[sampled_idx]

# 执行干预: do(cell_type = B细胞)
counterfactual_embs = encoder.intervene_and_transform(
    exogenous_embs=exo_embs,
    target_factor_idx=0,  # 细胞类型因子索引
    target_embs=sampled_target
)

print(f"反事实嵌入形状: {counterfactual_embs.shape}")
# 输出: torch.Size([100, 4, 64])
```

### 3. 多因子干预

```python
def multi_factor_intervention(encoder, exo_embs, interventions):
    """
    执行多因子干预

    Args:
        encoder: DisentanglementEncoder 实例
        exo_embs: (B, num_factor, out_dim) 原始外生嵌入
        interventions: List[Tuple[int, Tensor]] 干预列表 [(factor_idx, target_embs), ...]

    Returns:
        concept_embs: 干预后的概念嵌入
    """
    intervened = exo_embs.clone()

    for factor_idx, target_embs in interventions:
        intervened[:, factor_idx, :] = target_embs

    return encoder.causality_based_transform(intervened)

# 使用示例：同时干预细胞类型和批次
interventions = [
    (0, target_celltype_embs),  # 干预细胞类型
    (1, target_batch_embs),     # 干预批次
]
cf_embs = multi_factor_intervention(encoder, exo_embs, interventions)
```

### 4. 与扩散模型结合生成细胞

```python
# 假设已有训练好的扩散模型
diffusion_model = ...  # ZINBDiffusion 实例

# 获取反事实概念嵌入
counterfactual_embs = encoder.intervene_and_transform(exo_embs, 0, target_embs)

# 使用扩散模型生成细胞
# 注意：需要将 concept_embs 转换为 diffusion 模型期望的格式
cf_embs_flat = counterfactual_embs.view(batch_size, -1).unsqueeze(1)  # (B, 1, num_factor*out_dim)

# 采样生成
generated_cells = diffusion_model.sample_with_factor(
    concept_embs=cf_embs_flat,
    batch_size=batch_size
)
```

### 5. 无因果图模式（向后兼容）

```python
# 不传入 causal_dag，行为与修改前完全一致
encoder_no_causal = DisentanglementEncoder(
    profile_size=3000,
    out_dim=64,
    num_factor=4,
    label_categories=[10, 3, 5, 1]
    # causal_dag=None (默认)
)

# mask_recon_loss 将始终为 0
_, mask_loss, _, _, _ = encoder_no_causal(x, o)
assert mask_loss.item() == 0.0
```

---

## 常见因果图设计模式

### 模式1：层级因果结构

```python
# 疾病 → 细胞状态 → 基因表达
causal_dag = torch.tensor([
    [0, 0, 0, 0],  # disease: 根节点
    [1, 0, 0, 0],  # cell_state: 受 disease 影响
    [0, 1, 0, 0],  # gene_expr: 受 cell_state 影响
    [0, 0, 0, 0],  # unexplained
], dtype=torch.float32)
```

### 模式2：多父节点

```python
# 基因型和环境都影响表型
causal_dag = torch.tensor([
    [0, 0, 0, 0],  # genotype
    [0, 0, 0, 0],  # environment
    [1, 1, 0, 0],  # phenotype: 受两者影响
    [0, 0, 0, 0],  # unexplained
], dtype=torch.float32)
```

### 模式3：链式因果

```python
# A → B → C → D
causal_dag = torch.tensor([
    [0, 0, 0, 0],  # A
    [1, 0, 0, 0],  # B ← A
    [0, 1, 0, 0],  # C ← B
    [0, 0, 1, 0],  # D ← C
], dtype=torch.float32)
```

---

## 注意事项

1. **DAG约束**：因果图必须是有向无环图（DAG），否则 `(I-A)^(-1)` 可能不存在或不稳定

2. **因子顺序**：建议将根节点（无父节点的因子）放在前面，unexplained factor 放在最后

3. **矩阵可逆性**：确保 `(I-A)` 是可逆的，这在 A 是合法 DAG 的邻接矩阵时总是成立

4. **损失权重**：`mask_recon_loss` 的权重需要根据实验调整，建议从小值（如 0.1）开始

5. **设备一致性**：`causal_dag` tensor 需要与模型在同一设备上

---

## 参考

- CausCell: https://github.com/... (因果推理框架)
- 结构因果模型 (SCM): Pearl, J. (2009). Causality: Models, Reasoning, and Inference
