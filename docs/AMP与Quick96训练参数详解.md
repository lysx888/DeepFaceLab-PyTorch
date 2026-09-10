# AMP 与 Quick96 训练参数详解

> 本文档详细说明 AMP 和 Quick96 两种训练模型的全部参数。
> 界面位置：**step4 训练 → 模型类型**，选择 `AMP` 或 `Quick96` 后，下方参数区会切换为对应模型的参数。
> 两种模型的参数均保存在各自模型目录的配置文件中：`workspace/model/amp/AMP_training_config.json`、`workspace/model/quick96/Quick96_training_config.json`（训练时同步写入模型目录内的 `training_config.json`）。
> SAEHD 参数详见《SAEHD 训练参数详解》。

---

## 目录

1. [AMP（Adversarial Made Pretty）](#1-ampadversarial-made-pretty)
2. [Quick96](#2-quick96)
3. [AMP / Quick96 与 SAEHD 对比](#3-amp--quick96-与-saehd-对比)

---

## 1. AMP（Adversarial Made Pretty）

### 1.1 AMP 是什么

AMP（对抗式美化模型）是 DFL 的第二种模型架构，其特点是**用"源脸的五官"与"目标脸的五官"按 morph_factor 比例混合出中间表征**，再通过编码器/解码器重建。与 SAEHD 相比：

- 使用 `inter_dims`（Inter 维度）+ `morph_factor`（混合比例）两个 SAEHD 没有的架构字段
- `eyes_mouth_prio` 在 AMP 中**恒为 True**（代码硬编码，GUI 不显示该开关）
- `visibility_loss_power` 在 AMP 中恒为 0（不使用）
- 对 src/dst 角度差异较大的素材鲁棒性更好，但细节上限通常不如同配置的 SAEHD

### 1.2 参数一览表

| 参数 | 默认 | 范围 | 说明 |
|------|------|------|------|
| resolution | 224 | 64~640，步进 32，**必须 32 的倍数** | 训练分辨率 |
| face_type | wf | wf / head | 人脸裁剪类型 |
| batch_size | 8 | 1~32 | 每次迭代图像对数量 |
| ae_dims | 256 | 32~1024，步进 32 | 瓶颈维度 |
| inter_dims | 1024 | 32~2048，步进 32 | Inter 维度，**应 ≥ ae_dims** |
| e_dims | 64 | 16~256，步进 2 | 编码器基础通道维度 |
| d_dims | 64 | 16~256，步进 2 | 解码器基础通道维度 |
| d_mask_dims | 22 | 8~128，步进 2 | 遮罩解码器基础通道维度 |
| morph_factor | 0.5 | 0.1~0.5，步进 0.05 | 源/目标混合比例 |
| lr | 5e-5 | 1e-6~1e-2 | 学习率 |
| lr_dropout | n | n / y / ca | 学习率策略 |
| random_warp | True | — | 随机仿射变形 |
| random_src_flip | True | — | 源随机翻转 |
| random_dst_flip | True | — | 目标随机翻转 |
| uniform_yaw | False | — | 均匀偏航采样 |
| blur_out_mask | False | — | 模糊遮罩外区域 |
| ct_mode | none | none / rct / lct / mkl / idt / sot（**仅 6 项，无 rct-p / lab-match**） | 训练时颜色迁移 |
| gan_power | 0.0 | 0.0~5.0 | GAN 对抗损失强度 |
| gan_patch_size | 28 | 3~640 | GAN 判别器 Patch 大小 |
| gan_dims | 16 | 4~512 | GAN 判别器基础通道维度 |
| amp_mode | fp16 | fp32 / fp16 / bf16 | 混合精度 |
| gradient_accumulation_steps | 1 | 1~16 | 梯度累积步数 |
| clipgrad | False | — | 梯度裁剪 |
| enable_torch_compile | False | — | torch.compile 加速 |
| auto_batch | False | — | 自动批次探测 |
| pretrain | False | — | SRC-SRC 自重建预训练 |
| target_iter | 0 | 0~9999999 | 目标迭代数（0=无限） |
| backup_interval | 0 | 0~1000000 | 自动保存间隔 |

### 1.3 架构参数（模型创建后锁定）

以下参数在模型首次初始化后**锁定不可修改**（修改会导致模型结构不兼容，只能新建模型）：

```
resolution / face_type / ae_dims / inter_dims / e_dims / d_dims /
d_mask_dims / morph_factor / gan_patch_size / gan_dims / amp_mode
```

其中 `inter_dims` 与 `morph_factor` 是 AMP 特有的架构字段：

**inter_dims（Inter 维度）**
- 默认 1024，范围 32~2048（步进 32）
- Inter 层是 AMP 的核心混合层，维度越高表达能力越强，但显存与计算量线性增长
- **必须 ≥ ae_dims**，建议为 ae_dims 的 2~4 倍

**morph_factor（Morph 因子）**
- 默认 0.5，范围 0.1~0.5（步进 0.05）
- 控制 inter_src / inter_dst 两个中间表征的混合比例
- 0.5 = 各占一半；偏小则更像源脸，偏大则更偏向目标脸

### 1.4 与 SAEHD 不同的参数行为

- **ct_mode 只有 6 项**：`none / rct / lct / mkl / idt / sot`。SAEHD 有 `rct-p` 和 `lab-match`，AMP 没有
- **gan_patch_size 默认 28**（= 224/8，随 resolution 自适应；GUI 默认值 28），SAEHD 默认 16
- **eyes_mouth_prio 恒为 True**：AMP 训练时眼睛和嘴巴区域始终施加 300 倍 L1 损失，无开关
- **visibility_loss_power 恒为 0**：AMP 不使用可见性加权损失
- 其余参数（lr、lr_dropout、random_warp、amp_mode、GAN、pretrain 等）行为与 SAEHD 相同，参见《SAEHD 训练参数详解》对应章节

### 1.5 AMP 预训练行为

勾选 `pretrain`（SRC-SRC 预训练）时，代码强制：

- DST 数据集 = SRC 数据集（自重建）
- `uniform_yaw = True`（强制开启）
- `random_src_flip = True`、`random_dst_flip = True`（强制开启）
- `ct_mode = 'none'`（强制关闭颜色迁移）
- 预训练后模型有更好的初始化，可加速后续正式训练

### 1.6 AMP 推荐配置

| 显存 | resolution | batch_size | inter_dims | ae_dims | 说明 |
|------|-----------|-----------|------------|---------|------|
| 6GB | 160 | 2~4 | 512 | 192 | morph_factor 保持 0.5 |
| 12GB | 224 | 4~8 | 1024 | 256 | 标准配置（默认值） |
| 24GB+ | 320 | 8 | 1536 | 384 | 高细节 |

训练建议与 SAEHD 相同：初期开 random_warp → 中后期关 warp → 最后开 GAN；AMP 建议全程保持 `eyes_mouth_prio`（已硬编码开启）。

---

## 2. Quick96

### 2.1 Quick96 是什么

Quick96 是 DFL 的快速实验模型：**所有架构参数硬编码**（res=96、ae=128、e=64、d=64），训练速度极快、显存占用极小，用于快速验证数据质量、工作流是否顺畅，或小规模快速成片。不追求最高质量。

### 2.2 参数一览表

| 参数 | 默认 | 范围 | 说明 |
|------|------|------|------|
| batch_size | 4 | 1~16 | 每次迭代图像对数量（**唯一可调的训练规模参数**） |
| random_src_flip | True | — | 源随机翻转 |
| random_dst_flip | True | — | 目标随机翻转 |
| amp_mode | fp16 | fp32 / fp16 / bf16 | 混合精度 |
| target_iter | 0 | 0~9999999 | 目标迭代数（0=无限） |
| backup_interval | 0 | 0~1000000 | 自动保存间隔 |

### 2.3 硬编码参数（GUI 不显示、不可修改）

| 参数 | 固定值 | 说明 |
|------|--------|------|
| resolution | 96 | 训练分辨率固定 96 |
| face_type | wf | 固定宽全脸 |
| ae_dims | 128 | 固定瓶颈维度 |
| e_dims / d_dims | 64 | 固定编解码器维度 |
| d_mask_dims | 16 | 固定遮罩维度 |
| masked_training | True | 遮罩内训练固定开启 |
| random_warp | True | 随机变形固定开启 |
| uniform_yaw | False | 固定关闭 |
| ct_mode | none | 固定不迁移颜色 |
| eyes_mouth_prio | False | 固定关闭 |
| lr | 2e-4 | 学习率固定 2e-4（高于 SAEHD 的 5e-5，因为模型小、收敛快） |
| visibility_loss_power | 0.0 | 固定关闭 |

> 注意：step4 面板中 Quick96 的架构锁定集只有 `amp_mode`，即切换混合精度（fp16↔fp32）不影响模型结构兼容性；其余参数本就不存在或硬编码，无需锁定。

### 2.4 推荐配置

- 默认 batch_size=4 即可跑在 6GB 显存上
- 想要更快：batch_size 提到 8~16（显存允许时）
- 想要质量：Quick96 的天花板就是 96 分辨率，追求质量请用 SAEHD/AMP

---

## 3. AMP / Quick96 与 SAEHD 对比

| 维度 | SAEHD | AMP | Quick96 |
|------|-------|-----|---------|
| 架构字段 | archi(ae/e/d/d_mask_dims) | ae/inter/e/d/d_mask_dims/morph_factor | 全部硬编码 |
| 默认分辨率 | 256(wf)/384(head) | 224 | 96 |
| 特征混合 | 双解码器(DF)或双Inter(LIAE) | morph 混合源/目标表征 | 标准 AE |
| eyes_mouth_prio | 可开关 | **恒开** | 恒关 |
| ct_mode 选项 | 8 项 | 6 项 | 无（恒 none） |
| 质量上限 | 高 | 中高 | 低（快速验证用） |
| 训练速度 | 慢 | 中 | 极快 |
| 适用场景 | 正式换脸 | 角度差异大的素材 | 验证流程/快速出片 |
