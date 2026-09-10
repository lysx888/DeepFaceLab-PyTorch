# SAEHD 训练参数详细文档

> 本文档详细说明 SAEHD 换脸模型的所有训练参数、分阶段训练策略、以及不同显卡配置下的推荐参数。
> 本项目是 DFL（DeepFaceLab）SAEHD 的 PyTorch 原生重新实现，参数与 DFL 完全对齐，并增加了部分自研增强。

---

## 目录

1. [模型架构参数](#1-模型架构参数)
2. [优化器与学习率参数](#2-优化器与学习率参数)
3. [数据增强参数](#3-数据增强参数)
4. [人脸细节参数](#4-人脸细节参数)
5. [损失函数参数](#5-损失函数参数)
6. [GAN 对抗参数](#6-gan-对抗参数)
7. [训练控制参数](#7-训练控制参数)
8. [数值稳定性与显存参数](#8-数值稳定性与显存参数)
9. [分阶段训练详解](#9-分阶段训练详解)
10. [显卡配置推荐](#10-显卡配置推荐)
11. [通用训练建议](#11-通用训练建议)

---

## 1. 模型架构参数

> **重要**: 架构参数在模型创建时锁定，修改需要重新训练模型，无法复用旧权重。

### resolution（分辨率）

| 属性 | 值 |
|------|-----|
| 类型 | int |
| 默认 | 256 |
| 范围 | 64~640，必须是 16 的倍数（含 `-d` 或 `-t` 架构需 32 的倍数） |
| DFL 对应 | resolution |

**含义**：训练图像的分辨率（像素），决定模型输入输出的空间尺寸。分辨率越高，模型能保留的细节越多，但显存占用呈平方级增长。

**使用方法**：通常由 `face_type` 自动设置默认值（wf=256, head=384）。手动设置时需确保是 16 的倍数。

**推荐值**：
- `128`：快速迭代，适合前期调试和验证流程
- `192`：平衡质量与速度，低显存显卡可选
- `256`：**标准推荐**（wf 默认），质量与速度的最佳平衡
- `384`：高质量（head 默认），需要较大显存
- `512`：极高细节，仅适合 24GB+ 显卡

**注意事项**：
- 改变分辨率需要重新训练模型
- 含 `-d` 或 `-t` 架构时，分辨率会自动对齐为 32 的倍数
- 分辨率对显存的影响是平方级的：256→384 显存约增加 2.25 倍

---

### face_type（人脸类型）

| 属性 | 值 |
|------|-----|
| 类型 | str |
| 默认 | "wf" |
| 可选 | wf / head |
| DFL 对应 | face_type |

**含义**：人脸裁剪区域类型，决定训练时从原图中裁剪多大区域的人脸。同时自动设置默认分辨率。

**各选项详解**：
- `wf`（宽全脸，256px）：**最通用，推荐**，覆盖脸部及周边区域，包括额头和部分下巴
- `head`（整个头部，384px）：含头发区域，适合发际线融合，但训练难度大，**必须配合 XSeg 遮罩**

**使用方法**：绝大多数场景用 `wf`。仅当需要融合头发时考虑 `head`。`head` 类型要求 src 和 dst 数据集都有准确的 XSeg 遮罩。

---

### archi（AE 架构）

| 属性 | 值 |
|------|-----|
| 类型 | str |
| 默认 | "df" |
| 可选 | df / liae + 后缀选项（u/d/t/c） |
| DFL 对应 | archi |

**含义**：自编码器架构类型，决定 Encoder-Inter-Decoder 的组织方式。

**基础架构**：
- `df`（DeepFake）：双解码器架构。Encoder 共享，Inter 共享，Decoder_src 和 Decoder_dst 独立。**保留更多身份特征，推荐**。
- `liae`（Lightweight Inter AutoEncoder）：共享解码器架构。Encoder 共享，Inter_AB + Inter_B 两个 Inter 层，Decoder 共享。**能修复脸型差异较大的情况**。

**后缀选项（可多选组合）**：
- `u`：像素归一化（pixel_norm），增加面部相似度，提升泛化能力
- `d`：分辨率倍增（pixel_shuffle），用相同计算成本翻倍分辨率，**需 32 倍数分辨率**
- `t`：深层架构，更多 downscale 层和 ResidualBlock，更强的表达能力但更慢，**需 32 倍数分辨率**
- `c`：cos 激活函数（x*cos(x) 替代 leaky_relu），平滑的非线性激活

**常见组合**：
- `df`：标准 DeepFake，最常用
- `df-ud`：DeepFake + 像素归一化 + 分辨率倍增
- `liae-ud`：LIAE + 像素归一化 + 分辨率倍增
- `df-dt`：DeepFake + 分辨率倍增 + 深层

**DFL 默认**：`liae-ud`（DFL 原版默认）。本项目默认 `df`，因为 df 通常效果更好。

---

### ae_dims（瓶颈维度）

| 属性 | 值 |
|------|-----|
| 类型 | int |
| 默认 | 256 |
| 范围 | 32~1024，步进 32 |
| DFL 对应 | ae_dims |

**含义**：Inter 中间层（瓶颈层）的通道维度。所有人脸信息将被压缩到这个维度空间中。决定模型的容量（表达能力）。

**使用方法**：
- `256`：**标准值，推荐**。足以编码大部分人脸特征
- `384`：保留更多细节，适合复杂场景，但易过拟合且训练慢
- `512`：最大容量，仅适合大数据集 + 大显卡
- `128`：轻量级，适合小显卡或快速实验

**注意事项**：
- 如果 ae_dims 不足，模型会丢失某些特征（如闭眼无法识别）
- 更大的 ae_dims 需要更多显存，Inter 的两个 Linear 层是参数量大户
- 可以通过减小 ae_dims 来适配小显卡

---

### e_dims（编码器维度）

| 属性 | 值 |
|------|-----|
| 类型 | int |
| 默认 | 64 |
| 范围 | 16~256，步进 2 |
| DFL 对应 | e_dims |

**含义**：编码器（Encoder）的基础通道维度，控制特征提取能力。

**使用方法**：
- `64`：**标准值，推荐**
- `48`：轻量级，适合小显卡
- `96`/`128`：更强表达能力，更锐利的结果，但计算量和显存增加

---

### d_dims（解码器维度）

| 属性 | 值 |
|------|-----|
| 类型 | int |
| 默认 | 64 |
| 范围 | 16~256，步进 2 |
| DFL 对应 | d_dims |

**含义**：解码器（Decoder）的基础通道维度，控制生成细节的丰富度。

**使用方法**：
- `64`：**标准值，推荐**
- `48`：轻量级，适合小显卡
- `96`/`128`：更精细的细节，但计算量和显存增加

---

### d_mask_dims（遮罩解码器维度）

| 属性 | 值 |
|------|-----|
| 类型 | int |
| 默认 | 22 |
| 范围 | 8~128，步进 2 |
| DFL 对应 | d_mask_dims |

**含义**：遮罩解码器的基础通道维度，控制遮罩的精细度。

**使用方法**：
- `22`：**DFL 默认值**，通常为 d_dims / 3
- 如果手动标注了 dst 遮罩中的障碍物（如手、头发遮挡），可以增大此参数获得更好的遮罩质量

---

### batch_size（批次大小）

| 属性 | 值 |
|------|-----|
| 类型 | int |
| 默认 | 8 |
| 范围 | 1~32 |
| DFL 对应 | batch_size |

**含义**：每次迭代处理的图像对（src + dst）数量。越大梯度越稳定，但显存占用越大。

**使用方法**：根据 GPU 显存调整。
- RTX 3050 (6GB)：2~4
- RTX 3060 (12GB)：4~8
- RTX 3090/4090 (24GB)：8~16
- A100 (40/80GB)：16~32

**注意事项**：
- 过小（1~2）：训练可能震荡，梯度噪声大
- 过大：显存溢出（OOM）
- batch_size 对显存的影响是线性的

---

## 2. 优化器与学习率参数

### adabelief（AdaBelief 优化器）

| 属性 | 值 |
|------|-----|
| 类型 | str |
| 默认 | "adabelief" |
| 可选 | adabelief / rmsprop |
| DFL 对应 | adabelief |

**含义**：选择优化器类型（均对齐 DFL leras 实现，均支持 lr_dropout 固定 mask / lr_cos / clipnorm）。

**原理**：AdaBelief 用 `(g - m_t)²` 替代 `g²` 作为二阶矩估计，对噪声梯度更鲁棒，收敛更稳定。

**使用方法**：
- `adabelief`（**推荐**）：AdaBelief，更稳定的收敛，更高的泛化能力
- `rmsprop`：RMSprop，DFL 的备选优化器（无动量，建议配合 lr_dropout=y）

**注意**：本项目不提供 Adam/AdamW——其 bias correction 和 weight decay 与 DFL 的 AE 过拟合训练目标矛盾，无理论或实验优势。

**AdaBelief vs RMSprop 的本质区别**：

| 维度 | RMSprop | AdaBelief |
|------|---------|-----------|
| 状态变量 | accumulators（1 份） | ms_dict + vs_dict（2 份） |
| 一阶动量 | 无 | m_t = β₁·m + (1-β₁)·g |
| 二阶动量 | a = ρ·a + (1-ρ)·g² | v_t = β₂·v + (1-β₂)·(g - m_t)² |
| 更新公式 | -lr·g / √(a+ε) | -lr·m_t / √(v_t+ε) |

AdaBelief 不跟踪梯度本身的平方，而是跟踪**梯度与一阶动量预测值之间的偏差平方**。当梯度方向与动量一致时偏差小→步长大；梯度方向突变时>差大→步长自动缩小。这就是"Belief"（相信预测方向）的含义。

**显存开销**：每个可训练参数需要 2 份额外状态（m 和 v），RMSprop 只需 1 份。典型 SAEHD 模型额外显存约几十到一两百 MB，对 6GB 卡影响可控。

**DFL 官方训练建议**（重要）：
1. **启用后永远不要关闭**：一旦使用 AdaBelief 就应该一直保持开启。源码层面原因——优化器状态文件不兼容：AdaBelief 保存 `[iterations, ms_dict, vs_dict]`，RMSprop 保存 `[iterations, accumulators]`。切换时 `load_weights()` 会失败→触发 `init_weights()`→优化器状态从零开始，模型失去累积的梯度方向信息，训练出现明显质量回退
2. **不需要 lr_dropout**：AdaBelief 的 `(g-m_t)²` 已经对每个参数做了自适应步长归一化——梯度稳定的参数自动获得大步长，梯度波动的参数自动获得小步长。LRD 的稀疏正则化是多余的，两者叠加反而可能导致更新过于稀疏、收敛变慢
3. **不需要在 GAN 前先跑 LRD**：关闭 random_warp 后可直接启用 GAN，AdaBelief 会确保模型更准确自然地训练到更低 loss
4. **从旧模型切换时**：如果是预训练模型或重度训练过的模型，启用 AdaBelief 时**必须同时启用 random_warp=True**，让模型重新学习一切。源码层面原因——优化器状态重置后 ms_dict 和 vs_dict 全新初始化为零值，相当于优化器"失忆"，random_warp 提供数据增强让模型在更丰富的样本分布中重新适应 AdaBelief 的更新动态。如果只启用 AB 而保持 RW 关闭，效果可能不会改善甚至变差

**注意事项**：AdaBelief 需要略多显存（额外的 momentum 状态），但通常值得。

---

### lr（学习率）

| 属性 | 值 |
|------|-----|
| 类型 | float |
| 默认 | 5e-5 |
| 范围 | 1e-6~1e-2 |
| DFL 对应 | lr（DFL 硬编码 5e-5） |

**含义**：优化器的初始学习率，控制每次参数更新的步长。

**使用方法**：
- `5e-5`：**DFL 标准值，推荐**。配合 AdaBelief 使用
- `1e-4`：稍快收敛，适合预训练阶段
- `1e-5`：精细微调，训练后期可降低到此值

**注意事项**：
- 学习率过高 → 模型走捷径（不分离身份/属性），换脸无效
- 学习率过低 → 收敛极慢，可能陷入局部最优
- DFL 的设计哲学是"保守稳定"：极小 lr + 稍大 eps + 可选梯度裁剪

---

### lr_dropout（学习率衰减）

| 属性 | 值 |
|------|-----|
| 类型 | str |
| 默认 | "n" |
| 可选 | n / y / ca |
| DFL 对应 | lr_dropout |

**含义**：学习率策略（对 SRC 和 DST 对称生效）：
- `n` = 不启用（推荐初期）
- `y` = DFL 模式：30% 参数随机冻结 + 余弦波震荡（lr_cos=500）
- `ca` = 余弦退火：关闭冻结和震荡，学习率单调下降到 1%

**使用方法**：
- `n`：不启用（训练初期推荐）
- `y`：DFL 模式，训练后期开启可帮助收敛到更精细的结果，减少 subpixel shake
- `ca`：余弦退火，T_max=target_iter（target_iter>0 时），否则 10000 步；从开启时刻起满学习率单调退火到 lr/100。建议 target_iter 设为本次训练期望总步数

**DFL 建议的启用顺序**：先启用 lr_dropout → 再关闭 random_warp → 最后启用 GAN。

**注意事项**：
- 预训练时自动关闭（等价于强制 `n`，预训练时 lr_dropout 不生效）
- 启用后会有约 20% 的训练时间增加（需要更多迭代来收敛）
- **使用 AdaBelief 时不需启用 lr_dropout**：AdaBelief 会更准确自然地训练到更低 loss，LRD 是多余的。直接关闭 random_warp 后即可启用 GAN，无需先跑 LRD

---

## 3. 数据增强参数

### random_warp（随机变形）

| 属性 | 值 |
|------|-----|
| 类型 | bool |
| 默认 | True |
| DFL 对应 | random_warp |

**含义**：对训练图像施加随机仿射变形（旋转 ±10°、缩放 ±15%、平移 ±5%）+ 随机网格弹性形变。

**使用方法**：
- `True`（**必须开启**）：防止过拟合和提升泛化能力的关键
- `False`：仅在预训练或最终微调阶段考虑关闭

**注意事项**：
- 关闭 random_warp 会导致模型快速过拟合训练数据
- DFL 建议的训练顺序：先关闭 random_warp → 再启用 GAN
- 预训练时建议手动关闭（代码透传用户设置，不强制；DFL 建议预训练时关闭）

---

### random_src_flip / random_dst_flip（随机翻转）

| 属性 | 值 |
|------|-----|
| 类型 | bool |
| 默认 | True |
| DFL 对应 | random_src_flip / random_dst_flip |

**含义**：对 src/dst 图像随机水平翻转，增强数据多样性。

**使用方法**：始终开启，除非有特殊不对称需求（如单侧发型）。

**注意事项**：预训练时自动强制开启。

---

### random_hsv_power（随机色调偏移）

| 属性 | 值 |
|------|-----|
| 类型 | float |
| 默认 | 0.0 |
| 范围 | 0.0~1.0 |
| DFL 对应 | random_hsv_power |

**含义**：随机色调（Hue）/ 饱和度（Saturation）/ 亮度（Value）偏移强度。仅对 src 数据集生效。

**原理**：
- Hue：`±max(1, int(360 * power * 0.5))` 度偏移，然后 `%180`
- Saturation/Value：`±power * 255` 加法偏移，然后 clip(0, 255)

**使用方法**：
- `0.0`：不增强（默认）
- `0.05`：**DFL 推荐的典型值**，稳定颜色扰动
- `0.1~0.3`：较强增强，需要 src 数据集足够多样化

**注意事项**：
- 过大会导致色彩失真，降低颜色迁移的质量
- 预训练时建议手动关闭（代码透传用户设置，不强制；DFL 建议预训练用 0.0）

---

### ct_mode（颜色迁移）

| 属性 | 值 |
|------|-----|
| 类型 | str |
| 默认 | "none" |
| 可选 | none / rct / rct-p / lct / mkl / idt / sot / lab-match |
| DFL 对应 | ct_mode |

**含义**：训练时将 src 颜色迁移到 dst 色彩空间，减少 src 和 dst 之间的色调差异。仅对 src 数据集生效。

**各选项详解**：
- `none`：不迁移
- `lab-match`：**推荐**，色度匹配（L 通道 30% 迁移，AB 通道协方差匹配），自然温和
- `rct`：Reinhard 颜色迁移，快速，LAB 均值/标准差匹配
- `rct-p`：部分 Reinhard，自适应混合原始和迁移结果，更温和
- `lct`：线性颜色迁移，LAB 空间 PCA 匹配协方差
- `mkl`：MKL 迁移（Monge-Kantorovitch Linear），最优传输的线性近似，效果好
- `idt`：IDT 迭代迁移（Iterative Distribution Transfer），最精确但最慢
- `sot`：切片最优传输（Sliced Optimal Transport），质量好但慢

**使用方法**：
- 建议尝试所有模式找到最适合数据的方案
- `lab-match` 和 `mkl` 通常效果最好
- 预训练时建议手动关闭（代码透传用户设置，不强制；DFL 建议预训练用 none）

---

## 4. 人脸细节参数

### masked_training（遮罩训练）

| 属性 | 值 |
|------|-----|
| 类型 | bool |
| 默认 | True |
| DFL 对应 | masked_training |

**含义**：仅在人脸遮罩区域内计算损失，遮罩外区域不参与训练。

**使用方法**：
- `True`（**推荐**）：避免背景区域干扰训练，wf 和 head 类型时可用
- `False`：全图参与训练，背景区域也学习

**注意事项**：仅 `wf` 和 `head` 类型时此参数生效。

---

### eyes_mouth_prio（眼嘴优先）

| 属性 | 值 |
|------|-----|
| 类型 | bool |
| 默认 | False |
| DFL 对应 | eyes_mouth_prio |

**含义**：对眼睛和嘴巴区域施加 300 倍的损失权重。人眼对这些区域最敏感。

**使用方法**：
- `False`：不启用（训练初期推荐）
- `True`：训练中后期开启，可显著提升观感，修复"外星人眼睛"和错误眼神方向，提升牙齿细节

**注意事项**：训练初期开启可能导致模型过度关注眼嘴而忽略整体。

---

### uniform_yaw（均匀偏航采样）

| 属性 | 值 |
|------|-----|
| 类型 | bool |
| 默认 | False |
| DFL 对应 | uniform_yaw |

**含义**：均匀采样不同偏航角（yaw，左右转头）的人脸。当数据中正脸过多、侧脸过少时启用。

**原理**：
- 将 yaw 值（归一化到 [0, 1]）分成等宽 bin（数据量 >5000 时 128 bin）
- 每个 bin 的样本权重 = 1 / bin 内样本数
- 稀有的侧脸 bin（样本少）获得更高权重 → 采样概率提升
- 使用 WeightedRandomSampler（replacement=True）实现软均匀

**使用方法**：
- `False`：普通随机采样（默认）
- `True`：均匀 yaw 采样，可显著改善侧脸换脸效果

**注意事项**：
- 预训练时自动强制开启
- 如果数据集本身 yaw 分布均匀，开启后效果不明显但无害

---

### blur_out_mask（模糊遮罩外区域）

| 属性 | 值 |
|------|-----|
| 类型 | bool |
| 默认 | False |
| DFL 对应 | blur_out_mask |

**含义**：对遮罩外区域施加高斯模糊，减少遮罩边缘的突变，改善背景过渡自然度。

**使用方法**：
- `False`：不模糊（默认）
- `True`：模糊遮罩外区域，使背景在换脸后更平滑

**注意事项**：
- 需要 src 和 dst 都有准确的 XSeg 遮罩
- 预训练时建议关闭

---

### multiscale_loss_power（多尺度损失强度）

| 属性 | 值 |
|------|-----|
| 类型 | float |
| 默认 | 0.0 |
| 范围 | 0.0~10.0 |
| DFL 对应 | 无（自研增强） |

**含义**：多尺度重建损失强度。对预测和目标下采样到 64×64 计算额外 MSE 损失。

**原理**：
- 全分辨率损失：被遮挡区域产生无意义梯度
- 低分辨率损失：脸型轮廓等低频信息仍可见，梯度有效
- 特别适合大角度侧脸和模糊脸的场景

**使用方法**：
- `0.0`：不启用（默认，对齐 DFL）
- `1.0~3.0`：**推荐范围**，增强低频信息学习
- `>5.0`：过强，可能与主损失冲突

---

### visibility_loss_power（可见性损失强度）

| 属性 | 值 |
|------|-----|
| 类型 | float |
| 默认 | 0.0 |
| 范围 | 0.0~10.0 |
| DFL 对应 | 无（自研增强） |

**含义**：可见性加权损失强度。利用元数据中 `landmarks_106_visibility` 生成可见性 mask，可见区域梯度正常，遮挡区域梯度被抑制。

**使用方法**：
- `0.0`：不启用（默认，对齐 DFL）
- `1.0~5.0`：**推荐范围**，抑制遮挡区域的无意义梯度
- `>5.0`：过强，可能过度抑制有用梯度

**注意事项**：
- 仅对人工标注了可见性的数据生效
- insightface 自动提取的数据全可见（无效果但安全）
- 正脸（全可见）无效果，大角度侧脸自动抑制遮挡区域

---

## 5. 损失函数参数

### true_face_power（真脸判别强度）

| 属性 | 值 |
|------|-----|
| 类型 | float |
| 默认 | 0.0 |
| 范围 | 0.0~10.0 |
| DFL 对应 | true_face_power |
| 架构限制 | 仅 df 架构有效 |

**含义**：真脸判别器（CodeDiscriminator）强度。判别编码向量的真假，增强身份区分能力。

**使用方法**：
- `0.0`：不启用（默认）
- `0.01`：**DFL 推荐的典型值**，轻微增强身份区分
- `0.1`：较强的身份判别

**注意事项**：
- 仅 df 架构有效，liae 架构无此参数
- 预训练时自动关闭

---

### face_style_power（人脸风格强度）

| 属性 | 值 |
|------|-----|
| 类型 | float |
| 默认 | 0.0 |
| 范围 | 0.0~100.0 |
| DFL 对应 | face_style_power |

**含义**：人脸风格损失强度（基于 style_loss / Gram 矩阵）。学习预测脸的颜色风格向 dst 靠拢。

**使用方法**：
- `0.0`：不启用（默认）
- `0.001`：**起始推荐值**，从很小开始观察效果
- `1.0~10.0`：中等强度
- `10.0~100.0`：高强度

**注意事项**：
- **仅在 10k 迭代后启用**，当预测脸足够清晰时才开始学习风格
- 使用 `whole_face` 时必须配合 XSeg 训练好的遮罩
- 启用会增加模型崩溃的风险
- 预训练时自动关闭

---

### bg_style_power（背景风格强度）

| 属性 | 值 |
|------|-----|
| 类型 | float |
| 默认 | 0.0 |
| 范围 | 0.0~100.0 |
| DFL 对应 | bg_style_power |

**含义**：背景区域风格损失强度（遮罩外区域的 DSSIM + MSE）。学习遮罩外区域向 dst 靠拢。

**使用方法**：
- `0.0`：不启用（默认）
- `2.0`：**DFL 推荐的典型值**
- `1.0~10.0`：常用范围

**注意事项**：
- 使用 `whole_face` 时必须配合 XSeg 训练好的遮罩
- 可以使脸更像 dst
- 启用会增加模型崩溃的风险
- 预训练时自动关闭

---

## 6. GAN 对抗参数

### gan_power（GAN 强度）

| 属性 | 值 |
|------|-----|
| 类型 | float |
| 默认 | 0.0 |
| 范围 | 0.0~5.0 |
| DFL 对应 | gan_power |

**含义**：GAN 对抗损失强度（UNetPatchDiscriminator），提升生成图像的真实感和细节。

**使用方法**：
- `0.0`：不启用（训练初期推荐）
- `0.1`：**DFL 推荐的典型值**，轻微增强细节
- `0.5~1.0`：较强 GAN 效果
- `>1.0`：过强，可能导致伪影

**注意事项**：
- **仅在重建稳定后启用**（如 50k 迭代后）
- 启用后不要关闭
- 预训练时自动关闭
- **仅对 SRC 重建生效，DST 不参与 GAN 对抗**
- **启用时自动添加两项正则**（仅对 SRC）：TV 正则（1e-6）+ anti_mask MSE（0.02），用于抑制伪影
- **启用顺序取决于优化器**：
  - 使用 AdaBelief（推荐）：random_warp(False) → 直接 gan_power > 0，**不需要 lr_dropout**
  - 不使用 AdaBelief：lr_dropout(y) → random_warp(False) → gan_power > 0

---

### gan_patch_size（GAN 块大小）

| 属性 | 值 |
|------|-----|
| 类型 | int |
| 默认 | 16 |
| 范围 | 3~640 |
| DFL 对应 | gan_patch_size（DFL 默认 resolution // 8） |

**含义**：GAN 判别器的 Patch 大小。`find_archi()` 自动匹配最接近感受野的网络配置。

**使用方法**：
- `16`：**标准值**，适合 256 分辨率
- `32`：更大感受野，更高质量但更多显存
- 典型值：`resolution // 8`

---

### gan_dims（GAN 维度）

| 属性 | 值 |
|------|-----|
| 类型 | int |
| 默认 | 16 |
| 范围 | 4~64 |
| DFL 对应 | gan_dims |

**含义**：GAN 判别器的基础通道维度。

**使用方法**：
- `16`：**标准值，推荐**
- `32`/`64`：更大容量，更高质量但更多显存

---

## 7. 训练控制参数

### pretrain（预训练模式）

| 属性 | 值 |
|------|-----|
| 类型 | bool |
| 默认 | False |
| DFL 对应 | pretrain |

**含义**：使用预训练数据（大量不同人脸）进行预训练，学习通用人脸表征。

**预训练时自动设置**（代码强制，与 GUI 值无关）：
- `uniform_yaw = True`（强制开启均匀偏航采样）
- `random_src_flip = True`，`random_dst_flip = True`（强制开启翻转）
- `gan_power = 0.0`（强制关闭 GAN）
- `true_face_power = 0.0`（仅 df 架构，强制关闭）
- `face_style_power = 0.0`，`bg_style_power = 0.0`（强制关闭风格损失）
- `lr_dropout` 不生效（优化器按"未启用"处理，等价于 `n`）
- DST 数据集强制 = SRC 数据集（自重建），DST 侧强制 `ct_mode='none'`、`random_hsv_power=0.0`

**未强制、透传用户设置**（DFL 建议预训练时手动调整）：
- `random_warp`：透传用户值，DFL 建议预训练时关闭
- `random_hsv_power`：透传用户值，DFL 建议 0.0
- `ct_mode`：透传用户值，DFL 建议 none

**预训练数据源**：`workspace/pretrain_faces/faceset.pak`（名称锁定，训练开始时自动解压到同目录；该 pak 不存在时自动使用 src/dst 数据兜底预训练）

**使用方法**：
1. 在 step8（工具）中将对齐人脸目录打包为 `workspace/pretrain_faces/faceset.pak`（名称锁定，不可自定义）
2. 开启 `pretrain = True` 开始预训练
3. 预训练 50k~200k 迭代后关闭
4. **关闭 pretrain 时，Inter 层权重会自动重新初始化**（对齐 DFL），encoder 和 decoder 权重保留
5. 继续正常训练特定人物

**注意事项**：
- 预训练后模型有更好的初始化，可加速后续特定人物训练
- 预训练数据需要大量不同人脸（建议 >10000 张）
- **从预训练/旧模型切换到 AdaBelief 时**：必须同时启用 random_warp=True，让模型重新学习一切。如果只启用 AdaBelief 而保持 random_warp=False，效果可能不会改善甚至变差

---

### target_iter（目标迭代数）

| 属性 | 值 |
|------|-----|
| 类型 | int |
| 默认 | 0 |
| 范围 | 0~9999999 |
| DFL 对应 | target_iter |

**含义**：训练目标迭代次数。

**使用方法**：
- `0`：无限训练（手动停止）
- `100000~500000`：推荐范围（视数据量而定）

---

### backup_interval（保存间隔）

| 属性 | 值 |
|------|-----|
| 类型 | int |
| 默认 | 0 |
| 范围 | 0~1000000 |
| DFL 对应 | backup_interval |

**含义**：每隔多少次迭代自动保存模型备份。

**使用方法**：
- `0`：不自动保存（仅训练结束时保存）
- `10000~50000`：推荐范围

---

## 8. 数值稳定性与显存参数

### amp_mode（混合精度）

| 属性 | 值 |
|------|-----|
| 类型 | str |
| 默认 | "fp16" |
| 可选 | fp32 / fp16 / bf16 |
| DFL 对应 | 无（DFL 训练时 use_fp16=False，全 fp32） |

**含义**：混合精度训练模式，影响显存占用和计算速度。

**各选项详解**：
- `fp32`：全精度 float32，最稳定但最慢、最耗显存
- `fp16`：半精度 float16 + GradScaler，**推荐**，省显存加速，数值稳定性好
- `bf16`：BF16 半精度，7 位尾数精度不足，**易导致梯度爆炸，不推荐**

**注意事项**：
- 本项目 AMP 是自研功能，DFL 训练时全用 fp32
- fp16 下 Inter 层的 dense 计算强制 fp32（对齐 DFL 的数值稳定性）
- bf16 + clipgrad=False 时会自动强制开启 clipgrad=True 并警告

---

### clipgrad（梯度裁剪）

| 属性 | 值 |
|------|-----|
| 类型 | bool |
| 默认 | False |
| DFL 对应 | clipgrad |

**含义**：启用梯度全局 L2 范数裁剪（上限 1.0），防止梯度爆炸。

**使用方法**：
- `False`：不裁剪（默认，DFL 默认也是 False）
- `True`：裁剪梯度范数到 1.0，减少模型崩溃风险，但牺牲训练速度

**注意事项**：
- bf16 模式下 clipgrad=False 时会自动强制开启并警告
- 梯度裁剪会降低训练速度，仅在训练不稳定时启用

---

### enable_torch_compile（torch.compile 加速）

| 属性 | 值 |
|------|-----|
| 类型 | bool |
| 默认 | False |
| DFL 对应 | 无（自研功能） |

**含义**：启用 `torch.compile` 编译加速（PyTorch ≥ 2.0）。

**使用方法**：
- `False`：**默认关闭**
- `True`：启用编译加速

**注意事项**：
- 6GB 显存下 `aot_eager` 后端只做 AOT 编译不做算子融合，**无实际加速**
- 8GB+ 显存可用 `inductor` 后端获得真正加速（需 `PYTHONUTF8=1` 环境变量）
- 首次迭代触发编译（约 30~120 秒），编译失败自动降级为 eager 模式
- Windows 上有 GBK 编码 bug，需要 `PYTHONUTF8=1`

---

### auto_batch（自动批次探测）

| 属性 | 值 |
|------|-----|
| 类型 | bool |
| 默认 | False |
| DFL 对应 | 无（自研功能） |

**含义**：启用后自动探测当前显卡最大不 OOM 的 batch size。探测结果**不会超过**手动设置的批次大小。

**使用方法**：
- `False`：**默认关闭**，使用手动 batch_size
- `True`：每次训练开始时探测一次（约 10~30 秒），自动取"手动值"与"探测值"中较小者

**注意事项**：
- 探测在每次训练开始时进行，不影响后续迭代速度
- 显存吃紧时开启，可避免手动反复试错 OOM

---

### gradient_accumulation_steps（梯度累积步数）

| 属性 | 值 |
|------|-----|
| 类型 | int |
| 默认 | 1 |
| 范围 | 1~16 |
| DFL 对应 | 无（自研功能） |

**含义**：梯度累积步数。每 n 步才更新一次参数，等效 batch_size × n。

**使用方法**：
- `1`：不累积（默认）
- `2~8`：显存不够但想用大 batch 等效效果时开启

**注意事项**：
- 等效增大 batch size，梯度更稳定，但单次参数更新间隔变长
- 不增加显存占用（相比直接加大 batch_size）

---

## 9. 分阶段训练详解

DFL SAEHD 的训练是分阶段的，每个阶段有不同的参数配置和目标。训练路径取决于**架构选择（DF vs LIAE）**和**优化器选择（AdaBelief vs RMSprop）**。

### 9.1 架构选择：DF vs LIAE

| 维度 | DF（双解码器） | LIAE（共享解码器） |
|------|---------------|-------------------|
| Inter 数量 | 1 个（共享） | 2 个（inter_AB + inter_B） |
| Decoder 数量 | 2 个（decoder_src + decoder_dst） | 1 个（共享） |
| 身份分离机制 | 隐式，通过双解码器 | 显式，inter_AB/inter_B |
| 颜色一致性 | **差**，需 ct_mode | **好**，通常不需 ct_mode |
| true_face | ✓ 支持 | ✗ 不支持 |
| 脸型差异大 | 不适合 | **适合** |
| 换脸相似度 | **更高**（保留更多身份） | 稍低 |
| random_warp=False 时 | 无特殊行为 | **自动冻结 inter_AB** |

**选择建议**：
- **DF**：绝大多数场景推荐。换脸更像源脸，但需要 ct_mode 解决色斑
- **LIAE**：src 和 dst 脸型差异大时选择。颜色一致性更好，但换脸可能不够像

### 9.2 训练路径选择：AdaBelief vs RMSprop

| 维度 | AdaBelief（推荐） | RMSprop |
|------|-------------------|---------|
| 自适应步长 | ✓ (g-m_t)² 对噪声鲁棒 | ✗ g² 对噪声敏感 |
| 需要 LRD | **不需要** | 需要（GAN 前先开 LRD） |
| 收敛质量 | 更准确自然到更低 loss | 需要更多手动调参 |
| 显存开销 | 2 份状态变量 | 1 份状态变量 |
| 切换兼容性 | 启用后不可切回 RMSprop | — |

---

### 9.3 完整训练路线图

#### 路径 A：AdaBelief + DF（推荐组合）

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    AdaBelief + DF 完整训练路线图                          │
├──────────┬──────────────┬──────────────┬──────────────┬─────────────────┤
│  阶段0   │   阶段1      │   阶段2      │   阶段3      │   阶段4         │
│  预训练  │ 颜色+结构    │ 身份对齐     │ 眼嘴+关warp  │  GAN纹理增强    │
│ (可选)   │              │ (true_face)  │              │                 │
├──────────┼──────────────┼──────────────┼──────────────┼─────────────────┤
│ pretrain │ warp  ✓      │ warp  ✓      │ warp  ✗      │ warp  ✗        │
│ warp ✗   │ gan   0      │ gan   0      │ gan   0      │ gan   0.1      │
│ gan  0   │ em    ✗      │ em    ✗      │ em    ✓      │ em    ✓        │
│          │ tf    0      │ tf 0.01~0.05 │ tf   保持    │ tf   保持      │
│          │ ct  rct/mkl  │ ct  保持     │ ct  保持     │ ct  保持       │
│          │ lrd   n      │ lrd  n       │ lrd  n       │ lrd  n         │
│          │ AB   ✓       │ AB   ✓       │ AB   ✓       │ AB   ✓         │
├──────────┼──────────────┼──────────────┼──────────────┼─────────────────┤
│ 50k-200k │ 50k-150k     │ 20k-50k      │ 10k-30k      │ 50k-150k       │
└──────────┴──────────────┴──────────────┴──────────────┴─────────────────┘
 关键：AdaBelief 不需要 LRD！直接 warp(✓→✗) → GAN，无需中间 LRD 阶段
```

#### 路径 B：AdaBelief + LIAE

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    AdaBelief + LIAE 完整训练路线图                        │
├──────────┬──────────────┬──────────────┬─────────────────┐
│  阶段0   │   阶段1      │   阶段2      │   阶段3         │
│  预训练  │ 结构学习     │ 眼嘴+关warp  │  GAN纹理增强    │
│ (可选)   │              │              │                 │
├──────────┼──────────────┼──────────────┼─────────────────┤
│ pretrain │ warp  ✓      │ warp  ✗      │ warp  ✗        │
│ warp ✗   │ gan   0      │ gan   0      │ gan   0.1      │
│ gan  0   │ em    ✗      │ em    ✓      │ em    ✓        │
│          │ ct  none     │ ct  none     │ ct  none       │
│          │ lrd   n      │ lrd  n       │ lrd  n         │
│          │ inter_AB 训练│ inter_AB冻结 │ inter_AB冻结   │
├──────────┼──────────────┼──────────────┼─────────────────┤
│ 50k-200k │ 50k-150k     │ 10k-30k      │ 50k-150k       │
└──────────┴──────────────┴──────────────┴─────────────────┘
 关键：LIAE 无 true_face；关 warp 时自动冻结 inter_AB；通常不需 ct_mode
```

#### 路径 C：RMSprop（传统路径，需 LRD）

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    RMSprop 传统训练路线图                                 │
├──────────┬──────────┬──────────┬──────────┬──────────┬─────────────────┤
│  阶段0   │  阶段1   │  阶段2   │  阶段3   │  阶段4   │  阶段5         │
│  预训练  │ 结构学习 │ 眼嘴精修 │ LRD收敛  │ 关warp   │  GAN增强       │
├──────────┼──────────┼──────────┼──────────┼──────────┼─────────────────┤
│ warp ✗   │ warp ✓   │ warp ✓   │ warp ✓   │ warp ✗   │ warp ✗        │
│ gan  0   │ gan  0   │ gan  0   │ gan  0   │ gan  0   │ gan  0.1      │
│          │ em   ✗   │ em   ✓   │ em   ✓   │ em   ✓   │ em   ✓        │
│          │ lrd  n   │ lrd  n   │ lrd  y   │ lrd  y   │ lrd  y         │
├──────────┼──────────┼──────────┼──────────┼──────────┼─────────────────┤
│ 50k-200k │ 50k-100k │ 30k-60k  │ 20k-50k  │ 10k-20k  │ 50k-150k       │
└──────────┴──────────┴──────────┴──────────┴──────────┴─────────────────┘
 关键：必须先 LRD(y) → 再 warp(✗) → 最后 GAN，三步不能跳过
```

---

### 9.4 各阶段详细说明

#### 阶段 0：预训练（可选但推荐）

**目标**：用大量不同人脸训练通用人脸表征，为后续特定人物训练提供更好的初始化。

**前提条件**：准备预训练数据（>10000 张不同人脸的 aligned 图像）放入 `workspace/model/pretrain_faces/`。

**参数配置**：

| 参数 | 值 | 说明 |
|------|-----|------|
| pretrain | True | 启用预训练模式 |
| target_iter | 50000~200000 | 预训练迭代数 |
| adabelief | True | 从预训练开始就用 AdaBelief |
| batch_size | 尽量大 | 预训练不需要精细，batch 可大 |

**自动设置**（代码强制）：uniform_yaw=True、src/dst_flip=True、gan_power=0、true_face=0（仅 df）、face_style_power=0、bg_style_power=0、lr_dropout 无效化；DST=SRC 自重建。
**建议手动**：random_warp=False、random_hsv_power=0、ct_mode=none（代码透传用户设置，不强制）

**切换到阶段 1**：关闭 pretrain。Inter 层权重自动重新初始化（DF 重置 inter，LIAE 重置 inter_AB + inter_B），encoder/decoder 权重保留。迭代计数器重置为 0。

---

#### 阶段 1：基础重建（颜色 + 结构学习）

**目标**：模型学习基本的身份/属性分离和重建能力。loss 从高位快速下降。DF 架构同时通过 ct_mode 让 decoder_src 学习在 dst 色域下工作。

**参数配置**：

| 参数 | DF | LIAE | 说明 |
|------|-----|------|------|
| random_warp | True | True | **必须开启**，核心正则化 |
| random_src_flip | True | True | 数据增强 |
| random_dst_flip | True | True | 数据增强 |
| random_hsv_power | 0.0~0.05 | 0.0 | DF 可稍开，LIAE 不需要 |
| ct_mode | rct 或 mkl | none | **DF 从第一天就开！** LIAE 通常不需要 |
| adabelief | True | True | 保持 AdaBelief |
| lr_dropout | n | n | 不启用 |
| gan_power | 0.0 | 0.0 | 不启用 |
| face_style_power | 0.0 | 0.0 | 不启用 |
| bg_style_power | 0.0 | 0.0 | 不启用 |
| true_face_power | 0.0 | 0.0 | 不启用（仅 DF 有此参数） |
| eyes_mouth_prio | False | False | 结构都没学好，300× 权重会让模型走捷径 |
| uniform_yaw | 视数据而定 | 视数据而定 | 侧脸少则 True |
| masked_training | True | True | 遮罩内训练 |
| blur_out_mask | False | False | 初期不启用 |
| multiscale_loss_power | 0.0~2.0 | 0.0~2.0 | 可选，改善侧脸 |
| visibility_loss_power | 0.0~2.0 | 0.0~2.0 | 可选，抑制遮挡区域 |

**持续时间**：约 50k~150k 迭代（视数据量而定）。

**退出标志**：
- src_loss 和 dst_loss 持续下降并趋于平稳
- 预览图中 src→src 和 dst→dst 重建基本清晰可辨
- DF：颜色/色调基本一致（ct_mode 效果已显现）
- **判断标准**：loss 不再显著下降（如 1000 次迭代内改善 <1%）

**为什么这么配置**：
- `random_warp=True` 是最重要的参数！同一人的数据缺乏多样性，必须靠 warp 强制模型学习"人脸结构"而非"像素位置"
- `eyes_mouth_prio=False`：模型结构都没学好，300× 的眼嘴权重只会让模型走捷径
- DF 的 `ct_mode` 从阶段1就开：如果关了 ct_mode 训 DF，后期会出现严重色斑，几乎无法修复

---

#### 阶段 2：身份对齐（仅 DF）/ 眼嘴精修

**DF 架构 — 阶段 2：身份对齐（true_face）**

**目标**：通过 CodeDiscriminator 让 src 和 dst 的中间编码分布对齐，改善换脸质量。

| 参数 | 变更 | 说明 |
|------|------|------|
| true_face_power | 0.0 → 0.01 | 从 0.01 开始，观察后可升到 0.03~0.05 |
| 其他参数 | 保持 | 不变 |

**持续到何时**：
- 预览中换脸质量（pred_src_dst）显著改善
- D_code_loss 波动趋于稳定
- 典型迭代数：**20k~50k**

**注意事项**：
- true_face 是对抗性训练，不能和 eyes_mouth_prio 同时引入
- true_face_power **宁低勿高**，0.01 的效果可能已经很好
- D_code_loss 突然上升 → true_face_power 太高，降低一半

**LIAE 架构 — 阶段 2：眼嘴细节精修**

**目标**：精细修复眼嘴等关键区域（LIAE 无 true_face，直接进入眼嘴精修）。

|; | 参数 | 变更 | 说明 |
|------|------|------|
| eyes_mouth_prio | False → True | 300× 眼嘴 L1 权重 |
| 其他参数 | 保持 | 不变 |

**持续到何时**：
- 预览中眼睛清晰度、嘴巴形状明显改善
- 典型迭代数：**30k~60k**

---

#### 阶段 3：关闭 Random Warp（精准映射）

**目标**：学习精确的像素级映射，获得清晰细节。

**AdaBelief 路径**：直接关闭 random_warp，**不需要先开 LRD**。

**RMSprop 路径**：必须先开 LRD（阶段 2.5），等稳定后再关 warp。

| 参数 | AdaBelief | RMSprop | 说明 |
|------|-----------|---------|------|
| random_warp | True → False | True → False | **核心变化** |
| eyes_mouth_prio | True | True | 保持 |
| lr_dropout | n（不变） | y（已在阶段2.5开启） | AdaBelief 不需要 LRD |
| lr | 5e-5 或降至 3e-5 | 5e-5 | 可选降低学习率 |

**LIAE 特殊行为**：关闭 random_warp 时**自动冻结 inter_AB**（对齐 DFL）。inter_AB 不再接收梯度更新，其"身份理解"被锁定，只微调解码器。

**持续到何时**：
- 预览图像清晰度大幅提升
- **关键判断**：出现过拟合迹象（牙齿过度锐化、人工伪影）时立即停止
- 典型迭代数：**10k~30k**（不宜过长）

**最大坑位**：这个阶段不是越长越好！关闭 warp 后 loss 会持续下降——因为模型在过拟合训练数据。**判断标准是预览质量，不是 loss 数值！**

---

#### 阶段 4：GAN 纹理增强

**目标**：增加真实皮肤纹理、毛孔、光影细节。

| 参数 | 变更 | 说明 |
|------|------|------|
| gan_power | 0.0 → 0.1 | **DFL 推荐典型值** |
| gan_patch_size | 16~32 | resolution // 8 |
| gan_dims | 16 | 标准 |
| lr | 可降至 1e-5 | 进一步降低学习率 |

**持续到何时**：
- 预览皮肤纹理变得自然真实
- GAN loss 不再有有意义的变化（GAN 训练本质是博弈，loss 会来回波动）
- 出现伪影/奇怪纹理时降低 gan_power 或停止
- 典型迭代数：**50k~150k**

**注意事项**：
- GAN 启用后**不要关闭**
- GAN loss 不降反升是正常的！G 和 D 博弈决定了 loss 会来回波动
- 如果出现伪影，降低 gan_power
- GAN 需要额外显存，小显卡可能需要减小 batch_size

---

#### 阶段 5（可选）：风格损失

**目标**：学习预测脸的颜色风格向 dst 靠拢（face_style），背景区域向 dst 靠拢（bg_style）。

| 参数 | 变更 | 说明 |
|------|------|------|
| face_style_power | 0.0 → 0.001 | **从极小开始**，观察预览 |
| bg_style_power | 0.0 → 2.0 | DFL 推荐典型值 |

**注意事项**：
- **仅在 10k 迭代后启用**，当预测脸足够清晰时才开始学习风格
- 使用 whole_face 时必须配合 XSeg 训练好的遮罩
- 启用会增加模型崩溃的风险，出现伪影立即关闭

---

### 9.5 阶段切换判断信号

| 切换 | 信号 | 不要被误导 |
|------|------|-----------|
| 0→1 | 预训练 loss 稳定 | — |
| 1→2 | loss 进入缓慢下降，预览五官可辨 | 不需要等到 loss 完全不动 |
| 2→3 | 眼嘴细节清晰，整体重建良好（DF: true_face 稳定） | 不要追求 loss 最低点 |
| 3→4 | 清晰度足够，无伪影 | **最大坑：loss 还在降但已过拟合！看预览不要看 loss** |
| 4→5 | GAN 纹理稳定 | GAN loss 波动是正常的 |

### 9.6 推荐阶段迭代数

| 数据规模 | 阶段1 | 阶段2 | 阶段3 | 阶段4 | 总迭代 | 说明 |
|---------|-------|-------|-------|-------|--------|------|
| 小 (<1000) | 30k | 20k | 10k | 50k | 150k~200k | 快速收敛 |
| 中 (1k~5k) | 50k | 30k | 15k | 80k | 250k~350k | **标准** |
| 大 (>5k) | 80k | 40k | 20k | 120k | 350k~500k | 充分训练 |
| 超大 (>10k) | 100k | 50k | 30k | 200k | 500k~1000k | 最大质量 |

### 9.7 最重要的提醒

1. **AdaBelief 启用后永远不要关闭**：切换到 RMSprop 会丢失优化器状态，训练质量回退
2. **AdaBelief 不需要 LRD**：直接 warp(✓→✗) → GAN，简化训练流程
3. **从旧模型切换 AdaBelief 时必须同时开 random_warp**：让模型重新学习一切
4. **阶段3（关 warp）是过拟合重灾区**：1-2 万次迭代就能产生极低 loss，但预览可能已经崩了。**看预览，别看数字**
5. **GAN 阶段 loss 不降反升是正常的**：G 和 D 博弈本质决定了 loss 会来回波动
6. **DF 必须开 ct_mode**：DF 的双解码器天然存在颜色不一致，不开 ct_mode 几乎必然出现色斑
7. **LIAE 通常不需要 ct_mode**：LIAE 的共享解码器机制使颜色一致性更好
8. **LIAE 关 warp 时自动冻结 inter_AB**：对齐 DFL 行为，无需手动设置
9. **预训练切换时 Inter 层重新初始化**：DF 重置 inter，LIAE 重置 inter_AB + inter_B，encoder/decoder 保留

---

## 10. 显卡配置推荐

### 小显卡（6GB，如 RTX 3050）

```json
{
    "resolution": 256,
    "face_type": "wf",
    "batch_size": 2,
    "archi": "df",
    "ae_dims": 224,
    "e_dims": 48,
    "d_dims": 48,
    "d_mask_dims": 16,
    "amp_mode": "fp16",
    "clipgrad": false,
    "enable_torch_compile": false,
    "gradient_accumulation_steps": 1,
    "gan_power": 0.0,
    "gan_dims": 16,
    "gan_patch_size": 16,
    "target_iter": 250000,
    "backup_interval": 10000
}
```

**说明**：
- batch_size=2 是 6GB 显存的安全值
- 减小 ae_dims/e_dims/d_dims 以适配显存
- fp16 混合精度必须开启
- torch.compile 关闭（无加速）
- GAN 需要额外显存，可能需要 batch_size=1

---

### 中显卡（12GB，如 RTX 3060）

```json
{
    "resolution": 256,
    "face_type": "wf",
    "batch_size": 4,
    "archi": "df",
    "ae_dims": 256,
    "e_dims": 64,
    "d_dims": 64,
    "d_mask_dims": 22,
    "amp_mode": "fp16",
    "clipgrad": false,
    "enable_torch_compile": false,
    "gradient_accumulation_steps": 1,
    "gan_power": 0.0,
    "gan_dims": 16,
    "gan_patch_size": 16,
    "target_iter": 300000,
    "backup_interval": 15000
}
```

**说明**：
- 标准参数配置
- GAN 阶段可保持 batch_size=4

---

### 大显卡（24GB，如 RTX 3090/4090）

```json
{
    "resolution": 256,
    "face_type": "wf",
    "batch_size": 8,
    "archi": "df",
    "ae_dims": 256,
    "e_dims": 64,
    "d_dims": 64,
    "d_mask_dims": 22,
    "amp_mode": "fp16",
    "clipgrad": false,
    "enable_torch_compile": true,
    "gradient_accumulation_steps": 1,
    "gan_power": 0.0,
    "gan_dims": 16,
    "gan_patch_size": 32,
    "target_iter": 500000,
    "backup_interval": 20000
}
```

**说明**：
- 标准参数，batch_size 充裕
- 可启用 torch.compile（inductor 后端）获得加速
- GAN 阶段可使用更大 gan_patch_size

---

### 高质量配置（24GB+，追求最佳质量）

```json
{
    "resolution": 384,
    "face_type": "head",
    "batch_size": 4,
    "archi": "df-ud",
    "ae_dims": 384,
    "e_dims": 64,
    "d_dims": 64,
    "d_mask_dims": 22,
    "amp_mode": "fp16",
    "clipgrad": false,
    "enable_torch_compile": true,
    "gradient_accumulation_steps": 1,
    "gan_power": 0.0,
    "gan_dims": 32,
    "gan_patch_size": 32,
    "target_iter": 500000,
    "backup_interval": 20000
}
```

**说明**：
- head 类型 + 384 分辨率，含头发融合
- df-ud 架构，像素归一化 + 分辨率倍增
- 更大 ae_dims 保留更多细节
- 需要 XSeg 遮罩

---

### 超大显卡（40GB+，如 A100）

```json
{
    "resolution": 512,
    "face_type": "head",
    "batch_size": 8,
    "archi": "df-udt",
    "ae_dims": 512,
    "e_dims": 96,
    "d_dims": 96,
    "d_mask_dims": 32,
    "amp_mode": "fp16",
    "clipgrad": false,
    "enable_torch_compile": true,
    "gradient_accumulation_steps": 1,
    "gan_power": 0.0,
    "gan_dims": 32,
    "gan_patch_size": 32,
    "target_iter": 1000000,
    "backup_interval": 50000
}
```

**说明**：
- 512 分辨率，极高细节
- df-udt 深层架构
- 最大模型容量

---

## 11. 通用训练建议

### 数据准备

1. **src 数据集**（源人脸，要换上去的脸）：
   - 至少 500~5000 张不同角度、表情的 aligned 人脸
   - 光照条件多样
   - 质量越高越好，避免模糊、遮挡
   - 建议用 insightface 自动提取 + 对齐

2. **dst 数据集**（目标人脸，要被替换的脸）：
   - 包含视频中所有需要替换的帧的人脸
   - 角度和表情覆盖视频中的所有情况
   - 建议用 insightface 自动提取 + 对齐

3. **XSeg 遮罩**（可选但推荐）：
   - wf/head 类型建议训练 XSeg 遮罩
   - 手动标注障碍物（手、头发遮挡等）可提升遮罩质量

### 训练流程

1. **（可选）预训练**：用大量不同人脸预训练 50k~200k 迭代
2. **基础训练**：正常训练 50k~100k 迭代，直到重建清晰
3. **精细收敛**：lr_dropout(y) → random_warp(False) → eyes_mouth_prio(True)
4. **最终微调**：GAN(0.1) → true_face(0.01) → style(可选)
5. **导出模型**：训练满意后导出用于换脸

### 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 换脸无效（看起来像dst） | 学习率过高，模型走捷径 | 降低 lr 到 5e-5 |
| 训练崩溃/loss爆炸 | 梯度爆炸 | 启用 clipgrad=True，或改用 fp32 |
| 侧脸效果差 | 侧脸数据不足 | 启用 uniform_yaw=True |
| 眼睛不自然 | 眼部训练不足 | 启用 eyes_mouth_prio=True |
| 颜色不匹配 | src/dst 色调差异 | 设置 ct_mode=lab-match 或 mkl |
| 背景过渡不自然 | 遮罩边缘突变 | 启用 blur_out_mask=True |
| 显存不足(OOM) | 模型太大 | 减小 batch_size / ae_dims / e_dims / d_dims |
| 训练太慢 | 分辨率过高或未用AMP | 启用 amp_mode=fp16，降低分辨率 |

### DFL 对齐说明

本项目是 DFL SAEHD 的 PyTorch 原生重新实现，以下方面与 DFL 完全对齐：

- **优化器**：AdaBelief / RMSprop，eps 使用 `np.finfo(dtype).resolution`（float32: 1e-6, float16: 1e-3）
- **学习率**：默认 5e-5，与 DFL 一致
- **架构**：Encoder / Inter / Decoder 完全对齐，支持 df / liae + u/d/t/c 后缀
- **损失函数**：DSSIM + MSE + style_loss + GAN + true_face，完全对齐
- **数据增强**：random_warp / random_hsv / ct_mode / random_flip，完全对齐
- **预训练**：pretrain 模式参数自动设置，与 DFL 一致
- **pretrain 切换**：关闭 pretrain 时 Inter 层重新初始化，与 DFL 一致
- **LIAE 冻结**：LIAE + random_warp=False 时冻结 inter_AB，与 DFL 一致

**自研增强**（DFL 没有）：
- `amp_mode`：混合精度训练（fp16/fp32/bf16）
- `multiscale_loss_power`：多尺度重建损失
- `visibility_loss_power`：可见性加权损失
- `enable_torch_compile`：torch.compile 加速
- `auto_batch`：自动批次探测（约 10~30 秒，结果不超过手动 batch_size）
- `gradient_accumulation_steps`：梯度累积（等效 batch_size×n）
- `lr_dropout='ca'`：余弦退火学习率策略
- yaw 感知动态眉毛扩展系数
