# LIAE 架构深度分析与分阶段训练指南

> 基于 `D:\AI\Inswapper\faceswap\models\saehd\` 源码深度分析
>
> 生成日期：2026-08-06

---

## 目录

1. [LIAE 架构核心剖析](#一liae-架构核心剖析)
   - [数据流详解](#11-数据流详解)
   - [三个关键路径](#12-三个关键路径)
   - [各模块语义角色](#13-各模块的语义角色)
2. [关于"删 inter_AB"的真相](#二关于删-inter_ab-的真相)
3. [所有训练参数深度解析](#三所有训练参数深度解析)
   - [架构参数](#31-架构参数一次性设定不可改)
   - [训练控制参数](#32-训练控制参数可分阶段调整)
   - [损失函数参数](#33-损失函数参数)
   - [冻结参数](#34-冻结参数)
   - [损失函数详细组成](#35-损失函数详细组成从源码解析)
4. [LIAE 分阶段训练方案（核心）](#四liae-分阶段训练方案核心)
   - [阶段1：结构学习](#阶段1结构学习-structure-learning)
   - [阶段2：面部细节精修](#阶段2面部细节精修-detail-refinement)
   - [阶段3：LRD收敛-开启学习率丢弃](#阶段3lrd收敛--开启学习率丢弃-lr-dropout)
   - [阶段4：精准映射-关闭 Random Warp](#阶段4精修阶段--关闭-random-warp-fine-texture)
   - [阶段5：GAN 纹理增强](#阶段5gan-纹理增强-gan-texture-enhancement)
   - [阶段6：冻结身份微调-可选](#阶段6可选冻结身份--纯净学习-identity-freeze-fine-tune)
5. [架构修饰符 u/d/t/c 深度分析](#五架构修饰符-udtc-深度分析)
   - [u：像素归一化](#51-u--像素归一化-pixel-normalization)
   - [d：密集输出连接](#52-d--密集输出连接-dense-output-connections)
   - [t：更深网络](#53-t--更深网络-transformer-like-deeper)
   - [c：余弦激活](#54-c--余弦激活-cosine-activation)
   - [参数量与显存对比](#55-各变体参数量和计算量对比)
6. [各变体完整训练方案](#六各变体完整训练方案)
   - [方案 A：liae-ud（推荐标准方案）](#方案-aliae-ud推荐标准方案)
   - [方案 B：liae-u（省显存方案）](#方案-bliae-u省显存方案)
   - [方案 C：liae-d（不推荐）](#方案-cliae-d不推荐)
   - [方案 D：liae-udt（高分辨率推荐）](#方案-dliae-udt高分辨率推荐)
   - [方案 E：liae-ut / liae-dt（中间方案）](#方案-eliae-ut--liae-dt中间方案)
7. [架构选择决策树](#七架构选择决策树)
8. [总对比表与核心总结](#八总对比表与核心总结)

---

## 一、LIAE 架构核心剖析

### 1.1 数据流详解

```
                     ┌──────────────────────────────────┐
  warped_src ──────► │         Encoder (共享)            │
  warped_dst ──────► │   e_dims=64, 4/5次下采样          │──► [B, e_ch*8, 4×4或8×8]
                     └──────────────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      │
     ┌────────────────┐    ┌────────────────┐               │
     │   inter_AB     │    │   inter_B      │               │
     │ ae_dims=256    │    │ ae_dims=256    │               │
     │ out=ae*2=512   │    │ out=ae*2=512   │               │
     └───────┬────────┘    └───────┬────────┘               │
             │                     │                         │
    ┌────────┴────────┐   ┌───────┴────────┐               │
    ▼                 ▼   ▼                ▼               │
 [inter_AB, inter_AB]    [inter_B, inter_AB]               │
   src_code_cat           dst_code_cat                      │
    4×ae_dims ch           4×ae_dims ch                     │
         │                      │                           │
         ├──────────────────────┤                           │
         ▼                      ▼                           │
  ┌────────────────────────────────────────┐               │
  │           Decoder (共享)                │               │
  │    d_dims=64, d_mask_dims=22           │               │
  │    ├─ Image Branch → pred_src/dst      │               │
  │    └─ Mask Branch  → pred_mask         │               │
  └────────────────────────────────────────┘               │
```

### 1.2 三个关键路径

```python
# 路径1: src → src (重建源脸)
encoder(src) → inter_AB → [inter_AB, inter_AB] → decoder → pred_src_src

# 路径2: dst → dst (重建目标脸)
encoder(dst) → inter_B → [inter_B, inter_AB] → decoder → pred_dst_dst

# 路径3: dst → src (换脸!)
encoder(dst) → inter_AB → [inter_AB, inter_AB] → decoder → pred_src_dst
```

### 1.3 各模块的语义角色

| 模块 | 输入 | 作用 | 类比 |
|------|------|------|------|
| **encoder** | src/dst图像 | 共享特征提取，将人脸压缩到潜空间 | 通用人脸理解 |
| **inter_AB** | encoder输出 | **通用特征提取器**，提取身份+属性（表情/角度等），其输出在decoder position 1时触发"源脸生成模式" | 源脸生成触发器 |
| **inter_B** | encoder输出 | **目标环境适配器**，提取目标特定的颜色/光照等环境信息，其输出在decoder position 1时触发"目标脸生成模式" | 目标脸生成触发器 |
| **decoder** | [position1, position2]拼接 | **重建/生成**，通过position1的来源(AB vs B)决定生成"源脸"还是"目标脸" | 画家 |

**核心理解（从forward逻辑精确推导）**：
- `pred_src_src = decoder([AB(src), AB(src)])`：position1=AB → 源脸模式 → src重建
- `pred_dst_dst = decoder([B(dst), AB(dst)])`：position1=B → 目标脸模式 → dst重建
- `pred_src_dst = decoder([AB(dst), AB(dst)])`：position1=AB → 源脸模式，但内容来自dst → **换脸结果**

**关键发现**：
1. decoder通过**position 1的来源**（inter_AB vs inter_B）决定生成模式：
   - position1 = inter_AB的输出 → 触发"源脸生成模式"
   - position1 = inter_B的输出 → 触发"目标脸生成模式"
2. `pred_src_dst` 的输入只有 `inter_AB(dst)` 没有 `inter_B`！这说明换脸时，模型**完全忽略环境信息**，只从 dst 图像中提取"属性信息"（表情/角度等），再通过decoder以"源脸模式"生成。这既是 LIAE 颜色一致性好的原因，也是为什么有时候换脸不够像的原因。
3. inter_AB不仅提取"身份"，还提取"属性"（表情、角度等）。更准确的说法是"通用特征提取器"而非仅"身份特征提取器"。

---

## 二、关于"删 inter_AB"的真相

很多 DeepFaceLab 教程说要"删除 inter_AB"，这其实是一种**误解**。

### 为什么会有这个说法

在原始 DFL 的工作流中，LIAE 训练久了会出现一个问题：**inter_AB 过拟合**。inter_AB 同时接收 src 和 dst 数据训练，它会学到一种"折中"的身份表示——既不完全是 src 也不是 dst，而是一种"泛化的人脸"。这导致换脸时预测的 `pred_src_dst` 不够像 src。

### 正确做法

**方案A：冻结 inter_AB（推荐）**

```python
freeze_inter_AB = True  # 冻结身份特征层，只微调解码器
```

这让身份表示固定，解码器只优化如何更好地生成。`build_optimizers()` 中的逻辑会直接排除冻结模块的参数。

**方案B：利用冻结机制分阶段训练**

```python
# 阶段1: 冻结 inter_B → 先专注身份
freeze_inter_B = True

# 阶段2: 解冻 inter_B → 学习环境匹配
freeze_inter_B = False
```

---

## 三、所有训练参数深度解析

### 3.1 架构参数（一次性设定，不可改）

| 参数 | 默认值 | 作用 | 建议 |
|------|--------|------|------|
| `archi` | liae-ud | LIAE+像素归一化+密集连接 | **不要改**，liae-ud 是最优组合 |
| `resolution` | 256 | 训练分辨率 | 256 是推荐值，显存允许可上 384 |
| `ae_dims` | 256 | 瓶颈层维度，控制模型容量 | 256 足够，512 更精细但慢 |
| `e_dims` | 64 | 编码器通道数 | 64 标准，96/128 需更多显存 |
| `d_dims` | 64 | 解码器通道数 | 64 标准 |
| `d_mask_dims` | 22 | 遮罩解码器通道数 | 保持 22 即可 |

### 3.2 训练控制参数（可分阶段调整）

| 参数 | 作用机制 | 影响 |
|------|----------|------|
| `random_warp` | 对图像施加随机仿射变形（旋转/缩放/平移） | **核心正则化**：防止过拟合，强制模型学习泛化特征而非记忆像素 |
| `lr` | 基础学习率 | 决定收敛速度和稳定性 |
| `lr_dropout` | 随机将部分参数梯度置零（30%概率） | **正则化**：防止过拟合，帮助精细收敛。`lr_dropout_rate=0.3` |
| `lr_cos` | 余弦退火周期 | 学习率周期性变化，帮助跳脱局部最优 |
| `ct_mode` | 颜色迁移方法 | LIAE 通常不需要，DF 才需要 |
| `uniform_yaw` | 均匀采样不同偏航角 | 改善侧脸效果 |
| `eyes_mouth_prio` | 眼嘴区域 300× 权重 | 显著提升眼嘴细节 |

### 3.3 损失函数参数

| 参数 | 代码中的权重 | 作用 |
|------|-------------|------|
| `masked_training` | 控制是否用 mask | 只在人脸区域内计算 loss，**必须开启** |
| `gan_power` | 逐步从 0 升到设定值 | GAN 对抗损失，增加皮肤纹理真实感 |
| `gan_patch_size` | UNet 判别器 patch | 默认 16，控制判别器感受野 |
| `gan_dims` | 判别器通道 | 默认 16 |
| `vgg_perceptual_power` | vgg_w = power/50 | VGG 感知损失，改善"塑料感" |
| `true_face_power` | power × BCE | **仅 DF 架构有效**，LIAE 无此功能 |
| `ramp_start_ratio` | 渐进 GAN 起始比例 | 训练到 target_iter 的 20% 后才开始线性开启 GAN |

### 3.4 冻结参数

| 参数 | 冻结对象 | LIAE 训练策略意义 |
|------|----------|-----------------|
| `freeze_encoder` | Encoder 所有权重 | **不推荐**，编码器是共享的基础 |
| `freeze_inter_AB` | inter_AB | **后期冻结身份**，让解码器适配 |
| `freeze_inter_B` | inter_B | **后期冻结环境**，稳定颜色匹配 |
| `freeze_decoder_mask` | 遮罩解码器 | mask 训练稳定后冻结，防止退化 |

### 3.5 损失函数详细组成（从源码解析）

```python
# 在 _compute_losses 中，总损失 = src_loss + dst_loss + extra_style_loss + extra_masked_gan_loss

# 1. 结构损失（核心）
src_loss = DSSIM(target, pred) * 10     # SSIM损失，10×权重
         + MSE(target, pred) * 10       # MSE像素损失，10×权重
         + MSE(mask, pred_mask) * mask_weight  # mask损失，~16.0× (256分辨率)

# 2. 眼嘴优先 (eyes_mouth_prio=True 时)
src_loss += L1(target*em_mask, pred*em_mask) * 300   # 300×眼嘴L1损失！

# 3. VGG感知损失 (vgg_perceptual_power > 0 时)
vgg_w = vgg_perceptual_power / 50.0   # 例如 power=10 → w=0.2
src_loss += vgg_w * Sum(L1(VGG_feat(pred), VGG_feat(target)))

# 4. GAN损失 (gan_power > 0 时)
G_loss += effective_gan_power * BCE(D(pred), ones)  # 生成器对抗损失

# 5. GAN辅助损失（mask外区域）
extra_masked_gan_loss = 1e-6 * TV_MSE(pred)          # 总变分正则
                      + 0.02 * MSE(pred*anti_mask, target*anti_mask)  # mask外区域的MSE
```

---

## 四、LIAE 分阶段训练方案（核心）

### 完整训练路线图

```
┌────────────────────────────────────────────────────────────────────┐
│                      LIAE 完整训练路线图                           │
├───────────────┬──────────────┬──────────────┬─────────────────────┤
│   阶段1       │   阶段2      │   阶段3      │   阶段4(+5)         │
│   结构学习     │   细节精修    │   精准映射    │   GAN+冻结微调      │
├───────────────┼──────────────┼──────────────┼─────────────────────┤
│ warp  ✓       │ warp  ✓      │ warp  ✗      │ warp  ✗            │
│ gan   0       │ gan   0      │ gan   0      │ gan   0.1          │
│ em    ✗       │ em    ✓      │ em    ✓      │ em    ✓            │
│ lr   5e-5     │ lr  5e-5     │ lr  3e-5     │ lr  1e-5 / 5e-6   │
│ lrd   n       │ lrd  n       │ lrd  y       │ lrd  y             │
│ vgg   0       │ vgg  0       │ vgg  0       │ vgg  5.0 (可选)    │
│ freeze ✗      │ freeze ✗     │ freeze ✗     │ freeze_AB ✓(可选)  │
├───────────────┼──────────────┼──────────────┼─────────────────────┤
│ 5-15万次      │ 3-8万次      │ 1-3万次      │ 5-15万(+1-5万)     │
├───────────────┴──────────────┴──────────────┴─────────────────────┤
│ 切换判断：loss不降 → 预览质量判断 → 预览质量判断 → GAN伪影/预览   │
│ 注意：阶段3的切换不要在loss不降时才换，要看预览！                  │
└────────────────────────────────────────────────────────────────────┘
```

---

### 阶段1：结构学习 (Structure Learning)

**目标**：让模型学会基本的人脸重建和身份分离

**参数配置**：

```python
random_warp        = True      # 必须开启！核心正则化
gan_power          = 0.0       # 不开启GAN
vgg_perceptual_power = 0.0     # 不开启VGG
eyes_mouth_prio    = False     # 先不开启，模型还没学好基础
masked_training    = True      # 必须开启
uniform_yaw        = True      # 改善角度覆盖
lr                 = 5e-5      # 默认学习率
lr_dropout         = 'n'       # 不开启LRD
lr_cos             = 0         # 不开启余弦退火
random_hsv_power   = 0.0       # LIAE颜色好，不需要
ct_mode            = 'none'    # LIAE不需要颜色迁移
freeze_inter_AB    = False
freeze_inter_B     = False
freeze_encoder     = False
pretrain           = False
```

**持续到何时**：
- 观察 `src_loss` 和 `dst_loss` 持续下降
- 预览图像从模糊变得清晰可辨
- **判断标准**：loss 不再显著下降（比如 1000 次迭代内改善 <1%）
- 典型迭代数：**5万-15万次**

**为什么这么配置**：
- `random_warp=True` 是 LIAE 架构最重要的参数！它让每张训练图都有随机仿射变形，强制模型学习"人脸结构"而非"像素位置"。源码中 `pretrain=True` 时自动关闭 warp，因为预训练用大量不同人脸，本身就有足够多样性。但正式训练时，同一人的数据缺乏多样性，必须靠 warp。
- `gan_power=0` 确保第一阶段只关注重建质量，GAN 会引入不稳定因素
- `eyes_mouth_prio=False` 因为模型结构都没学好，300× 的眼嘴权重只会让模型走捷径

---

### 阶段2：面部细节精修 (Detail Refinement)

**目标**：精细修复眼嘴等关键区域，提升面部细节

**参数配置**：

```python
random_warp        = True      # 仍然开启，保持泛化
gan_power          = 0.0       # 暂不开启GAN
eyes_mouth_prio    = True      # ← 核心变化！开启眼嘴优先
masked_training    = True
uniform_yaw        = True
lr                 = 5e-5      # 可以保持，如果震荡则降至 3e-5
lr_dropout         = 'n'       # 暂不开启
lr_cos             = 0
vgg_perceptual_power = 0.0
```

**持续到何时**：
- 观察预览中眼睛清晰度、嘴巴形状明显改善
- 眼嘴细节不再显著提升
- loss 进入缓慢下降期
- 典型迭代数：**3万-8万次**

**为什么这么配置**：
- 源码中 `eyes_mouth_prio=True` 会给眼嘴区域增加 **300× L1 损失权重**！
  ```python
  src_loss += (target*em_mask - pred*em_mask).abs().mean() * 300
  ```
  这是非常大的权重，必须在模型已经有基本重建能力后才开启。否则模型会"走捷径"——拼命优化眼嘴而忽略其他区域，导致肤色不均、结构变形。

---

### 阶段3：LRD收敛 — 开启学习率丢弃 (LR Dropout)

**目标**：在保持random_warp的同时开启LRD，让模型先适应稀疏梯度更新

> **⚠️ DFL原版建议**：DFL的help_message明确说"Enabled it before `disable random warp` and before GAN"，即**先开LRD再关warp**，不要同时进行。

**参数配置**：

```python
random_warp        = True      # 保持开启！先不关
eyes_mouth_prio    = True      # 保持
gan_power          = 0.0
masked_training    = True
uniform_yaw        = True      # 保持
lr                 = 5e-5      # 保持
lr_dropout         = 'y'       # ← 核心变化！开启LRD
lr_cos             = 500       # ← DFL原版开启lrd时自动设为500，我们需手动设
vgg_perceptual_power = 0.0
```

**持续到何时**：
- loss在LRD开启后短暂波动然后重新稳定
- 预览质量保持稳定或略有提升
- 典型迭代数：**2万-5万次**

**为什么这么配置**：
- `lr_dropout='y'` 开启LRD，30%概率随机丢弃梯度更新，等价于正则化
- **DFL原版行为差异**：DFL原版在`lr_dropout='y'`时自动设置`lr_cos=500`（余弦退火周期500），我们代码需要手动设置lr_cos
- 先在warp开启时让模型适应LRD的稀疏梯度，避免关warp时同时面临两个变化导致训练不稳定

---

### 阶段4：精修阶段 — 关闭 Random Warp (Fine-Texture)

**目标**：学习精确的像素级映射，获得清晰细节

**参数配置**：

```python
random_warp        = False     # ← 核心变化！关闭变形
eyes_mouth_prio    = True      # 保持
gan_power          = 0.0
masked_training    = True
uniform_yaw        = False     # 可以关闭，专注于已有角度
lr                 = 3e-5      # ← 降低学习率
lr_dropout         = 'y'       # 保持LRD
lr_cos             = 500       # 保持
vgg_perceptual_power = 0.0
freeze_inter_AB    = True      # ← 建议手动冻结，模拟DFL原版行为
```

**持续到何时**：
- 预览图像的清晰度大幅提升
- **关键判断**：出现了过拟合的迹象（如牙齿过度锐化、出现人工伪影）时就立即停止
- 典型迭代数：**1万-3万次**（不宜过长）

**为什么这么配置**：
- `random_warp=False` 是双刃剑：
  - 好处：模型不再看到变形后的图像，可以精确学习原始像素映射，清晰度飙升
  - 坏处：失去数据增强，模型开始记忆训练数据的像素位置，**容易过拟合**
- `lr=3e-5` 降低学习率防止过拟合加速
- `lr_dropout='y'` 保持LRD正则化

**⚠️ 自动冻结inter_AB的行为差异**：
- **DFL原版**（第335-338行）：`random_warp=False`时**自动**从trainable_weights中排除inter_AB，无需手动设置
- **PyTorch参考版**（第751-758行）：同样自动冻结
- **我们代码**：没有自动冻结逻辑，需要**手动设置`freeze_inter_AB=True`**来模拟DFL行为
- 如果不设置freeze_inter_AB，inter_AB在关warp后仍会继续训练，可能导致与DFL不同的行为

**⚠️ 最大坑位**：这个阶段不是越长越好！关闭 warp 后 loss 会持续下降——因为模型在过拟合训练数据。**判断标准是预览质量，不是 loss 数值！**

---

### 阶段5：GAN 纹理增强 (GAN Texture Enhancement)

**目标**：增加真实皮肤纹理、毛孔、光影细节

**参数配置**：

```python
random_warp        = False     # 保持关闭
eyes_mouth_prio    = True      # 保持
gan_power          = 0.1       # ← 核心变化！开启GAN
gan_patch_size     = 16        # 默认即可
gan_dims           = 16        # 默认即可
vgg_perceptual_power = 5.0     # ← 可选开启，减少塑料感
masked_training    = True
lr                 = 1e-5      # ← 进一步降低学习率
lr_dropout         = 'y'       # 保持
ramp_start_ratio   = 0.2       # 渐进GAN的起始比例
target_iter        = 50000     # 设置目标迭代数，用于渐进GAN
freeze_inter_AB    = True      # 保持冻结
```

**持续到何时**：
- 观察预览的皮肤纹理变得自然真实
- GAN loss（D_gan_loss）不再有意义地变化（GAN 训练本质是博弈，不像普通 loss 一样单调下降）
- 出现伪影/奇怪纹理时降低 gan_power 或停止
- 典型迭代数：**5万-15万次**

**为什么这么配置**：
- 渐进 GAN 机制（源码 `compute_effective_gan_power`）：
  ```python
  # 从 target_iter * ramp_start_ratio 开始，GAN强度从0按sigmoid平滑过渡增长到 gan_power
  ramp_start = target_iter * 0.2  # 如50k*0.2=10k迭代后开始
  progress = (iter_count - ramp_start) / (target_iter - ramp_start)
  smooth = 1/(1+e^(-(progress-0.5)*10))  # sigmoid平滑，不是线性！
  effective_power = gan_power * smooth
  ```
  这意味着你需要设置一个 `target_iter` 值，训练从 `target_iter*0.2` 到 `target_iter` 之间 GAN 强度按sigmoid曲线逐步提升。如果 `target_iter=0`（默认），GAN 直接全开。

- GAN 的额外 mask 外区域正则（源码）：
  ```python
  extra_masked_gan_loss = 1e-6 * TV_MSE(pred) + 0.02 * MSE(pred*anti_mask, target*anti_mask)
  ```
  这确保 GAN 不会在非人脸区域产生奇怪的纹理。

- `vgg_perceptual_power=5.0` 时，实际权重 = 5.0/50 = 0.1，这是很轻的感知约束，帮助减少"塑料感"

---

### 阶段6（可选）：冻结身份 — 纯净学习 (Identity Freeze Fine-tune)

**目标**：锁定身份表示，微调其他部分以适配 dst 环境

**参数配置**：

```python
freeze_inter_AB     = True      # ← 核心！冻结通用特征提取器
freeze_inter_B      = False     # 保持环境学习能力
encoder             = False     # 保持编码器可训练
random_warp         = False
gan_power           = 0.05~0.1  # 可以降低GAN强度
eyes_mouth_prio     = True
lr                  = 5e-6      # 非常低的学习率
lr_dropout          = 'y'
```

**持续到何时**：
- 预览中换脸效果的身份一致性更好
- 环境融合更自然
- 典型迭代数：**1万-5万次**

**为什么这么配置（这就是"删 inter_AB"的真正含义）**：
- 源码中 `freeze_inter_AB=True` 时，`build_optimizers()` 会跳过冻结模块，`inter_AB` 不再接收梯度更新，其"身份理解"被锁定
- 在原始 DFL 中，有些人会删除 inter_AB 的权重文件然后重新训练。但 `freeze_inter_AB` 是更优雅的做法——保留已学好的身份表示，只微调其他部分。
- **注意**：如果在阶段4已设置`freeze_inter_AB=True`，此阶段是在其基础上进一步降低学习率做微调

---

### 切换阶段的关键判断信号

| 切换 | 信号 | 不要被误导 |
|------|------|-----------|
| 1→2 | src/dst loss 进入缓慢下降，预览中五官可辨 | 不需要等到 loss 完全不动 |
| 2→3 | 眼嘴细节清晰，整体重建良好 | 不要追求 loss 最低点 |
| 3→4 | LRD开启后loss短暂波动然后重新稳定 | LRD开启初期loss波动是正常的 |
| 4→5 | 清晰度足够，无伪影 | **最大的坑：loss 还在降但已经过拟合了！看预览不要看 loss！** |
| 5→6 | GAN纹理稳定，身份相似度和环境融合需微调 | 可选阶段 |

### 最重要的提醒

1. **别照搬教程的"删 inter_AB"！** 代码中的 `freeze_inter_AB` 才是正确做法。
2. **GAN 阶段 loss 不降反升是正常的！** GAN 博弈本质决定了 generator loss 和 discriminator loss 会来回波动。
3. **阶段4是过拟合重灾区！** random_warp 关闭后，模型 1-2 万次迭代就能产生极低 loss，但预览可能已经崩了。**看预览，别看数字**。
4. **LIAE 比 DF 对颜色/光照相容性更好**，所以 `ct_mode='none'` 是正确的默认值。
5. **smart_stop_detector** 在代码中默认开启（window=500, threshold=0.1%），它会在收敛时提醒你。但在 GAN 阶段 loss 波动大，它的判断可能不准确。
6. **DFL原版在关warp时自动冻结inter_AB**，我们代码需手动设置`freeze_inter_AB=True`。
7. **DFL原版在开启lrd时自动设置lr_cos=500**，我们代码需手动设置。
8. **pretrain切换行为**：DFL原版在pretrain从True→False时，会重新初始化inter_AB和inter_B，并重置iter为0。我们代码也有类似逻辑（`on_pretrain_override`），但未覆盖random_flip=True。

---

## 五、架构修饰符 u/d/t/c 深度分析

架构通过 `archi` 参数设置，格式为 `liae` + 修饰符组合。

### 5.1 `u` — 像素归一化 (Pixel Normalization)

**源码位置**：`Encoder.forward()` 第 133-134 行

```python
if self._use_u:
    x = pixel_norm(x)   # 将编码向量归一化到单位长度
```

**实际作用**：将 encoder 输出的每个样本的特征向量除以自身的 L2 范数，强制所有编码落在超球面上。这等价于给潜空间加了强正则化，防止 encoder 输出极端值。

**训练影响**：
- 阶段1：训练比无 `u` 的变体**明显更稳定**，loss 下降更平滑。可以稍微激进一点的学习率（比如 7e-5 而不是 5e-5）
- 阶段2-4：与标准方案几乎一致
- **零参数成本，只带来好处。强烈推荐所有变体都加上。**

### 5.2 `d` — 密集输出连接 (Dense Output Connections)

**两处生效**：

1. **Inter 层**（第 154 行）— 瓶颈更紧凑：
   ```python
   lowest_dense_res = resolution // (32 if 'd' in opts else 16)
   ```
   无 `d`：Inter 输出 16×16 网格。有 `d`：Inter 输出 8×8 网格 → 更紧凑的瓶颈。

2. **Decoder 输出层**（第 254-263 行）— 多路卷积 + depth_to_space：
   ```python
   if self._use_d:
       x_pre_dts = torch.cat((self.out_conv(x),    # 1×1 conv
                      self.out_conv1(x),            # 3×3 conv
                      self.out_conv2(x),            # 3×3 conv
                      self.out_conv3(x)), dim=1)    # 3×3 conv → 4路12通道
       x = depth_to_space(x_pre_dts, 2)             # 12ch→3ch，同时2×上采样
       x = torch.sigmoid(x)
   ```
   无 `d`：单层 1×1 卷积出 3 通道。有 `d`：**4 个不同感受野的卷积核**各自产生输出，然后通过 depth_to_space 融合并再做一次 2× 上采样。

**训练影响**：
- 阶段1：无显著差异，瓶颈在 encoder+inter
- 阶段2：多路卷积各路径独立学习，**开眼嘴优先效果更显著**
- 阶段4：⚠️ depth_to_space 多一次上采样 → 精细但更易过拟合。**比无 `d` 短约 20-30%**
- 阶段5：GAN 对有 `d` 的输出更好——depth_to_space 避免 checkerboard 伪影
- **特殊警告**：阶段4 要留意预览中是否出现"网格状"伪影（depth_to_space 典型副作用），一旦出现立即进入阶段5

### 5.3 `t` — 更深网络 (Transformer-like Deeper)

**三处生效**：

1. **Encoder**（第 102-114 行）— 5 次下采样 + 2 个残差块：
   ```python
   if use_t:
       self.down1~down5           # 5次下采样 (vs 普通4次)
       self.res1, self.res5       # 2个残差块 (vs 普通0个)
   ```

2. **Inter**（第 162-165 行）— 取消 upscale1：
   ```python
   if not use_t:
       self.upscale1 = Upscale(...)   # 有t时不创建
   ```

3. **Decoder**（第 199-206 行）— 增加一组 upscale+residual + mask 分支：
   ```python
   if use_t:
       self.upscale1~upscale3      # 3组上采样 (vs 普通2组)
       self.res1~res3              # 3个残差块 (vs 普通2个)
   ```

**训练影响（这是差异最大的修饰符）**：

- **阶段1**：⚠️ 需要 **1.5-2 倍**迭代次数。更深网络梯度更难传导。学习率降到 **4e-5**。前期 loss 下降慢是正常的，不要误判。
- **阶段2**：额外的 residual block 让眼嘴优先梯度传得更远，细节学习更好。时长与标准相当。
- **阶段4**：⚠️ **过拟合风险最大！** 5 层 encoder + 3 个 decoder residual block → 极易记忆训练数据。关 warp 后每 **2k 次必须检查预览**！时长缩短到标准方案的 **50-60%**（约 5k-15k 次）。
- **阶段5**：高频表达能力最强，GAN 增益最大。延长到标准方案的 **1.3-1.5 倍**。
- **额外建议**：搭配 `lr_cos=50000`，帮助深网络跳脱局部最优。
- **分辨率要求**：5 次下采样后特征图极小，建议 **分辨率 ≥ 256**，推荐 384+。

### 5.4 `c` — 余弦激活 (Cosine Activation)

**源码**（第 26-31 行）：全部用 `x * cos(x)` 替代 `LeakyReLU(0.1)`。

**训练影响**：
- 余弦函数梯度行为复杂（`cos(x) - x*sin(x)`），输入较大时梯度可能消失或爆炸
- 阶段1：训练可能**很不稳定**，loss 频繁震荡
- 如果一定要用：学习率降到 **2e-5~3e-5**，开启 `clipgrad=True`
- **不推荐新手使用，属于实验性功能**

### 5.5 各变体参数量和计算量对比

256 分辨率下的粗略估算：

| 变体 | Encoder | Inter | Decoder | 总参数量 | 相对VRAM |
|------|---------|-------|---------|---------|----------|
| `liae`（裸） | 4 down | 1 upscale + 2 dense | 2 upscale+res, 1 out_conv | 基准 100% | 100% |
| `liae-u` | +pixel_norm（0参数） | 同 | 同 | ~100% | ~100% |
| `liae-d` | 同 | Inter 压缩到 8×8 | +3 个 out_conv | ~110% | ~108% |
| `liae-t` | 5 down + 2 res | **无** upscale1 | +1 upscale+res | ~135% | ~130% |
| `liae-ud` | +pixel_norm | +Inter 压缩 + 3 out_conv | +3 out_conv | ~110% | ~108% |
| `liae-ut` | +pixel_norm, 5down+2res | 无 upscale1 | +1 upscale+res | ~135% | ~130% |
| `liae-dt` | 5down+2res | Inter 压缩 8×8, 无 upscale1 | +1 upscale+res, +3 out_conv | ~150% | ~142% |
| `liae-udt` | +pixel_norm, 5down+2res | Inter 压缩 8×8, 无 upscale1 | +1 upscale+res, +3 out_conv | ~150% | ~142% |

> `t` 变体下 encoder 输出分辨率减半（多一次下采样），decoder 起始特征图更小，建议分辨率 ≥ 256。

---

## 六、各变体完整训练方案

### 方案 A：`liae-ud`（推荐标准方案）

```python
# 阶段1：结构学习 (50k-150k)
random_warp=True, gan_power=0.0, vgg_perceptual_power=0.0
eyes_mouth_prio=False, masked_training=True, uniform_yaw=True
lr=5e-5, lr_dropout='n', lr_cos=0, ct_mode='none'

# 阶段2：细节精修 (30k-80k)
# 变化：eyes_mouth_prio=True

# 阶段3：LRD收敛 (20k-50k)
# 变化：lr_dropout='y', lr_cos=500

# 阶段4：精准映射 (10k-30k)
# 变化：random_warp=False, lr=3e-5, uniform_yaw=False, freeze_inter_AB=True

# 阶段5：GAN纹理增强 (50k-150k)
# 变化：gan_power=0.1, vgg_perceptual_power=5.0, lr=1e-5
# 设置 target_iter=50000, ramp_start_ratio=0.2

# 阶段6：冻结身份微调（可选）(10k-50k)
# 变化：lr=5e-6（freeze_inter_AB已在阶段4开启）
```

---

### 方案 B：`liae-u`（省显存方案）

与方案 A **几乎完全一致**。差异：
- 少了 `d` 的 depth_to_space 输出，阶段4 可**稍微延长 20%**（不容易出网格伪影）
- 阶段5 GAN 效果略逊于 `liae-ud`，但不影响训练节奏
- 显存省约 8-10%

---

### 方案 C：`liae-d`（不推荐）

少了 `u` 的像素归一化，encoder 可能输出极端值，训练稳定性下降。

```python
# 阶段1：结构学习 (80k-200k)  ← 更长
lr=3e-5  # ← 降低学习率

# 阶段2：细节精修 (30k-80k)
lr=3e-5

# 阶段3：LRD收敛 (15k-40k)
lr_dropout='y', lr_cos=500

# 阶段4：精准映射 (8k-20k)  ← 更短，防止过拟合
lr=2e-5, freeze_inter_AB=True

# 阶段5：GAN纹理增强 (50k-150k)
gan_power=0.1, lr=1e-5
```

**不推荐原因**：无 `u` 归一化，训练稳定性下降 30%+，`liae-ud` 只多 8% VRAM 换来巨大稳定性提升。

---

### 方案 D：`liae-udt`（高分辨率推荐，384+）

**这是差异最大的方案！**

```python
# 阶段1：结构学习 (100k-300k)  ← 2倍！
random_warp=True, gan_power=0.0, vgg_perceptual_power=0.0
eyes_mouth_prio=False, masked_training=True, uniform_yaw=True
lr=4e-5       # ← 降低学习率
lr_dropout='n', lr_cos=50000  # ← 加入余弦退火
ct_mode='none'

# 阶段2：细节精修 (40k-100k)  ← 1.3倍
# 变化：eyes_mouth_prio=True, lr=4e-5

# 阶段3：LRD收敛 (20k-50k)
# 变化：lr_dropout='y', lr_cos=500

# 阶段4：精准映射 (5k-15k)  ← 一半！⚠️
# 变化：random_warp=False, lr=2e-5, freeze_inter_AB=True
# ⚠️ 每 2k 次检查一次预览！

# 阶段5：GAN纹理增强 (80k-200k)  ← 1.5倍
# 变化：gan_power=0.1, vgg_perceptual_power=5.0, lr=1e-5

# 阶段6：冻结微调（可选）(10k-50k)
# 变化：lr=5e-6（freeze_inter_AB已在阶段4开启）
```

**关键改动理由**：
1. **阶段1×2**：更深网络需要更多迭代让所有层收敛
2. **阶段4÷2**：5 层 encoder + 3 个 decoder residual block → **极易过拟合**
3. **阶段5×1.5**：`t` 的高频表达能力在 GAN 阶段才真正释放
4. `lr_cos=50000`：帮助深网络跳脱局部最优

---

### 方案 E：`liae-ut` / `liae-dt`（中间方案）

- `liae-ut`：取方案 D，但阶段3 可以比 `udt` 稍长 20%（少了 `d` 的 depth_to_space 伪影风险）
- `liae-dt`：阶段1 用方案 D 的长度，阶段3 用方案 D 的长度，阶段4 介于方案 C 和 D 之间

---

## 七、架构选择决策树

```
你追求什么？
│
├─ 通用效果最好 → liae-ud（无脑选这个）
│
├─ 显存紧张 → liae-u（省 ~15% VRAM）
│
├─ 高分辨率 384+ → liae-udt（训练时间翻倍但细节更好）
│   └─ 还想要更细纹理 → liae-udt + gan_power=0.15~0.2
│
├─ 超高分辨率 512+ → liae-udt 必选
│   └─ 注意：t 变体下 encoder 输出 16×16(512/32)
│       → Inter 无上采样 → 8×8 出 → decoder
│       └─ 无 t: encoder 输出 32×32(512/16) → Inter 上采样 → 64×64 出
│          → decoder 输入分辨率太高，参数量爆炸
│
├─ 追求极致清晰度（高VRAM） → liae-udt + ae_dims=384 + e_dims=96 + d_dims=96
│
└─ 实验性/研究 → liae-udc（余弦激活，行为不同，可能有意想不到的效果）
```

---

## 八、总对比表与核心总结

### 各变体分阶段训练对比

|  | `liae-ud` (基准) | `liae-u` | `liae-udt` | `liae-d` (无u) |
|---|---|---|---|---|
| 阶段1 长度 | 50k-150k | 50k-150k | **100k-300k** ⬆ | 80k-200k |
| 阶段1 学习率 | 5e-5 | 5e-5 | **4e-5** | 3e-5 |
| 阶段2 长度 | 30k-80k | 30k-80k | **40k-100k** | 30k-80k |
| 阶段3 长度(LRD) | 20k-50k | 20k-50k | **20k-50k** | 15k-40k |
| 阶段4 长度(关warp) | 10k-30k | 12k-35k | **5k-15k** ⚠️⬇ | 8k-20k |
| 阶段4 过拟合风险 | 中 | 低 | **极高** ⚠️ | 高 |
| 阶段5 长度(GAN) | 50k-150k | 50k-120k | **80k-200k** ⬆ | 50k-150k |
| GAN 收益 | 标准 | 标准 | **最大** | 较好 |
| 推荐分辨率 | 128-384 | 128-384 | **256-512** | 128-384 |
| 推荐 lr_cos | 0→500(阶段3) | 0→500(阶段3) | **50k** | 0→500(阶段3) |

### 核心总结

1. **分阶段训练大框架对所有 LIAE 变体一致**：warp → eyes_mouth → LRD → no_warp → GAN
2. **`u` 几乎无影响**，无脑加上
3. **`d` 影响阶段4过拟合速度和阶段5纹理表现**，阶段4 略短
4. **`t` 差异最大**：阶段1翻倍、阶段4减半、阶段5加长
5. **阶段4 是最大的坑**：看预览别看 loss，过拟合时 loss 仍在下降
6. **GAN 阶段 loss 波动是正常的**，博弈本质决定
7. **`freeze_inter_AB` 才是"删 inter_AB"的正确实现**
8. **LIAE 不需要 color transfer**，`ct_mode='none'` 是正确的
9. **DFL原版在关warp时自动冻结inter_AB**，我们代码需手动设置`freeze_inter_AB=True`
10. **DFL原版在开启lrd时自动设置lr_cos=500**，我们代码需手动设置
11. **先开LRD再关warp**（DFL建议顺序），不要同时进行

---

> **文档生成基于源码**：
> - `faceswap/models/saehd/saehd_arch.py` — 架构定义
> - `faceswap/models/saehd/saehd_model.py` — 训练逻辑与损失函数
> - `faceswap/business/saehd_trainer.py` — 训练循环
> - `faceswap/business/base_trainer.py` — 基类训练器
> - `faceswap/gui_app/saehd_param_defs.py` — 参数定义与说明
> - `faceswap/models/saehd/losses.py` — 损失函数实现
> - `faceswap/models/saehd/discriminators.py` — GAN 判别器
> - `faceswap/core/saehd_utils.py` — 工具函数
> - `faceswap/business/smart_stop_detector.py` — 智能停止检测
