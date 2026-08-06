# DF 架构深度分析与分阶段训练指南

> 基于 `D:\AI\Inswapper\faceswap\models\saehd\` 源码深度分析
>
> 生成日期：2026-08-06

---

## 目录

1. [DF 架构核心剖析](#一df-架构核心剖析)
   - [数据流详解](#11-数据流详解)
   - [三个关键路径](#12-三个关键路径)
   - [DF vs LIAE 架构关系对比](#13-df-vs-liae-架构关系对比)
2. [DF 专属机制深度分析](#二df-专属机制深度分析)
   - [true_face_power：CodeDiscriminator 的工作原理](#21-true_face_powercodediscriminator-的工作原理)
   - [ct_mode：为什么 DF 强烈推荐开颜色迁移](#22-ct_mode为什么-df-强烈推荐开颜色迁移)
   - [双解码器：src vs dst 的分离训练](#23-双解码器src-vs-dst-的分离训练)
3. [DF 所有参数深度解析](#三df-所有参数深度解析)
4. [DF 分阶段训练方案（核心）](#四df-分阶段训练方案核心)
   - [阶段1：颜色+结构学习](#阶段1颜色结构学习)
   - [阶段2：身份对齐——true_face](#阶段2身份对齐true_face)
   - [阶段3：眼嘴细节精修](#阶段3眼嘴细节精修)
   - [阶段4：关闭warp——精准映射](#阶段4关闭-warp精准映射)
   - [阶段5：GAN 纹理增强](#阶段5gan-纹理增强)
   - [阶段6：冻结inter微调（可选）](#阶段6冻结-inter-微调可选)
5. [DF 架构修饰符 u/d/t/c 的影响](#五df-架构修饰符-udtc-的影响)
6. [DF 各变体完整训练方案](#六df-各变体完整训练方案)
7. [DF vs LIAE 终极对比与选择指南](#七df-vs-liae-终极对比与选择指南)
8. [核心总结](#八核心总结)

---

## 一、DF 架构核心剖析

### 1.1 数据流详解

```
                     ┌──────────────────────────────────┐
  warped_src ──────► │         Encoder (共享)            │
  warped_dst ──────► │   e_dims=64, 4/5次下采样          │──► encoder_flat [B, e_ch*8*H*W]
                     └──────────────────────────────────┘
                                     │
                                     ▼
                     ┌──────────────────────────────────┐
                     │          Inter (共享)             │
                     │   dense1: flat→ae_dims           │
                     │   dense2: ae_dims→ae_dims*H2*W2  │
                     │   upscale1: 2×上采样 (无t时)      │──► code [B, ae_dims, H2, W2]
                     └──────┬───────────────┬───────────┘
                            │               │
                     src_code               dst_code
                     (src→inter)            (dst→inter)
                            │               │
              ┌─────────────┘               └─────────────┐
              ▼                                           ▼
   ┌──────────────────────┐               ┌──────────────────────┐
   │    Decoder_SRC       │               │    Decoder_DST       │
   │  d_dims=64           │               │  d_dims=64           │
   │  d_mask_dims=22      │               │  d_mask_dims=22      │
   │  ├ pred_src_src      │               │  ├ pred_dst_dst      │
   │  ├ pred_src_srcm     │               │  ├ pred_dst_dstm     │
   │  ├ pred_src_dst  ←───┤──── dst_code   │  └ ...              │
   │  └ pred_src_dstm     │               │                      │
   └──────────────────────┘               └──────────────────────┘
```

**DF 与 LIAE 的本质区别**：

| 维度 | DF (Dual-decoder) | LIAE |
|------|-------------------|------|
| Inter 数量 | **1 个**（共享） | 2 个（inter_AB + inter_B） |
| Inter 输出通道 | `ae_dims` | `ae_dims * 2` |
| Decoder 数量 | **2 个**（decoder_src, decoder_dst） | 1 个（共享） |
| Decoder 输入通道 | `ae_dims` | `ae_dims * 4`（两路拼接） |
| 身份/环境分离 | ❌ 隐式，通过双解码器 | ✓ 显式，inter_AB/inter_B |
| 颜色一致性 | ❌ 差，需 ct_mode | ✓ 好，不需 ct_mode |
| true_face 机制 | ✓ CodeDiscriminator | ❌ 不支持 |
| 冻结机制 | freeze_inter | freeze_inter_AB / freeze_inter_B |

### 1.2 三个关键路径

```python
# 路径1: src → src (重建源脸)
encoder(src) → inter → decoder_src → pred_src_src

# 路径2: dst → dst (重建目标脸)
encoder(dst) → inter → decoder_dst → pred_dst_dst

# 路径3: dst → src (换脸!) — 核心差异
encoder(dst) → inter → decoder_src → pred_src_dst
#                     ^^^^^^^^^^^
#                     关键：dst的中间编码 送入 src的专属解码器
```

**路径3的核心含义**：DF 换脸的逻辑是"用 src 专属的解码器去解码 dst 的中间表示"。这意味着：

1. `decoder_src` 被训练成"解码出 src 人脸"的专家
2. 当你把 dst 图像的 inter 输出喂给 `decoder_src` 时，它会尝试"以 src 的方式"生成人脸
3. 但颜色/光照信息是 dst 的，所以颜色可能不匹配 → 这就是**色斑**的根源

### 1.3 DF vs LIAE 架构关系对比

```
┌─────────────────────────────────────────────────────────────────┐
│                        DF 架构                                  │
│                                                                 │
│  Encoder ──► Inter ──┬──► Decoder_SRC ──► pred_src             │
│                      │                                          │
│                      └──► Decoder_DST ──► pred_dst             │
│                                                                 │
│  特点：单瓶颈 + 双解码器                                         │
│  换脸路径：dst→Encoder→Inter→Decoder_SRC = 身份迁移             │
│  问题：颜色从属于decoder_src，dst颜色丢失                        │
├─────────────────────────────────────────────────────────────────┤
│                       LIAE 架构                                 │
│                                                                 │
│  Encoder ──┬──► Inter_AB (身份) ──┬─► cat ──► Decoder ──► pred │
│            └──► Inter_B  (环境) ──┘                              │
│                                                                 │
│  特点：双瓶颈 + 单解码器                                         │
│  换脸路径：dst→Encoder→Inter_AB→[AB,AB]→Decoder = 提取身份      │
│  优势：环境信息(inter_B)独立，颜色自然                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、DF 专属机制深度分析

### 2.1 true_face_power：CodeDiscriminator 的工作原理

这是 DF **独有的**身份对齐机制。源码位置：`saehd_model.py` 第 119-125 行（构建）和第 326-333 行（损失计算）。

#### 构建

```python
# 第122-125行：仅DF架构且true_face_power≠0时创建
if c.true_face_power != 0.0 and c.archi_type == 'df':
    code_res = self.inter.get_out_res()    # 例如 256÷16×2=32
    code_ch = self.inter.get_out_ch()      # ae_dims, 例如 256
    self.code_discriminator = CodeDiscriminator(in_ch=code_ch, code_res=code_res)
```

`CodeDiscriminator`（`discriminators.py` 第 7-26 行）是一个简单的下采样卷积网络：
- 输入：`[B, ae_dims, code_res, code_res]` 的特征图
- 结构：`code_res//8 + 1` 次 stride=2 的卷积 + 1×1 输出
- 输出：`[B, 1, ~4, ~4]` 的判别热力图

#### 损失计算

```python
# 第326-333行
if c.true_face_power != 0.0 and c.archi_type == 'df':
    src_code_d = self.code_discriminator(fw['src_code'])  # src编码的判别
    dst_code_d = self.code_discriminator(fw['dst_code'])  # dst编码的判别

    # 生成器损失：让src_code变得"像"dst_code（骗过判别器）
    G_loss += c.true_face_power * BCE(src_code_d, ones)

    # 判别器损失：学会区分src_code和dst_code
    D_code_loss = 0.5 * (BCE(dst_code_d, ones) + BCE(src_code_d.detach(), zeros))
```

#### 工作机制图解

```
┌──────────────────────────────────────────────────────────────┐
│                CodeDiscriminator 工作原理                     │
│                                                              │
│  生成器目标：                                                 │
│    让 src_code 骗过判别器 → BCE(src_code_d, 1) → 最小化      │
│    ↓                                                         │
│    含义：让 src 和 dst 的中间编码分布趋同                     │
│    ↓                                                         │
│    效果：decoder_src 能更好地解码 dst 的脸                    │
│                                                              │
│  判别器目标：                                                 │
│    区分 src 和 dst → BCE(dst,1) + BCE(src,0) → 最小化       │
│    ↓                                                         │
│    效果：防止生成器把所有编码都变得一样（保持多样性）           │
│                                                              │
│  本质上：这是一个编码级别的 GAN，让 src 和 dst 的编码          │
│  分布对齐，从而改善换脸质量                                    │
└──────────────────────────────────────────────────────────────┘
```

#### 对训练的影响

- `true_face_power` 是**对抗性损失**，不宜过早开启
- 需要在模型有一定重建能力后才引入（类似 patch-gan 的道理）
- 值范围：**0.01~0.1**，过大会导致编码崩溃（所有编码趋同）
- 同时优化 `G_loss`（生成器）和 `D_code_loss`（判别器），与其他优化器交替更新

### 2.2 ct_mode：为什么 DF 强烈推荐开颜色迁移

#### 根本原因

```
DF 换脸时：
  dst图像 → encoder → inter → decoder_src → 换脸结果
                                        ^^^
                                    src专属解码器

decoder_src 只见过 src 的颜色分布 → 它输出的颜色永远偏向 src
但输入是 dst 的编码 → 内容像 src，颜色像 src，光照错位 → 色斑
```

#### 解决方案

```
ct_mode 的作用（训练时）：
  src原图 → [颜色迁移: src→dst色域] → 变形后src → encoder → inter → decoder_src

这样 decoder_src 学习的是：在 dst 色域下如何重建 src 人脸
换脸时自然就匹配 dst 的光照/色调 → 消除色斑
```

#### 各种 ct_mode 推荐的来源

| ct_mode | 方法 | 速度 | 质量 | 适用场景 |
|---------|------|------|------|----------|
| `rct` | Reinhard 颜色迁移（LAB均值/标准差匹配） | 🔥快 | ⭐⭐⭐ | **最常用推荐** |
| `rct-p` | 部分Reinhard（50%混合原图） | 🔥快 | ⭐⭐ | 保守方案 |
| `lct` | 线性迁移（PCA协方差匹配） | 🔥快 | ⭐⭐⭐ | 备选推荐 |
| `mkl` | MKL 最优传输线性近似 | 中等 | ⭐⭐⭐⭐ | 高质量 |
| `sot` | 切片最优传输 | 慢 | ⭐⭐⭐⭐⭐ | 最高质量 |
| `idt` | 迭代迁移 | 最慢 | ⭐⭐⭐⭐⭐ | 极端情况 |

> **DF 强烈推荐开 ct_mode，LIAE 通常不需要**。DFL原版不强制，但实践中DF不开ct_mode几乎必然出现色斑。

### 2.3 双解码器：src vs dst 的分离训练

```python
# 第94-99行：两个独立的Decoder实例
self.decoder_src = Decoder(in_ch=inter_out_ch, d_ch=c.d_dims, ...)
self.decoder_dst = Decoder(in_ch=inter_out_ch, d_ch=c.d_dims, ...)
```

两个解码器**结构完全相同**（同一个 Decoder 类），但参数独立，各自接收不同角色的数据训练：

| Decoder | 训练时接收的输入 | 学到的能力 |
|---------|----------------|-----------|
| decoder_src | src_code（src 的编码） | 解码出 src 的脸 |
| decoder_dst | dst_code（dst 的编码） | 解码出 dst 的脸 |

**关键推论**：
- `decoder_src` **只能在 src 数据上训练其 src→src 重建能力**
- `decoder_dst` **只能在 dst 数据上训练其 dst→dst 重建能力**
- 换脸时 `decoder_src(dst_code)` 是一个"跨域"操作 → decoder_src 从未被训练过如何解码 dst_code → 这就是为什么需要 true_face + ct_mode

---

## 三、DF 所有参数深度解析

### 3.1 DF 特有的参数

| 参数 | 作用 | DF 中的意义 |
|------|------|------------|
| `true_face_power` | 编码级别的对抗损失 | **DF 核心参数**：0.01~0.1，让 src 和 dst 的中间编码分布对齐 |
| `ct_mode` | 颜色迁移 | **DF 强烈推荐**：推荐 `rct` 或 `lct`，消除色斑 |
| `freeze_inter` | 冻结 Inter 层 | 后期冻结瓶颈层，与 LIAE 的 freeze_inter_AB 类似 |

### 3.2 DF 与 LIAE 共享的参数（DF 中需要调整的）

| 参数 | DF 中的建议 | 与 LIAE 的差异 |
|------|------------|---------------|
| `random_hsv_power` | 0.0~0.05 | LIAE 保持 0.0，DF 可稍微开启（因为颜色是弱点） |
| `archi` | `df-ud`（推荐）| DF 基础架构 |

### 3.3 架构参数（一次性设定）

| 参数 | 默认值 | DF 建议 |
|------|--------|---------|
| `archi` | df-ud | DF 也可以加 t 变体（df-udt），推理同上篇 |
| `resolution` | 256 | 同 LIAE |
| `ae_dims` | 256 | 同 LIAE |
| `e_dims` | 64 | 同 LIAE |
| `d_dims` | 64 | 同 LIAE |

---

## 四、DF 分阶段训练方案（核心）

### 完整训练路线图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        DF 完整训练路线图                                  │
├──────────┬──────────────┬──────────────┬──────────────┬─────────────────┤
│  阶段1   │   阶段2      │   阶段3      │   阶段4      │   阶段5(+6)     │
│ 颜色+结构 │ 身份对齐     │ 眼嘴细节     │ 关闭warp     │  GAN+冻结微调   │
├──────────┼──────────────┼──────────────┼──────────────┼─────────────────┤
│ warp  ✓  │ warp  ✓      │ warp  ✓      │ warp  ✗      │ warp  ✗        │
│ gan   0  │ gan   0      │ gan   0      │ gan   0      │ gan   0.1      │
│ em    ✗  │ em    ✗      │ em    ✓      │ em    ✓      │ em    ✓        │
│ tf    0  │ tf   0.01~0.05│ tf   保持    │ tf   0.005↓  │ tf   保持      │
│ ct  rct  │ ct  rct      │ ct  rct      │ ct  rct      │ ct  rct        │
│ lr 5e-5  │ lr  5e-5     │ lr  5e-5     │ lr  3e-5     │ lr  1e-5       │
│ lrd  n   │ lrd  n       │ lrd  n       │ lrd  y       │ lrd  y         │
├──────────┼──────────────┼──────────────┼──────────────┼─────────────────┤
│ 50k-150k  │ 20k-50k     │ 30k-60k      │ 10k-20k      │ 50k-150k(+10k)  │
├──────────┴──────────────┴──────────────┴──────────────┴─────────────────┤
│ ⚠️ 注意：ct_mode 从阶段1贯穿始终！DF不像LIAE，强烈建议不要关颜色迁移！     │
│ ⚠️ 注意：阶段4关闭warp后，true_face_power必须降低！                      │
└──────────────────────────────────────────────────────────────────────────┘
```

---

### 阶段1：颜色+结构学习

**目标**：建立基本的 src↔dst 重建能力，同时通过 ct_mode 让 decoder_src 学习在 dst 色域下工作。

**参数配置**：

```python
random_warp         = True       # 核心正则化
gan_power           = 0.0
eyes_mouth_prio     = False      # 结构都还没学好
true_face_power     = 0.0        # 先不开，编码还没稳定
ct_mode             = 'rct'      # ← DF专属：从第一天就开！
masked_training     = True
uniform_yaw         = True
random_hsv_power    = 0.0        # 可以稍后开启
lr                  = 5e-5
lr_dropout          = 'n'
freeze_inter        = False
```

**持续到何时**：
- src_loss 和 dst_loss 持续下降并趋于平稳
- 预览中 src→src 和 dst→dst 重建基本可辨
- 颜色/色调基本一致（ct_mode 的效果应该已经显现）
- 典型迭代数：**5万-15万次**

**为什么比 LIAE 多了 ct_mode**：
- 源码注释说的很清楚：`DF架构建议开启以缓解色斑问题`（saehd_param_defs.py 第 103 行）
- DF 的双解码器天然存在颜色不一致问题
- ct_mode 从阶段1就让 decoder_src 适应 dst 的光照条件
- **如果关了 ct_mode 训 DF，阶段 4-5 会出现严重的色斑，几乎无法修复**

---

### 阶段2：身份对齐——true_face

**目标**：通过 CodeDiscriminator 让 src 和 dst 的中间编码分布对齐，改善换脸质量。

**参数配置**：

```python
random_warp         = True       # 保持开启！
gan_power           = 0.0
eyes_mouth_prio     = False      # 暂不开启
true_face_power     = 0.03       # ← 核心变化！从0.01开始，逐步升到0.05
ct_mode             = 'rct'
masked_training     = True
uniform_yaw         = True
lr                  = 5e-5
lr_dropout          = 'n'
freeze_inter        = False
```

**持续到何时**：
- 预览中换脸质量（SD = pred_src_dst）显著改善
- 不再有明显的身伪失真和颜色偏差
- D_code_loss 波动趋于稳定（不再大起大落）
- 典型迭代数：**2万-5万次**

**为什么这个阶段特殊**：
- true_face 是一个**对抗性训练**。CodeDiscriminator 是判别器，G 和 D 交替优化
- 它不能和 eyes_mouth_prio 同时引入——两个都引入会互相干扰
- 如果在阶段1就开 true_face，编码还没稳定就开始对抗，训练容易崩溃
- true_face_power 从 0.01 开始，观察 5000 次后可以升到 0.03~0.05

**观察信号**：
- D_code_loss 如果突然上升很多 → true_face_power 太高，降低到一半
- G_loss 如果突然上升（排除其他因素）→ CodeDiscriminator 太强，降低 tf

> **true_face_power 的调节原则**：宁低勿高。0.01 的效果可能已经很好，不用追求 0.1。

---

### 阶段3：眼嘴细节精修

**目标**：在身份对齐的基础上，精细修复眼睛和嘴巴区域。

**参数配置**：

```python
random_warp         = True       # 保持
gan_power           = 0.0
eyes_mouth_prio     = True       # ← 核心变化！
true_face_power     = 0.03       # 保持阶段2的值
ct_mode             = 'rct'
masked_training     = True
uniform_yaw         = True
lr                  = 5e-5
lr_dropout          = 'n'
```

**持续到何时**：
- 眼睛清晰度、嘴巴形状明显改善
- 眼嘴细节不再提升
- 典型迭代数：**3万-6万次**

**注意**：眼嘴优先（300× L1 权重）和 true_face（编码级对抗）**可能产生轻微冲突**。如果发现眼睛突然变差，可能需要在两者之间取舍——先关 em 再微调 true_face，或者降低 true_face 再开 em。

---

### 阶段4：关闭 warp——精准映射

**目标**：精确学习像素级映射，获得清晰的面部细节。

> **⚠️ DFL原版建议**：先开LRD（warp仍开）再关warp。DFL的help_message说"Enabled it before `disable random warp`"。建议在阶段3和4之间增加一个LRD过渡阶段（warp=True, lrd='y', lr_cos=500）。

**参数配置**：

```python
random_warp         = False      # ← 核心变化
gan_power           = 0.0
eyes_mouth_prio     = True
true_face_power     = 0.01       # ← ⚠️ 降低！降到阶段2的一半以下
ct_mode             = 'rct'
masked_training     = True
uniform_yaw         = False
lr                  = 3e-5       # ← 降低学习率
lr_dropout          = 'y'        # ← 开启LRD
lr_cos              = 500        # ← DFL原版开启lrd时自动设为500，我们需手动设
freeze_inter        = False
```

**持续到何时**：
- 清晰度大幅提升
- **关键判断**：出现色斑或过拟合迹象立即停止！
- 典型迭代数：**1万-2万次**（比 LIAE 更短！）

**为什么 DF 的这个阶段比 LIAE 更短且更危险**：

1. **DF 的过拟合风险比 LIAE 更高**。原因：
   - 双解码器 → decoder_src 关闭 warp 后会快速过拟合 src 的特定颜色/纹理
   - 没有 inter_B 的环境分离 → 颜色信息混在单一编码中 → 更容易"记忆"颜色
   - true_face 的 CodeDiscriminator 在关闭 warp 后也可能过拟合

2. **色斑是 DF 关 warp 的第一大杀手**：
   - 关 warp 后 decoder_src 记住 src 的颜色分布 → 换脸出现色斑
   - 即使有 ct_mode，关 warp 后的过拟合仍可能绕过 ct_mode 的正则化

3. **降低 true_face_power 的原因**：
   - 关 warp 后，G 和 D 都更容易过拟合各自的分布
   - CodeDiscriminator 可能快速"学会区分"两个过拟合的编码 → 产生反向效果

**⚠️ DFL行为差异**：
- **DFL原版**在`lr_dropout='y'`时自动设置`lr_cos=500`，我们代码需手动设置
- **pretrain切换**：DFL原版在pretrain从True→False时重新初始化inter并重置iter为0，我们代码也有类似逻辑但未覆盖random_flip=True

---

### 阶段5：GAN 纹理增强

**目标**：增加真实皮肤纹理和毛孔细节。

**参数配置**：

```python
random_warp         = False
gan_power           = 0.1        # ← 核心变化
gan_patch_size      = 16
gan_dims            = 16
eyes_mouth_prio     = True
true_face_power     = 0.005~0.01 # 保持低值或关掉
ct_mode             = 'rct'
masked_training     = True
lr                  = 1e-5
lr_dropout          = 'y'
ramp_start_ratio    = 0.2
target_iter         = 50000
```

**持续到何时**：
- 皮肤纹理自然真实
- 出现伪影或奇怪纹理时停止
- 典型迭代数：**5万-15万次**

**GAN 对 DF 的特殊挑战**：
- GAN 判别器也作用在 `pred_src_src`（src 的 mask 内区域）上
- DF 的 pred_src_src 颜色本身就偏向 src → GAN 可能**放大色差**
- 如果 GAN 阶段出现严重色斑 → 提高 ct_mode 质量（换 mkl 或 sot）或关闭 GAN
- 可以开启 `vgg_perceptual_power=5.0` 来约束 GAN，减少塑料感

---

### 阶段6：冻结 inter 微调（可选）

**目标**：锁定中间编码，微调解码器以进一步适配 dst。

**参数配置**：

```python
freeze_inter        = True       # ← 核心！冻结瓶颈层
random_warp         = False
gan_power           = 0.05~0.1
eyes_mouth_prio     = True
true_face_power     = 0.0        # 冻结 inter 后不需要了
ct_mode             = 'rct'
lr                  = 5e-6       # 非常低
lr_dropout          = 'y'
```

**持续到何时**：
- 换脸效果稳定，身份一致性更好
- 环境融合更自然
- 典型迭代数：**1万-5万次**

**为什么 DF 用 freeze_inter 而不是 freeze_inter_AB**：
- DF 只有一个 inter，冻结它意味着冻结所有中间编码 → decoder_src 和 decoder_dst 不再接收新的编码信息
- 相当于在 DF 中实现类似 LIAE 的"冻结身份"效果
- `freeze_inter=True` + `true_face_power=0`（inter 已冻结，code discriminator 无意义）

---

### DF 各阶段切换判断信号总览

| 切换 | 信号 | DF 特有注意事项 |
|------|------|---------------|
| 1→2 | src/dst loss 平稳，SRC→SRC 重建可辨 | 颜色一致性 OK 才进入阶段2 |
| 2→3 | 换脸预览中人脸结构和色调基本正确 | true_face 效果达到预期 |
| 3→4 | 眼嘴细节清晰 | 确认换脸预览无严重色斑 |
| 4→5 | 清晰度足够 | ⚠️ **色斑出现立即停止阶段4！** |
| 5→6 | GAN 纹理稳定 | 可选 |

---

## 五、DF 架构修饰符 u/d/t/c 的影响

DF 和 LIAE 共用相同的 `Encoder`、`Inter`、`Decoder` 类，所以 `u`/`d`/`t`/`c` 修饰符在代码层面的作用是**完全相同的**。

### 5.1 修饰符在 DF 中的特殊表现

| 修饰符 | DF 中的额外影响 | 建议 |
|--------|---------------|------|
| `u` | 像素归一化稳定编码 → 帮 CodeDiscriminator 更稳定 | **强烈推荐** |
| `d` | depth_to_space 输出 → 关 warp 后色斑风险更高 | 阶段4 更短 |
| `t` | 更深网络 → 更多参数，换脸质量可能更好但参数量翻倍 | 高分辨率专用 |
| `c` | 余弦激活 → 可能产生不稳定颜色表现 | 不推荐 |

### 5.2 DF 各变体参数对比

| 变体 | VRAM | 阶段1 | 阶段2 | 阶段3 | 阶段4 | 阶段5 | 推荐 |
|------|------|-------|-------|-------|-------|-------|------|
| df-ud | 100% | 50k-150k | 20k-50k | 30k-60k | 10k-20k | 50k-150k | ⭐推荐 |
| df-u | ~92% | 50k-150k | 20k-50k | 30k-60k | 10k-20k | 50k-120k | 省显存 |
| df-udt | ~142% | 100k-300k | 30k-60k | 40k-80k | 8k-15k | 80k-200k | 高分辨率 |

> DF 需要两倍的 Decoder 内存（decoder_src + decoder_dst），所以相同参数下 DF 比 LIAE 多约 15-30% VRAM。

---

## 六、DF 各变体完整训练方案

### 方案 A：`df-ud`（DF 推荐方案）

```python
# 阶段1：颜色+结构学习 (50k-150k)
random_warp=True, gan_power=0.0, em=False, true_face_power=0.0
ct_mode='rct', masked_training=True, uniform_yaw=True
lr=5e-5, lr_dropout='n'

# 阶段2：身份对齐 (20k-50k)
# 变化：true_face_power=0.03 (从0.01逐步升)

# 阶段3：眼嘴细节 (30k-60k)
# 变化：em=True

# 阶段4：精准映射 (10k-20k) ⚠️
# 变化：random_warp=False, lr=3e-5, lrd='y', true_face_power=0.01

# 阶段5：GAN纹理 (50k-150k)
# 变化：gan_power=0.1, lr=1e-5, true_face_power=0.005~0.01
# 设置 target_iter=50000, ramp_start_ratio=0.2

# 阶段6：冻结inter (10k-50k, 可选)
# 变化：freeze_inter=True, true_face_power=0.0, lr=5e-6
```

### 方案 B：`df-u`（省显存）

与方案 A 一致，阶段4 可延长 20%（无 `d` 的 depth_to_space 伪影风险）。

### 方案 C：`df-udt`（高分辨率 384+）

```python
# 阶段1：颜色+结构学习 (100k-300k) ← 2倍
lr=4e-5, lr_cos=50000, 其他同方案A

# 阶段2：身份对齐 (30k-60k)
true_face_power=0.03

# 阶段3：眼嘴细节 (40k-80k)
em=True

# 阶段4：精准映射 (8k-15k) ⚠️ 更短！
lr=2e-5, lrd='y', true_face_power=0.005

# 阶段5：GAN (80k-200k)
gan_power=0.1, lr=1e-5

# 阶段6：冻结inter (10k-50k)
freeze_inter=True, true_face_power=0.0, lr=5e-6
```

---

## 七、DF vs LIAE 终极对比与选择指南

### 架构本质区别

| 维度 | DF | LIAE |
|------|----|------|
| **核心理念** | 双解码器各自专精 | 双瓶颈层显式分离 |
| **身份保留** | ⭐⭐⭐⭐ 强（decoder_src 专精） | ⭐⭐⭐ 中（inter_AB 共享训练） |
| **颜色一致性** | ⭐⭐ 弱（强烈推荐 ct_mode） | ⭐⭐⭐⭐ 强（inter_B 显式建模） |
| **训练稳定性** | ⭐⭐⭐ 中（ct_mode+ture_face 互相制衡） | ⭐⭐⭐⭐⭐ 强（架构天然稳定） |
| **换脸相似度** | ⭐⭐⭐⭐⭐ 强 | ⭐⭐⭐ 中 |
| **颜色匹配** | ⭐⭐ 取决于 ct_mode | ⭐⭐⭐⭐ 好 |
| **显存占用** | 高（双 decoder） | 低（单 decoder + 双 inter） |
| **训练时间** | 较长（6 阶段） | 中等（4-5 阶段） |
| **分阶段复杂度** | 高（ct_mode + true_face 调节） | 低（参数变化更直观） |

### 选择决策树

```
你的优先级是什么？
│
├─ 换脸后"像不像 src"最重要 → 选 DF
│   ├─ 能接受颜色微调和 ct_mode 调节 → df-ud
│   ├─ 高分辨率 → df-udt
│   └─ 不想调 ct_mode → 回到 LIAE
│
├─ 颜色自然/视频一致性最重要 → 选 LIAE
│   └─ 对眼嘴/GAN 阶段有更高宽容度
│
├─ 新手/不想折腾 → LIAE（少 2 个阶段，少色斑风险）
│
├─ src 和 dst 肤色差异大 → LIAE（颜色分离更好）
│
└─ src 和 dst 肤色接近 → DF（身份利用更充分）
```

### 关键警告

| 场景 | DF | LIAE |
|------|----|----|
| src 白人 + dst 亚洲人 | ❌ **色斑重灾区** | ✓ 颜色好 |
| src 和 dst 肤色相近 | ✓ 理想场景 | ✓ 也可以 |
| 追求极致像 src | ✓ **DF 的强项** | 中 |
| 视频换脸（帧间一致性） | 中（可能帧间色差）| ✓ 更好 |
| 训练时间有限 | ❌ 需要更多阶段 | ✓ 阶段少 |
| 显存不足 | ❌ VRAM 少时不行 | ✓ 更省显存 |

---

## 八、核心总结

### DF 训练十大黄金法则

1. **ct_mode 从第一天开到训练结束**。这是 DF 的第一原则，关了几乎必然出严重色斑。
2. **ct_mode 推荐 rct 或 lct**。rct 速度快效果好，lct 也可靠。mkl/sot/idt 留给极端情况。
3. **true_face_power 宁低勿高**。0.01-0.03 对大多数场景足够，盲目追求 0.1 容易训练崩盘。
4. **true_face 必须在 warp 开启时引入**。等编码稳定后再开（阶段2），不要在阶段1就开。
5. **关 warp 时必须降低 true_face_power**。阶段3的 0.03 要降到阶段4的 0.01 或更低。
6. **DF 的关 warp 阶段（阶段4）比 LIAE 更短**。双解码器过拟合更快，必须频繁检查预览。
7. **DF 的色斑是关 warp 后第一杀手**。一旦预览出现局部色块 → 立刻开始 GAN 阶段或回到阶段3。
8. **DF 整体需要更多训练迭代**。6 个阶段 vs LIAE 的 4-5 个阶段。
9. **DF 比 LIAE 更耗显存**。两个独立的 Decoder 比 LIAE 的单 Decoder + 双 Inter 多约 15-30% 参数。
10. **true_face_power=0 时 CodeDiscriminator 不会被创建**。所以如果决定不开 true_face，显存会省一点。

### DF 训练对比 LIAE 的关键差异速查

| | DF | LIAE |
|---|---|---|
| 阶段数 | **6 个**（多身份对齐+冻结） | 4-5 个 |
| ct_mode | **必开**（rct/lct） | 不需要 |
| true_face | ✓ 核心参数 | ❌ 不支持 |
| 关warp风险 | **极高**（色斑） | 中 |
| 关warp时长 | 10k-20k | 10k-30k |
| 冻结方式 | freeze_inter | freeze_inter_AB |
| 推荐架构 | df-ud | liae-ud |
| 推荐新手 | ❌ | ✓ |

### DF 训练的总建议

如果你追求**换脸后和 src 长得足够像**，DF 是正确的选择——但必须接受以下代价：

- 更长的训练时间和更多阶段
- ct_mode 的额外设置和调节
- true_face_power 的精细控制
- 更高的显存占用
- 关 warp 后色斑的持续风险

如果你优先追求**自然的颜色融合和训练稳定性**，LIAE 是更好的选择。

---

> **文档生成基于源码**：
> - `faceswap/models/saehd/saehd_model.py` — DF forward + true_face + GAN + loss 完整逻辑
> - `faceswap/models/saehd/saehd_arch.py` — Encoder/Inter/Decoder 架构定义
> - `faceswap/models/saehd/discriminators.py` — CodeDiscriminator + UNetPatchDiscriminator
> - `faceswap/models/saehd/losses.py` — dssim/ms_ssim/style_loss/VGG 损失函数
> - `faceswap/business/saehd_trainer.py` — TrainingConfig + SAEHDTrainer
> - `faceswap/business/base_trainer.py` — train_one_step + 优化器调度
> - `faceswap/gui_app/saehd_param_defs.py` — 全部参数定义与 tooltip 说明
> - `faceswap/core/saehd_utils.py` — compute_effective_gan_power 等工具函数
