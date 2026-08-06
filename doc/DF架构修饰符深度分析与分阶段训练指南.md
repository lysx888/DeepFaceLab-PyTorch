# DF 架构修饰符 (`u`/`d`/`t`/`c`) 深度对比与分阶段训练指南

> 基于 `saehd_arch.py` 和 `saehd_model.py` 源码级别分析
> 阅读前建议先读完《DF架构深度分析与分阶段训练指南》

---

## 零、前置知识：DF 与 LIAE 修饰符的核心差异

修饰符在 Encoder 和 Inter 中的行为与 LIAE **完全一致**（共用同一份代码），但在 DF 中有 **三个关键差异**：

| 维度 | LIAE | DF（本文重点） |
|------|------|---------------|
| Decoder 数量 | 1 个（共享） | **2 个**（`decoder_src` + `decoder_dst`） |
| 修饰符对 decoder 的开销 | ×1 | **×2**（两个 decoder 各自应用） |
| 瓶颈层 | inter_AB + inter_B（双路径） | inter（单路径，共享） |
| 身份对齐机制 | inter_AB 分离身份 | `true_face_power` 判别器 |

**核心结论**：修饰符对 decoder 的任何改动，在 DF 中成本翻倍；但对 inter/encoder 的改动成本与 LIAE 相同。

---

## 二、各修饰符在 DF 源码中的实际作用

### `u` —— 像素归一化 (Pixel Normalization)

**生效位置**：仅 Encoder（与 LIAE 完全相同）

```python:131:134:D:\AI\Inswapper\faceswap\models\saehd\saehd_arch.py
x = x.reshape(x.shape[0], -1)

if self._use_u:
    x = pixel_norm(x)
```

**DF 中的影响**：
- 编码器输出全部归一化到超球面 → 共享 inter 入口处的特征分布更均匀
- **对 DF 特别重要**：DF 的 inter 是唯一瓶颈（没有 inter_AB/inter_B 的双路径分离），`u` 帮助防止 encoder 把 SRC 和 DST 特征映射到非常不同的尺度上去
- 零参数，零额外开销

---

### `d` —— 密集输出连接 (Dense Output Connections)

**两处生效**：

#### ① Inter 层 —— 瓶颈更紧凑

```python:154:D:\AI\Inswapper\faceswap\models\saehd\saehd_arch.py
lowest_dense_res = resolution // (32 if 'd' in opts else 16)
```

有 `d`：瓶颈网格缩小一倍（如 256→8×8；无 `d` 则是 16×16）。

#### ② 两个 Decoder 的图像输出端 —— 四路卷积 + depth_to_space

```python:254:263:D:\AI\Inswapper\faceswap\models\saehd\saehd_arch.py
if self._use_d:
    x_pre_dts = torch.cat((self.out_conv(x),      # 1×1 conv
                   self.out_conv1(x),              # 3×3 conv
                   self.out_conv2(x),              # 3×3 conv
                   self.out_conv3(x)), dim=1)      # 3×3 conv  → 4路12通道
    x = depth_to_space(x_pre_dts, 2)               # 12ch → 3ch，同时2x上采样
    x = torch.sigmoid(x)
```

无 `d`：单层 1×1 卷积 → sigmoid。

#### ③ Mask 分支也受影响

```python:221:236:D:\AI\Inswapper\faceswap\models\saehd\saehd_arch.py
self.upscalem1 = Upscale(d_mask_ch * 8, d_mask_ch * (8 if use_t else 4), ...)
if use_t:
    self.upscalem2 = Upscale(d_mask_ch * 8, d_mask_ch * 4, ...)
    self.upscalem3 = Upscale(d_mask_ch * 4, d_mask_ch * 2, ...)
    if use_d:
        self.upscalem4 = Upscale(d_mask_ch * 2, d_mask_ch, ...)  # 多一层
else:
    self.upscalem2 = Upscale(d_mask_ch * 4, d_mask_ch * 2, ...)
    if use_d:
        self.upscalem3 = Upscale(d_mask_ch * 2, d_mask_ch, ...)  # 多一层
```

`d` 让 mask 分支多一个 upscale 层（更精细的 mask 预测）。

**DF 中的独特开销**：两个 decoder 各一套 4 路 out_conv + mask 扩展，成本 **×2**。

---

### `t` —— 更深网络 (Transformer-like deeper)

**三处生效**（与 LIAE 完全相同，但 decoder 部分 ×2）：

#### ① Encoder：5 次下采样 + 2 个残差块

```python:102:109:D:\AI\Inswapper\faceswap\models\saehd\saehd_arch.py
if use_t:
    self.down1 = Downscale(in_ch, e_ch, kernel_size=5, ...)
    self.res1 = ResidualBlock(e_ch, kernel_size=3, ...)
    self.down2 = Downscale(e_ch, e_ch * 2, ...)
    self.down3 = Downscale(e_ch * 2, e_ch * 4, ...)
    self.down4 = Downscale(e_ch * 4, e_ch * 8, ...)
    self.down5 = Downscale(e_ch * 8, e_ch * 8, ...)
    self.res5 = ResidualBlock(e_ch * 8, kernel_size=3, ...)
```

#### ② Inter：取消 upscale1

```python:162:163:D:\AI\Inswapper\faceswap\models\saehd\saehd_arch.py
if not use_t:
    self.upscale1 = Upscale(ae_out_ch, ae_out_ch, ...)  # 有t时不创建
```

#### ③ 两个 Decoder：各加一对 upscale+residual

```python:199:210:D:\AI\Inswapper\faceswap\models\saehd\saehd_arch.py
if use_t:
    self.upscale1 = Upscale(d_ch * 8, d_ch * 8, ...)
    self.res1 = ResidualBlock(d_ch * 8, ...)
    self.upscale2 = Upscale(d_ch * 8, d_ch * 4, ...)
    self.res2 = ResidualBlock(d_ch * 4, ...)
    self.upscale3 = Upscale(d_ch * 4, d_ch * 2, ...)
    self.res3 = ResidualBlock(d_ch * 2, ...)
else:
    self.upscale1 = Upscale(d_ch * 8, d_ch * 4, ...)
    self.res1 = ResidualBlock(d_ch * 4, ...)
    self.upscale2 = Upscale(d_ch * 4, d_ch * 2, ...)
    self.res2 = ResidualBlock(d_ch * 2, ...)
```

`t` 变体 decoder 有 **3 对** upscale+residual（无 `t` 只有 2 对）。

Mask 分支同样多一层：
```python:222:224:D:\AI\Inswapper\faceswap\models\saehd\saehd_arch.py
if use_t:
    self.upscalem2 = Upscale(d_mask_ch * 8, d_mask_ch * 4, ...)
    self.upscalem3 = Upscale(d_mask_ch * 4, d_mask_ch * 2, ...)
```

**DF 中的独特开销**：两个 decoder 各加一对 upscale+res，成本约为 LIAE 的 **2.2 倍**。

---

### `c` —— 余弦激活 (Cosine Activation)

全部用 `x * cos(x)` 替代 `LeakyReLU(0.1)`。行为与 LIAE 完全相同。实验性，不推荐常规使用。

---

## 三、各变体参数量和显存对比（256 分辨率估算）

| 变体 | Encoder | Inter | 2×Decoder | 总参数量 | 相对VRAM |
|------|---------|-------|-----------|---------|---------|
| `df`（裸） | 4 down | 1 us, 2 dense | 2×(2 us+res, 1 out) | 100% | 100% |
| `df-u` | +pixel_norm | 同 | 同 | ~100% | ~100% |
| `df-d` | 同 | Inter压缩→res/32 | 2×(3 out_conv, mask+1us) | ~118% | ~122% |
| `df-t` | 5 down+2 res | 无 upscale1 | 2×(3 us+res 对), mask+1us | ~170% | ~175% |
| `df-ud` | +pixel_norm | Inter压缩 | 2×(3 out_conv, mask+1us) | ~118% | ~122% |
| `df-ut` | +pixel_norm, 5down+2res | 无 upscale1 | 2×(3 us+res 对), mask+1us | ~170% | ~175% |
| `df-dt` | 5down+2res | Inter压缩+无us1 | 2×(3 us+res 对+4 out_conv) | ~195% | ~210% |
| `df-udt` | +pixel_norm, 5down+2res | Inter压缩+无us1 | 2×(3 us+res 对+4 out_conv) | ~195% | ~210% |

> **关键对比**：`df-udt` 的 VRAM 需求是 `liae-udt` 的约 **1.5 倍**（因为两个 decoder）。

---

## 四、关键分辨率约束

```python:77:81:D:\AI\Inswapper\faceswap\models\saehd\saehd_model.py
has_d = 'd' in c.archi_opts
divisor = 32 if has_d else 16
if c.resolution < 64 or c.resolution % divisor != 0:
    raise ValueError(...)
```

| 架构 | 分辨率约束 | 合法值示例 |
|------|-----------|-----------|
| 无 `d` | 16 的倍数 | 64, 80, 96, 112, **128**, 144, 160, 176, 192, 208, 224, 240, **256**... |
| 有 `d` | **32** 的倍数 | 64, 96, **128**, 160, 192, 224, **256**, 288, 320, 352, 384... |

---

## 五、各变体下游分辨率与 bottleneck 大小

默认维度 `ae_dims=256, e_dims=64, d_dims=64`：

| 分辨率 | 变体 | Encoder 输出 | Inter(bottleneck) | Inter 输出 | Decoder 起始 |
|--------|------|-------------|-------------------|-----------|-------------|
| 128 | df | 8×8×512 | 8×8×256 → dense | 16×16×256 | 16×16 |
| 128 | df-d | 8×8×512 | **4×4**×256 → dense | 8×8×256 | 8×8 |
| 128 | df-t | **4×4**×512 | 8×8×256 → dense | 8×8×256 | 8×8 |
| 128 | df-udt | **4×4**×512 | **4×4**×256 → dense | **4×4**×256 | **4×4** ⚠️ |
| 256 | df | 16×16×512 | 16×16×256 | 32×32×256 | 32×32 |
| 256 | df-d | 16×16×512 | 8×8×256 | 16×16×256 | 16×16 |
| 256 | df-t | 8×8×512 | 16×16×256 | 16×16×256 | 16×16 |
| 256 | df-udt | 8×8×512 | 8×8×256 | 8×8×256 | 8×8 |
| 384 | df-ud | 24×24×512 | 12×12×256 | 24×24×256 | 24×24 |
| 384 | df-udt | 12×12×512 | 12×12×256 | 12×12×256 | 12×12 |

**关键发现**：
- `df-udt` 在 128 分辨率下 bottleneck 仅 4×4 — 太小了！两个 decoder 要从 4×4 重建到 128×128，训练极其困难
- `df-t` 变体建议 **256 分辨率起步**，`df-udt` 建议 **320 分辨率起步**
- `df-d`（无 t）在 128 分辨率下是可行的（8×8 bottleneck）

---

## 六、各变体对 DF 六个训练阶段的差异化影响

### 速查总表

| 阶段 | `u` 的影响 | `d` 的影响 | `t` 的影响 | `c` 的影响 |
|------|-----------|-----------|-----------|-----------|
| **阶段1** 颜色+结构 | 收敛更稳，可用稍高LR | 无明显差异 | ⚠️ **需 1.5-2× 迭代** | 可能震荡，降 LR |
| **阶段2** 身份对齐 | 对 code_discriminator 无直接影响 | inter 瓶颈更小 → true_face 信号更强 | 深 encoder → 代码判别器更强力 | 不确定 |
| **阶段3** 眼嘴细节 | 无影响 | 多路 out_conv 捕获眼嘴更好 | 残差块保留高频好 | 不确定 |
| **阶段4** 关 warp | 正则化够，可更短 | ⚠️ 过拟合风险高 | ⚠️⚠️ **过拟合风险极高** | 不确定 |
| **阶段5** GAN | 无影响 | GAN+depth_to_space 纹理好 | **纹理优势最大** | 可能伪影 |
| **阶段6** 冻结 inter | 无影响 | **必须做！** inter 太小 | **强烈建议！** | 无影响 |

---

### 6.1 `u` (像素归一化) —— DF 下仍然必选

**本质**：零参数正则化。

**DF 独特优势**：
- DF 只有 **一个共享 inter**，比 LIAE 的双瓶颈更依赖 encoder 输出稳定性
- `u` 确保 SRC 和 DST 特征映射到同一超球面 → inter 学习更容易
- 与 LIAE 一样：阶段 1 可用稍高 LR（6e-5 而非 5e-5），其余阶段一致

**结论**：`u` 对 DF 甚至比 LIAE 更重要，**必选**。

---

### 6.2 `d` (密集输出连接) —— DF 中成本较高但值得

**DF 独特考量**：

**阶段1-3**：与 LIAE 基本一致，无特殊变化。

**阶段2（身份对齐）的特殊交互**：
- `d` 让 inter 瓶颈从 res/16 压缩到 res/32
- 更小的瓶颈 → 代码判别器更难区分 SRC/DST 特征 → **`true_face_power` 的对抗信号更强**
- 这其实是好事：更强的对抗让 decoder_src 更努力地提取身份信息
- 但需要把 `true_face_power` 调得更保守：**0.005~0.02**（vs 标准 0.01~0.03）

**阶段4（关 warp）的特殊风险**：
- `d` 的 4 路 out_conv 各自有不同感受野 → 关 warp 后容易过拟合到训练集的特定纹理模式
- 加上 DF 有两个 decoder → 过拟合的参数量翻倍
- **阶段4 应该缩短到 5k-12k**（vs 标准 10k-20k）

**阶段5（GAN）**：
- depth_to_space 输出 + GAN 判别器 → 纹理非常自然
- 两个 decoder 各自受益于 GAN，效果显著

**阶段6（冻结 inter）**：
- `d` 让 inter 瓶颈更小 → inter 承载的信息更紧凑 → **必须冻结 inter**
- 冻结后两个 decoder 各自微调到最优

**特殊建议**：
- 预览中注意是否出现**网格状伪影**（depth_to_space 的典型副作用）
- 一旦出现 → 降低学习率或进入下一阶段
- `df-d` 的 ct_mode 建议用 `rct`（更强的颜色迁移可以掩盖微小的深度到空间伪影）

---

### 6.3 `t` (更深网络) —— DF 下差异最大的修饰符

**这是对 DF 影响最大的修饰符**。

#### 阶段1（颜色+结构）—— 显著变慢

- 5 层 encoder + 3 对 decoder upscale/res → **参数量是裸 DF 的 1.7 倍**
- 两个 decoder 都需要独立学习 → **收敛时间是裸 DF 的 2 倍**
- 建议学习率降到 **3e-5 ~ 4e-5**
- 128 分辨率不推荐 `df-t`：decoder 起始仅 8×8，太难

```
df-t 阶段1: warp✓ gan0 ct=rct em✗ lr=4e-5 lrd=n  → 100k-300k (2倍!)
```

#### 阶段2（身份对齐）—— true_face 判别器更强力

- 更深 encoder 产生的代码特征更丰富 → `code_discriminator` 的判别能力更强
- 好处：身份对齐效果更好
- 风险：`true_face_power` 过高会导致 decoder_src 过度关注身份而忽略颜色
- **建议 `true_face_power` 降到 0.005~0.015**（标准 0.01~0.03）

```
df-t 阶段2: true_face=0.01 lr=4e-5  → 30k-80k
```

#### 阶段3（眼嘴细节）—— 残差块优势

- `t` 的 3 对 residual block（vs 普通 2 对）让高频细节传播更充分
- 眼嘴优先（300× L1）的梯度能穿过更多残差连接
- 迭代数不需要额外延长，质量更好

#### 阶段4（关 warp）—— ⚠️ DF-t 最危险的阶段

这是 `df-t` 最容易翻车的阶段，比 `liae-t` 更危险：

| 因素 | LIAE-t | DF-t | 原因 |
|------|--------|------|------|
| Decoder 数量 | 1 个 | **2 个** | 翻倍的过拟合参数量 |
| 残差块总数 | 3 个 | **6 个**（2×3） | 每个残差块都能"记忆"训练数据 |
| 瓶颈深度 | inter_AB+inter_B | **单 inter** | 共享瓶颈更容易被 decoder 利用 |

**DF-t 阶段4 的铁律**：
- 迭代数：**3k-8k**（标准 DF 是 10k-20k，减半再减半）
- **每 1k 次检查一次预览**
- 判断标准不是 loss，而是：
  1. 出现**色斑**（颜色不均匀的块）→ 立即停止
  2. 牙齿/眼睛开始变模糊 → 立即停止
  3. SRC 预览中出现 DST 的颜色污染 → 立即停止
- 一旦出现任何上述信号，立刻进入阶段5（GAN 可以"修复"轻微过拟合）

```
df-t 阶段4: warp✗ gan0 true_face=0.005 lr=2e-5 lrd=y  → 3k-8k ⚠️
```

#### 阶段5（GAN）—— `t` 的纹理优势最大化

- 这是 `df-t` 投资回报最高的阶段
- 2 个深度 decoder 在 GAN 对抗下能产生最细腻的纹理
- **建议阶段5 延长到 80k-200k**
- GAN 阶段能部分"修复"阶段4 的轻微过拟合

```
df-t 阶段5: warp✗ gan=0.1 true_face=0.005 lr=1e-5 lrd=y  → 80k-200k
```

#### 阶段6（冻结 inter）—— 强烈建议

- `df-t` 的 inter 是单点瓶颈 → 冻结后两个 decoder 各自微调
- 迭代数：20k-80k

```
df-t 阶段6: freeze_inter=true lr=5e-6  → 20k-80k
```

---

### 6.4 `c` (余弦激活) —— 实验性，DF 下不推荐

- 余弦梯度 `cos(x) - x*sin(x)` 在深层网络中更不稳定
- DF 有两个 decoder → 梯度不稳定被放大
- 如果强行用：学习率降到 2e-5，开启 `clipgrad=True`

---

## 七、不同变体的完整训练方案

### 方案 A：`df-ud`（推荐，DF 标准方案）

参照《DF架构深度分析与分阶段训练指南》的标准六阶段方案。

```
阶段1: warp✓ gan0 ct=rct em✗ lr=5e-5  lrd=n  tf=0     → 50k-150k
阶段2: warp✓ gan0 ct=rct em✗ lr=5e-5  lrd=n  tf=0.02  → 20k-50k
阶段3: warp✓ gan0 ct=rct em✓ lr=5e-5  lrd=n  tf=0.02  → 30k-60k
阶段4: warp✗ gan0 ct=rct em✓ lr=3e-5  lrd=y  tf=0.005 → 5k-12k (d缩短)
阶段5: warp✗ gan0.1 em✓   lr=1e-5  lrd=y  tf=0.005 → 50k-150k
阶段6: freeze_inter=true   lr=5e-6  tf=0               → 20k-50k
```

---

### 方案 B：`df-u`（省显存方案）

与方案 A 几乎一致，差异：
- 少了 `d`，阶段4 可以稍长：8k-20k（不容易出现 depth_to_space 网格伪影）
- 阶段5 GAN 纹理略逊于 `df-ud`，但不影响整体质量
- 适合显存吃紧时使用

```
阶段1: warp✓ gan0 ct=rct em✗ lr=5e-5  lrd=n  tf=0     → 50k-150k
阶段2: warp✓ gan0 ct=rct em✗ lr=5e-5  lrd=n  tf=0.02  → 20k-50k
阶段3: warp✓ gan0 ct=rct em✓ lr=5e-5  lrd=n  tf=0.02  → 30k-60k
阶段4: warp✗ gan0 ct=rct em✓ lr=3e-5  lrd=y  tf=0.005 → 8k-20k
阶段5: warp✗ gan0.1 em✓   lr=1e-5  lrd=y  tf=0.005 → 50k-120k
阶段6: freeze_inter=true   lr=5e-6  tf=0               → 20k-50k
```

---

### 方案 C：`df-d`（不推荐⚠️，缺少 `u` 的稳定化）

少了 `u` 的正则化，encoder 输出可能极端。DF 共享 inter 的特性让这个问题更严重——SRC 和 DST 的特征尺度不一致时 inter 学习困难。

```
阶段1: warp✓ gan0 ct=rct em✗ lr=3e-5  lrd=n  tf=0     → 80k-200k (更长)
阶段2: warp✓ gan0 ct=rct em✗ lr=3e-5  lrd=n  tf=0.01  → 30k-60k (tf降低)
阶段3: warp✓ gan0 ct=rct em✓ lr=3e-5  lrd=n  tf=0.01  → 40k-80k
阶段4: warp✗ gan0 ct=rct em✓ lr=2e-5  lrd=y  tf=0.005 → 3k-8k (极短⚠️)
阶段5: warp✗ gan0.1 em✓   lr=1e-5  lrd=y  tf=0.005 → 50k-150k
阶段6: freeze_inter=true   lr=5e-6  tf=0               → 20k-50k
```

**不推荐理由**：加 `u` 只多 0 参数，但稳定性提升 30%+。

---

### 方案 D：`df-udt`（高分辨率推荐，320+）

**这是差异最大、风险最高的方案。只建议有经验的用户使用。**

**前提条件**：
- 分辨率 ≥ 320（256 勉强可行但 bottleneck 仅 8×8）
- VRAM ≥ 16GB（384 分辨率下约需要 18-22GB）
- 有足够的耐心（总训练时间是 `df-ud` 的 1.8~2.2 倍）

```
阶段1: warp✓ gan0 ct=rct em✗ lr=3e-5  lrd=n  tf=0     → 120k-350k (2倍!)
阶段2: warp✓ gan0 ct=rct em✗ lr=3e-5  lrd=n  tf=0.008 → 40k-100k (tf降低!)
阶段3: warp✓ gan0 ct=rct em✓ lr=3e-5  lrd=n  tf=0.008 → 50k-100k
阶段4: warp✗ gan0 ct=rct em✓ lr=2e-5  lrd=y  tf=0.003 → 2k-6k (极短!!!⚠️)
阶段5: warp✗ gan0.1 em✓   lr=1e-5  lrd=y  tf=0.003 → 100k-250k (1.5倍)
阶段6: freeze_inter=true   lr=5e-6  tf=0               → 30k-100k
```

**关键调整理由**：

1. **阶段1×2**：两个深度 decoder 需要更长时间收敛
2. **true_face_power 降到 0.005~0.01**：更深 encoder 让代码判别器更强大，不需要高 tf
3. **阶段4÷3**：2 个 decoder × 3 对残差块 = **极易过拟合**。每 500~1000 次检查一次预览
4. **阶段5×1.5**：`t` 的高频表达能力在 GAN 阶段充分释放
5. 建议搭配 `lr_cos=50000`（余弦退火），帮助深网络跳出局部最优

**阶段4 的死亡信号（出现即停止）**：
- ❌ 色斑（局部颜色异常块）
- ❌ 牙齿/眼白开始变浑浊
- ❌ SRC 输出中出现 DST 的颜色基调
- ❌ 预览中人脸轮廓开始变形
- ❌ Mask 预测出现明显退化（mask 边界变模糊）

---

### 方案 E：`df-ut` / `df-dt`（中间方案）

#### `df-ut`（有 t 无 d）

介于方案 A 和方案 D 之间。少了 `d` 的 depth_to_space 输出 → 阶段4 风险比 `df-udt` 低，但比 `df-ud` 高。

```
阶段1: warp✓ gan0 ct=rct em✗ lr=4e-5  lrd=n  tf=0     → 100k-300k
阶段2: warp✓ gan0 ct=rct em✗ lr=4e-5  lrd=n  tf=0.01  → 30k-80k
阶段3: warp✓ gan0 ct=rct em✓ lr=4e-5  lrd=n  tf=0.01  → 40k-80k
阶段4: warp✗ gan0 ct=rct em✓ lr=2e-5  lrd=y  tf=0.005 → 4k-10k
阶段5: warp✗ gan0.1 em✓   lr=1e-5  lrd=y  tf=0.005 → 80k-200k
阶段6: freeze_inter=true   lr=5e-6  tf=0               → 30k-80k
```

#### `df-dt`（有 d 有 t，无 u）

与方案 D 一致但稳定性差。**不推荐**，要么加 `u` 生成 `df-udt`。

---

## 八、架构选择决策树（DF 版）

```
你追求什么？分辨率多少？显存多大？
│
├─ 通用效果最好，128-384 分辨率
│   └─ df-ud（无脑选这个）
│       显存 ≥ 8GB → 256 分辨率
│       显存 = 6GB → 128-192 分辨率
│
├─ 显存紧张，128-256 分辨率
│   └─ df-u（省 ~20% VRAM）
│       质量影响：~5% 纹理损失（可接受）
│
├─ 高分辨率 320+，显存充裕（16GB+），追求极致细节
│   └─ df-udt（训练时间翻倍，风险高）
│       ├─ 320 分辨率 → 起点 10×10 bottleneck
│       ├─ 384 分辨率 → 起点 12×12 bottleneck（推荐）
│       └─ 512 分辨率 → 起点 16×16 bottleneck（理想！但 VRAM 爆炸）
│
├─ 高分辨率 320+，显存一般（12GB）
│   └─ df-ut（比 udt 省 ~20% VRAM，阶段4 风险也低一些）
│
├─ 中高分辨率 256-384，想要好纹理但不敢上 t
│   └─ df-ud（d 的 depth_to_space 已经能显著改善纹理）
│
├─ 追求极致身份相似度
│   └─ df-udt（t 变体 + 代码判别器 = 最强身份对齐）
│       └─ true_face_power=0.008~0.015（不要太高）
│
└─ 实验/研究
    └─ df-udc（余弦激活，行为不同）
        但强烈不建议在生产中使用
```

---

## 九、DF 各变体阶段长度对比总表

| 阶段 | `df-ud` (基准) | `df-u` | `df-d` (无u) | `df-udt` | `df-ut` |
|------|---------------|--------|-------------|----------|---------|
| 阶段1 迭代 | 50k-150k | 50k-150k | 80k-200k | **120k-350k** | 100k-300k |
| 阶段1 学习率 | 5e-5 | 5e-5 | 3e-5 | **3e-5** | 4e-5 |
| 阶段2 迭代 | 20k-50k | 20k-50k | 30k-60k | **40k-100k** | 30k-80k |
| 阶段2 tf_power | 0.02 | 0.02 | 0.01 | **0.008** | 0.01 |
| 阶段3 迭代 | 30k-60k | 30k-60k | 40k-80k | **50k-100k** | 40k-80k |
| 阶段4 迭代 | 5k-12k | 8k-20k | 3k-8k | **2k-6k** ⚠️ | 4k-10k |
| 阶段4 风险 | 中 | 低 | 高 | **极高** ⚠️ | 高 |
| 阶段4 tf_power | 0.005 | 0.005 | 0.005 | **0.003** | 0.005 |
| 阶段5 迭代 | 50k-150k | 50k-120k | 50k-150k | **100k-250k** | 80k-200k |
| 阶段6 迭代 | 20k-50k | 20k-50k | 20k-50k | **30k-100k** | 30k-80k |
| GAN 纹理收益 | 好 | 标准 | 很好 | **极好** | 很好 |
| 推荐分辨率 | 128-384 | 128-384 | 128-384 | **320-512** | 256-384 |
| 推荐 VRAM | 8GB+ | 6GB+ | 8GB+ | **16GB+** | 10GB+ |
| 训练稳定性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

---

## 十、DF 修饰符的核心安全守则

### 黄金法则

1. **`u` 必加**：零成本，纯收益。对于共享 inter 的 DF 尤为重要
2. **`d` 看显存**：显存够就加，纹理收益明显；不够就 `df-u`
3. **`t` 看分辨率+显存**：320 以下不建议；VRAM 需 12GB+
4. **`c` 不要碰**：实验性的，DF 双 decoder 会放大不稳定性
5. **阶段4 是鬼门关**：`t` 变体在此最容易翻车，宁可早切不可贪
6. **true_face_power 随架构调整**：`t` 变体用更低的值（0.005~0.01）

### `d` + `t` 的组合效应

```
df-udt 的 mask 分支路径（最复杂情况）：

图像分支：upscale0→res0→us1→res1→us2→res2→us3→res3→[4路out_conv]→dts→sigmoid
Mask分支：upscalem0→uscm1→uscm2→uscm3→uscm4→out_convm→sigmoid
                    ↑t加   ↑基础  ↑t加  ↑d加

Mask 分辨率路径（256 下）：
  输入 8×8 → 16→32→64→128→256(如果d)
  d 额外让 mask 分辨率达到完整 256，提供最精细的 mask
```

---

## 十一、DF 与 LIAE 修饰符选择的交叉对比

| 需求 | 推荐 LIAE | 推荐 DF | 原因 |
|------|----------|---------|------|
| 通用/入门 | `liae-ud` | `df-ud` | 最稳定，最成熟 |
| 显存紧张 | `liae-u` | `df-u` | 省 decoder 开销 |
| 高分辨率细节 | `liae-udt` | `df-udt` | `t` 提供最强高频表达 |
| 换脸相似度优先 | `liae-ud` | **`df-udt`** | DF + t 身份对齐最强 |
| 颜色一致性优先 | **`liae-ud`** | `df-ud` | LIAE 天然颜色好 |
| 训练时间紧迫 | `liae-ud` | `df-ud` | 最快收敛 |
| 512+ 超高分辨率 | `liae-udt` | `df-udt` | `t` 变体 encoder 下采样合理 |

**终极建议**：

- **90% 的用户**：用 `df-ud` 或 `liae-ud`，选择哪个取决于你更在乎相似度（DF）还是颜色一致性（LIAE）
- **追求极致+有经验+有硬件**：用 `df-udt` 或 `liae-udt`（384 分辨率 + 16GB VRAM）
- **永远不加 `c`**：除非你在做研究
- **永远加 `u`**：无论选什么架构，`u` 是必选项
