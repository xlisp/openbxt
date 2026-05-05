# OpenBXT — 面向天文图像的开源 AI 反卷积工具

OpenBXT 是 [BlurXTerminator (BXT)](https://www.rc-astro.com/software/bxt/) 核心思想的开源 PyTorch 实现：
一个**专门为线性天文图像设计**的 AI 反卷积 / 像差校正网络。

它**不是**一个生成式超分辨率模型。设计目标是：
**只恢复图像中真实存在（受光学带宽限制）的低对比细节，绝不凭空构造结构。**

---

## 目录

- [一、为什么需要 OpenBXT](#一为什么需要-openbxt)
- [二、特性](#二特性)
- [三、安装](#三安装)
- [四、快速上手](#四快速上手)
- [五、目录结构](#五目录结构)
- [六、关键设计解释](#六关键设计解释)
  - [6.1 反卷积本质上是病态问题](#61-反卷积本质上是病态问题)
  - [6.2 残差有界输出：不会凭空捏造结构](#62-残差有界输出不会凭空捏造结构)
  - [6.3 PSF 由网络作为条件输入](#63-psf-由网络作为条件输入)
  - [6.4 空间变化的像差合成](#64-空间变化的像差合成)
  - [6.5 星点 / 星云双头分离](#65-星点--星云双头分离)
  - [6.6 多分量损失函数](#66-多分量损失函数)
  - [6.7 在线合成数据管线](#67-在线合成数据管线)
  - [6.8 分块推理 + 余弦窗融合](#68-分块推理--余弦窗融合)
- [七、与原版 BXT 的差异](#七与原版-bxt-的差异)
- [八、训练建议](#八训练建议)
- [九、模型如何避免"幻觉"](#九模型如何避免幻觉)
- [十、许可证](#十许可证)

---

## 一、为什么需要 OpenBXT

通用的 AI 锐化工具（Topaz, Photoshop AI 等）虽然普及，但它们存在两个根本性问题：

1. **训练数据不含天文图像**——模型无法正确处理星点（点扩散函数）。
2. **属于生成式模型**——为了"看起来更锐"，会编造原图中不存在的细节。

天文摄影对真实性要求极高：一颗本不存在的恒星、或一段被网络"想象"出来的星云结构，会让科研观测和深空摄影都失去意义。
BXT 通过精心设计的网络架构与训练流程解决了这个问题；OpenBXT 则把这套方法学开源化。

---

## 二、特性

- 基于 U-Net 的 PSF 条件残差预测网络
- 通过学习的 star map 联合处理星点与非星点区域
- 支持**空间变化**的 PSF 与光学像差：
  - 散焦、慧差、像散、三叶差、球差（基于 Zernike 多项式）
  - 大气视宁度（Moffat / Gaussian）
  - 引导误差（运动模糊）
  - 横向 / 纵向色差
- 在线合成训练管线 — 不需要真值 sharp/blur 配对
- 任意大图的分块推理 + 重叠融合
- 支持 FITS / TIFF / PNG I/O

---

## 三、安装

```bash
pip install -r requirements.txt
```

主要依赖：`torch`, `numpy`, `scipy`, `astropy`, `pillow`, `tifffile`, `tqdm`。

CPU 可训练但很慢，建议 NVIDIA GPU（任意 6GB+ 显存）或 Apple Silicon。

---

## 四、快速上手

### 训练

```bash
python train.py \
    --data_dir /path/to/sharp/linear/images \
    --out_dir runs/openbxt_v1 \
    --epochs 200 \
    --batch_size 8
```

输入目录里放**已对齐 / 已堆叠 / 锐利的线性**天文图像（FITS 优先，TIFF/PNG 也行）。
模糊与像差由代码在线合成，因此不需要人工准备 (blur, sharp) 配对。

### 推理

```bash
python infer.py \
    --input my_image.fits \
    --output my_image_sharp.fits \
    --weights runs/openbxt_v1/best.pt \
    --sharpen_nonstellar 0.9 \
    --sharpen_stellar 0.5
```

`sharpen_stellar` 与 `sharpen_nonstellar` 是 0~1 的强度滑块，分别控制
星点与星云的锐化幅度。这对应 BXT 中"对非星点施加更多锐化以避免 ringing"
的设计——见 [6.5](#65-星点--星云双头分离)。

---

## 五、目录结构

```
openbxt/
  __init__.py
  model.py        — U-Net、PSF 编码器、双头输出
  psf.py          — Moffat / Gaussian / Zernike PSF
  aberrations.py  — 空间变化的像差合成
  dataset.py      — 在线合成训练数据集
  star_detect.py  — DAOFIND 风格星点检测 + 经验 PSF 估计
  losses.py       — 通量守恒 + 边缘保持 + 星点掩膜的组合损失
  inference.py    — 分块推理引擎
  utils.py        — FITS / TIFF / PNG IO
train.py
infer.py
README.md
README-ZH.md
requirements.txt
```

---

## 六、关键设计解释

下面这一节是 OpenBXT 与一般"图像锐化网络"最本质的区别所在。

### 6.1 反卷积本质上是病态问题

数学上反卷积是个 ill-posed 问题：对同一张模糊输入，存在**无穷多张**清晰图像，
重新模糊后都能得到完全相同的输入。哪一张是对的？

经典算法（Richardson-Lucy、van Cittert 等）必须靠**先验知识**或**迭代约束**来选择一个解；
深度学习方法则把这种先验编码到网络权重里。

**OpenBXT 的策略是用"恢复真实细节"而非"猜想细节"作为先验**。具体怎么做？看下面几节。

### 6.2 残差有界输出：不会凭空捏造结构

`openbxt/model.py` 中 `OpenBXT` 输出的不是直接的 sharp 图像，而是一张**有界残差**：

```python
residual = torch.tanh(self.head_residual(d1)) * self.residual_scale  # residual_scale=0.5
sharp = (x + residual).clamp(0.0, 1.5)
```

- `tanh` 将每个像素的修正值限制在 `[-residual_scale, +residual_scale]`。
- 输出 = 输入 + 有界残差。

**为什么这样设计？**

- 当网络对某个区域**不确定**时（比如训练分布外的纹理），它会自然倾向于输出 ~0 残差，
  也就是把输入原样保留，而不是"赌一把"生成新结构。
- 这与生成式模型截然不同：生成式 GAN/Diffusion 的输出是"从零开始画出来的清晰图像"，
  而 OpenBXT 的输出是"对输入的微调"。错误模式的最坏情况是**没锐化**，而不是**幻觉**。

### 6.3 PSF 由网络作为条件输入

`PSFEncoder`（`model.py`）把一个 K×K 的 PSF 核压缩成 128 维向量，再通过
`ResBlock` 中的 FiLM 风格调制（`scale, shift`）注入网络的每一层：

```python
# ResBlock.forward
if self.cond is not None and c is not None:
    scale, shift = self.cond(c).chunk(2, dim=-1)
    h = h * (1 + scale[..., None, None]) + shift[..., None, None]
```

- **训练时**：合成阶段同时生成 (blurry, sharp, psf) 三元组，把 psf 作为条件输入，
  让网络学会"在这种 PSF 下应该怎么反卷积"。
- **推理时**：对任意一张真实图像，先用 `star_detect.estimate_psf` 从图中检测到的
  星点裁剪窗口求平均得到经验 PSF，再喂给网络。

这正对应 BXT 文档中所说："every star is a copy of the PSF"——星点就是 PSF 的真实样本。

### 6.4 空间变化的像差合成

真实望远镜的 PSF **不是**全图常数。视场边缘的星点比中心更宽、更扁、有更明显的慧差，
原因是慧差、像散、场曲、横向色差等像差与视场角强相关。

很多经典反卷积方法假设 PSF 在全图相同——这是它们效果差的主要原因之一。

OpenBXT 在 `openbxt/aberrations.py` 中**显式建模**了这个空间变化：

```python
# 在 G×G 网格上为每个 tile 单独生成一个 Zernike 像差 PSF
for gy in range(Gy):
    for gx in range(Gx):
        ny = (gy + 0.5) / Gy * 2 - 1
        nx = (gx + 0.5) / Gx * 2 - 1
        r = sqrt(ny**2 + nx**2)          # 归一化场半径
        scale = 1.0 + edge_aberration_boost * r   # 边缘像差更强
        coeffs = { 4..11: N(0, sigma * scale) }   # 散焦、像散、慧差、三叶差、球差
        # 慧差通常沿径向指向外
        coeffs[7] += radial_coma * ny
        coeffs[8] += radial_coma * nx
        kernels[gy, gx] = aberrated_psf(...)
```

然后用一个**双线性加权**的逐 tile 卷积把整张图卷出来：

```python
# 每个 tile 的卷积输出按距离到该 tile 中心的双线性权重融合
out = sum(conv_g * w_g for g in grid) / sum(w_g)
```

这样每个像素其实是受最近 4 个 tile 中心 PSF 加权的结果，得到平滑变化的空间像差。

为了让网络学会**根据视场位置**做不同的修正，输入端拼接了 3 个 CoordConv 通道
（`y`, `x`, `r`，均归一化到 [-1, 1]）：

```python
# model.py
coords = coord_channels(B, H, W, x.device, x.dtype)  # (B, 3, H, W)
h0 = self.in_conv(torch.cat([x, coords], dim=1))
```

### 6.5 星点 / 星云双头分离

星点和星云是两类完全不同的目标：

- **星点**：本身近似点源，与 PSF 直接卷积。强反卷积容易在边缘产生**黑环（ringing）**。
- **星云 / 星系**：低对比度延展结构，希望把所有可恢复的细节都拉出来。

经典反卷积一刀切，结果就是经常要么星点边缘有黑环，要么星云不够锐。

OpenBXT 的网络有**两个输出头**：

```python
# model.py 输出
{
    "sharp":      x + residual,           # 锐化结果
    "residual":   有界残差,
    "star_map":   sigmoid(star_logits),    # 星点概率图 (B, 1, H, W)
    "star_logits": ...,
}
```

推理时通过两个独立强度滑块对残差做空间加权：

```python
# 推理时 mix 残差
mix = star_map * stellar_strength + (1 - star_map) * nonstellar_strength
residual = residual * mix
```

- `stellar_strength = 0.5, nonstellar_strength = 0.9`：星云全力锐化，星点温和处理（避免 ringing）。
- `stellar_strength = 1.0, nonstellar_strength = 0.0`：只缩小星点不动星云。
- `stellar_strength = 0.0, nonstellar_strength = 1.0`：只锐化星云，星点保持原状。

`star_map` 由 `head_starmap` 输出，由 `losses.star_bce` 监督——而真值 star_mask
是在训练时用 LoG 检测器从 sharp 图直接生成的（`star_detect.detect_stars` + `stellar_mask`）。

### 6.6 多分量损失函数

`openbxt/losses.py` 中的 `OpenBXTLoss` 由四部分组成，每一部分都对应一个明确的物理 / 视觉目标：

```python
total =  w_l1   * l1_term       # ① 加权 L1 重建
       + w_grad * grad_term     # ② 梯度（边缘）匹配
       + w_flux * flux_term     # ③ 局部通量守恒
       + w_star * star_term     # ④ 星点掩膜 BCE
```

#### ① 加权 L1

```python
weight = 1.0 + star_weight_factor * star_mask   # star_weight_factor=5.0
l1_term = ((sharp - target).abs() * weight).mean()
```

星点像素加权 6×。星点在图像中面积小但视觉重要；不加权的话 L1 主要被星云背景主导，
网络不会认真学习星点。

#### ② 梯度损失

```python
def gradient_loss(pred, target):
    pgx, pgy = grad(pred);  tgx, tgy = grad(target)
    return |pgx - tgx| + |pgy - tgy|
```

匹配 Sobel 一阶梯度，鼓励网络在 target 真有边缘的地方恢复出对应的边缘——
而不是把所有梯度都拉强。这是与"暴力锐化"的关键区别。

#### ③ 局部通量守恒

```python
flux = avg_pool(pred, k=9) vs avg_pool(target, k=9)  # L1
```

天文图像中**总能量必须守恒**。如果反卷积偷偷改变了某个 9×9 邻域的平均亮度，
那就是凭空增减光子——本质上等于幻觉或丢光。这一项是把"科学正确"约束硬编码进损失。

#### ④ 星点 BCE

监督 `star_logits` 输出，让 `star_map` 真的是星点位置的概率图，
为推理时的双滑块控制提供基础。

### 6.7 在线合成数据管线

`openbxt/dataset.py` 中的 `SyntheticAstroDataset` 不要求用户准备 (blur, sharp) 配对：

```
sharp 图 (用户提供)
   │
   ├── auto_stretch + 随机 crop + 翻转
   ├── synthesize_blurred()   # 6.4 的空间变化像差合成
   │     ├── 网格化 Zernike PSF
   │     ├── 可选运动模糊（引导误差）
   │     ├── 通道独立的色差 + 视宁度抖动
   │     ├── Poisson 散粒噪声 + 高斯读出噪声
   │     └── 子像素平移（横向色差）
   ├── detect_stars + stellar_mask  → 真值星点掩膜
   └── 输出 {blurry, sharp, psf, star_mask}
```

由于训练分布**只包含真实天文图像 + 真实物理模糊**，网络永远不会学到通用图像先验
（人脸纹理、文字结构等），自然就避免了在天文图像上"幻想"出非天文结构。

### 6.8 分块推理 + 余弦窗融合

真实天文图像常常是 6000×4000 甚至更大，不能一次喂进网络。
`openbxt/inference.py` 的 `deconvolve` 实现了：

1. 按 `tile`（默认 256）和 `overlap`（默认 32）滑窗分块。
2. 边缘 tile 不够大时用 `reflect padding` 补到 `tile`。
3. 每个 tile 用一个 1D 余弦窗的外积做权重，重叠区平滑融合：

```python
# inference.py
ramp = 0.5 - 0.5 * cos(linspace(0, π, overlap))  # 在重叠区从 0 平滑升到 1
```

4. 同时融合 `sharp` 和 `star_map`，最后除以累计权重。

这样可以处理任意分辨率，且 tile 边缘不会有可见的拼接缝。

---

## 七、与原版 BXT 的差异

| 项目 | 原版 BXT | OpenBXT |
|------|---------|---------|
| 平台 | PixInsight 插件，闭源 | PyTorch 库，全开源 |
| 训练数据 | 作者私有大型数据集 | 用户自备 sharp 图 + 在线合成 blur |
| AI 框架 | 自定义 + CoreML / TensorFlow | 纯 PyTorch（CUDA / MPS / CPU） |
| 像差范围 | 工业级（含 drizzle 上采样、详细色差等） | 标准 Zernike + 视宁度 + 运动 + 色差 |
| GPU 加速 | macOS 自动 / Win/Linux 需额外配置 | 用 PyTorch 默认即可 |
| 修改 / 微调 | 不可 | 任意 |

OpenBXT 不追求在数值精度上**完全匹配**原版 BXT——那需要训练规模与作者多年的工程
经验。但作为一个**透明的、可复现的、可改造的**实现，它提供了一条路径让科研、教学、
天文摄影社区都可以理解、复现并改进这套方法。

---

## 八、训练建议

- **数据**：尽量收集多个不同望远镜 / 焦距 / 视场 / 主题（星云、星系、星团）的 sharp 图。
  100~500 张 6k 级图像就能训出可用模型；越多越好。
- **图像质量**：所有训练图必须是**线性**（未做非线性拉伸）的，且本身已经尽量锐利。
  一般来说叠加大量帧 + 已经做过传统反卷积的图最佳。
- **裁切尺寸 `--crop`**：默认 256，对应感受野约 32 像素。视场更大时可上调到 384 / 512
  以让网络学到更长程的结构。
- **批大小 `--batch_size`**：6GB 显存约 4~6，12GB 约 8~12，24GB 可上 16~24。
- **训练时长**：默认 100 epoch + 余弦退火。在 RTX 3090 + 200 张图的设置下约 1~2 天。
- **PSF 大小 `--psf_size`**：默认 33，覆盖到 ±16 像素。如果观测系统视宁度差或像差大可改成 49。

---

## 九、模型如何避免"幻觉"

> 这是这类工具最常见的质疑。这里给出一个清晰的答案。

OpenBXT 通过**三道独立防线**抑制幻觉：

1. **训练分布只包含真实天文图像 + 真实物理模糊**
   网络从没见过非天文的训练样本，因此不存在"通用图像先验"可被错误使用。

2. **输出是有界残差**（[6.2](#62-残差有界输出不会凭空捏造结构)）
   不确定时退化为输入原样，而不是从零生成。最坏情况是**没锐化**而不是**虚假结构**。

3. **通量守恒损失**（[6.6](#66-多分量损失函数) ③）
   无法在不被惩罚的前提下偷偷改变局部平均亮度。任何凭空多出的"恒星"都会因为打破
   局部光通量守恒而被损失项压制。

这三道防线**任意一个失效都还有另外两个兜底**，使 OpenBXT 在工程上保持"求真"而非"求美"的取向。

---

## 十、许可证

MIT License。

欢迎 issue / PR。如果你用 OpenBXT 做了科研工作，请引用本仓库；如果发现某种新的像差或
观测系统能被加进合成管线，更欢迎贡献到 `openbxt/aberrations.py`。
