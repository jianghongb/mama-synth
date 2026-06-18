# Pipeline Flowchart — v17 (GAN + SDEdit Diffusion Refinement)

## 训练阶段

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Pix2PixHD GAN (v14, 已完成)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  pre ──→ resize 512 ──→ GlobalGenerator ──→ fake = pre + Δ      │
│                         (174M params)                           │
│  Loss = GAN + VGG + Feat + MSEC×50 + Tumor×10                  │
│  Data: data_split_v4 (2811 cases, breast mask, intensity_aug)   │
│  Output: checkpoints/mamasynth_v14/latest_net_G.pth             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (frozen)
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: Diffusion Refiner (v17)                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each (pre, gt) pair:                                       │
│                                                                 │
│    ① gan_out = frozen_GAN(pre)           # coarse synthesis     │
│    ② t ~ Uniform(0, T)                   # random timestep      │
│    ③ ε ~ N(0, I)                         # random noise         │
│    ④ noisy_gt = √ᾱt · gt + √(1-ᾱt) · ε  # forward diffusion   │
│    ⑤ ε̂ = RefinerUNet([noisy_gt, pre, gan_out], t)               │
│    ⑥ Loss = MSE(ε̂, ε)                    # noise prediction     │
│                                                                 │
│  RefinerUNet: 18M params, 3ch→1ch, cosine schedule, T=1000     │
│  Optimizer: AdamW, lr=1e-4, 100 epochs                          │
│  Output: checkpoints/refiner_v17/refiner_latest.pth             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 推理阶段 (Docker, T4 GPU)

```
┌─────────────────────────────────────────────────────────────────┐
│  Input: pre-contrast .mha                                        │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: GAN (~1s)                                              │
│                                                                 │
│  pre ──→ resize 512 ──→ Pix2PixHD ──→ gan_out (coarse)         │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Stage 2: SDEdit Refinement (~5-8s)                              │
│                                                                 │
│  ① 对 gan_out 加噪声 (strength=0.3 → t_start=300):              │
│     x_T = √ᾱ₃₀₀ · gan_out + √(1-ᾱ₃₀₀) · ε                   │
│                                                                 │
│  ② DDIM 去噪 20 步 (t=300 → t=0):                               │
│     for t in [300, 285, 270, ..., 15, 0]:                       │
│       ε̂ = RefinerUNet([x_t, pre, gan_out], t)                   │
│       x₀_pred = (x_t - √(1-ᾱt)·ε̂) / √ᾱt                     │
│       x_{t-1} = DDIM_step(x₀_pred, ε̂, t)                       │
│                                                                 │
│  ③ result = x₀ (refined output)                                 │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  resize back → output.mha                                        │
│                                                                 │
│  Total: <10s per case ✅                                         │
│  Weights: GAN 696MB + Refiner ~72MB = ~768MB                    │
└─────────────────────────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Output: synthetic-contrast .mha                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 设计思路

```
GAN alone:     pre ────────────────────────────→ output (快但模糊边界)
                                                    ↑
SDEdit:        pre ──→ GAN ──→ +noise ──→ denoise ─┘ (保留全局结构，精修细节)
```

- **GAN 提供全局结构**：整体亮度、增强区域位置
- **Diffusion 精修细节**：肿瘤边界锐度、纹理真实感
- **strength=0.3**：只加 30% 噪声，不破坏 GAN 的全局合成质量
- **GAN output 作为 condition**：refiner 始终知道 GAN 给了什么，只需学残差修正

## 超参数调节

| 参数 | 默认 | 作用 | 调节方向 |
|------|------|------|---------|
| `MAMA_SDEDIT_STRENGTH` | 0.3 | 噪声强度 | ↑更多修改，↓更保守 |
| `MAMA_DDIM_STEPS` | 20 | 去噪步数 | ↑质量更好但更慢 |
| `T` (训练) | 1000 | 总时间步 | 通常不改 |
| `base_ch` | 64 | UNet 宽度 | ↑更大模型 |

## 向后兼容

- 若 `refiner_latest.pth` 不存在 → 自动退回纯 GAN 推理
- 环境变量控制：`MAMA_REFINER_PATH` 指向权重文件
