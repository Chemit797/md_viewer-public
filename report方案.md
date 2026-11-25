## Part E: Report 框架 (7 章)

按照要求的 7 个必写章节 + 前面的封面页。

---

### 封面页
- 课程名: XMUM 2604 Natural Language Processing / Deep Learning
- 项目名: **EmotionLens — Cross-Domain Facial Expression Recognition**
- 组号 + 组员名单

---

### Chapter 1: Abstract (150-200 词)

**写什么**：一篇小摘要，扫一眼就知道我们干了什么。

**结构**：
1. 问题: FER 模型在真实人脸上表现骤降
2. 方法: 多源数据融合 + 6 组控制变量实验
3. 关键发现: 标签质量 (+35pp) 和数据多样性 (+20pp) 是主导因素, 架构几乎无用
4. 最优结果: FERPlus test 89.0%, 真实人脸 64.5%
5. 应用: EmotionLens 实时系统, 5 个示例镜头, 开放架构

---

### Chapter 2: Introduction (0.7 页)

**写什么**：交代背景 → 核心挑战 → 我们的方案。

**结构**：
- 第 1 段: FER 的应用价值 (课堂监控、人机交互、在线监考……规模、客观、24/7)
- 第 2 段: 核心挑战——lab 训练的模型在真实人脸上崩溃 (domain gap)
- 第 3 段: 课程基线——老师建议 FER2013+CNN/VGG16 → 我们复现 → 真实人脸 11.6%
- 第 4 段: 我们的方法概述——多源数据融合 + 控制变量实验定位瓶颈 + 部署为开放应用
- 第 5 段: 论文结构 (Section 3 Literature Review, Section 4 Data, ...)

---

### Chapter 3: Literature Review (1.5 页)

**写什么**：不是罗列论文。要写成「我们选了 X，为什么 X 比 Y 好」。

**三条线**：

**线 1 — FER 数据集演进** (为什么用 FERPlus 而不是 FER2013)
- FER2013 (Goodfellow et al., 2013): 首个大规模 FER 数据集，但单标注者 → 噪声严重
- FERPlus (Barsoum et al., 2016): 同样图像 + 10 人投票 → 软标签 → 标注更可靠
- RAF-DB (Li & Deng, 2019): 真实场景 + 高质量标注 → 弥补 lab 数据偏向
- 关键论点: "标签噪声"和"域间差距"是两个独立问题，需要分别解决

**线 2 — 架构选择** (为什么 ResNet18)
- VGG16 (Simonyan & Zisserman, 2014): 经典但 138M 参数，推理慢
- ResNet (He et al., 2016): 残差连接 → 缓解梯度消失；ImageNet 预训练 → 特征迁移
- EfficientNet (Tan & Le, 2019): 轻量但我们在同条件下只训到 41% (远逊 ResNet18 的 79.9%)
- **如何体现 "justification"**：不是"ResNet18 是最好的"，而是"我们实测了 VGG/ResNet/EfficientNet，ResNet18 在参数效率、训练稳定性和最终性能之间平衡最优"

**线 3 — 跨域泛化**
- Domain shift: 训练分布 ≠ 部署分布
- Subject identity bias: 模型学会认人而不是认表情
- 正则化方法: mixup (Zhang et al., 2018), label smoothing (Szegedy et al., 2016), AdamW (Loshchilov & Hutter, 2019)

**参考文献清单** (至少引这些):
1. Goodfellow et al., 2013 — FER2013
2. Barsoum et al., 2016 — FERPlus
3. Li & Deng, 2019 — RAF-DB
4. He et al., 2016 — ResNet
5. Simonyan & Zisserman, 2014 — VGG
6. Tan & Le, 2019 — EfficientNet
7. Zhang et al., 2018 — mixup
8. Szegedy et al., 2016 — label smoothing (Inception-v3)
9. Loshchilov & Hutter, 2019 — AdamW
10. Ekman & Friesen, 1971 — 7 basic emotions
11. Chen et al., 2022 — cross-domain FER
12. Zhang et al., 2016 — MTCNN (face detection)

---

### Chapter 4: Data Set (1.5 页)

**写什么**：四个数据源 + 自采集的详细描述。

**结构**：

**4.1 公开数据集**
- FER2013: 35,887 张, 48×48 灰度, 7 类硬标签, CSV → 仅用于 Exp1 baseline
- FERPlus: 35,803 张, 同上图片但 10 人投票软标签, parquet 格式, 8 类 (我们丢弃 contempt → 7 类)
- RAF-DB: 15,344 张, 真实场景人脸, 7 类硬标签

**4.2 自采集数据 (Self-Captured)** — 对应 "Explain collection of dataset"
- 采集设备: 6 名组员各用笔记本自带摄像头
- 环境: 教室/宿舍室内自然光, 距离屏幕 40-60cm
- 流程: 录制视频 → 脚本按间隔抽帧 → 弱模型预标 → **人工逐帧审核** (这是关键, 说明标注质量有保证)
- 最终: 1,123 张 reviewed frames, 7 类表情
- Per-user distribution: [插入 per_user_emotion_bars.png]
- Split: 80/20 stratified random split

**4.3 数据统计总表**
```
               Train      Val       Test      Total
FER2013        28,709     3,589     3,589     35,887
FERPlus        28,709     3,546     3,546     35,803
RAF-DB         ~12,000    —         ~3,000    ~15,344
Self           899        224       —         1,123
```

**4.4 预处理**
- 所有图像统一: 224×224 RGB, ImageNet 归一化 (μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225])
- FER2013/FERPlus: 48×48 灰度 → resize → 3 通道复制
- RAF-DB: 原始 RGB → resize
- Self: 原始 RGB → MTCNN 人脸裁剪 → resize

---

### Chapter 5: Implementation (4 页) ⭐ 最重要

**写什么**：这是老师关注的核心。必须覆盖:
- Calculate loss, fine tune and reduce loss
- Adjust parameters, fine tune parameters
- Implement gradient, slicing, sampling
- Describe the working of each section of code
- Provide justification of using specific algorithm

**策略**：表 > 公式 > 文字。公式一行就够，别展开推导。每小节压缩到半页内。

**结构**：

**5.1 模型架构**

```
ResNet18 (He et al., 2016):
  - torchvision.models.resnet18(weights="IMAGENET1K_V1")
  - 移除 1000-class FC → Linear(512, 7)
  - 输入: 224×224 RGB, ImageNet 归一化
  - 参数量: 11,180,103
  - 为什么选它: 11M 参数 (VGG16 的 1/12), 残差连接防梯度消失, 
    ImageNet 预训练特征可迁移到 FER 任务
```

**5.2 训练配方** — 对应课程 "loss / gradient / fine-tune / parameters"

**(a) Loss Function — 两种 loss 的对比是故意的实验设计**

```
FER2013 (硬标签): CrossEntropyLoss
  loss = -log(p_model[true_class])
  问题: 标注者可能标错，但 loss 强制模型 100% 相信那个标签

FERPlus (软标签): Soft CrossEntropy
  loss = -Σ_k p_voter(k) * log(p_model(k))
  p_voter 是 10 人投票的概率分布 [0.0, 0.1, 0.7, 0.1, ...]
  好处: 模型不必 100% 相信单一标签 → 对标注噪声更鲁棒
```

**(b) Optimizer & Gradient**

```
AdamW (Loshchilov & Hutter, 2019):
  - 解耦权重衰减: θ = θ - η*(m̂/√v̂ + λθ)
  - β1=0.9, β2=0.999, lr=3e-4, wd=1e-4
  - 比 SGD 收敛更快, 比 Adam 正则化更好

Mixed Precision Training (AMP):
  - torch.amp.GradScaler
  - 前向传播用 FP16 (加速), 反向传播时 scale loss → unscale gradient
    → optimizer step, 防止小梯度 underflow
  - 对应课程 "implement gradient" 要求
```

**(c) Learning Rate Schedule**

```
CosineAnnealingLR: η_t = η_min + 0.5*(η_max-η_min)*(1 + cos(t/T_max * π))
T_max = 35 (epochs), η_max = 3e-4
理由: 无 extra hyperparams, 平滑衰减, 最后几个 epoch 探索更精细的 minima
对应课程 "adjust parameters, fine tune parameters"
```

**(d) Sampling — 对应课程 "slicing, sampling"**

```
WeightedRandomSampler:
  - 每类权重 w_class = 1 / count(class)
  - 少数类 (disgust, fear) 过采样, 多数类 (happiness) 欠采样
  - replacement=True, num_samples = len(dataset)
  - 解决 FER2013/FERPlus 的数据不平衡
```

**(e) Regularization**

```
Label Smoothing (ε=0.1):
  y_smooth = (1-ε) * y_hard + ε/7
  防止模型过自信 → 提升泛化

mixup (α=0.2):
  x̃ = λ*x_i + (1-λ)*x_j
  ỹ = λ*y_i + (1-λ)*y_j, λ ~ Beta(α,α)
  训练在"虚拟样本"上 → 提升对域偏移的鲁棒性
```

**(f) Data Augmentation**

```
RandomHorizontalFlip(p=0.5)
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
RandomErasing(p=0.1, scale=(0.02, 0.1))
Test-time augmentation: H-flip averaging
```

**(g) 超参数总表**

| 参数 | 值 | 理由 |
|------|-----|------|
| Optimizer | AdamW | 解耦 weight decay, 训练更稳定 |
| lr | 3e-4 | 经试探, 比 1e-4 收敛更快且不震荡 |
| weight decay | 1e-4 | 标准值 |
| LR schedule | CosineAnnealing, T=35 | 平滑, 无需调额外参数 |
| Epochs | 35 | 观察 val_acc 在此附近饱和 |
| Batch size | 128 (VGG16: 64) | VGG16 受限 VRAM |
| Loss | CE (hard) / Soft CE | 对应两种标签类型 |
| Label smoothing | 0.1 | Inception-v3 原论文推荐值 |
| mixup α | 0.2 | 较小 α 产生更保守的混合 |
| Hardware | RTX 4060 Laptop (8GB) | 本地训练 |

**5.3 实验设计** ⭐ **这是这一章的灵魂**

```
我们设计了 6 组实验，每次只改变一个变量：

┌───────┬────────────┬─────────────┬──────────────────┐
│ 实验  │ 架构       │ 训练数据     │ 控制的变量        │
├───────┼────────────┼─────────────┼──────────────────┤
│ Exp1  │ VGG16      │ FER2013     │ — (baseline)      │
│ Exp1b │ ResNet18   │ FER2013     │ 只变架构           │
│ Exp2  │ ResNet18   │ FERPlus     │ 只变标签质量       │
│ Exp3  │ ResNet18   │ F+R+S       │ 只变数据多样性     │
│ Exp4  │ 全部 4 个  │ —           │ 跨域评测           │
│ Exp5  │ FER 库     │ FER2013     │ 第三方 baseline    │
└───────┴────────────┴─────────────┴──────────────────┘

训练配方在 6 组实验中完全统一 (Exp1 的 VGG16 仅调整 bs=64)。
```

**5.4 应用部署 (简述)**

```
后端: FastAPI + uvicorn + WebSocket
  - engine.py: YuNet 人脸检测 → ResNet18 推理 → EMA 时间平滑
  - lenses/: 5 个独立镜头插件, 消费同一条情绪流
前端: 原生 HTML/CSS/JS + Swiper.js 轮播
公网: Cloudflare Tunnel (HTTPS → getUserMedia 可用)
一键启动: python main.py
```

**代码引用注释要求**: 每个关键模块（loss 实现、sampler、gradient scaling、mixup）需在代码中注释来源 URL 或论文。

---

### Chapter 6: Results and Analysis (5 页) ⭐ 最精彩 — 图多字少

**策略**：每张图占 ~1/3 页 + 下面只写 3 行 bullet（看到什么 → 说明什么 → 结论）。不写段落。

**图表** (8 张, 6 张主图 + 2 张补充):

**6.1 控制变量链 (核心图表) — `control_variable_chain.png`**
```
[图: 瀑布图, 11.6%→9.2%→44.4%→64.5%, 每一步标 delta]
• VGG16+FER2013 = 11.6% 基线, ResNet18+FER2013 = 9.2% (架构 −2.3pp, 无用)
• ResNet18+FERPlus = 44.4% (标签质量 +35.2pp, 最大单步提升)
• ResNet18+F+R+S = 64.5% (数据多样性 +20.0pp, 恢复负情绪)
• 总 +52.9pp, 每一步只变一个变量
```

**6.2 架构消融 — `architecture_ablation.png`**
```
[图: VGG16 vs ResNet18, FER2013 test + Self data 双柱]
• 同数据同配方换架构: lab +0.5pp, self −2.3pp
• 架构对真实泛化的贡献是负的
```

**6.3 Per-Class Recall — `per_class_recall_all.png` + `runs/self_eval/self_confusion_matrix.png`**
```
[图1: 4模型 × 7类 recall 分组柱状图]
[图2: 最优模型 F+R+S 在 self data 上的混淆矩阵]
• FERPlus-only: sadness/disgust/fear recall = 0%（完全不认）
• F+R+S: 恢复到 53-60%（加数据后学会）
• 混淆矩阵: 7 类都有预测, 不再 collapse
```

**6.4 跨域评测 — `cross_domain_radar.png` + 大表**
```
[图: 雷达图, 5模型 × 3测试集]
[表: 完整数字]
• 每模型在自己训练域最优; 只有 F+R+S 跨域可用
```

**6.5 Domain Gap — `domain_gap.png`**
```
[图: lab acc vs self acc 配对柱状图, gap 标注]
• 从 58pp gap → 25pp gap: 加真实数据缩小一半以上
```

**6.6 贡献分解 — `contribution_breakdown.png`**
```
[图: 水平条形图, 架构/标签/数据 各贡献多少]
• 架构 −2.3pp | 标签 +35.2pp | 数据 +20.0pp = 净 +52.9pp
```

**6.7 FER 库 baseline**
```
文字即可 (不需要图):
FER 库 (F=52.6%, disgust=0%, anger=1.8%) → 和 VGG16+FER2013 一致
不是实现问题, 是 FER2013 数据造不出能泛化的模型
```

---

### Chapter 7: Conclusion (0.5-1 页)

**写什么**：三段收束。

**第 1 段 — 做了什么**
> We built a cross-domain facial expression recognition system trained on three complementary data sources (FERPlus soft labels, RAF-DB real-world faces, and a self-captured set of 1,123 frames). Through six controlled experiments, we decomposed the contributors to real-face generalization and found that label quality (+35.2pp) and data diversity (+20.0pp) dominate, while architecture alone contributes essentially nothing (−2.3pp).

**第 2 段 — 关键发现的意义**
> The key insight is not that our model reaches 64.5% on real faces — it is that we now know *why*. The chain of control-variable experiments provides a causal decomposition: upgrading from hard labels to 10-voter soft labels is the single largest lever; adding diverse real-world faces is the second. This finding is actionable for anyone building FER systems.

**第 3 段 — 应用哲学**
> EmotionLens demonstrates that a well-trained FER engine opens a wide design space. The five lenses — crowd statistics, emotion alert, cinema diary, speech coach, mimic game — are not the product; they are proofs that the engine works across different contexts. The engine is reusable, the lenses are swappable, and the use cases are limited only by the user's imagination.

---

## Part F: 图表清单 (8 张, 全部已生成)

| # | 文件名 | 内容 | 放哪 | 大小 |
|---|--------|------|------|------|
| 1 | `control_variable_chain.png` | 控制变量链瀑布图 | 6.1 | ~1/3p |
| 2 | `architecture_ablation.png` | VGG vs ResNet 同数据对比 | 6.2 | ~1/3p |
| 3 | `per_class_recall_all.png` | 4 模型 per-class recall | 6.3 | ~1/2p |
| 4 | `self_confusion_matrix.png` | 最优模型 self confusion matrix | 6.3 | ~1/3p |
| 5 | `cross_domain_radar.png` | 5 模型 × 3 测试集 雷达 | 6.4 | ~1/3p |
| 6 | `domain_gap.png` | lab vs self gap 柱状图 | 6.5 | ~1/3p |
| 7 | `contribution_breakdown.png` | 架构/标签/数据贡献分解 | 6.6 | ~1/4p |
| 8 | `per_user_emotion_bars.png` | Per-user 表情分布 | 4.2 | ~1/3p |

图表位置: `runs/report_charts/` (前 7 张) + `runs/self_eval/` (第 8 张)

## Part G: 最终页数分配

| 章节 | 页数 | 图/表 | 策略 |
|------|:---:|:---:|------|
| 封面 + 目录 | 2p | — | — |
| Abstract | 0.5p | — | 150-200 词 |
| 1. Introduction | 0.7p | — | 三段缩两句 |
| 2. Literature Review | 1.5p | — | 三条线, 每条一段 |
| 3. Data Set | 1.5p | 1 表 + 1 图 | 文字从简, 图说话 |
| **4. Implementation** | **4p** | 1 大表 | 公式一行, 表 > 文字 |
| **5. Results** | **5p** | **6 图 + 1 表** | 每图 3 行 bullet |
| 6. Conclusion | 0.5p | — | 三段收束 |
| References | 1.5p | — | 10-12 篇 |
| **总计** | **17.2p** | 8 图 + 2 表 | ✅ 15-20 中间偏稳 |

还可以加 1 张图到 18p: `runs/ablation/ablation_comparison.png` (F-only vs F+R+S 消融) 放在 6.3 旁。

---

## Part H: 

**我们做了什么**：
> 从老师给的 FER2013+VGG16 出发 (真实人脸 11.6%)，通过 6 组控制变量实验证明了"标签质量是第一瓶颈 (+35pp)，数据多样性是第二瓶颈 (+20pp)，架构几乎没用 (−2pp)"。最优模型达到 FERPlus 89.0% / 真实人脸 64.5%。部署为 EmotionLens：1 个推理引擎 + 5 个可替换的分析镜头，核心是引擎不是镜头。




https://who.1ms.dog/zh-cn/
