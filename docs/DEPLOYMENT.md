# NIDS-Net 部署与使用文档

记录本仓库在本机（Ubuntu + RTX 4090，CUDA 驱动 13.2）上的完整部署过程、自定义数据集接入方式，以及训练/推理脚本的用法。所有踩过的坑都已经固化进 [setup_nids_env.sh](setup_nids_env.sh)，正常情况下不需要重复排查。

## 目录

- [1. 环境搭建](#1-环境搭建)
- [2. 数据集约定](#2-数据集约定)
- [3. 准备自己的数据集](#3-准备自己的数据集)
- [4. 三条推理/训练路径](#4-三条推理训练路径)
- [5. 输出结果说明](#5-输出结果说明)
- [6. 已知问题 / 踩坑记录](#6-已知问题--踩坑记录)
- [7. 命令速查](#7-命令速查)

---

## 1. 环境搭建

### 前置条件
- Linux + NVIDIA GPU（本机 RTX 4090，CUDA 驱动 13.2；只要驱动新于 CUDA 11.8 即可，向下兼容）
- 已安装 `conda`
- 能访问 GitHub / HuggingFace / PyPI / conda 官方源（GroundingDINO 权重从 HuggingFace 拉取，DINOv2 从 `torch.hub` 拉取）

### 一键安装
```bash
cd /workspace/NIDS-Net
./setup_nids_env.sh            # 建环境 + 装依赖 + 跑一次官方 demo 验证
./setup_nids_env.sh --no-demo  # 只装环境，不跑 demo
```

脚本是**幂等**的：conda 环境、已下载的权重、已编译的 detectron2 都会跳过重复安装/下载，可以放心重复执行。

### 脚本做了什么（按顺序）

| 步骤 | 内容 | 为什么 |
|---|---|---|
| 1 | `conda create -n nids python=3.9.18` | 仓库锁定的 Python 版本 |
| 2 | **先装** `pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8` | 必须最先装，避免后续 pip 依赖把 torch 版本带偏 |
| 3 | 装完整 `cuda-toolkit`（`nvidia/label/cuda-11.8.0` 频道） | 只装 `cuda-nvcc` 缺 `cuda_runtime.h`/`cusparse.h` 等头文件，编译 detectron2/GroundingDINO 会报 `fatal error` |
| 4 | 装 conda-forge 的 `gcc/g++ 11` 工具链 | 系统自带 gcc 13 太新，nvcc 11.8 编译期直接拒绝 |
| 5 | 装 `xformers`（conda xformers 频道，自动匹配 torch2.2.1+cu11.8） | GroundingDINO/DINOv2 用到 |
| 6 | `pip install mkl==2024.0.0` | conda 默认装的 mkl 2025 移除了 torch 2.2.1 依赖的符号，会报 `undefined symbol: iJIT_NotifyEvent` |
| 7 | pip 装剩余依赖（`omegaconf`、`fvcore`、`pytorch-lightning` 等） | `pytorch-lightning==1.8.1` 的 wheel 元数据在新版 pip 下无法解析，改装不锁版本的新版（demo 用不到这个包） |
| 8 | 源码编译 `detectron2`（`--no-build-isolation`，环境变量指向 gcc11 + `TORCH_CUDA_ARCH_LIST` + `CUDA_HOME`） | 不加 `--no-build-isolation` 的话 pip 会在隔离环境里编译，看不到已装的 torch |
| 9 | 装 `segment-anything`、`supervision==0.20.0` | SAM 推理 + 可视化工具 |
| 10 | `pip install "numpy<2"` | `pycocotools`/`detectron2` 是按 numpy1.x ABI 编译的预编译包，numpy2.x 下会崩 |
| 11 | 在仓库根目录跑 `python setup.py install` | 装 RoboKit，自动 `pip install` GroundingDINO + MobileSAM，并下载它们的权重到 `ckpts/gdino/`、`ckpts/mobilesam/` |
| 12 | 下载 SAM ViT-H 权重（~2.4GB）到 `ckpts/sam_weights/` | |
| 13 | 预热并 patch DINOv2 的 `torch.hub` 缓存代码 | DINOv2 官方 `main` 分支用了 `float \| None` 这种 Python 3.10+ 语法，Python 3.9 下 `import` 直接报 `TypeError`；脚本会在缓存文件头部插入 `from __future__ import annotations` 修复 |
| 14 | 全量 import 验证 + 跑一次 `demo_eval_gdino_FFA.py` | 确认整条链路可用 |

手动验证环境是否正常：
```bash
conda run -n nids python -c "
import torch, torchvision, detectron2, numpy
from groundingdino.models import build_model
import mobile_sam
print(torch.__version__, torch.cuda.is_available(), detectron2.__version__, numpy.__version__)
"
```

---

## 2. 数据集约定

NIDS-Net 的模板（template/训练）数据固定用这个目录结构（[utils/instance_det_dataset.py](utils/instance_det_dataset.py) 里 `InstanceDataset` 的读取逻辑）：

```
<root>/Objects/
├── <物体1名字>/
│   ├── images/   001.jpg 002.jpg ...
│   └── masks/    001.png 002.png ...
├── <物体2名字>/
│   ├── images/
│   └── masks/
...
```

**硬性要求**（代码里写死，不是建议）：
1. `masks/xxx.png` 文件名必须和 `images/xxx.jpg` 一一对应（代码里直接用字符串替换 `images`→`masks`、`.jpg`→`.png`）。mask 是纯前景二值图——这个方法叫 FFA（Foreground Feature Averaging），是把 DINOv2 patch 特征在 mask 内做平均，背景没抠干净会直接污染模板特征。
2. **每个物体的模板图片数量必须一致**。无论是生成模板特征还是训练 adapter，代码都用 `总模板数 // 物体数` 反推每个物体的模板数（`num_example`），数量不齐会导致标签错位。
3. 物体文件夹按名字字母顺序排序后，索引 `i`（0-based）就是该物体的 `category_id`。所有下游脚本（`run_no_adapter.py`、`train_adapter.py`）都依赖这个顺序生成 `object_names` 列表，不要在训练完之后重命名/增删文件夹。

推理用的场景图片没有格式要求，但要注意：`utils/inference_utils.py` 里的 `get_object_proposal` 用文件名最后一个 `_` 分隔的片段解析 `image_id`（要求是数字），文件名必须形如 `xxx_018.jpg`，不能用任意字符串结尾（比如 `xxx_resized.jpg` 会直接报 `ValueError`）。

---

## 3. 准备自己的数据集

### 3.1 官方 InsDet-Full 数据集

README 指向的 Google Drive 文件夹：
https://drive.google.com/drive/folders/1rIRTtqKJGCTifcqJFSVvFshRb-sB0OzP

结构：`Background/`（背景图）+ `Objects/`（100 个物体，每个 `images/`+`masks/`）+ `Scenes/`（真实场景照片 + Pascal-VOC 风格 xml 标注，可用于验证）。

**匿名脚本化下载会被限流**：用 `gdown` 批量拉取几十个文件后，Google Drive 会返回 "Cannot retrieve the public link... may have had many accesses"，冷却几分钟也不一定解封。**推荐用登录了 Google 账号的浏览器手动下载**（网页端下载没有这个限制）：
1. 打开上面的文件夹链接
2. 多选需要的 `Objects/<物体>/` 子文件夹 + 需要的场景图，右键"下载"（Google 会自动打包成 zip，超大文件夹会自动分成多个 zip）
3. 用 Python 的 `zipfile` 模块按需抽取，不用整个解压：
   ```python
   import zipfile, os
   targets = ["InsDet-FULL/Objects/001_binder_clips_median/", ...]  # 想要的前缀
   with zipfile.ZipFile("InsDet-FULL-xxx-001.zip") as z:
       for name in z.namelist():
           if any(name.startswith(t) for t in targets):
               rel = name[len("InsDet-FULL/"):]
               out = os.path.join("database", rel)
               os.makedirs(os.path.dirname(out), exist_ok=True)
               with z.open(name) as src, open(out, "wb") as dst:
                   dst.write(src.read())
   ```

### 3.2 自己拍的物体

- 每个物体拍 10~24 张不同角度/光照的照片（官方数据集是每个物体 24 张）
- 用 SAM（仓库已装好 `segment_anything`）交互式点选，或 labelme/GIMP 手抠，产出对应的二值 mask
- 按 [第2节](#2-数据集约定) 的目录结构摆放
- 场景图（要检测的图）不需要任何标注，`GroundingDINO` 用通用 prompt `"objects"` 自动出候选框

---

## 4. 三条推理/训练路径

本仓库整理出的示例数据在 `example_dataset/`：
```
example_dataset/
├── train/
│   ├── Objects/<8个物体>/{images,masks}   # 模板图，来自官方 InsDet-Full
│   └── adapter/                            # train_adapter.py 的产出物（见下）
└── inference/
    └── scene_018.jpg                       # 验证用真实场景照片，8个物体都在图里
```

### 路径 A：官方 demo（训练无关，用官方预算好的特征）
```bash
conda run -n nids python demo_eval_gdino_FFA.py
```
用的是从 Box 下载的 `obj_FFA/object_features_vitl14_reg.json`（InsDet 100 个物体的官方特征），跟 `example_dataset` 无关，仅用于验证环境本身可用。

### 路径 B：不训练（Training-Free FFA）
```bash
conda run -n nids python run_no_adapter.py
```
- 读 `example_dataset/train/Objects/` → DINOv2 前景特征平均（FFA，纯前向推理，无训练）→ 缓存到 `obj_FFA/example_no_adapter_features.json`
- 读 `example_dataset/inference/*.jpg` → GroundingDINO+SAM 出候选框 → FFA 特征 → 和模板做余弦相似度 + stable matching
- 结果写到 `exps/example_no_adapter/`

### 路径 C：训练 adapter + 用 adapter 推理（两个脚本，训练/推理彻底解耦）

**第一步，训练**（[train_adapter.py](train_adapter.py)）：
```bash
conda run -n nids python train_adapter.py
```
- 复用路径 B 同样的 FFA 模板特征生成逻辑
- 用 InfoNCE 对比损失训练一个两层 `WeightAdapter`（200 epoch，GPU 上几秒钟）
- 产出物全部存进 `example_dataset/train/adapter/`：
  - `weights.pth` — adapter 权重
  - `adapted_features.json` — adapter 处理后的模板特征
  - `raw_features.json` — 训练前的原始 FFA 特征（缓存，供重跑训练复用）
  - `meta.json` — 物体名单、超参数、训练最终 loss

**第二步，推理**（[infer_with_adapter.py](infer_with_adapter.py)）：
```bash
conda run -n nids python infer_with_adapter.py
```
- **只读** `example_dataset/train/adapter/` 里的三个产出文件，不会再碰训练原图
- 场景候选框特征同样过一遍 adapter 再匹配
- 结果写到 `exps/example_with_adapter/`

这样拆分之后，"训练"和"部署推理"是两个独立环节：训练机器上跑完 `train_adapter.py`，只要把 `example_dataset/train/adapter/` 这一个文件夹拷到推理机器上，`infer_with_adapter.py` 就能直接跑，不需要把原始训练图片也搬过去。

---

## 5. 输出结果说明

每条路径的输出目录下都有：
- `<图片名>_pred.jpg` — 检测框可视化（框 + 类别名 + 置信度）
- `predictions.json` — 结构化结果：`bbox`（xywh）、`category_id`、`category_name`、`score`

`SCORE_THRESHOLD = 0.5` 写死在两个脚本开头，过滤掉置信度过低的匹配（stable matching 本身会强制给每个候选框分配一个类别，哪怕它其实什么都不是；阈值用来把明显不靠谱的匹配砍掉）。想看全部候选框可以把阈值调成 0 或直接看 stable matching 之前的 `sims` 矩阵。

**训练 vs 不训练 adapter 的对比**：在 8 个物体（彼此长得完全不一样）的样例上实测，两条路径都是 8/8 全部检测正确，训练 adapter 对置信度**没有提升，反而普遍略降**（比如 `002_binder_clips_small` 从 0.74 降到 0.66~0.67）。这是符合预期的——adapter 的价值在模板库里有很多**相似/易混淆**物体时才能体现出来；本样例物体区分度天然很高，adapter 在小样本、无难例的情况下容易轻微过拟合到模板图的干净背景分布。想验证 adapter 的真实收益，需要往 `example_dataset/train/Objects/` 里加几个长得像的同类物体（比如同款不同口味的商品）再对比。

---

## 6. 已知问题 / 踩坑记录

| 现象 | 原因 | 解决 |
|---|---|---|
| `pip install pytorch-lightning==1.8.1` 报 `No matching distribution found` | 该 wheel 元数据里 `torch (>=1.9.*)` 用了非法的版本号语法，新版 pip（≥24.1）拒绝解析 | 装不锁版本的新版即可，demo 代码路径用不到这个包 |
| `torch` import 报 `undefined symbol: iJIT_NotifyEvent` | conda 装的 `mkl 2025` 移除了 torch 2.2.1 依赖的符号 | `pip install mkl==2024.0.0` |
| 编译 detectron2/GroundingDINO 报 nvcc 拒绝识别 gcc 版本 | 系统默认 gcc 13，nvcc 11.8 只认到 gcc 12 | `conda install -c conda-forge gxx_linux-64=11 gcc_linux-64=11`，编译时 `CC=`/`CXX=` 指过去 |
| 编译报 `fatal error: cuda_runtime.h`／`cusparse.h` 找不到 | 只装了 `cuda-nvcc`，没有完整开发头文件 | 装完整 `cuda-toolkit`（`nvidia/label/cuda-11.8.0` 频道） |
| `pip install -e .` 编译 detectron2 报 `ModuleNotFoundError: No module named 'torch'` | `-e`（editable）安装内部会递归调一次 `pip install --use-pep517`，丢失外层的 `--no-build-isolation` | 改用非 editable 的 `pip install --no-build-isolation .` |
| `import detectron2`/`pycocotools` 报 numpy ABI 不兼容警告甚至崩溃 | 这些是按 numpy1.x ABI 预编译的 wheel，某些依赖（如新版 `opencv-python`）会把 numpy 升到 2.x | 最后统一 `pip install "numpy<2"` |
| `torch.hub.load('facebookresearch/dinov2', ...)` 报 `TypeError: unsupported operand type(s) for \|` | DINOv2 官方仓库 `main` 分支现在用了 `float \| None` 这种 Python 3.10+ 语法 | 给缓存到 `~/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/{attention,block}.py` 的文件头部插入 `from __future__ import annotations` |
| `conda install --force-reinstall pytorch` 之后 `torchvision` import 报 `ImportError: cannot import name 'DiagnosticOptions'` | 之前 pip 误装过 torch 2.8.0，其残留文件混进了 `torch/onnx/_internal/exporter/`，conda 重装 2.2.1 时没清干净 | `rm -rf` 整个 `torch` 包目录后重装，不要让 pip 在 conda 装 torch 之前碰过 torch |
| Google Drive `gdown` 批量下载报 "Cannot retrieve the public link... may have had many accesses" | 匿名下载触发反滥用限流，几十个文件后就会拦，冷却时间不确定（观察到 90 秒不够） | 用登录了账号的浏览器手动下载，见 [3.1](#31-官方-insdet-full-数据集) |
| `get_object_proposal` 报 `ValueError: invalid literal for int()` | 场景图文件名最后一段必须是数字（比如 `test_002.jpg`），代码硬编析 `文件名.split('_')[-1]` 当 `image_id` | 场景图文件名改成 `xxx_<数字>.jpg` 格式 |
| GroundingDINO 每次 `import robokit.ObjDetection` 都打印一行 `FATAL Flags parsing error: Unknown command line flag 'user'` | `robokit/ObjDetection.py` 里有一行遗留代码 `os.system("python setup.py build develop --user")`，在 import 时无条件执行，但这里没有对应的 setup.py 上下文 | 无害，可以忽略；不影响后续检测逻辑 |

---

## 7. 命令速查

```bash
# 环境
./setup_nids_env.sh                              # 一键装环境（幂等，可重复跑）
conda activate nids

# 官方 demo（验证环境）
python demo_eval_gdino_FFA.py

# 自定义数据集，不训练
python run_no_adapter.py

# 自定义数据集，训练 adapter + 用它推理
python train_adapter.py
python infer_with_adapter.py
```
