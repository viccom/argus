# ultralytics CPU 训练环境搭建记录（2026-10-21）

> 结论先行：**训练环境已在共用 venv 上以纯增量方式搭好并全链冒烟通过**——预训练权重加载 → 训练
> → val → predict → 导出 ONNX → onnxruntime 直跑验证，全程 CPU、限 6 线程、总耗时 <1 分钟。
> 共用验证工具链零变动（pip freeze diff 仅 6 行纯新增）。27 类自训路线的环境侧障碍已清除。

## 一、背景

27 类工地车型自训路线（见 `model-download-candidates-20261021.md` §十）需要 ultralytics 训练
能力。本机无 GPU（KVM 虚拟机，仅 Cirrus 虚拟显卡），需回答两件事：CPU 能否支撑训练流程；对
共用 venv（全工作区 python 导出/验证环境）是否有破坏性。本文记录搭建与验证全过程。

## 二、环境基线（搭建前，已就绪部分）

| 项 | 状态 |
|---|---|
| venv | `/home/ncpe/dongit/yolo-rs/.venv`（Python 3.12.3，全工作区共用） |
| 训练核心栈 | torch 2.14.0+cu130（CUDA 不可用，CPU 跑无碍）、torchvision 0.29.0、ultralytics 8.4.142、opencv 4.10.0.84、matplotlib、pillow、ultralytics-thop、onnxslim 0.1.96（导出 ONNX 用）均已装 |
| 缺失 | pandas / scipy / seaborn / py-cpuinfo（ultralytics 训练路径的画图与指标依赖） |
| 硬件 | Xeon Gold 6226R 16 核 / 31G / 磁盘空闲 325G；负载 0.23（空闲） |

## 三、关键坑：裸装会把共享 numpy 顶成 2.x（dry-run 抓获）

```console
$ pip install --dry-run pandas scipy seaborn py-cpuinfo
Would install numpy-2.5.4 pandas-3.0.6 ...   # ← numpy 1.26.4 会被强顶成 2.5.4！
```

pandas 3.0 起要求 numpy≥2，裸装最新版会升级共享 numpy 1.26.4，危及 opencv/onnx 工具链
（编译 ABI 兼容性不保证）。**必须钉版本**再装，dry-run 复核：

```console
$ pip install --dry-run "pandas<3" "scipy<1.17" seaborn py-cpuinfo
Would install pandas-2.3.3 py-cpuinfo-9.0.0 pytz-2026.5 scipy-1.16.3 seaborn-0.13.2 tzdata-2026.5
# numpy 未出现 → 纯增量，可实装
```

## 四、实际变更清单与零破坏证明

实装命令（清华镜像一次性参数，**未改任何 pip 配置文件**；pypi 直连 Fastly 下载 scipy 卡 8 分钟未完，镜像秒级）：

```bash
~/dongit/yolo-rs/.venv/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  "pandas<3" "scipy<1.17" seaborn py-cpuinfo
```

freeze diff（安装前快照 `~/dongit/yolo-train/pip-before.txt` 对照 after，**仅 6 行纯新增、零版本变动**）：

```
> pandas==2.3.3      > scipy==1.16.3     > seaborn==0.13.2
> py-cpuinfo==9.0.0  > pytz==2026.5      > tzdata==2026.5
```

共享关键包复核：numpy 1.26.4 / onnx 1.16.2 / onnxruntime 1.29.0 / torch 2.14.0 /
ultralytics 8.4.142 / opencv 4.10.0.84 / timm 1.0.30 **全部原样**。卸载即
`pip uninstall pandas scipy seaborn py-cpuinfo pytz tzdata`，环境回到搭建前。

## 五、冒烟全链（脚本在 `~/dongit/yolo-train/`）

数据用 ultralytics 官方 **coco8**（8 张 COCO 图自动下载，548K），只证管线不求精度。
资源纪律：`OMP_NUM_THREADS=6` + `torch.set_num_threads(6)` + `workers=0`（共享机限核，
与 cargo `--jobs 6` 同思想）。

```bash
cd ~/dongit/yolo-train
curl -sL -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
OMP_NUM_THREADS=6 .venv路径/bin/python smoke_train.py    # 训练→val
OMP_NUM_THREADS=6 .venv路径/bin/python smoke_export.py   # predict→导出→ort 验证
```

实测结果（23:33，空闲机）：

| 阶段 | 耗时 | 结果 |
|---|---|---|
| torch 导入 + 加载 yolov8n.pt | ~8s | COCO 80 类预训练 |
| 训练 1 epoch @640px batch4 | ~15s | `runs/smoke-cpu/weights/best.pt` |
| 独立 val | ~1s | mAP50=0.888（4 图拟合，数值无意义只证管线） |
| predict | 1.6s | `000000000009.jpg` 检出 5 目标 |
| 导出 ONNX（onnxslim simplify） | 1.2s | `best.onnx` 13M |
| onnxruntime 直跑 | <0.1s | 见下 |

ort 验证对齐 **argus 模型准入前置门**（shape + 值域）：

```
输入 images [1,3,640,640] → 输出 output0 [1,84,8400]
值域 [0.000, 640.234]   # >1 的分量即像素级坐标，非归一化——argus 解码器兼容布局
```

即：训练产物经同一道门检验，与 slots.toml 入槽链路天然衔接（forklift 件已实证入槽）。

## 六、踩坑记录

1. **pandas 3.0 的 numpy 2.x 强升级**（本文第三节）：装包前必 dry-run，钉 `pandas<3`。
2. **pypi 直连极慢**：本机无 pip 镜像配置，scipy 40MB 直连 Fastly 8 分钟未完；用
   `-i https://pypi.tuna.tsinghua.edu.cn/simple` 一次性参数解决，不动全局配置。
3. **`settings.update()` 进程内不生效于当次下载**：`DATASETS_DIR` 在 ultralytics import 时
   从 settings.json 固化，训练脚本里 update 只对**下次进程**生效——首跑 coco8 落到了
   `~/dongit/datasets/`。已搬入沙箱 `yolo-train/datasets/`，settings.json 现已持久化
   `datasets_dir=~/dongit/yolo-train/datasets`。**正式训练的数据 yaml 一律写绝对路径
   `path:`，不依赖该设置**。
4. **`.cache` 标签缓存含绝对路径**：数据集挪位后必须删 `labels/*.cache`，否则扫到旧路径报错。
5. **NNPACK 警告无害**：`Could not initialize NNPACK! Unsupported hardware` 是 torch CPU
   conv 在本机的固定告警，不影响训练。
6. **pillow 探测误报**：import 名是 `PIL` 不是 `pillow`，用 `importlib.find_spec` 探测时
   探 pip 名会假报缺失。

## 七、CPU 训练能力分级（容量规划）

| 规模 | 本机 CPU 实测量级 | 判定 |
|---|---|---|
| 管线冒烟（8 图 1 epoch） | <1 分钟 | ✅ 已证 |
| 试点：每类几十~两百张、v8n、20–30 epochs | 1–3 小时 | ✅ 本机可做 |
| 正式：迁移学习+冻结骨干+27 类全量 | 限 6 核约 1 天量级 | ⚠️ 可做但占机长，低峰跑 |
| 从零 v8s 100 epochs | 多天连续满载 | ❌ 应借 GPU（A100 机器/云端） |

> 分级数字为公开吞吐参考值量级（本机仅实测了冒烟档），正式训练前建议先跑 1 epoch
> 全量数据实测单 epoch 耗时再定总工期。

正式训练命令模板（沙箱内，custom yaml 绝对路径）：

```bash
yolo detect train model=$BASE/yolov8n.pt data=$BASE/site27.yaml \
  epochs=60 imgsz=640 batch=8 workers=4 device=cpu \
  pretrained=$BASE/yolov8n.pt freeze=10 project=$BASE/runs name=site27
```

产物走向：`best.pt` → `yolo export format=onnx imgsz=640 simplify=True` → argus 三道
前置门 → `slots.toml` 入槽（零代码发版）。

## 八、沙箱产物清单（`~/dongit/yolo-train/`，共 37M）

```
yolov8n.pt            6.3M  COCO 预训练权重（正式训练的起点）
smoke_train.py              冒烟脚本一：训练→val
smoke_export.py             冒烟脚本二：predict→导出→ort 验证
pip-before.txt / -after.txt 安装前后 freeze 快照（零破坏证据）
datasets/coco8/       548K  官方冒烟数据集
runs/smoke-cpu/             训练产物（args.yaml + best/last.pt + best.onnx + 曲线图）
runs/detect/val/            val 产物（混淆矩阵等）
```

均不入库（工作区 scratch）；删除整个目录 + 卸载 6 个新包即完全回滚。

## 九、后续增量（2026-10-26，vision-trainops M1 需要）

本文记录的是 2026-10-21 的搭建过程；以下是此后按同一纪律（先 dry-run）追加的包，
共享关键包（numpy 1.26.4 / scipy 1.16.3 / opencv 4.10.0.84 / torch 2.14.0 /
onnx 1.16.2 / onnxruntime 1.29.0）复核后均未被顶：

| 包 | 用途 | dry-run 结论 |
|---|---|---|
| supervision 0.26.0（+ defusedxml 0.7.1） | vision-trainops M1 评估的旁证口径（独立实现算 mAP/F1，与 pycocotools 一致） | 纯增量，不动任何现有包版本 |
| pytest 9.1.1（+ pluggy / iniconfig） | vision-trainops 仓库自检入口 `scripts/selftest.py` 的测试运行器 | 纯增量，不动任何现有包版本 |

> supervision 依赖较轻（numpy/scipy/matplotlib/pyyaml/pillow/opencv/requests/tqdm
> 均已满足），**不会触发第三节记录的 pandas 3.0 → numpy 2.x 强升级路径**。
