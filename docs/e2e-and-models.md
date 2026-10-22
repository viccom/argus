# 端到端验证与模型工程笔记

> 来源:argus 11 个推理端点从零到全量端到端打通的实操记录。
> 环境:Ubuntu 24.04 / 16C31G 共享虚拟机 / rust-onnx-infer (ort 2.0.0-rc.13, load-dynamic) / ORT 1.26.0。

## 1. 模型下载

### 1.1 找下载源的优先级

1. **先读消费库的源码注释**。rust-onnx-infer 的 `license_plate.rs` 文档注释里就写着"已核实可下载"的直链(hpc203 仓库的 mnet_plate / Final_LPRNet_model),省掉全部搜索。上游作者往往已经踩过找模型的坑。
2. **官方组织仓库**:OpenCV Zoo 全系模型已迁到 HuggingFace `opencv/` 组织(YuNet、SFace、PP-HumanSeg 等),还有 `opencv/opencv_contribution` 收录 caffe→onnx 官方转换版(east、wechat_qr、yolov8n 等)。
3. **第三方预导出**:onnx-community、Xenova 等把常见模型转好了(depth-anything、yolov8n-pose 的 `onnx/model.onnx`)。注意核对输出格式是否与消费代码匹配(见 1.4)。
4. **都没有 → 自己导出/转换**(见 §2)。

### 1.2 国内加速

| 途径 | 用法 |
|---|---|
| HF 镜像 | 域名替换 `huggingface.co` → `hf-mirror.com`。下载 `/resolve/main/<path>`;列文件 `/api/models/<repo>` 取 `.siblings[].rfilename`;搜索 `/api/models?author=opencv&search=qr` |
| GitHub release 资产 | `gh release download v8.3.0 -R ultralytics/assets -p 'yolov8n-obb.pt'`(gh 已认证,比匿名 API 稳;api.github.com 匿名易限流) |
| GitHub raw 失败时 | 依次试直连 → `ghproxy.net` 前缀。`gh api` 走认证通道通常直连可用 |
| 国内 model zoo CDN | Rockchip `rknn_model_zoo` 的 download_model.sh 指向联想云盘,下载快但模型可能被改过(见 1.4) |

### 1.3 搜源技巧

- **GitHub code search 不索引二进制**:`filename:xxx.onnx` 恒为 0 结果。但**搜文件名字符串引用**有效:`gh api -X GET 'search/code?q="yolov8n-obb.onnx"'` 能定位到引用它的 model zoo 配置(axelera 的 yaml、Rockchip 的 download_model.sh 都是这么找到的),里面往往就有下载 URL。
- 仓库消失的替代路径:wilinz/wxscan-weights(HF)已删 → ① HF `search=wxscan`/`author=wilinz` 找 fork/改名;② GitHub 搜同名仓库(可能只剩 tflite);③ 找原始 caffe 模型自己转(OpenCV 官方就是这么做的,`opencv_contribution` 里连 `convert_to_onnx.py` 都给你);④ 查消费库 issue/注释里的 SHA256,用于校验找到的镜像是否同一文件。

### 1.4 下载后必做验证

```bash
# ① 魔数:ONNX 是 protobuf,0x08 开头合法;404 页/HTML 一眼识别
xxd -l 4 -p model.onnx        # 期望 080X12XX
file model.onnx               # "data" 合法;出现 "HTML" 即假文件

# ② 签名核对:输入输出名/shape 必须与消费代码的假设一致
uv venv /tmp/onnx-check-env && uv pip install --python /tmp/onnx-check-env/bin/python onnx
/tmp/onnx-check-env/bin/python -c "
import onnx
m = onnx.load('model.onnx')
for i in m.graph.input:  print('in :', i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim])
for o in m.graph.output: print('out:', o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim])
"
```

签名不符的典型事故(全部实际发生过):

| 拿到的 | 期望的 | 后果 |
|---|---|---|
| Rockchip 版 yolov8n-obb:3 个裁剪中间头输出 `[1,79,H,W]`+`[1,1,8400]` | 标准 `[1, 4+1+nc, 8400]` 单输出 | 后处理报"通道数 1 < 6" |
| OpenCV 版 wechat_qr:输入 `[1,1,384,384]`、输出 `detection_output [1,1,1,7]`(含 PriorBox/DetectionOutput) | 输入 416、输出 `mbox_loc [1,32448]` + `mbox_conf_flatten [1,16224]` | 无法接入,需图手术(§2.3) |
| YOLOv8 End2End 导出 `[1,M,7]` vs 传统 `[1,20,8400]` | 消费方认其一 | 布局判定分支不同,先 dump shape 再对号 |

## 2. 模型导出

### 2.1 环境(yolo-rs/.venv 现成可用)

```
Python 3.12 / torch 2.14.0 / ultralytics 8.4.142 / onnx 1.16.2 / onnxslim 0.1.96
```

官方权重:`gh release download v8.3.0 -R ultralytics/assets -p 'yolov8n-obb.pt'`。

### 2.2 Ultralytics 导出

```python
from ultralytics import YOLO
YOLO('yolov8n-obb.pt').export(format='onnx', imgsz=640)   # 5.1s, 11.9MB, CPU 环境即可
```

要点:**优先官方导出而非第三方转换版**。官方 `yolov8n-obb` 导出为传统布局 `[1,20,8400]`(4 box + 1 angle + 15 DOTA 类),是 rust-onnx-infer 测试覆盖的路径;第三方(axelera End2End `[1,300,7]`、Rockchip 裁剪头)兼容性各异。导出后务必按 §1.4 核签名。

### 2.3 ONNX 图手术(改官方转换版适配私有后处理)

实例:OpenCV 官方版 wechat_qr(384 输入 + PriorBox/DetectionOutput 尾层)→ rust-onnx-infer 需要的裁剪版(416 输入 + 中间张量输出)。用 onnx 库四步:

1. **改输入尺寸**:`graph.input[0]` 各 dim 384→416。anchors 换算:416 → 26×26(step16) + 4×13×13(step32) = 1352 位置 × 6 框 = 8112 anchors → loc `[1,32448]`、conf `[1,16224]`(384 则 1152×6=6912 → 27648/13824)。
2. **重算 Reshape/Flatten 常量**:`cls1/flatten_shape [1,6912]→[1,8112]` 等 11 个 initializer 同步改。
3. **裁尾**:从新输出 `{mbox_loc, mbox_conf_flatten}` 做反向可达分析,删除不可达节点(PriorBox×5 + Concat + DetectionOutput 共 7 个)。
4. **重建 `graph.output`**(ValueInfoProto 声明新 shape)→ `onnx.checker.check_model` 校验 → 保存。

已知无害副作用:ORT 加载时警告 `Expected shape from model of {1,27648} does not match actual shape of {1,32448}`——shape inference 缓存的旧值,实际 Reshape 用的是改过的 initializer,推理正确。

## 3. 端到端测试

### 3.1 ORT 动态库(load-dynamic 路线)

- 运行时需 `ORT_DYLIB_PATH` 指向 libonnxruntime.so;yolo-rs 发行版 tarball 自带 1.26.0 可直接解压用。
- **版本检查坑**:报 `Failed to load ONNX Runtime dylib: BadVersion { version_str: "1.26.0" }` 时,不是库坏了。ort-sys 的 `ORT_API_VERSION = 17 + Σ(api-N features)`;ort 2.0.0-rc.13 的检查用 ORT 库的 **release minor** 与 `ORT_API_VERSION` 比较,而 ORT 1.18 起 **API version = release minor + 1**。启了 `api-27`(=27)就会拒绝 1.26.0(minor=26)。**处置:消费方 feature 降为 `api-26`**,ORT 库向后兼容,接口无损。

### 3.2 服务生命周期

- **引擎实例缓存在 OnceCell,且后处理失败也会缓存**(失败发生在推理后而非加载时)。**替换模型文件后必须重启服务**,否则一直在跑内存里的旧模型——症状是"换了文件、报错一字不差"。
- 环境变量在每次新 shell 里不持久:`source .env.build && source .env.runtime && 启动+测试` 必须同一调用内完成;长驻服务用后台任务方式启动。

### 3.3 测试方法

```bash
# 冒烟顺序:健康 → 逐端点 → 错误路径 → 引擎清单
curl -s localhost:7042/health -H "X-API-Key: $KEY"
curl -s -X POST localhost:7042/detect -H "X-API-Key: $KEY" -F "file=@testdata/bus.jpg" | jq '{status, count: .data.count}'
curl -s -X POST localhost:7042/qr      -H "X-API-Key: $KEY" -F "file=@testdata/bus.jpg"   # 未配模型应返回明确"模型未配置"
curl -s "localhost:7042/health?deep=true" -H "X-API-Key: $KEY"                            # deep 额外返回已配置能力清单(configured)
```

- **count=0 也是通路验证**:bus.jpg 无车牌/无旋转目标,/license-plate、/detect/obb 返回空是正确行为,模型加载+推理+后处理链路已打通。
- **强验证用官方 pipeline 做参照**:同一输入跑 `YOLO('xxx.onnx').predict()`,对比目标数与框坐标,两侧一致即后处理等价。合成图(手绘旋转矩形)是域外样本,官方模型同样检不出,别用它当否定证据。
- **断言前先看结构**:`jq '.data | keys'` 确认 DTO 字段名,再写路径断言——嵌套字段(如 `.data.depth.mean`)写错路径会误判为"返回 null"。
- **优先读服务日志**:tracing 里有布局判定打印(`OBB 输出布局: Traditional（shape=[1, 20, 8400]）`),排错比猜快。

## 4. 踩坑速查

| 症状 | 根因 | 处置 |
|---|---|---|
| `BadVersion { version_str: "1.26.0" }` | ort release-minor vs API-version 错位比较 | feature `api-27`→`api-26` |
| OBB 报"通道数 1 < 6" | 模型是裁剪头/多输出,`run_inference` 取最后一个输出 | 换标准导出版 |
| 换了模型文件行为不变 | OnceCell 缓存旧引擎实例 | 重启服务 |
| curl 后台启动服务无输出 | shell 环境不持久 / 子进程脱离 | source 与启动同调用;`run_in_background` |
| `xxd` 首字节非 0x08 | 下载到 404 页(ghproxy 失败时) | 换镜像重下 |
| ORT 警告 shape 不匹配(图手术后) | shape inference 缓存旧值 | 无害,忽略 |
| pip3 不存在但有 uv | Ubuntu 24.04 默认 | `uv venv` + `uv pip install --python <venv>/bin/python` |
| `/detect` bbox 零宽(x≈0)、类名是数字、SAHI count=0 | 模型是 Xenova/yolo-rs 变体:检测头输出**归一化坐标**(cx∈[0,1]),解码器按 640 像素约定反算即崩塌;类名元数据缺失 | `YOLO('yolov8n.pt').export(format='onnx', imgsz=640)` 重导出标准版(判别法:python 直跑输出看 cx 值域 0~1 vs 0~640) |
| 分类置信度系统性偏低(如 0.58 的图只出 0.38)、`no result above threshold`,但 python 同预处理直跑分数正常 | rust-onnx-infer `resize` 曾把 Linear 映射到 fast_image_resize `ResizeAlg::Interpolation`(**固定核**快速算法),大比率缩小(>2x)跳过核外源像素→混叠,质量等效 nearest;检测框不敏感所以 /detect 长期掩盖此 bug | 已修(`ResizeAlg::Convolution`,核随缩放比拉伸);判别法:python 跑 bilinear vs nearest 两变体对照分数,若 rust 分数≈nearest 即中招;诊断手段:临时 ignored 测试打印 `preprocess` 通道均值(与 python 对照)+ `run_inference` 原始 argmax |
| /human-parsing 只返回唯一部件"Hat"且占比=人像前景占比 | **模型语义错配**:引擎是 ATR 18 类人衣解析(ImageNet 归一化/512 输入,配套 segformer_b2_clothes),却喂了 pp_humanseg 人像**分割**模型(单通道 0/1 掩码)——掩码值 1 映射到标签表[1]="Hat",整个人被标成帽子 | 换 `segformer_b2_clothes.onnx`(mattmdjaga/segformer_b2_clothes);判别法:近景多人图若唯一部件且占比≈前景占比即中招 |
| hf-mirror 下载大模型得到损坏文件(onnx load 报 Wire format corrupt) | 大文件(>100MB)经镜像下载可能**静默截断**(56MB/110MB 无报错) | 下载后必须校验:大小对照 HF 页面 + `xxd -l 4` magic + onnx 加载;不完整 `curl -L --retry 3` 重下 |
| 前端显示"未检出目标"但后端日志识别成功 | **前后端字段契约脱节**:后端 /license-plate 返回 `data.plates`,前端 plate 卡片按 kind="boxes" 读 `data.detections` | 前端新增 kind="plates" 读正确字段;此类问题根修是后端显式契约(如 depth preview_cols 先例),前端不猜字段名 |
| /classify 对多主体场景图返回空 predictions(引擎有推理、日志正常) | 分类引擎默认置信度阈值 0.5,`predict_top_k` 先过阈值再取 top-k;场景图概率分散后 top-1 < 0.5 即返回空——**文档化语义(空结果非错误)**,但消费方易误判为服务故障 | 单主体图验证(猫图 top1 0.74 正常);需要场景图分类时给引擎调低构造阈值(另立任务) |
| /detect/obb 对地面照片恒 0 框 | 唯一可用模型 yolov8n-obb 是 **DOTA 航拍 15 类**(飞机/船/小车…),地面视角属域错配,0 框是模型的正确行为 | 端点链路用深检验证;正向验证需航拍/俯拍图(含 DOTA 类目标) |
| warden 服务级环境变量只注入了第一个键 | services.toml 服务级字段名是 **`environment`**;写成 `env` 被 serde **静默忽略**(`env` 是 `[daemon]` 段全局变量键)——排查法:`tr '\0' '\n' < /proc/<pid>/environ` 对照 toml | 字段名改 `environment`;此类静默丢失建议 warden 后续加 unknown-field 告警 |
| reranker 复用 embedding runner 直接失败 / 分数恒错 | cross-encoder(reranker)与 embedding 接口形态三处不同:**输出仅 `logits [B,1]`**(无 `last_hidden_state`,复用 `runner::new` 构造即失败);**pair 模板 `<s>q</s></s>d</s>`**(query+doc 拼单条输入,非逐条独立编码);**tokenizer `<pad>=1` 非 BERT 约定的 0**(jina/bge 两族 reranker 均如此,pad 值错了 attention 语义全偏) | fork 走 `new_for_output("logits", Some(1))` 构造期校验末维=1;`encode_pairs` 批内最长 padding 且 pad id 从 tokenizer 解析(不硬编码);logits→sigmoid 取分(fork `1800743`,argus `/rerank`) |
| reranker int8 分数每次微变、与 fp32 不逐位一致 | int8 量化下 padding 长度/pad 组成影响 logits,**批组成或顺序变化实测抖动 0.08-0.33**(2026-10-22,非 ulp 级;fp32 更小);同批重放则逐位确定 | 排序指标实测与 fp32 无差,消费方取 top-N 不受影响;**分数缓存须以整批(query+texts 全数组)为键**,勿按单条 text 缓存 |
| texts 仅 32 条、总 token 884(远小于 8192)的单请求实测 9.7s | 批内按最长序列 padding,真实计算量是 `B × max_pair_len` 而非 Σ token——31 条短文本+1 条 512 token 时 Σ=884 可过 Σ 预算,但 padding 后 `32×512=16384`(2026-10-22 实测) | 预算指标用 **`B×max_pair_len`**(含 query 与特殊 token 的最长 pair 编码长度×条数),argus `/rerank` 按 `B×max_pair_len≤8192` 拒绝(超预算 bad_request);有效条数随文本长度变化(全 512 token 最多 16 条) |
