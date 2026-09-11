# vision-ext GPU 镜像（多阶段：源码编译 → CUDA 运行时）
#
# 构建（仓库根目录执行，deploy/lib/ 需先放入 onnxruntime-linux-x64-gpu-1.26.0/）：
#   docker build --build-arg GIT_HASH=$(git rev-parse --short HEAD) \
#     -f deploy/docker/Dockerfile -t vision-ext:0.1.0-gpu .
# 运行（宿主机需 NVIDIA 驱动 ≥525.60 + nvidia-container-toolkit）：
#   docker run -d --gpus all -p 7042:7042 vision-ext:0.1.0-gpu
# 无 GPU 机器验证容器链路（ORT CUDA EP 在无驱动时硬失败，不静默回退）：
#   docker run --rm -e VISION_EXT_DEVICE=cpu -p 7042:7042 vision-ext:0.1.0-gpu
#
# 基础镜像可从已有机器流式导入（如 yhpc，免外网下载）：
#   ssh yhpc docker save nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 | docker load

# ---- 构建阶段：ubuntu 22.04 + rustup ----
# 不用 rust:1-bookworm（glibc 2.36）：编译产物可能引用 GLIBC_2.35/2.36 符号，
# 在 glibc 2.35 的运行时镜像上无法加载。构建/运行同用 22.04（2.35）消除该风险。
FROM ubuntu:22.04 AS builder

# CARGO_BUILD_JOBS：共享机上限制并行度防拖垮整机；独立构建机可 --build-arg 提高
ARG CARGO_BUILD_JOBS=6
# 版本标识注入（构建上下文不含 .git，build.rs 读取这两个环境变量）
ARG GIT_HASH=""
ARG GIT_DIRTY=""
ENV CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS} \
    VISION_EXT_BUILD_GIT_HASH=${GIT_HASH} \
    VISION_EXT_BUILD_GIT_DIRTY=${GIT_DIRTY}

# apt 源换国内镜像（本站到 docker CDN 仅 ~0.7MB/s；tuna 走内网出口快得多）
RUN sed -i 's|//archive.ubuntu.com|//mirrors.tuna.tsinghua.edu.cn|g; s|//security.ubuntu.com|//mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup \
       RUSTUP_UPDATE_ROOT=https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup \
       curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH=/root/.cargo/bin:${PATH}

# crates.io 换 rsproxy sparse 索引（字节跳动国内镜像）
RUN mkdir -p /root/.cargo && printf '[source.crates-io]\nreplace-with = "rsproxy-sparse"\n\n[source.rsproxy-sparse]\nregistry = "sparse+https://rsproxy.cn/index/"\n' > /root/.cargo/config.toml

WORKDIR /build
# 依赖缓存层：先以 dummy main 编译全部依赖（含 rust-onnx-infer git 依赖，其
# api-26 + resize 卷积修复补丁由 Cargo.lock 锁定的远端 commit 提供），源码
# 变更时不重编依赖
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src && echo "fn main() {}" > src/main.rs \
    && cargo build --release --features cuda --locked && rm -rf src
COPY build.rs ./
COPY web ./web
COPY src ./src
RUN touch build.rs && cargo build --release --features cuda --locked \
    && strip target/release/vision-ext

# ---- 运行阶段：CUDA 12.8 + cuDNN 9.8 ----
# ORT CUDA EP 要求 CUDA 12.x + cuDNN 9 主版本；小版本对吞吐影响巨大：
# yolo-rs A100 实测 cuDNN 9.8（12.8.1 镜像）~500 QPS，9.1（12.4.1 镜像）仅 ~256，
# 新版 cuDNN 卷积内核对该负载快约 2 倍。勿回退低版本基础镜像
# （二分实验见 yolo-rs/deploy/README.md「容器性能实测」）。
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# HEALTHCHECK 依赖 curl；apt 源同上换国内镜像
RUN sed -i 's|//archive.ubuntu.com|//mirrors.tuna.tsinghua.edu.cn|g; s|//security.ubuntu.com|//mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/target/release/vision-ext /app/vision-ext
COPY deploy/lib/onnxruntime-linux-x64-gpu-1.26.0/lib/ /app/lib/
COPY models/ /app/models/

# ORT providers_cuda.so 与主库同目录，ORT 自动发现；CUDA/cuDNN 运行时由基镜像提供。
# vision-ext 按显式路径配置引擎（不扫目录），11 个能力的 14 个模型路径必须逐个给出
ENV ORT_DYLIB_PATH=/app/lib/libonnxruntime.so.1.26.0 \
    VISION_EXT_DEVICE=cuda \
    VISION_EXT_MODELS_DIR=/app/models \
    VISION_EXT_PORT=7042 \
    VISION_EXT_DETECTION_MODEL=/app/models/yolov8n.onnx \
    VISION_EXT_CLASSIFICATION_MODEL=/app/models/yolov8n-cls.onnx \
    VISION_EXT_OBB_MODEL=/app/models/yolov8n-obb.onnx \
    VISION_EXT_FACE_DET_MODEL=/app/models/face_detection_yunet_2023mar.onnx \
    VISION_EXT_FACE_REC_MODEL=/app/models/face_recognition_sface_2021dec.onnx \
    VISION_EXT_OCR_DET_MODEL=/app/models/ppocrv5_det.onnx \
    VISION_EXT_OCR_REC_MODEL=/app/models/ppocrv5_rec.onnx \
    VISION_EXT_OCR_DICT=/app/models/ppocrv5_dict.txt \
    VISION_EXT_PLATE_DET_MODEL=/app/models/mnet_plate.onnx \
    VISION_EXT_PLATE_REC_MODEL=/app/models/Final_LPRNet_model.onnx \
    VISION_EXT_QR_MODEL=/app/models/wechat_qr_detect.onnx \
    VISION_EXT_POSE_MODEL=/app/models/yolov8n-pose.onnx \
    VISION_EXT_DEPTH_MODEL=/app/models/depth_anything_v2_small.onnx \
    VISION_EXT_HUMAN_PARSING_MODEL=/app/models/segformer_b2_clothes.onnx \
    # 防护性默认：A100 与 vLLM 共享时不设上限会 arena 膨胀致死（config.rs P0 实测）；
    # 16GB 覆盖 11 引擎任意并发，docker run -e VISION_EXT_GPU_MEM_LIMIT_MB=... 覆盖
    VISION_EXT_GPU_MEM_LIMIT_MB=16384

EXPOSE 7042
# 浅检即可判活（深检跑真实推理，不适合周期探针）
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:7042/health || exit 1

ENTRYPOINT ["/app/vision-ext"]
