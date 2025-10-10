# InfiniteTalk Worker Docker Image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 設定環境變數
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 設定工作目錄
WORKDIR /workspace/InfiniteTalk

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 升級 pip
RUN pip3 install --no-cache-dir --upgrade pip

# 複製 requirements.txt（如果有的話）
COPY requirements.txt* ./

# 安裝 Python 依賴
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install --no-cache-dir \
    diffusers \
    transformers \
    accelerate \
    xformers \
    opencv-python \
    pillow \
    numpy \
    requests \
    python-dotenv \
    nvidia-ml-py3 \
    peft

# 複製應用程式代碼
COPY . .

# 創建必要的目錄
RUN mkdir -p temp_downloads outputs

# 健康檢查
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# 啟動 Worker
CMD ["python3", "worker.py"]
