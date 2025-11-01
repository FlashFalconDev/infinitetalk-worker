# InfiniteTalk Worker - RTX 5090 安裝指南

**針對 NVIDIA RTX 5090 優化的完整部署文檔**

---

## 📋 目錄

1. [系統需求](#系統需求)
2. [關鍵配置說明](#關鍵配置說明)
3. [安裝步驟](#安裝步驟)
4. [RTX 5090 特定配置](#rtx-5090-特定配置)
5. [驗證安裝](#驗證安裝)
6. [常見問題](#常見問題)
7. [性能優化](#性能優化)

---

## 系統需求

### ✅ 已驗證的環境

| 組件 | 規格 | 說明 |
|------|------|------|
| **GPU** | NVIDIA RTX 5090 (32GB VRAM) | 計算能力 12.0 (Blackwell) |
| **驅動** | 581.57+ | 支援 CUDA 12.8 的最新驅動 |
| **作業系統** | Ubuntu 24.04 LTS | 其他 Linux 發行版亦可 |
| **Python** | 3.10 - 3.12 | 建議使用 3.11 或 3.12 |
| **CUDA** | 12.8 | 透過 PyTorch 安裝 |
| **硬碟空間** | 250GB+ | 模型約 235GB |
| **記憶體** | 32GB+ | 推薦 64GB |

### 🎯 最低配置

- **GPU**: 24GB+ VRAM
- **磁碟**: 250GB 可用空間
- **RAM**: 32GB 系統記憶體

---

## 關鍵配置說明

### 🔥 RTX 5090 的特殊性

RTX 5090 使用 **Blackwell 架構 (SM 12.0)**，與一些現有 AI 框架存在兼容性問題：

1. **Flash Attention 兼容性**
   - ⚠️ 當前版本的 Flash Attention 2.x 不完全支援 SM 12.0
   - ✅ **解決方案**: 禁用 Flash Attention，使用 xformers 的標準 attention

2. **PyTorch 版本**
   - ✅ 需要 **PyTorch 2.9.0+** 以支援 CUDA 12.8
   - ✅ 使用 `cu128` 版本

3. **transformers 版本鎖定**
   - ⚠️ 必須使用 **4.45.2**
   - 原因: 更新版本要求 PyTorch 2.6+ (尚不穩定)

---

## 安裝步驟

### 第 1 步: 環境準備

```bash
# 更新系統
sudo apt update && sudo apt upgrade -y

# 安裝必要套件
sudo apt install -y git wget curl build-essential python3-pip python3-venv

# 克隆專案
cd ~
git clone https://github.com/FlashFalconDev/infinitetalk-worker.git
cd infinitetalk-worker
```

### 第 2 步: 創建虛擬環境

```bash
# 創建虛擬環境
python3 -m venv venv

# 啟用虛擬環境
source venv/bin/activate

# 升級 pip
pip install --upgrade pip setuptools wheel
```

### 第 3 步: 安裝 PyTorch (RTX 5090 專用)

**⚠️ 關鍵步驟！必須先安裝 PyTorch，其他依賴才能正確編譯**

```bash
# 安裝 PyTorch 2.9.0 + CUDA 12.8 (RTX 5090 必須)
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

驗證安裝：

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'CUDA 版本: {torch.version.cuda}')
"
```

**預期輸出**:
```
PyTorch: 2.9.0+cu128
CUDA 可用: True
GPU: NVIDIA GeForce RTX 5090
CUDA 版本: 12.8
```

### 第 4 步: 安裝其他依賴

```bash
# 安裝除 flash_attn 外的所有依賴
grep -v "flash_attn" requirements.txt > requirements_temp.txt
pip install -r requirements_temp.txt

# 🔧 修復 transformers 版本 (關鍵!)
pip install 'transformers==4.45.2' --force-reinstall

# 安裝 Hugging Face CLI
pip install huggingface-hub
```

### 第 5 步: 配置環境變數

創建 `.env` 文件：

```bash
cp .env.example .env
nano .env
```

**RTX 5090 專用配置**:

```ini
# API 設定
INFINITETALK_API_BASE=https://host.flashfalcon.info
INFINITETALK_WORKER_TOKEN=你的_token_這裡

# Multi-GPU 設定 (如果只有一張 5090，設為 false)
ENABLE_MULTI_GPU=false
NUM_WORKERS=1

# ⚠️ RTX 5090 關鍵設定 - 禁用 Flash Attention
XFORMERS_DISABLE_FLASH_ATTN=1
XFORMERS_FORCE_DISABLE_TRITON=1
ATTN_BACKEND=xformers

# PyTorch CUDA 記憶體優化
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**⚠️ 重要**: 將 `你的_token_這裡` 替換為實際的 Worker Token

### 第 6 步: 下載模型文件

**總大小約 235GB，需時 2-4 小時**

```bash
# 創建模型目錄
mkdir -p weights
cd weights

# 1. 下載 chinese-wav2vec2-base (1.5GB)
huggingface-cli download TencentGameMate/chinese-wav2vec2-base \
    --local-dir ./chinese-wav2vec2-base

# 下載 PR #1 的額外文件
huggingface-cli download TencentGameMate/chinese-wav2vec2-base \
    model.safetensors --revision refs/pr/1 \
    --local-dir ./chinese-wav2vec2-base

# 2. 下載 RealESRGAN (2.4MB)
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth

# 3. 下載 Wan2.1-I2V-14B-480P (~77GB)
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
    --local-dir ./Wan2.1-I2V-14B-480P

# 4. 下載 InfiniteTalk (~157GB) - 最大的模型
huggingface-cli download MeiGen-AI/InfiniteTalk \
    --local-dir ./InfiniteTalk

# 5. 下載 LoRA 文件 (354MB)
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
    FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors \
    --local-dir-use-symlinks False \
    --local-dir ./

# 返回專案根目錄
cd ..
```

查看下載結果：

```bash
du -sh weights/*
```

**預期輸出**:
```
1.5G    weights/chinese-wav2vec2-base
77G     weights/Wan2.1-I2V-14B-480P
157G    weights/InfiniteTalk
354M    weights/Wan2.1_I2V_14B_FusionX_LoRA.safetensors
2.4M    weights/realesr-animevideov3.pth
```

---

## RTX 5090 特定配置

### 🔧 worker.py 中的關鍵設定

文件已包含 RTX 5090 優化配置 (worker.py:18-23):

```python
# Configure PyTorch CUDA memory allocator to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Disable xformers Flash Attention (RTX 5090 compatibility)
os.environ['XFORMERS_DISABLE_FLASH_ATTN'] = '1'
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'
```

### 📊 記憶體管理

RTX 5090 有 32GB VRAM，足夠運行完整模型：

```python
# 在 Python 環境中設定
import torch
torch.cuda.empty_cache()  # 清理未使用的記憶體
torch.backends.cudnn.benchmark = True  # 自動尋找最佳演算法
```

### ⚡ Attention 機制選擇

由於 Flash Attention 兼容性問題，使用 xformers 的標準實作：

| Attention 類型 | RTX 5090 支援 | 性能 | 記憶體 |
|---------------|-------------|------|--------|
| **Flash Attention 2** | ❌ 不兼容 | 最快 | 最省 |
| **xformers (標準)** | ✅ **推薦** | 快 | 中等 |
| **PyTorch 原生** | ✅ 兼容 | 較慢 | 較高 |

---

## 驗證安裝

### 1️⃣ 檢查依賴版本

```bash
source venv/bin/activate

python -c "
import torch
import transformers
import diffusers
import xformers

print('=' * 60)
print('✅ 依賴檢查')
print('=' * 60)
print(f'PyTorch:      {torch.__version__}')
print(f'CUDA:         {torch.version.cuda}')
print(f'cuDNN:        {torch.backends.cudnn.version()}')
print(f'transformers: {transformers.__version__}')
print(f'diffusers:    {diffusers.__version__}')
print(f'xformers:     {xformers.__version__}')
print('=' * 60)
print(f'GPU:          {torch.cuda.get_device_name(0)}')
print(f'VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
print(f'計算能力:      {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}')
print('=' * 60)
"
```

**預期輸出**:
```
============================================================
✅ 依賴檢查
============================================================
PyTorch:      2.9.0+cu128
CUDA:         12.8
cuDNN:        91002
transformers: 4.45.2
diffusers:    0.35.1
xformers:     0.0.33+5d4b92a5.d20251029
============================================================
GPU:          NVIDIA GeForce RTX 5090
VRAM:         31.84 GB
計算能力:      12.0
============================================================
```

### 2️⃣ 檢查模型文件

```bash
ls -lh weights/
```

應該看到：
- ✅ chinese-wav2vec2-base/ (目錄)
- ✅ Wan2.1-I2V-14B-480P/ (目錄)
- ✅ InfiniteTalk/ (目錄)
- ✅ Wan2.1_I2V_14B_FusionX_LoRA.safetensors (354MB)
- ✅ realesr-animevideov3.pth (2.4MB)

### 3️⃣ 測試運行

```bash
source venv/bin/activate
python worker.py
```

**成功啟動的標誌**:

```
======================================================================
🚀 初始化 InfiniteTalk Worker v7.3.3
🆔 Worker ID: your-worker-id
🌐 主 API: https://www.flashfalcon.info
🔄 備用 API: https://host.flashfalcon.info
📊 GPU 監控: ✅ 已啟用
======================================================================
🔌 測試連線...
✅ 連線成功
📥 載入模型（只執行一次）...
✅ wav2vec2 完成
✅ InfiniteTalk 完成
🎉 模型已常駐！
💓 心跳線程已啟動（每 60 秒）
======================================================================
✅ Worker 準備就緒!
======================================================================
🤖 InfiniteTalk Worker 運行中...
```

---

## 常見問題

### ❓ Q1: Flash Attention 錯誤

**錯誤訊息**:
```
RuntimeError: FlashAttention only supports Ampere GPUs or newer
```

**解決方案**:

確認 `.env` 文件中已設置：
```ini
XFORMERS_DISABLE_FLASH_ATTN=1
XFORMERS_FORCE_DISABLE_TRITON=1
```

### ❓ Q2: CUDA Out of Memory

**錯誤訊息**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**解決方案**:

1. 確認記憶體配置：
```bash
nvidia-smi
```

2. 添加環境變數：
```ini
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
```

3. 重啟 worker

### ❓ Q3: transformers 版本衝突

**錯誤訊息**:
```
ERROR: transformers requires torch>=2.6.0
```

**解決方案**:

強制重裝 transformers 4.45.2：
```bash
pip install 'transformers==4.45.2' --force-reinstall
```

### ❓ Q4: 驅動版本過舊

**錯誤訊息**:
```
NVIDIA driver 555.xx does not support CUDA 12.8
```

**解決方案**:

更新 NVIDIA 驅動至 581.57+：

```bash
# Ubuntu
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-581
sudo reboot
```

### ❓ Q5: 模型加載失敗

**錯誤訊息**:
```
FileNotFoundError: LoRA file not found
```

**解決方案**:

檢查 LoRA 文件路徑：
```bash
ls -lh weights/Wan2.1_I2V_14B_FusionX_LoRA.safetensors
```

如果不存在，重新下載：
```bash
cd weights
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
    FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors \
    --local-dir-use-symlinks False \
    --local-dir ./
```

---

## 性能優化

### 🚀 RTX 5090 優化建議

#### 1. CUDA 圖形優化

在 `model_service.py` 中啟用 CUDA graphs (如果支援):

```python
# 適用於穩定的推理工作負載
torch.cuda.synchronize()
with torch.cuda.graph(graph):
    output = model(input)
```

#### 2. 混合精度訓練

RTX 5090 對 FP16/BF16 有優異支援：

```python
from torch.cuda.amp import autocast

with autocast():
    output = model(input)
```

#### 3. 批次處理優化

根據 VRAM 調整 batch size：

| VRAM | 推薦 Batch Size | Quality |
|------|----------------|---------|
| 32GB | 2-4 | High |
| 24GB | 1-2 | Balanced |
| <24GB | 1 | Low |

#### 4. 記憶體碎片優化

已在 worker.py 中配置：

```python
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

額外優化：

```bash
# 定期清理記憶體
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
```

#### 5. TensorRT 加速 (進階)

如需極致性能，可考慮使用 TensorRT：

```bash
pip install tensorrt
```

### 📊 性能基準

在 RTX 5090 上的預期性能：

| 任務 | 解析度 | 時間 | VRAM 使用 |
|------|--------|------|-----------|
| 圖片+音頻 → 影片 | 480p | 2-3 分鐘 | ~18GB |
| 圖片+音頻 → 影片 | 720p | 4-5 分鐘 | ~24GB |
| 圖片+音頻 → 影片 | 1080p | 8-10 分鐘 | ~28GB |

---

## 生產環境部署

### 🔄 使用 systemd 服務

創建服務文件：

```bash
sudo nano /etc/systemd/system/infinitetalk-worker.service
```

內容：

```ini
[Unit]
Description=InfiniteTalk Worker Service (RTX 5090)
After=network.target

[Service]
Type=simple
User=你的使用者名稱
WorkingDirectory=/home/你的使用者名稱/infinitetalk-worker
Environment="PATH=/home/你的使用者名稱/infinitetalk-worker/venv/bin"
ExecStart=/home/你的使用者名稱/infinitetalk-worker/venv/bin/python worker.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/infinitetalk-worker.log
StandardError=append:/var/log/infinitetalk-worker.error.log

[Install]
WantedBy=multi-user.target
```

啟用服務：

```bash
sudo systemctl daemon-reload
sudo systemctl enable infinitetalk-worker
sudo systemctl start infinitetalk-worker
sudo systemctl status infinitetalk-worker
```

查看日誌：

```bash
sudo journalctl -u infinitetalk-worker -f
```

---

## 維護與監控

### 📈 GPU 監控

即時監控 RTX 5090：

```bash
# 每秒更新
watch -n 1 nvidia-smi

# 詳細資訊
nvidia-smi dmon -s pucvmet
```

### 🔍 Worker 監控

```bash
# 查看進程
ps aux | grep worker.py

# 查看日誌
tail -f worker.log

# 檢查記憶體使用
free -h
```

### 🧹 定期維護

```bash
# 清理 PyTorch 緩存
python -c "import torch; torch.cuda.empty_cache()"

# 清理臨時文件
rm -rf temp_downloads/*
rm -rf outputs/*

# 更新依賴 (謹慎!)
pip list --outdated
```

---

## 故障排除清單

執行以下命令診斷問題：

```bash
# 1. 檢查 GPU
nvidia-smi

# 2. 檢查 CUDA
nvcc --version

# 3. 檢查 Python 環境
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 4. 檢查模型文件
ls -lh weights/

# 5. 檢查環境變數
cat .env

# 6. 檢查端口占用
sudo netstat -tulpn | grep python

# 7. 檢查磁碟空間
df -h

# 8. 檢查記憶體
free -h
```

---

## 更新日誌

### v7.3.3 (2025-10-31)

✅ **RTX 5090 完整支援**
- 添加 CUDA 12.8 支援
- 禁用 Flash Attention 以兼容 SM 12.0
- 優化記憶體管理
- 更新依賴至最新穩定版本

---

## 相關文檔

- 📖 [SMOOTH_DEPLOYMENT_GUIDE.md](./SMOOTH_DEPLOYMENT_GUIDE.md) - 通用部署指南
- 📖 [LORA_DOWNLOAD_GUIDE.md](./LORA_DOWNLOAD_GUIDE.md) - LoRA 下載詳解
- 📖 [README_OFFICIAL_DEPLOYMENT.md](./README_OFFICIAL_DEPLOYMENT.md) - 官方部署流程
- 🐛 [CRASH_DIAGNOSIS.md](./CRASH_DIAGNOSIS.md) - 崩潰診斷

---

## 支援與社群

- **GitHub Issues**: https://github.com/FlashFalconDev/infinitetalk-worker/issues
- **官方專案**: https://github.com/MeiGen-AI/InfiniteTalk
- **Email**: support@flashfalcon.info

---

## 致謝

- **NVIDIA** - RTX 5090 及 CUDA 工具鏈
- **MeiGen-AI** - InfiniteTalk 官方專案
- **Wan-AI** - Wan 影片生成模型
- **PyTorch Team** - CUDA 12.8 支援

---

**文檔版本**: 1.0.0
**最後更新**: 2025-10-31
**測試環境**: Ubuntu 24.04 LTS + RTX 5090 32GB
**維護者**: FlashFalcon Development Team

---

## 快速啟動檢查清單

在開始前，確認以下項目：

- [ ] NVIDIA 驅動 581.57+ 已安裝
- [ ] Python 3.10-3.12 已安裝
- [ ] 至少 250GB 可用硬碟空間
- [ ] 虛擬環境已創建並啟用
- [ ] PyTorch 2.9.0+cu128 已安裝
- [ ] transformers 4.45.2 已安裝
- [ ] .env 文件已配置 (Token 已填入)
- [ ] XFORMERS_DISABLE_FLASH_ATTN=1 已設置
- [ ] 所有模型文件已下載 (~235GB)
- [ ] LoRA 文件存在 (354MB)
- [ ] 測試運行成功，模型已加載

**全部打勾？恭喜，可以開始使用了！** 🎉
