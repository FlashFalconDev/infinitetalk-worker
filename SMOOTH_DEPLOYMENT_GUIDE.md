# InfiniteTalk Worker 順暢部署流程

**實戰驗證版本** - 基於實際部署經驗總結的完整流程

**驗證日期**: 2025-10-30
**環境**: vast.ai GPU 實例
**Python**: 3.12
**CUDA**: 12.1

---

## 📋 目錄

1. [環境要求](#環境要求)
2. [關鍵問題與解決方案](#關鍵問題與解決方案)
3. [完整安裝流程](#完整安裝流程)
4. [模型下載詳解](#模型下載詳解)
5. [驗證與啟動](#驗證與啟動)
6. [常見錯誤處理](#常見錯誤處理)

---

## 環境要求

### 硬件要求
- **GPU**: NVIDIA GPU (推薦 24GB+ VRAM)
- **磁盤空間**: 至少 250GB 可用
- **內存**: 32GB+ RAM 推薦

### 軟件要求
- **操作系統**: Linux (Ubuntu 20.04+)
- **Python**: 3.10 - 3.12
- **CUDA**: 12.1
- **Git**: 已安裝
- **網絡**: 穩定的國際網絡連接 (用於下載模型)

---

## 關鍵問題與解決方案

### ⚠️ 問題 1: 依賴安裝順序錯誤

**錯誤現象**:
```
ModuleNotFoundError: No module named 'torch'
error: subprocess-exited-with-error during flash_attn setup.py
```

**根本原因**:
`flash_attn` 編譯時需要 torch,但 `pip install -r requirements.txt` 按字母順序安裝,導致 flash_attn 先於 torch 安裝。

**解決方案**:
```bash
# 1. 先安裝 PyTorch
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. 安裝其他依賴 (排除 flash_attn)
grep -v "flash_attn" requirements.txt > requirements_temp.txt
pip install -r requirements_temp.txt
rm requirements_temp.txt
```

---

### ⚠️ 問題 2: transformers 版本衝突 (最嚴重)

**錯誤現象**:
```
ValueError: Due to a serious vulnerability issue in torch.load,
even with weights_only=True, we now require users to upgrade
torch to at least v2.6. See: CVE-2025-32434
```

**根本原因**:
transformers 4.56+ 添加了安全檢查,要求 torch>=2.6,但 PyTorch 2.6 尚未發布!

**嘗試過的失敗方案**:
- ❌ 升級 torch 到 2.5.1 → 仍然要求 2.6
- ❌ 降級 transformers 到 4.40.0 → 缺少 Gemma2PreTrainedModel
- ❌ 使用 transformers 4.56.2 → 同樣的安全檢查

**最終解決方案**:
```bash
# 使用 transformers 4.45.2 (最後一個兼容 torch 2.4.1 的版本)
pip install 'transformers==4.45.2' --force-reinstall
```

**關鍵版本組合**:
```
torch==2.4.1+cu121
transformers==4.45.2  ← 必須固定這個版本！
diffusers==0.35.1
xformers==0.0.28
```

---

### ⚠️ 問題 3: 錯誤的模型倉庫名稱

**錯誤現象**:
```
401 Client Error: Unauthorized
for url: https://huggingface.co/api/models/MeiGen-AI/Wan2.1-I2V-14B-480P
```

**根本原因**:
早期文檔使用了錯誤的倉庫名稱。

**正確的倉庫映射**:
| 模型 | ❌ 錯誤倉庫 | ✅ 正確倉庫 |
|------|-----------|-----------|
| Wan2.1-I2V-14B-480P | MeiGen-AI/Wan2.1-I2V-14B-480P | **Wan-AI**/Wan2.1-I2V-14B-480P |
| InfiniteTalk | ✓ MeiGen-AI/InfiniteTalk | MeiGen-AI/InfiniteTalk |

---

### ⚠️ 問題 4: LoRA 文件路徑錯誤

**錯誤現象**:
```
404 Not Found
for url: https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX/resolve/main/Wan2.1_I2V_14B_FusionX_LoRA.safetensors
```

**根本原因**:
LoRA 文件在子目錄 `FusionX_LoRa/` 中,不在主目錄。

**正確的下載路徑**:
```bash
# ❌ 錯誤 (主目錄不存在)
https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX/resolve/main/Wan2.1_I2V_14B_FusionX_LoRA.safetensors

# ✅ 正確 (在 FusionX_LoRa 子目錄)
https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX/blob/main/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors
```

**正確的下載命令**:
```bash
cd /workspace/weights
huggingface-cli download vrgamedevgirl84/Wan14BT2VFusioniX \
    FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors \
    --local-dir . --local-dir-use-symlinks False

# 移動到正確位置
mv FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors .
rm -rf FusionX_LoRa
```

---

## 完整安裝流程

### 步驟 1: 克隆倉庫

```bash
cd /workspace
git clone https://github.com/FlashFalconDev/infinitetalk-worker.git
cd infinitetalk-worker
```

---

### 步驟 2: 按正確順序安裝依賴 ⭐

**重要**: 必須按照此順序安裝,否則會出錯！

```bash
# 2.1 先安裝 PyTorch (必須第一步)
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 2.2 安裝其他依賴 (排除 flash_attn)
grep -v "flash_attn" requirements.txt > requirements_temp.txt
pip install -r requirements_temp.txt
rm requirements_temp.txt

# 2.3 修復 transformers 版本衝突 (關鍵步驟!)
pip install 'transformers==4.45.2' --force-reinstall

# 2.4 驗證關鍵包版本
python -c "import torch; print(f'torch: {torch.__version__}')"
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
python -c "import diffusers; print(f'diffusers: {diffusers.__version__}')"
```

**預期輸出**:
```
torch: 2.4.1+cu121
transformers: 4.45.2
diffusers: 0.35.1
```

---

### 步驟 3: 配置環境變量

```bash
# 複製配置文件
cp .env.example .env

# 編輯配置文件
nano .env
```

**必須配置的項目**:
```env
INFINITETALK_API_BASE=https://host.flashfalcon.info
INFINITETALK_WORKER_TOKEN=你的token這裡
ENABLE_MULTI_GPU=false
NUM_WORKERS=2
```

---

### 步驟 4: 下載模型 (按順序) ⭐

創建模型目錄:
```bash
mkdir -p /workspace/weights
cd /workspace/weights
```

#### 4.1 chinese-wav2vec2-base (2.9GB)

```bash
huggingface-cli download TencentGameMate/chinese-wav2vec2-base \
    --local-dir ./chinese-wav2vec2-base

# 下載 PR #1 的額外文件
huggingface-cli download TencentGameMate/chinese-wav2vec2-base \
    model.safetensors --revision refs/pr/1 \
    --local-dir ./chinese-wav2vec2-base
```

**驗證**:
```bash
du -sh chinese-wav2vec2-base
# 應顯示: 2.9G
```

---

#### 4.2 RealESRGAN (2.4MB)

```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth
```

**驗證**:
```bash
ls -lh realesr-animevideov3.pth
# 應顯示: 2.4M
```

---

#### 4.3 Wan2.1-I2V-14B-480P (~62GB) ⚠️

**注意**: 使用正確的倉庫名 `Wan-AI`！

```bash
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
    --local-dir ./Wan2.1-I2V-14B-480P
```

**下載時間**: 約 30-60 分鐘 (取決於網速)

**驗證**:
```bash
du -sh Wan2.1-I2V-14B-480P
# 應顯示: 62G 左右

# 檢查關鍵文件
ls Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth
ls Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth
ls Wan2.1-I2V-14B-480P/diffusion_pytorch_model-*.safetensors | wc -l
# 應該有 7 個 safetensors 文件
```

---

#### 4.4 InfiniteTalk (~160GB) ⚠️

**最大的模型,需要最長時間下載**

```bash
huggingface-cli download MeiGen-AI/InfiniteTalk \
    --local-dir ./InfiniteTalk
```

**下載時間**: 約 1-3 小時 (取決於網速)

**驗證**:
```bash
du -sh InfiniteTalk
# 應顯示: 160G 左右

# 檢查關鍵文件 (worker 需要 single 版本)
ls -lh InfiniteTalk/single/infinitetalk.safetensors
# 應顯示: 9.3G
```

---

#### 4.5 FusionX LoRA (354MB) ⚠️ 必須

**重要**: 這是必須的文件,不能省略！

```bash
cd /workspace/weights

# 下載 (會下載到 FusionX_LoRa 子目錄)
huggingface-cli download vrgamedevgirl84/Wan14BT2VFusioniX \
    FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors \
    --local-dir . --local-dir-use-symlinks False

# 移動到正確位置
mv FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors .
rm -rf FusionX_LoRa
```

**驗證**:
```bash
ls -lh Wan2.1_I2V_14B_FusionX_LoRA.safetensors
# 應顯示: 354M
```

---

### 步驟 5: 最終驗證

```bash
cd /workspace/weights
ls -lh

# 預期輸出:
# chinese-wav2vec2-base/         (目錄, 2.9GB)
# InfiniteTalk/                  (目錄, ~160GB)
# Wan2.1-I2V-14B-480P/          (目錄, ~62GB)
# Wan2.1_I2V_14B_FusionX_LoRA.safetensors  (354MB)
# realesr-animevideov3.pth      (2.4MB)

# 檢查總大小
du -sh /workspace/weights
# 應顯示: ~230GB
```

---

## 驗證與啟動

### 啟動 Worker

```bash
cd /workspace/infinitetalk-worker

# 前台運行 (測試用)
python worker.py

# 或後台運行 (生產環境)
nohup python worker.py > worker.log 2>&1 &
```

### 查看日志

```bash
tail -f worker.log
```

### 成功啟動的標誌

日誌應該顯示:
```
2025-10-30 XX:XX:XX - INFO - ✅ GPU 監控已啟用 (偵測到 1 個 GPU)
2025-10-30 XX:XX:XX - INFO - 🚀 初始化 InfiniteTalk Worker v7.3.3
2025-10-30 XX:XX:XX - INFO - ✅ 連線成功
2025-10-30 XX:XX:XX - INFO -    使用 API: https://www.flashfalcon.info
2025-10-30 XX:XX:XX - INFO - ================================================================================
2025-10-30 XX:XX:XX - INFO - 🎨 LoRA 配置:
2025-10-30 XX:XX:XX - INFO -    路徑: weights/Wan2.1_I2V_14B_FusionX_LoRA.safetensors
2025-10-30 XX:XX:XX - INFO -    狀態: ✅ 已載入
2025-10-30 XX:XX:XX - INFO -    大小: 354.0 MB
2025-10-30 XX:XX:XX - INFO - ================================================================================
2025-10-30 XX:XX:XX - INFO - 📥 載入 wav2vec2...
2025-10-30 XX:XX:XX - INFO - ✅ wav2vec2 完成
2025-10-30 XX:XX:XX - INFO - 📥 載入 InfiniteTalk...
2025-10-30 XX:XX:XX - INFO - ✅ InfiniteTalk 完成
2025-10-30 XX:XX:XX - INFO - 🎉 模型已常駐！
```

**如果看到這些訊息,表示部署成功！** ✅

---

## 常見錯誤處理

### 錯誤 1: torch 版本要求 2.6

```
ValueError: torch >= 2.6 required
```

**解決**:
```bash
pip install 'transformers==4.45.2' --force-reinstall
```

---

### 錯誤 2: LoRA 文件未找到

```
FileNotFoundError: ❌ LoRA 檔案不存在
```

**解決**:
```bash
cd /workspace/weights
huggingface-cli download vrgamedevgirl84/Wan14BT2VFusioniX \
    FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors \
    --local-dir . --local-dir-use-symlinks False
mv FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors .
```

---

### 錯誤 3: models_t5 文件未找到

```
FileNotFoundError: 'weights/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth'
```

**原因**: Wan 模型下載未完成

**解決**:
```bash
# 檢查下載進程
ps aux | grep "huggingface-cli download Wan-AI"

# 如果沒有進程,重新下載
cd /workspace/weights
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
    --local-dir ./Wan2.1-I2V-14B-480P
```

---

### 錯誤 4: CUDA Out of Memory

```
torch.cuda.OutOfMemoryError
```

**解決方案**:
1. 檢查 GPU 內存: `nvidia-smi`
2. 清理其他進程
3. 使用更低的品質設定 (ultra_fast, turbo, fast)

---

## 自動化腳本

為了簡化部署,項目提供了自動化腳本:

### 一鍵部署

```bash
cd /workspace/infinitetalk-worker
./deploy_official.sh
```

這個腳本會:
1. ✅ 檢查環境
2. ✅ 按正確順序安裝依賴
3. ✅ 修復 transformers 版本
4. ✅ 配置 .env
5. ✅ 下載所有模型
6. ✅ 驗證安裝

### 只下載模型

```bash
cd /workspace/infinitetalk-worker
./download_models_official.sh
```

---

## 完整時間估算

| 步驟 | 時間 | 說明 |
|------|------|------|
| 1. 克隆倉庫 | 1 分鐘 | |
| 2. 安裝依賴 | 10-15 分鐘 | 包括 PyTorch 編譯 |
| 3. 配置環境 | 2 分鐘 | 手動編輯 .env |
| 4.1 chinese-wav2vec2-base | 5 分鐘 | 2.9GB |
| 4.2 RealESRGAN | 10 秒 | 2.4MB |
| 4.3 Wan2.1-I2V-14B-480P | 30-60 分鐘 | 62GB |
| 4.4 InfiniteTalk | 60-180 分鐘 | 160GB |
| 4.5 LoRA | 1 分鐘 | 354MB |
| 5. 驗證與啟動 | 5 分鐘 | 模型載入 |
| **總計** | **2-4 小時** | 主要取決於網速 |

---

## 關鍵成功要素 ⭐

1. **依賴安裝順序**: PyTorch 必須先安裝
2. **transformers 版本**: 必須固定在 4.45.2
3. **模型倉庫名稱**: Wan-AI (不是 MeiGen-AI)
4. **LoRA 文件路徑**: FusionX_LoRa 子目錄
5. **磁盤空間**: 確保至少 250GB 可用

---

## 故障排除檢查清單

部署失敗時,按順序檢查:

- [ ] Python 版本是 3.10-3.12
- [ ] CUDA 版本是 12.1
- [ ] PyTorch 版本是 2.4.1+cu121
- [ ] transformers 版本是 4.45.2
- [ ] 磁盤空間足夠 (250GB+)
- [ ] 網絡連接穩定
- [ ] .env 文件已配置 token
- [ ] 所有 5 個模型文件已下載
- [ ] LoRA 文件在正確位置
- [ ] Wan 模型包含 models_t5 文件

---

## 目錄結構參考

```
/workspace/
├── infinitetalk-worker/              # Worker 代碼
│   ├── worker.py                     # 主程序
│   ├── model_service.py              # 模型服務
│   ├── .env                          # 配置文件 (需手動創建)
│   ├── requirements.txt              # 依賴清單
│   ├── deploy_official.sh            # 一鍵部署 ⭐
│   ├── download_models_official.sh   # 模型下載 ⭐
│   ├── SMOOTH_DEPLOYMENT_GUIDE.md    # 本文檔
│   └── LORA_DOWNLOAD_GUIDE.md        # LoRA 詳細指南
│
└── weights/                          # 模型目錄 (~230GB)
    ├── chinese-wav2vec2-base/        # 2.9GB
    │   ├── config.json
    │   ├── model.safetensors
    │   └── ...
    ├── Wan2.1-I2V-14B-480P/         # ~62GB
    │   ├── models_t5_umt5-xxl-enc-bf16.pth  ← 關鍵文件
    │   ├── Wan2.1_VAE.pth
    │   ├── diffusion_pytorch_model-*.safetensors (7個)
    │   └── ...
    ├── InfiniteTalk/                 # ~160GB
    │   ├── single/
    │   │   └── infinitetalk.safetensors  ← Worker 使用這個
    │   ├── multi/
    │   └── ...
    ├── Wan2.1_I2V_14B_FusionX_LoRA.safetensors  # 354MB ← 必須
    └── realesr-animevideov3.pth      # 2.4MB
```

---

## 相關文檔

- **LORA_DOWNLOAD_GUIDE.md** - LoRA 詳細下載指南
- **README_OFFICIAL_DEPLOYMENT.md** - 官方部署文檔
- **DEPLOYMENT_ISSUES_LOG.md** - 問題記錄和解決方案
- **STATUS_AND_NEXT_STEPS.md** - 當前狀態

---

## 獲取支持

- **GitHub Issues**: https://github.com/FlashFalconDev/infinitetalk-worker/issues
- **官方倉庫**: https://github.com/MeiGen-AI/InfiniteTalk
- **Email**: support@flashfalcon.info

---

**文檔版本**: 1.0
**創建日期**: 2025-10-30
**驗證狀態**: ✅ 實戰驗證通過
**維護者**: FlashFalcon Dev Team

**特別感謝**: 本文檔基於實際部署過程中遇到的所有問題和解決方案編寫,確保後續部署一次成功。
