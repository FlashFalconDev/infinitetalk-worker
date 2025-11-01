# InfiniteTalk Worker

**音頻驅動的視頻生成 Worker** - 基於官方 MeiGen-AI/InfiniteTalk

---

## 🔥 RTX 5090 用戶請注意！

**如果你使用 NVIDIA RTX 5090 (Blackwell 架構)**，請優先閱讀：

### 👉 [RTX_5090_INSTALLATION_GUIDE.md](./RTX_5090_INSTALLATION_GUIDE.md) ⚡

這份指南包含：
- ✅ CUDA 12.8 + PyTorch 2.9.0 配置
- ✅ Flash Attention 兼容性解決方案
- ✅ RTX 5090 專屬優化設置
- ✅ 已在 RTX 5090 32GB 上完整驗證

**RTX 5090 與舊版 GPU 配置不同，請務必使用專用指南！**

---

## 🚀 快速開始 (其他 GPU)

### 推薦閱讀順序

1. **SMOOTH_DEPLOYMENT_GUIDE.md** ⭐ **← 從這裡開始！**
   - 完整的實戰驗證流程
   - 包含所有已知問題的解決方案
   - 詳細的步驟說明和驗證命令
   - 預計時間: 2-4 小時

2. **LORA_DOWNLOAD_GUIDE.md**
   - LoRA 文件的詳細下載指南
   - LoRA 是必須的,不能省略

3. **README_OFFICIAL_DEPLOYMENT.md**
   - 官方部署流程文檔
   - 與 SMOOTH_DEPLOYMENT_GUIDE 互補

---

## ⚡ 一鍵部署

如果你想直接開始:

```bash
cd /workspace/infinitetalk-worker
./deploy_official.sh
```

這個腳本會自動:
- ✅ 檢查環境
- ✅ 按正確順序安裝依賴
- ✅ 修復版本衝突
- ✅ 下載所有模型 (~230GB)
- ✅ 驗證安裝

---

## 📋 環境要求

### 一般 GPU (RTX 3090, 4090, A100 等)
- **GPU**: NVIDIA GPU (24GB+ VRAM 推薦)
- **磁盤**: 至少 250GB 可用空間
- **內存**: 32GB+ RAM
- **Python**: 3.10 - 3.12
- **CUDA**: 12.1

### RTX 5090 專用要求 🔥
- **GPU**: NVIDIA RTX 5090 (32GB VRAM)
- **驅動**: 581.57+
- **CUDA**: 12.8 (透過 PyTorch 2.9.0)
- **特殊配置**: 禁用 Flash Attention
- **詳細說明**: 請參考 [RTX_5090_INSTALLATION_GUIDE.md](./RTX_5090_INSTALLATION_GUIDE.md)

---

## 🔑 關鍵成功要素

在開始部署前,請務必了解這些關鍵點:

1. **依賴安裝順序**
   - ⚠️ PyTorch **必須**先安裝
   - 原因: flash_attn 編譯時需要 torch

2. **transformers 版本**
   - ⚠️ 必須固定在 **4.45.2**
   - 原因: 4.56+ 要求 torch>=2.6 (尚未發布)

3. **模型倉庫名稱**
   - ⚠️ Wan 模型在 **Wan-AI** (不是 MeiGen-AI)
   - 錯誤的倉庫名會導致 401 錯誤

4. **LoRA 文件**
   - ⚠️ LoRA 是**必須的**,不能省略
   - 正確路徑: `FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors`

詳細說明請參考 `SMOOTH_DEPLOYMENT_GUIDE.md`

---

## 📚 完整文檔列表

| 文檔 | 用途 | 推薦度 |
|------|------|--------|
| **RTX_5090_INSTALLATION_GUIDE.md** 🔥 | **RTX 5090 專用安裝指南** | ⭐⭐⭐⭐⭐ |
| **SMOOTH_DEPLOYMENT_GUIDE.md** | 完整安裝流程 (其他 GPU) | ⭐⭐⭐⭐⭐ |
| **LORA_DOWNLOAD_GUIDE.md** | LoRA 詳細下載指南 | ⭐⭐⭐⭐ |
| **README_OFFICIAL_DEPLOYMENT.md** | 官方部署文檔 | ⭐⭐⭐ |
| **DEPLOYMENT_ISSUES_LOG.md** | 問題記錄和解決方案 | ⭐⭐⭐ |
| **STATUS_AND_NEXT_STEPS.md** | 當前狀態 | ⭐⭐ |

---

## 🛠️ 手動安裝步驟概要

如果你想手動安裝,按此順序:

```bash
# 1. 克隆倉庫
git clone https://github.com/FlashFalconDev/infinitetalk-worker.git
cd infinitetalk-worker

# 2. 安裝 PyTorch (必須第一步!)
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. 安裝其他依賴
grep -v "flash_attn" requirements.txt > requirements_temp.txt
pip install -r requirements_temp.txt

# 4. 修復 transformers 版本 (關鍵!)
pip install 'transformers==4.45.2' --force-reinstall

# 5. 配置 .env
cp .env.example .env
nano .env  # 填入你的 token

# 6. 下載模型
./download_models_official.sh

# 7. 啟動 worker
python worker.py
```

**詳細說明請參考 SMOOTH_DEPLOYMENT_GUIDE.md**

---

## ⏱️ 預計時間

| 階段 | 時間 |
|------|------|
| 安裝依賴 | 10-15 分鐘 |
| 下載模型 | 2-4 小時 |
| 總計 | **2-4 小時** |

*主要取決於網速*

---

## ✅ 驗證安裝

啟動成功時,日誌應顯示:

```
✅ GPU 監控已啟用
✅ 連線成功
✅ wav2vec2 完成
✅ InfiniteTalk 完成
🎉 模型已常駐！
```

---

## 🐛 遇到問題？

1. 先查看 **SMOOTH_DEPLOYMENT_GUIDE.md** 的"常見錯誤處理"章節
2. 檢查 **DEPLOYMENT_ISSUES_LOG.md**
3. 提交 Issue: https://github.com/FlashFalconDev/infinitetalk-worker/issues

---

## 📞 支持

- **GitHub**: https://github.com/FlashFalconDev/infinitetalk-worker
- **官方倉庫**: https://github.com/MeiGen-AI/InfiniteTalk
- **Email**: support@flashfalcon.info

---

## 🙏 致謝

- **MeiGen-AI** - InfiniteTalk 官方項目
- **Wan-AI** - Wan 視頻生成模型
- 所有貢獻者和測試者

---

**最後更新**: 2025-10-30
**版本**: 7.3.3
**狀態**: ✅ 生產就緒
