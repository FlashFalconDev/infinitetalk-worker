# LoRA 文件下載指南

**重要**: LoRA 文件是 InfiniteTalk Worker 必須的組件,不能省略！

---

## 📥 下載方式

### 正確的倉庫位置

❌ **錯誤** (主目錄不存在):
```
https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX/resolve/main/Wan2.1_I2V_14B_FusionX_LoRA.safetensors
```

✅ **正確** (在 FusionX_LoRa 子目錄):
```
https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX/blob/main/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors
```

---

## 🚀 下載命令

### 方式 A: 使用 huggingface-cli (推薦)

```bash
cd /workspace/weights

# 下載文件 (會下載到 FusionX_LoRa 子目錄)
huggingface-cli download vrgamedevgirl84/Wan14BT2VFusioniX \
    FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors \
    --local-dir . \
    --local-dir-use-symlinks False

# 移動到正確位置
mv FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors .
rm -rf FusionX_LoRa

# 驗證
ls -lh Wan2.1_I2V_14B_FusionX_LoRA.safetensors
# 應顯示: -rw-rw-r-- 1 root root 354M ...
```

### 方式 B: 使用 Python

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="vrgamedevgirl84/Wan14BT2VFusioniX",
    filename="FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors",
    local_dir="/workspace/weights",
    local_dir_use_symlinks=False
)
```

---

## ℹ️ 文件信息

| 項目 | 值 |
|------|-----|
| **文件名** | Wan2.1_I2V_14B_FusionX_LoRA.safetensors |
| **大小** | 354 MB (371,093,736 bytes) |
| **倉庫** | vrgamedevgirl84/Wan14BT2VFusioniX |
| **路徑** | FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors |
| **格式** | SafeTensors |
| **用途** | FusionX LoRA 權重,用於增強 Wan 模型品質 |

---

## 🔍 為什麼 LoRA 是必須的？

根據官方代碼 (`model_service.py`):

```python
self.wan_i2v = wan.InfiniteTalkPipeline(
    config=cfg,
    checkpoint_dir=self.ckpt_dir,
    device_id=0,
    rank=0,
    lora_dir=[self.lora_dir],      # 必須參數
    lora_scales=[1.0],              # 必須參數
    infinitetalk_dir=self.infinitetalk_dir
)
```

LoRA 文件用於:
- 提高視頻生成品質
- 減少色彩偏移
- 優化長視頻生成效果
- 配合 InfiniteTalk 音頻同步

---

## ✅ 驗證安裝

```bash
# 檢查文件是否存在
ls -lh /workspace/weights/Wan2.1_I2V_14B_FusionX_LoRA.safetensors

# 檢查文件大小
du -h /workspace/weights/Wan2.1_I2V_14B_FusionX_LoRA.safetensors
# 應顯示: 354M

# 檢查文件完整性 (可選)
sha256sum /workspace/weights/Wan2.1_I2V_14B_FusionX_LoRA.safetensors
```

---

## 🐛 常見問題

### Q: 為什麼之前的文檔說 LoRA 可選？
A: 那是錯誤的信息。根據實際代碼和用戶要求,LoRA 是必須的組件。

### Q: 能否使用其他來源的 LoRA？
A: 可以,但需確保:
- 文件名為 `Wan2.1_I2V_14B_FusionX_LoRA.safetensors`
- 放在 `/workspace/weights/` 目錄
- 大小約 354MB
- 格式為 SafeTensors

### Q: LoRA 下載失敗怎麼辦？
A: 嘗試以下方案:
1. 檢查網絡連接
2. 使用 HuggingFace Token (如果需要)
3. 嘗試從 mirror 站點下載
4. 檢查磁盤空間是否足夠

---

## 📝 部署腳本集成

LoRA 下載已集成到:
- `download_models_official.sh` - 第 5 步自動下載
- `deploy_official.sh` - 一鍵部署時自動處理

如果使用自動腳本,LoRA 會自動下載和配置。

---

**創建日期**: 2025-10-30
**驗證狀態**: ✅ 已在實際環境測試
**文件大小**: 354 MB
**下載時間**: 約 1-2 分鐘
