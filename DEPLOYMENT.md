# InfiniteTalk Worker 部署指南

## 快速部署（新機器）
```bash
# 1. 克隆專案
git clone https://github.com/FlashFalconDev/infinitetalk-worker.git
cd infinitetalk-worker

# 2. 設定環境變數
cp .env.example .env
nano .env  # 編輯填入 API_BASE 和 TOKEN

# 3. 執行部署
bash DEPLOY.sh

# 4. 啟動 Worker
nohup python worker.py > worker.log 2>&1 &
tail -f worker.log
```

## 環境需求

- Python: $(cat PYTHON_VERSION.txt)
- CUDA: 12.1+
- GPU: NVIDIA (支援 CUDA)

## 疑難排解

### 問題：依賴安裝失敗
解決：確認 Python 版本正確

### 問題：模型載入失敗
解決：檢查 /workspace/weights 目錄是否存在

### 問題：無法連線 API
解決：檢查 .env 中的 API_BASE 和 TOKEN 是否正確
