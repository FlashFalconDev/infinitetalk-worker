# InfiniteTalk Worker 部署指南

## 🚀 快速部署（新機器 3 步驟）
```bash
# 1. 克隆專案
git clone https://github.com/FlashFalconDev/infinitetalk-worker.git
cd infinitetalk-worker

# 2. 設定環境變數
cp .env.example .env
nano .env  # 編輯填入你的 API_BASE 和 TOKEN

# 3. 執行部署
bash DEPLOY.sh
```

## 🎬 啟動 Worker
```bash
nohup python worker.py > worker.log 2>&1 &
tail -f worker.log
```

## 📋 環境需求

- **Python**: 見 PYTHON_VERSION.txt
- **CUDA**: 12.1+
- **GPU**: NVIDIA (支援 CUDA)
- **磁碟**: 至少 50GB（用於模型）

## 🔧 環境變數說明

編輯 `.env` 檔案：
```bash
# API 基礎網址
INFINITETALK_API_BASE=https://host.flashfalcon.info

# Worker 認證 Token
INFINITETALK_WORKER_TOKEN=your_actual_token_here
```

## 📊 監控與管理
```bash
# 查看 Worker 狀態
ps aux | grep worker.py

# 查看即時日誌
tail -f worker.log

# 停止 Worker
pkill -f worker.py

# 重啟 Worker
pkill -f worker.py && nohup python worker.py > worker.log 2>&1 &
```

## 🐛 疑難排解

### 問題：依賴安裝失敗
**解決**：確認 Python 版本與 PYTHON_VERSION.txt 一致

### 問題：模型載入失敗
**解決**：
```bash
# 檢查 weights 目錄
ls -la /workspace/weights/

# 執行模型下載
bash download_models.sh
```

### 問題：無法連線 API
**解決**：檢查 .env 中的配置是否正確

### 問題：Worker 啟動後立即退出
**解決**：查看日誌找出原因
```bash
cat worker.log
```

## 📦 更新部署
```bash
cd /workspace/InfiniteTalk
git pull
bash DEPLOY.sh
pkill -f worker.py
nohup python worker.py > worker.log 2>&1 &
```

## 🔄 完整重置
```bash
cd /workspace
rm -rf InfiniteTalk
# 然後重新執行快速部署步驟
```
