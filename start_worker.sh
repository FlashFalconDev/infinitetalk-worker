#!/bin/bash

# InfiniteTalk Worker 啟動腳本

echo "🚀 啟動 InfiniteTalk Worker"
echo "=" | head -c 50
echo ""

# 檢查 .env 檔案
if [ ! -f .env ]; then
    echo "❌ 找不到 .env 檔案"
    echo "請執行: cp .env.example .env 並填入配置"
    exit 1
fi

# 載入環境變數
set -a
source .env
set +a

# 檢查 Token
if [ -z "$INFINITETALK_WORKER_TOKEN" ]; then
    echo "❌ 未設定 INFINITETALK_WORKER_TOKEN"
    echo "請在 .env 中填入從 Admin 後台複製的 Token"
    exit 1
fi

# 檢查 Python 環境
if [ ! -d "infinitetalk-env" ]; then
    echo "❌ 找不到 Python 環境: infinitetalk-env"
    exit 1
fi

# 啟動
echo "✅ 配置檢查通過"
echo "🔄 啟動 Worker..."
echo ""

source infinitetalk-env/bin/activate
python worker.py
