#!/bin/bash

# InfiniteTalk Worker 啟動腳本（非 Docker 版本）

set -e

cd /workspace/InfiniteTalk

echo "🚀 啟動 InfiniteTalk Worker"
echo "============================"
echo ""

# 檢查環境
if [ ! -f .env ]; then
    echo "❌ 找不到 .env 文件"
    echo "   請複製: cp .env.example .env"
    echo "   然後編輯填入 Token"
    exit 1
fi

# 載入環境變數
set -a
source .env
set +a

if [ -z "$INFINITETALK_WORKER_TOKEN" ]; then
    echo "❌ 未設定 INFINITETALK_WORKER_TOKEN"
    exit 1
fi

# ✅ 檢查虛擬環境（支援多種名稱）
VENV_DIR=""
for dir in infinitetalk-env venv env .venv; do
    if [ -d "$dir" ]; then
        VENV_DIR="$dir"
        break
    fi
done

# ✅ 如果沒有虛擬環境，直接使用系統 Python
if [ -n "$VENV_DIR" ]; then
    echo "✅ 使用虛擬環境: $VENV_DIR"
    source $VENV_DIR/bin/activate
else
    echo "⚠️  沒有找到虛擬環境，使用系統 Python"
fi

echo "✅ 環境檢查通過"
echo "🔄 啟動 Worker..."
echo ""

# 啟動
nohup python worker.py > worker.log 2>&1 &

WORKER_PID=$!
sleep 2

# 檢查是否啟動成功
if ps -p $WORKER_PID > /dev/null; then
    echo "✅ Worker 已啟動"
    echo "   PID: $WORKER_PID"
    echo "   日誌: tail -f worker.log"
    echo ""
    echo "停止命令: ./stop_worker.sh"
    echo "查看狀態: ./status_worker.sh"
else
    echo "❌ Worker 啟動失敗"
    echo "   查看日誌: cat worker.log"
    exit 1
fi
