#!/bin/bash

echo "📊 InfiniteTalk Worker 狀態"
echo "============================"
echo ""

# 檢查進程
PIDS=$(pgrep -f "python worker.py")

if [ -z "$PIDS" ]; then
    echo "❌ Worker 未運行"
    echo ""
    echo "啟動命令: ./start_worker.sh"
    exit 0
fi

echo "✅ Worker 運行中"
echo ""

# 顯示進程資訊
echo "進程資訊:"
ps aux | grep "python worker.py" | grep -v grep | awk '{printf "  PID: %s\n  CPU: %s%%\n  MEM: %s%%\n  運行時間: %s\n", $2, $3, $4, $10}'

echo ""

# 顯示最近日誌
if [ -f worker.log ]; then
    echo "最近日誌 (最後 20 行):"
    echo "----------------------------------------"
    tail -20 worker.log
    echo "----------------------------------------"
    echo ""
    echo "完整日誌: tail -f worker.log"
fi

echo ""
echo "停止命令: ./stop_worker.sh"
