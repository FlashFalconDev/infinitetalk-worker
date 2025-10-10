#!/bin/bash

echo "⛔ 停止 InfiniteTalk Worker"
echo ""

# 查找進程
PIDS=$(pgrep -f "python worker.py")

if [ -z "$PIDS" ]; then
    echo "⚠️  沒有找到運行中的 Worker"
    exit 0
fi

echo "找到 Worker 進程:"
ps aux | grep "python worker.py" | grep -v grep

echo ""
read -p "確定要停止嗎？(y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    pkill -f "python worker.py"
    sleep 2
    echo "✅ Worker 已停止"
else
    echo "❌ 取消操作"
fi
