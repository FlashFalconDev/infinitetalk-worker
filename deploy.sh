#!/bin/bash

# InfiniteTalk Worker 部署腳本

set -e

echo "🚀 InfiniteTalk Worker 部署工具"
echo "================================"
echo ""

# 檢查 Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安裝"
    echo "   安裝方式: https://docs.docker.com/engine/install/"
    exit 1
fi

# 檢查 docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose 未安裝"
    echo "   安裝方式: sudo apt-get install docker-compose"
    exit 1
fi

# 檢查 NVIDIA Docker Runtime
if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA Docker Runtime 未配置"
    echo "   安裝方式: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

echo "✅ 環境檢查通過"
echo ""

# 檢查 .env 文件
if [ ! -f .env ]; then
    echo "⚠️  找不到 .env 文件"
    echo ""
    echo "正在創建 .env..."
    cp .env.docker.example .env
    echo "✅ .env 已創建"
    echo ""
    echo "📝 請編輯 .env 並填入 Token:"
    echo "   nano .env"
    echo ""
    echo "然後重新執行: ./deploy.sh"
    exit 1
fi

# 檢查 Token
if grep -q "your_token_from_admin_here" .env; then
    echo "❌ 請先在 .env 中設定 INFINITETALK_WORKER_TOKEN"
    echo "   編輯: nano .env"
    exit 1
fi

echo "📦 開始構建 Docker 映像..."
docker-compose build

echo ""
echo "✅ 構建完成"
echo ""
echo "🚀 啟動選項:"
echo "1. 啟動單個 Worker: docker-compose up -d"
echo "2. 啟動多個 Worker: docker-compose --profile multi-gpu up -d"
echo "3. 查看日誌: docker-compose logs -f"
echo "4. 停止: docker-compose down"
echo ""

read -p "現在要啟動嗎？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 啟動 Worker..."
    docker-compose up -d
    
    echo ""
    echo "✅ Worker 已啟動"
    echo ""
    echo "📊 查看狀態:"
    docker-compose ps
    
    echo ""
    echo "📋 實用命令:"
    echo "  查看日誌: docker-compose logs -f"
    echo "  重啟: docker-compose restart"
    echo "  停止: docker-compose down"
    echo "  進入容器: docker-compose exec infinitetalk-worker bash"
fi
