#!/bin/bash
set -e

echo "🚀 InfiniteTalk Worker 標準部署流程"
echo "===================================="
echo ""

# 檢查 Python 版本
REQUIRED_PYTHON=$(cat PYTHON_VERSION.txt 2>/dev/null || echo "Unknown")
CURRENT_PYTHON=$(python --version 2>&1)
echo "📌 Python 版本: $CURRENT_PYTHON"
echo "   要求版本: $REQUIRED_PYTHON"
echo ""

# 1. 安裝確切版本的依賴
echo "📦 [1/4] 安裝依賴套件..."
pip install -r requirements_exact_working.txt

# 2. 設定環境
echo "🔧 [2/4] 設定環境..."
mkdir -p /workspace/weights
rm -f weights
ln -sf /workspace/weights weights

# 3. 驗證安裝
echo "✅ [3/4] 驗證安裝..."
python -c "
import sys
sys.path.insert(0, '/workspace/InfiniteTalk')
try:
    import worker
    print('✅ Worker 模組導入成功')
except Exception as e:
    print(f'❌ 失敗: {e}')
    sys.exit(1)
"

# 4. 檢查環境變數
echo "⚙️  [4/4] 檢查環境變數..."
if [ ! -f .env ]; then
    echo "⚠️  .env 不存在，請先設定："
    echo ""
    echo "   cp .env.example .env"
    echo "   nano .env  # 編輯填入正確的 API_BASE 和 TOKEN"
    echo ""
    exit 1
fi

echo ""
echo "✅ 部署完成！"
echo ""
echo "🚀 啟動指令:"
echo "   nohup python worker.py > worker.log 2>&1 &"
echo "   tail -f worker.log"
echo ""
echo "📊 監控指令:"
echo "   ps aux | grep worker.py"
echo "   tail -f worker.log"
