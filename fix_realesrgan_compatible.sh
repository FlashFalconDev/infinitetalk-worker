#!/bin/bash
set -e

echo "🔧 修復 Real-ESRGAN（保持 InfiniteTalk 相容性）..."

# 停止 worker
echo "1️⃣ 停止 worker..."
pkill -9 python || true
sleep 2

# 記錄目前版本
echo "📊 目前版本:"
pip list | grep -E "torch|xformers|numpy"

# 降級到相容版本
echo ""
echo "2️⃣ 安裝相容版本..."
pip uninstall -y torch torchvision torchaudio xformers numpy

# 安裝正確的 PyTorch 版本（配合 xformers 0.0.28）
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# 重新安裝 xformers
pip install xformers==0.0.28

# 安裝相容的 numpy
pip install "numpy<2.0"

# 安裝 Real-ESRGAN
echo ""
echo "3️⃣ 安裝 Real-ESRGAN..."
pip install basicsr==1.4.2
pip install realesrgan==0.3.0
pip install facexlib gfpgan opencv-python-headless

# 驗證
echo ""
echo "4️⃣ 驗證安裝..."
python3 << 'PYEOF'
try:
    import torch
    import xformers
    import numpy as np
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    import cv2
    
    print("✅ 所有套件安裝成功")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   xformers: {xformers.__version__}")
    print(f"   numpy: {np.__version__}")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
except Exception as e:
    print(f"❌ 錯誤: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
PYEOF

# 檢查模型
echo ""
echo "5️⃣ 檢查模型檔案..."
mkdir -p weights
if [ ! -f "weights/realesr-animevideov3.pth" ]; then
    echo "下載 Real-ESRGAN 模型..."
    cd weights
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth
    cd ..
else
    echo "✅ 模型已存在"
fi

echo ""
echo "✅ 修復完成！"
echo ""
echo "📊 最終版本:"
pip list | grep -E "torch|xformers|numpy|realesrgan|basicsr"

echo ""
echo "🚀 現在可以啟動 worker:"
echo "   python worker.py"
