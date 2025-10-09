#!/bin/bash
set -e

echo "🔧 強制修復 Real-ESRGAN（鎖定所有版本）..."

# 停止 worker
echo "1️⃣ 停止 worker..."
pkill -9 python || true
sleep 2

# 完全卸載相關套件
echo "2️⃣ 卸載所有相關套件..."
pip uninstall -y torch torchvision torchaudio xformers numpy basicsr realesrgan facexlib gfpgan opencv-python opencv-python-headless || true

# 按順序安裝（避免依賴衝突）
echo ""
echo "3️⃣ 安裝 PyTorch 組件..."
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --no-deps
pip install xformers==0.0.28

# 強制安裝 numpy 1.x 並防止升級
echo ""
echo "4️⃣ 鎖定 numpy 版本..."
pip install "numpy>=1.20,<2.0"

# 安裝其他依賴（但不自動升級 numpy）
echo ""
echo "5️⃣ 安裝其他依賴..."
pip install --no-deps basicsr==1.4.2
pip install --no-deps realesrgan==0.3.0

# 手動安裝 basicsr 和 realesrgan 的依賴（排除會升級 numpy 的）
pip install addict future lmdb pyyaml requests scikit-image scipy tb-nightly tqdm yapf
pip install opencv-python-headless
pip install facexlib gfpgan --no-deps

# 補充缺失的依賴
pip install Pillow

# 最後再次確保 numpy 是 1.x
pip install --force-reinstall "numpy>=1.20,<2.0"

echo ""
echo "6️⃣ 驗證安裝..."
python3 << 'PYEOF'
import sys

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    
    import torchvision
    print(f"✅ torchvision: {torchvision.__version__}")
    
    import xformers
    print(f"✅ xformers: {xformers.__version__}")
    
    import numpy as np
    print(f"✅ numpy: {np.__version__}")
    if np.__version__.startswith('2.'):
        print("❌ 警告: numpy 版本仍是 2.x")
        sys.exit(1)
    
    # 測試 torchvision.transforms.functional_tensor
    try:
        from torchvision.transforms import functional as F
        print(f"✅ torchvision.transforms 正常")
    except ImportError as e:
        print(f"⚠️  torchvision.transforms 問題: {e}")
    
    from realesrgan import RealESRGANer
    print(f"✅ Real-ESRGAN 可用")
    
    from basicsr.archs.rrdbnet_arch import RRDBNet
    print(f"✅ BasicSR 可用")
    
    import cv2
    print(f"✅ OpenCV 可用")
    
    print(f"✅ CUDA 可用: {torch.cuda.is_available()}")
    
    print("\n✅ 所有套件安裝成功且相容")
    
except Exception as e:
    print(f"❌ 錯誤: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF

if [ $? -eq 0 ]; then
    echo ""
    echo "7️⃣ 檢查模型檔案..."
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
    pip list | grep -E "torch|xformers|numpy|realesrgan|basicsr" | head -10
    
    echo ""
    echo "🚀 現在可以啟動 worker:"
    echo "   python worker.py"
else
    echo ""
    echo "❌ 驗證失敗，請檢查錯誤訊息"
    exit 1
fi
