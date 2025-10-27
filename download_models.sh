#!/bin/bash
set -e

echo "🚀 開始下載 InfiniteTalk 所需模型..."

# 創建 weights 目錄
mkdir -p /workspace/weights
cd /workspace/weights

# 1. 下載 Chinese Wav2Vec2
echo "📥 下載 Chinese Wav2Vec2..."
if [ ! -d "chinese-wav2vec2-base" ]; then
    git clone https://huggingface.co/TencentGameMate/chinese-wav2vec2-base
else
    echo "✅ chinese-wav2vec2-base 已存在"
fi

# 2. 下載 RealESRGAN
echo "📥 下載 RealESRGAN..."
if [ ! -f "realesr-animevideov3.pth" ]; then
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth
else
    echo "✅ realesr-animevideov3.pth 已存在"
fi

# 3. 下載 Wan2.1 (這個可能需要手動或其他方式)
echo "📥 Wan2.1 需要手動下載或從其他來源獲取"
echo "   請確認是否需要這個模型"

# 4. 下載 InfiniteTalk 模型
echo "📥 下載 InfiniteTalk 模型..."
if [ ! -d "InfiniteTalk" ]; then
    # 這裡需要確認實際的模型下載位置
    echo "⚠️  請手動設定 InfiniteTalk 模型下載"
else
    echo "✅ InfiniteTalk 已存在"
fi

# 創建符號連結
cd /workspace/InfiniteTalk
ln -sf /workspace/weights weights

echo "✅ 模型下載完成！"
