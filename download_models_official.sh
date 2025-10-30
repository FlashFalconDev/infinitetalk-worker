#!/bin/bash
set -e

# ============================================================================
# InfiniteTalk 官方模型下载脚本
# 基于 MeiGen-AI/InfiniteTalk 官方仓库
# ============================================================================

echo "========================================================================"
echo "  InfiniteTalk 官方模型下载"
echo "========================================================================"
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 创建 weights 目录
WEIGHTS_DIR="/workspace/weights"
mkdir -p "$WEIGHTS_DIR"
cd "$WEIGHTS_DIR"

echo -e "${BLUE}模型将下载到: $WEIGHTS_DIR${NC}"
echo ""

# 显示磁盘空间
echo "当前磁盘空间:"
df -h "$WEIGHTS_DIR" | tail -1
echo ""

# 检查 huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${RED}错误: huggingface-cli 未安装${NC}"
    echo "安装方法: pip install huggingface-hub"
    exit 1
fi

echo "将下载以下模型（官方来源）:"
echo ""
echo "  [1] chinese-wav2vec2-base      (2.9 GB)  - TencentGameMate"
echo "  [2] realesr-animevideov3.pth   (2.4 MB)  - GitHub Release"
echo "  [3] Wan2.1-I2V-14B-480P        (~30 GB)  - Wan-AI"
echo "  [4] InfiniteTalk               (~160 GB) - MeiGen-AI"
echo ""
echo "  总计: ~200 GB"
echo ""
read -p "确认下载所有模型？(y/N): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "取消下载"
    exit 0
fi

# ============================================================================
# 1. Chinese Wav2Vec2
# ============================================================================
echo ""
echo -e "${BLUE}[1/4] 下载 chinese-wav2vec2-base (2.9 GB)...${NC}"
if [ -d "chinese-wav2vec2-base" ]; then
    echo -e "${YELLOW}目录已存在，跳过${NC}"
else
    echo "从 TencentGameMate/chinese-wav2vec2-base 下载..."
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base \
        --local-dir ./chinese-wav2vec2-base

    echo "下载 PR #1 的额外文件..."
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base \
        model.safetensors --revision refs/pr/1 \
        --local-dir ./chinese-wav2vec2-base

    echo -e "${GREEN}✓ chinese-wav2vec2-base 下载完成${NC}"
fi

# ============================================================================
# 2. RealESRGAN
# ============================================================================
echo ""
echo -e "${BLUE}[2/4] 下载 RealESRGAN (2.4 MB)...${NC}"
if [ -f "realesr-animevideov3.pth" ]; then
    echo -e "${YELLOW}文件已存在，跳过${NC}"
else
    echo "从 GitHub Release 下载..."
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth
    echo -e "${GREEN}✓ RealESRGAN 下载完成${NC}"
fi

# ============================================================================
# 3. Wan2.1-I2V-14B-480P
# ============================================================================
echo ""
echo -e "${BLUE}[3/4] 下载 Wan2.1-I2V-14B-480P (~30 GB)...${NC}"
if [ -d "Wan2.1-I2V-14B-480P" ]; then
    echo -e "${YELLOW}目录已存在，跳过${NC}"
else
    echo "从 Wan-AI/Wan2.1-I2V-14B-480P 下载..."
    echo "这将需要一些时间，请耐心等待..."

    # ⚠️ 重要：使用正确的仓库名 Wan-AI，不是 MeiGen-AI
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
        --local-dir ./Wan2.1-I2V-14B-480P

    echo -e "${GREEN}✓ Wan2.1-I2V-14B-480P 下载完成${NC}"
fi

# ============================================================================
# 4. InfiniteTalk
# ============================================================================
echo ""
echo -e "${BLUE}[4/4] 下载 InfiniteTalk (~160 GB)...${NC}"
if [ -d "InfiniteTalk" ]; then
    echo -e "${YELLOW}目录已存在，跳过${NC}"
else
    echo "从 MeiGen-AI/InfiniteTalk 下载..."
    echo -e "${RED}这是最大的模型，需要数小时下载！${NC}"
    echo ""

    huggingface-cli download MeiGen-AI/InfiniteTalk \
        --local-dir ./InfiniteTalk

    echo -e "${GREEN}✓ InfiniteTalk 下载完成${NC}"
fi

# ============================================================================
# 完成
# ============================================================================
echo ""
echo "========================================================================"
echo -e "${GREEN}  模型下载完成！${NC}"
echo "========================================================================"
echo ""
echo "已下载的模型:"
ls -lh "$WEIGHTS_DIR"
echo ""
echo "磁盘使用情况:"
du -sh "$WEIGHTS_DIR"/* 2>/dev/null
echo ""

# ============================================================================
# 5. LoRA 文件（必须）
# ============================================================================
echo ""
echo -e "${BLUE}[5/5] 下载 LoRA 文件 (354 MB)...${NC}"
if [ -f "$WEIGHTS_DIR/Wan2.1_I2V_14B_FusionX_LoRA.safetensors" ]; then
    echo -e "${GREEN}✓ LoRA 文件已存在${NC}"
else
    echo "从 vrgamedevgirl84/Wan14BT2VFusioniX 下载..."
    huggingface-cli download vrgamedevgirl84/Wan14BT2VFusioniX \
        FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors \
        --local-dir "$WEIGHTS_DIR" --local-dir-use-symlinks False

    # 移动到正确位置
    if [ -f "$WEIGHTS_DIR/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors" ]; then
        mv "$WEIGHTS_DIR/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors" "$WEIGHTS_DIR/"
        rm -rf "$WEIGHTS_DIR/FusionX_LoRa"
        echo -e "${GREEN}✓ LoRA 文件下载完成${NC}"
    else
        echo -e "${RED}✗ LoRA 文件下载失败${NC}"
    fi
fi

echo "下一步:"
echo "  1. 确认 .env 文件配置正确"
echo "  2. 启动 worker: nohup python worker.py > worker.log 2>&1 &"
echo "  3. 查看日志: tail -f worker.log"
echo ""
