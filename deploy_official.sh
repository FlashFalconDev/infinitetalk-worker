#!/bin/bash
set -e

# ============================================================================
# InfiniteTalk Worker 官方完整部署脚本
# 基于 MeiGen-AI/InfiniteTalk 官方仓库
# 日期: 2025-10-30
# ============================================================================

echo "========================================================================"
echo "  InfiniteTalk Worker 官方部署脚本"
echo "========================================================================"
echo ""

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ============================================================================
# 第 1 步: 环境检查
# ============================================================================
echo -e "${YELLOW}[1/5] 环境检查...${NC}"
echo ""

# 检查 Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}错误: Python 未安装${NC}"
    exit 1
fi
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python: $PYTHON_VERSION"

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU 已检测到"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    echo -e "${YELLOW}⚠ 警告: NVIDIA GPU 未检测到${NC}"
fi

# 检查 huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${YELLOW}⚠ 安装 huggingface-hub...${NC}"
    pip install huggingface-hub
fi

# 检查磁盘空间 (需要至少 200GB)
AVAILABLE_GB=$(df -BG /workspace | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -lt 200 ]; then
    echo -e "${RED}警告: 磁盘空间不足 200GB，当前可用: ${AVAILABLE_GB}GB${NC}"
    read -p "继续？(y/N): " confirm
    if [ "$confirm" != "y" ]; then
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}✓ 环境检查完成${NC}"
echo ""

# ============================================================================
# 第 2 步: 安装依赖 (修复版本冲突)
# ============================================================================
echo -e "${YELLOW}[2/5] 安装依赖包...${NC}"
echo ""

# 2.1 先安装 PyTorch
echo "安装 PyTorch 2.4.1 + CUDA 12.1..."
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 2.2 安装其他依赖（排除 flash_attn）
if [ -f "requirements.txt" ]; then
    echo "安装其他依赖包..."
    grep -v "flash_attn" requirements.txt > /tmp/requirements_temp.txt
    pip install -r /tmp/requirements_temp.txt
    rm /tmp/requirements_temp.txt
fi

# 2.3 修复 transformers 版本冲突（关键步骤！）
echo "修复 transformers 版本..."
pip install 'transformers==4.45.2' --force-reinstall

echo ""
echo -e "${GREEN}✓ 依赖安装完成${NC}"
echo ""

# ============================================================================
# 第 3 步: 配置环境变量
# ============================================================================
echo -e "${YELLOW}[3/5] 配置环境...${NC}"
echo ""

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✓ 已创建 .env 文件"
        echo -e "${YELLOW}请编辑 .env 文件填入你的 INFINITETALK_WORKER_TOKEN${NC}"
        echo ""
        read -p "按回车继续..."
    else
        echo -e "${RED}错误: .env.example 未找到${NC}"
        exit 1
    fi
else
    echo "✓ .env 文件已存在"
fi

# 检查 token 是否配置
if grep -q "your_token_here" .env 2>/dev/null || ! grep -q "INFINITETALK_WORKER_TOKEN" .env 2>/dev/null; then
    echo -e "${YELLOW}⚠ 警告: Token 可能未配置${NC}"
    echo "请确认 .env 文件中的 INFINITETALK_WORKER_TOKEN 已设置"
    echo ""
fi

echo ""
echo -e "${GREEN}✓ 环境配置完成${NC}"
echo ""

# ============================================================================
# 第 4 步: 下载模型文件（官方来源）
# ============================================================================
echo -e "${YELLOW}[4/5] 下载模型文件 (官方来源)...${NC}"
echo ""
echo "这将下载约 200GB 的模型文件，需要 2-4 小时"
echo ""
read -p "是否现在下载？(y/N): " download_now

if [ "$download_now" == "y" ] || [ "$download_now" == "Y" ]; then
    mkdir -p /workspace/weights
    cd /workspace/weights

    echo ""
    echo -e "${BLUE}[1/5] 下载 chinese-wav2vec2-base (2.9GB)...${NC}"
    if [ ! -d "chinese-wav2vec2-base" ]; then
        huggingface-cli download TencentGameMate/chinese-wav2vec2-base \
            --local-dir ./chinese-wav2vec2-base

        # 下载 PR #1 的额外文件
        huggingface-cli download TencentGameMate/chinese-wav2vec2-base \
            model.safetensors --revision refs/pr/1 \
            --local-dir ./chinese-wav2vec2-base
        echo -e "${GREEN}✓ 完成${NC}"
    else
        echo -e "${GREEN}✓ 已存在，跳过${NC}"
    fi

    echo ""
    echo -e "${BLUE}[2/5] 下载 RealESRGAN (2.4MB)...${NC}"
    if [ ! -f "realesr-animevideov3.pth" ]; then
        wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth
        echo -e "${GREEN}✓ 完成${NC}"
    else
        echo -e "${GREEN}✓ 已存在，跳过${NC}"
    fi

    echo ""
    echo -e "${BLUE}[3/5] 下载 Wan2.1-I2V-14B-480P (~30GB)...${NC}"
    echo "这将需要一些时间，请耐心等待..."
    if [ ! -d "Wan2.1-I2V-14B-480P" ]; then
        # 使用正确的仓库名：Wan-AI/Wan2.1-I2V-14B-480P
        huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
            --local-dir ./Wan2.1-I2V-14B-480P
        echo -e "${GREEN}✓ 完成${NC}"
    else
        echo -e "${GREEN}✓ 已存在，跳过${NC}"
    fi

    echo ""
    echo -e "${BLUE}[4/5] 下载 InfiniteTalk (~160GB)...${NC}"
    echo -e "${RED}警告: 这是最大的模型，需要数小时下载！${NC}"
    if [ ! -d "InfiniteTalk" ]; then
        huggingface-cli download MeiGen-AI/InfiniteTalk \
            --local-dir ./InfiniteTalk
        echo -e "${GREEN}✓ 完成${NC}"
    else
        echo -e "${GREEN}✓ 已存在，跳过${NC}"
    fi

    echo ""
    echo -e "${BLUE}[5/5] 检查 LoRA 文件...${NC}"
    # LoRA 文件的官方链接已失效，需要手动处理
    if [ ! -f "Wan2.1_I2V_14B_FusionX_LoRA.safetensors" ]; then
        echo -e "${YELLOW}⚠ LoRA 文件未找到${NC}"
        echo "官方链接已失效，worker 代码需要修改使 LoRA 可选"
        echo "见 DEPLOYMENT_GUIDE_FIXED.md 第 5 步"
    else
        echo -e "${GREEN}✓ LoRA 文件存在${NC}"
    fi

    cd /workspace/infinitetalk-worker

    echo ""
    echo "模型下载结果:"
    ls -lh /workspace/weights/
    echo ""
    echo "磁盘使用:"
    du -sh /workspace/weights/* 2>/dev/null

else
    echo -e "${YELLOW}跳过模型下载${NC}"
    echo "稍后可以运行: ./download_models_official.sh"
fi

echo ""
echo -e "${GREEN}✓ 模型配置完成${NC}"
echo ""

# ============================================================================
# 第 5 步: 验证安装
# ============================================================================
echo -e "${YELLOW}[5/5] 验证安装...${NC}"
echo ""

# 验证关键包
echo "验证核心包版本:"
python -c "
import sys
try:
    import torch
    print(f'  ✓ torch:        {torch.__version__}')
except ImportError:
    print('  ✗ torch: 未安装')
    sys.exit(1)

try:
    import transformers
    print(f'  ✓ transformers: {transformers.__version__}')
except ImportError:
    print('  ✗ transformers: 未安装')
    sys.exit(1)

try:
    import diffusers
    print(f'  ✓ diffusers:    {diffusers.__version__}')
except ImportError:
    print('  ✗ diffusers: 未安装')
    sys.exit(1)

try:
    import dotenv
    print(f'  ✓ python-dotenv: 已安装')
except ImportError:
    print('  ✗ python-dotenv: 未安装')
"

echo ""
echo -e "${GREEN}✓ 验证完成${NC}"
echo ""

# ============================================================================
# 完成
# ============================================================================
echo "========================================================================"
echo -e "${GREEN}  部署完成！${NC}"
echo "========================================================================"
echo ""
echo "下一步操作:"
echo ""
echo "  1. 确认 .env 配置:"
echo "     nano .env"
echo ""
echo "  2. (可选) 修改代码使 LoRA 可选:"
echo "     见 DEPLOYMENT_GUIDE_FIXED.md 第 5 步"
echo ""
echo "  3. 启动 worker:"
echo "     nohup python worker.py > worker.log 2>&1 &"
echo ""
echo "  4. 查看日志:"
echo "     tail -f worker.log"
echo ""
echo "  5. 验证运行:"
echo "     ps aux | grep worker.py"
echo ""
echo "详细文档:"
echo "  • DEPLOYMENT_GUIDE_FIXED.md - 完整部署指南"
echo "  • STATUS_AND_NEXT_STEPS.md - 当前状态和问题"
echo "  • DEPLOYMENT_ISSUES_LOG.md - 问题记录"
echo ""
echo "获取帮助:"
echo "  • GitHub: https://github.com/FlashFalconDev/infinitetalk-worker"
echo "  • 官方: https://github.com/MeiGen-AI/InfiniteTalk"
echo ""
