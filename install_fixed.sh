#!/bin/bash
set -e

# InfiniteTalk Worker 修复版安装脚本
# 基于实际部署经验创建
# 日期: 2025-10-30

echo "=========================================="
echo "InfiniteTalk Worker 安装脚本 (修复版)"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 步骤 1: 检查环境
echo -e "${YELLOW}[1/5] 检查环境...${NC}"
python --version || { echo -e "${RED}错误: Python 未安装${NC}"; exit 1; }
nvidia-smi > /dev/null 2>&1 || { echo -e "${RED}警告: NVIDIA GPU 未检测到${NC}"; }
echo -e "${GREEN}✓ 环境检查完成${NC}"
echo ""

# 步骤 2: 安装 PyTorch（必须先安装！）
echo -e "${YELLOW}[2/5] 安装 PyTorch 2.4.1 + CUDA 12.1...${NC}"
echo "这一步很重要，因为 flash_attn 依赖 torch"
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 \
    || { echo -e "${RED}错误: PyTorch 安装失败${NC}"; exit 1; }
echo -e "${GREEN}✓ PyTorch 安装完成${NC}"
echo ""

# 步骤 3: 安装其他依赖（排除 flash_attn）
echo -e "${YELLOW}[3/5] 安装其他依赖...${NC}"
if [ -f "requirements.txt" ]; then
    echo "创建临时 requirements 文件（排除 flash_attn）"
    grep -v "flash_attn" requirements.txt > requirements_temp.txt

    echo "安装依赖包（这可能需要几分钟）..."
    pip install -r requirements_temp.txt

    rm requirements_temp.txt
    echo -e "${GREEN}✓ 依赖安装完成${NC}"
else
    echo -e "${RED}错误: requirements.txt 未找到${NC}"
    exit 1
fi
echo ""

# 步骤 4: 修复 transformers 版本（关键步骤！）
echo -e "${YELLOW}[4/5] 修复 transformers 版本冲突...${NC}"
echo "降级 transformers 到 4.45.2（兼容 torch 2.4.1）"
pip install 'transformers==4.45.2' --force-reinstall
echo -e "${GREEN}✓ Transformers 版本修复完成${NC}"
echo ""

# 步骤 5: 验证安装
echo -e "${YELLOW}[5/5] 验证安装...${NC}"
python -c "import torch; print(f'torch: {torch.__version__}')" || { echo -e "${RED}错误: torch 导入失败${NC}"; exit 1; }
python -c "import transformers; print(f'transformers: {transformers.__version__}')" || { echo -e "${RED}错误: transformers 导入失败${NC}"; exit 1; }
python -c "import diffusers; print(f'diffusers: {diffusers.__version__}')" || { echo -e "${RED}错误: diffusers 导入失败${NC}"; exit 1; }
echo -e "${GREEN}✓ 所有核心包验证通过${NC}"
echo ""

# 显示最终状态
echo "=========================================="
echo -e "${GREEN}安装完成！${NC}"
echo "=========================================="
echo ""
echo "已安装的关键版本:"
python -c "
import torch, transformers, diffusers
print(f'  • torch:        {torch.__version__}')
print(f'  • transformers: {transformers.__version__}')
print(f'  • diffusers:    {diffusers.__version__}')
"
echo ""
echo "下一步:"
echo "  1. 配置环境变量:"
echo "     cp .env.example .env"
echo "     nano .env"
echo ""
echo "  2. 下载模型文件（见 DEPLOYMENT_GUIDE_FIXED.md）"
echo ""
echo "  3. 启动 worker:"
echo "     nohup python worker.py > worker.log 2>&1 &"
echo ""
echo "详细文档:"
echo "  • DEPLOYMENT_GUIDE_FIXED.md - 完整部署指南"
echo "  • DEPLOYMENT_ISSUES_LOG.md - 问题记录和解决方案"
echo ""
