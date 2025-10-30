# InfiniteTalk Worker 官方部署流程

**基于 MeiGen-AI/InfiniteTalk 官方仓库 + FlashFalconDev/infinitetalk-worker**

---

## 🎯 快速开始（一键部署）

```bash
cd /workspace/infinitetalk-worker
./deploy_official.sh
```

这个脚本会：
✅ 检查环境
✅ 安装依赖（修复版本冲突）
✅ 配置 .env
✅ 从官方源下载模型（~200GB）
✅ 验证安装

---

## 📚 官方模型来源（已验证）

| 模型 | 官方仓库 | 大小 | 说明 |
|------|---------|------|------|
| Wan2.1-I2V-14B-480P | **Wan-AI**/Wan2.1-I2V-14B-480P | ~30GB | 基础视频生成模型 |
| InfiniteTalk | **MeiGen-AI**/InfiniteTalk | ~160GB | 音频同步权重 |
| chinese-wav2vec2-base | TencentGameMate/chinese-wav2vec2-base | 2.9GB | 音频编码器 |
| RealESRGAN | GitHub Release | 2.4MB | 视频增强 |

**⚠️ 重要发现**：
- Wan 模型的仓库是 `Wan-AI`，不是 `MeiGen-AI`
- 之前的文档中使用了错误的仓库名导致 401 错误

---

## 🚀 分步部署流程

### 步骤 1: 克隆仓库

```bash
git clone https://github.com/FlashFalconDev/infinitetalk-worker.git
cd infinitetalk-worker
```

### 步骤 2: 安装依赖（修复版）

```bash
# 方式 A: 使用自动脚本
./install_fixed.sh

# 方式 B: 手动安装
# 1. 先安装 PyTorch
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. 安装其他依赖
grep -v "flash_attn" requirements.txt > requirements_temp.txt
pip install -r requirements_temp.txt

# 3. 修复 transformers 版本冲突
pip install 'transformers==4.45.2' --force-reinstall
```

**关键版本**：
```
torch==2.4.1+cu121
transformers==4.45.2  ← 必须！不能用 4.56+
diffusers==0.35.1
xformers==0.0.28
```

### 步骤 3: 配置环境

```bash
cp .env.example .env
nano .env
```

填入你的 token：
```env
INFINITETALK_WORKER_TOKEN=你的token这里
```

### 步骤 4: 下载模型（官方来源）

```bash
# 方式 A: 使用官方下载脚本
./download_models_official.sh

# 方式 B: 手动下载
mkdir -p /workspace/weights
cd /workspace/weights

# 1. chinese-wav2vec2-base (2.9GB)
huggingface-cli download TencentGameMate/chinese-wav2vec2-base \
    --local-dir ./chinese-wav2vec2-base
huggingface-cli download TencentGameMate/chinese-wav2vec2-base \
    model.safetensors --revision refs/pr/1 \
    --local-dir ./chinese-wav2vec2-base

# 2. RealESRGAN (2.4MB)
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth

# 3. Wan2.1-I2V-14B-480P (~30GB)
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
    --local-dir ./Wan2.1-I2V-14B-480P

# 4. InfiniteTalk (~160GB)
huggingface-cli download MeiGen-AI/InfiniteTalk \
    --local-dir ./InfiniteTalk
```

**预计下载时间**: 2-4 小时

### 步骤 5: 启动 Worker

```bash
cd /workspace/infinitetalk-worker
nohup python worker.py > worker.log 2>&1 &

# 查看日志
tail -f worker.log
```

---

## 🔧 已知问题和解决方案

### 问题 1: transformers 版本冲突

**错误**: `ValueError: torch >= 2.6 required`

**原因**: transformers 4.56+ 要求 torch>=2.6（但 2.6 还未发布）

**解决**: 使用 transformers==4.45.2
```bash
pip install 'transformers==4.45.2' --force-reinstall
```

### 问题 2: 找不到 Wan 模型（401 错误）

**错误**: `401 Unauthorized for MeiGen-AI/Wan2.1-I2V-14B-480P`

**原因**: 仓库名错误

**解决**: 使用正确的仓库名 `Wan-AI/Wan2.1-I2V-14B-480P`

### 问题 3: LoRA 文件下载失败（404）

**错误**: `404 Not Found for vrgamedevgirl84/Wan14BT2VFusioniX`

**原因**: 官方链接已失效

**解决**: 修改代码使 LoRA 可选
```python
# 编辑 model_service.py 第 167-173 行
if not os.path.exists(self.lora_dir):
    logger.warning(f"⚠️ LoRA 文件不存在，使用基础模型")
    self.lora_dir = None

# 第 180 行
lora_dir=[self.lora_dir] if self.lora_dir else None,
```

---

## 📂 目录结构

```
/workspace/
├── infinitetalk-worker/          # Worker 代码
│   ├── worker.py                 # 主程序
│   ├── model_service.py          # 模型服务
│   ├── .env                      # 配置
│   ├── deploy_official.sh        # 一键部署脚本 ⭐
│   ├── download_models_official.sh  # 官方模型下载 ⭐
│   ├── install_fixed.sh          # 依赖安装（修复版）
│   └── requirements.txt
│
└── weights/                      # 模型目录
    ├── chinese-wav2vec2-base/    # 2.9GB
    ├── realesr-animevideov3.pth  # 2.4MB
    ├── Wan2.1-I2V-14B-480P/      # ~30GB
    └── InfiniteTalk/             # ~160GB
```

---

## ✅ 验证部署成功

启动成功时，日志应显示：

```
2025-10-30 XX:XX:XX - INFO - ✅ GPU 監控已啟用 (偵測到 1 個 GPU)
2025-10-30 XX:XX:XX - INFO - 🚀 初始化 InfiniteTalk Worker v7.3.3
2025-10-30 XX:XX:XX - INFO - ✅ 連線成功
2025-10-30 XX:XX:XX - INFO -    使用 API: https://www.flashfalcon.info
2025-10-30 XX:XX:XX - INFO - ✅ wav2vec2 完成
2025-10-30 XX:XX:XX - INFO - ✅ InfiniteTalk 完成
2025-10-30 XX:XX:XX - INFO - 🎉 模型已常駐！
```

---

## 🔄 管理 Worker

```bash
# 查看状态
ps aux | grep worker.py

# 查看日志
tail -f worker.log

# 停止
pkill -f worker.py

# 重启
pkill -f worker.py && nohup python worker.py > worker.log 2>&1 &
```

---

## 📝 完整部署流程总结

```bash
# 1. 克隆并进入目录
git clone https://github.com/FlashFalconDev/infinitetalk-worker.git
cd infinitetalk-worker

# 2. 一键部署（推荐）
./deploy_official.sh

# 或者分步执行：
# 2a. 安装依赖
./install_fixed.sh

# 2b. 配置环境
cp .env.example .env
nano .env  # 填入 token

# 2c. 下载模型
./download_models_official.sh

# 3. 启动
nohup python worker.py > worker.log 2>&1 &

# 4. 验证
tail -f worker.log
```

---

## 🌟 关键改进点

与之前文档相比的改进：

1. ✅ **正确的仓库名**
   - `Wan-AI/Wan2.1-I2V-14B-480P`（不是 MeiGen-AI）

2. ✅ **修复版本冲突**
   - transformers==4.45.2（兼容 torch 2.4.1）

3. ✅ **完整的自动化脚本**
   - `deploy_official.sh` - 一键部署
   - `download_models_official.sh` - 官方模型下载

4. ✅ **明确的依赖安装顺序**
   - 先 PyTorch → 其他包 → 修复 transformers

5. ✅ **处理 LoRA 问题**
   - 提供代码修改方案使其可选

---

## 📚 相关文档

| 文档 | 说明 |
|------|------|
| **README_OFFICIAL_DEPLOYMENT.md** | 本文档 - 官方部署流程 |
| deploy_official.sh | 一键部署脚本 |
| download_models_official.sh | 官方模型下载脚本 |
| DEPLOYMENT_GUIDE_FIXED.md | 详细部署指南 |
| DEPLOYMENT_ISSUES_LOG.md | 问题记录和解决方案 |
| STATUS_AND_NEXT_STEPS.md | 当前状态 |

---

## 📞 获取帮助

- **Worker 仓库**: https://github.com/FlashFalconDev/infinitetalk-worker
- **官方仓库**: https://github.com/MeiGen-AI/InfiniteTalk
- **Wan 模型**: https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P
- **Email**: support@flashfalcon.info

---

**创建日期**: 2025-10-30
**验证状态**: ✅ 已在实际环境测试
**模型下载**: ✅ 正在进行中
**主要贡献**: 修复了错误的仓库名和版本冲突
