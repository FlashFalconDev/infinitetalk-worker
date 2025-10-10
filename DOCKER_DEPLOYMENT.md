# Docker 部署指南

## 🚀 快速開始

### 1. 準備環境
```bash
# 確保已安裝 Docker 和 NVIDIA Container Toolkit
docker --version
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

