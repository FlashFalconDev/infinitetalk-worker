# InfiniteTalk Worker v8.0.0

🚀 高性能的 InfiniteTalk 視頻生成 Worker，支援完整 GPU 監控、Token 認證和 **Multi-GPU 並行處理**。

---

## ⚠️ 重要提醒

### 📖 完整部署指南

**⭐ 強烈推薦**: [DEPLOYMENT_GUIDE_COMPLETE.md](./DEPLOYMENT_GUIDE_COMPLETE.md) - **包含所有實際驗證的修復步驟**

此指南記錄了從安裝到成功生成影片的完整流程，包含：
- ✅ Python 3.12 兼容性修復
- ✅ Flash Attention fallback 實作
- ✅ CUDA 記憶體優化
- ✅ 所有遇到的問題和解決方案
- ✅ 已在實際環境測試通過

**簡化版**: [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) - 基本部署流程

### ⚠️ 關鍵注意事項

此 Worker 需要配合 **[MeiGen-AI/InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk)** 官方倉庫的模型使用！

- ❌ **錯誤做法**: 只克隆此倉庫（會缺少 ~160GB 的模型權重）
- ✅ **正確做法**: 按照 [DEPLOYMENT_GUIDE_COMPLETE.md](./DEPLOYMENT_GUIDE_COMPLETE.md) 的完整流程部署

---

## ✨ 主要功能

- 🔐 **Token 認證**: 安全的 Bearer Token 認證機制
- 🔥 **Multi-GPU 並行處理** (v8.0.0 新功能):
  - 支援多 GPU 並行影片生成
  - 自動 GPU 調度和負載均衡
  - 吞吐量可提升 2x-4x（取決於 GPU 數量）
  - 輕鬆切換單/多 GPU 模式
  - 詳見: [MULTI_GPU_GUIDE.md](./MULTI_GPU_GUIDE.md)
- 📊 **GPU 監控**: 完整的 GPU 性能監控
  - 使用率、溫度、功率
  - 時鐘頻率、風扇轉速
  - 記憶體使用詳情
- 💓 **心跳系統**: 自動保持在線狀態（每 60 秒）
- ⚙️ **環境配置**: 支援 .env 檔案配置
- 🎨 **品質控制**: 支援多種品質預設（fast/balanced/high/ultra）

## 🚀 快速開始

### 1. 準備環境
```bash
# 克隆倉庫
git clone https://github.com/FlashFalconDev/infinitetalk-worker.git
cd infinitetalk-worker

# 安裝依賴
pip install -r requirements.txt

# 可選：安裝 GPU 監控
pip install nvidia-ml-py3
2. 配置 Worker
bash# 複製配置範例
cp .env.example .env

# 編輯配置，填入 Token
nano .env
在 .env 中設定：
envINFINITETALK_API_BASE=https://host.flashfalcon.info
INFINITETALK_WORKER_TOKEN=your_token_from_admin
3. 啟動 Worker
方式 1: 使用腳本（推薦）
bash./start_worker.sh
方式 2: 手動啟動
bashnohup python worker.py > worker.log 2>&1 &
方式 3: 前台運行（測試）
bashpython worker.py
📊 管理 Worker
查看狀態
bash./status_worker.sh
查看日誌
bashtail -f worker.log
停止 Worker
bash./stop_worker.sh
重啟 Worker
bash./stop_worker.sh
./start_worker.sh
🔧 配置說明
環境變數
變數必須說明預設值INFINITETALK_WORKER_TOKEN✅Worker Token（從 Admin 後台獲取）-INFINITETALK_API_BASE❌API Base URLhttps://host.flashfalcon.infoWORKER_ID❌Worker 識別碼自動生成LOG_LEVEL❌日誌級別INFO
獲取 Token

登入 Admin 後台：https://host.flashfalcon.info/admin/
進入「Worker 主機」管理
點擊「新增 Worker」
填寫資訊並保存
複製顯示的 Token

📦 系統需求

Python: 3.10+
GPU: NVIDIA GPU with CUDA 11.8+
VRAM: 建議 24GB+ （根據品質設定）
儲存: 10GB+ 可用空間
系統: Ubuntu 20.04+ / CentOS 7+

🛠️ 依賴套件
核心依賴：
torch>=2.0.0
diffusers>=0.25.0
transformers>=4.36.0
accelerate>=0.25.0
xformers>=0.0.23
requests>=2.31.0
python-dotenv>=1.0.0
可選依賴：
nvidia-ml-py3>=7.352.0  # GPU 監控
完整列表請見 requirements.txt
🐛 故障排除
Token 無效
bash# 檢查 .env 配置
cat .env | grep INFINITETALK_WORKER_TOKEN

# 確認 Token 正確且未被停用
GPU 監控不可用
bash# 安裝 nvidia-ml-py3
pip install nvidia-ml-py3

# 測試 GPU
python -c "import pynvml; pynvml.nvmlInit(); print('GPU OK')"
Worker 無法連線
bash# 檢查網路
ping host.flashfalcon.info

# 檢查 Token
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://host.flashfalcon.info/ai/api/worker/heartbeat
查看詳細日誌
bash# 實時日誌
tail -f worker.log

# 最近 100 行
tail -100 worker.log

# 搜尋錯誤
grep ERROR worker.log
常見錯誤
錯誤訊息原因解決方法Invalid or inactive tokenToken 無效或已停用檢查 Admin 後台，重新生成 TokenWorker does not support modelWorker 未配置此模型在 Admin 後台添加支援的模型CUDA out of memoryGPU 記憶體不足降低品質設定或使用更大的 GPUConnection refusedAPI 無法連接檢查網路和 API Base URL
📊 性能監控
GPU 狀態
bash# 系統層級
nvidia-smi

# Worker 內部（顯示詳細資訊）
./status_worker.sh
任務統計
在 Admin 後台查看：

總任務數
完成任務數
失敗任務數
成功率
平均處理時間

效能指標
Worker 會自動收集並回報：

GPU 使用率
GPU 溫度
GPU 功率使用
GPU 記憶體使用
時鐘頻率
風扇轉速

🔄 更新
更新 Worker
bash# 1. 停止 Worker
./stop_worker.sh

# 2. 備份配置
cp .env .env.backup

# 3. 拉取最新代碼
git pull origin main

# 4. 更新依賴
pip install -r requirements.txt --upgrade

# 5. 恢復配置
cp .env.backup .env

# 6. 重啟
./start_worker.sh
檢查版本
bash# 查看當前版本
git describe --tags

# 查看 Worker 版本
grep "worker_version" worker.py
🔒 安全建議
保護 Token

不要將 .env 提交到 Git

bash   echo ".env" >> .gitignore

定期更換 Token

建議每 3 個月更換一次
在 Admin 後台重新生成
更新 .env 並重啟 Worker


限制檔案權限

bash   chmod 600 .env
防火牆設定
bash# 只允許必要的出站連線
sudo ufw allow out 443/tcp  # HTTPS
sudo ufw allow out 80/tcp   # HTTP (可選)
日誌管理
bash# 定期清理日誌
find . -name "worker.log" -mtime +7 -delete

# 或使用 logrotate
sudo nano /etc/logrotate.d/infinitetalk-worker
🏗️ 架構說明
工作流程
Worker 啟動
    ↓
註冊/驗證 Token
    ↓
載入模型（只執行一次）
    ↓
啟動心跳線程（每 60 秒）
    ↓
輪詢任務（每 30 秒）
    ↓
處理任務
    ├─ 下載圖片和音頻
    ├─ 生成影片
    ├─ 上傳結果
    └─ 回報完成
    ↓
清理臨時檔案
    ↓
繼續輪詢
目錄結構
infinitetalk-worker/
├── worker.py              # 主程序
├── model_service.py       # 模型服務
├── start_worker.sh        # 啟動腳本
├── stop_worker.sh         # 停止腳本
├── status_worker.sh       # 狀態腳本
├── check_env.sh           # 環境檢查
├── .env                   # 配置（不要提交）
├── .env.example           # 配置範例
├── requirements.txt       # 依賴列表
├── README.md              # 本文件
├── CHANGELOG.md           # 更新日誌
├── temp_downloads/        # 臨時下載目錄
├── outputs/               # 輸出目錄
└── worker.log             # 日誌檔案
📝 開發指南
本地測試
bash# 1. 設定測試 Token
export INFINITETALK_WORKER_TOKEN=test_token

# 2. 前台運行（方便調試）
python worker.py

# 3. 查看日誌
tail -f worker.log
修改配置
編輯 worker.py 中的配置：
python# 輪詢間隔（秒）
def run(self, poll_interval=30):

# 心跳間隔（秒）
time.sleep(60)
自訂品質預設
在後端 Admin 配置不同的品質參數。
📝 版本歷史
v7.2 (2025-10-10)

✅ Token 認證機制
✅ 完整 GPU 監控（nvidia-ml-py3）
✅ 心跳系統（每 60 秒）
✅ 部署腳本（start/stop/status）
✅ 環境變數配置（.env）
✅ 錯誤修正（JSON 序列化）

v7.1 (2025-10-09)

✅ 品質參數支援（fast/balanced/high/ultra）
✅ 參數字典覆蓋功能
✅ .gitignore 清理

v7.0 (2025-10-08)

✅ 初始版本
✅ InfiniteTalk 模型整合
✅ LoRA 加速支援
✅ 基本任務處理流程

詳細更新日誌請見 CHANGELOG.md
🤝 貢獻
歡迎提交 Issue 和 Pull Request！
貢獻指南

Fork 本倉庫
創建 feature 分支：git checkout -b feature/amazing-feature
提交變更：git commit -m 'Add amazing feature'
推送分支：git push origin feature/amazing-feature
開啟 Pull Request

📄 授權
Apache-2.0 License
📞 支援

GitHub Issues: https://github.com/FlashFalconDev/infinitetalk-worker/issues
Email: support@flashfalcon.info
文檔: https://github.com/FlashFalconDev/infinitetalk-worker/wiki

🙏 致謝

InfiniteTalk 模型開發團隊
Diffusers 社群
所有貢獻者


🎉 享受使用 InfiniteTalk Worker！
Made with ❤️ by FlashFalcon Team
