# Changelog

All notable changes to InfiniteTalk Worker will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [7.2] - 2025-10-10

### Added
- Token 認證機制（Bearer Token in Headers）
- 完整 GPU 監控支援（nvidia-ml-py3）
  - GPU 使用率、溫度、功率監控
  - 時鐘頻率、風扇轉速監控
  - 記憶體使用詳細資訊
  - 進程數量統計
- 獨立心跳線程（每 60 秒自動更新）
- .env 配置檔案支援（python-dotenv）
- 部署腳本集
  - start_worker.sh: 啟動腳本
  - stop_worker.sh: 停止腳本
  - status_worker.sh: 狀態查看
  - check_env.sh: 環境診斷
- 詳細的錯誤提示和日誌
- 完整的文檔（README.md）

### Changed
- API 端點從 
