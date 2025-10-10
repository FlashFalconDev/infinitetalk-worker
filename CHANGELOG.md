# Changelog

All notable changes to InfiniteTalk Worker will be documented in this file.

## [7.2] - 2025-10-10

### Added
- Token 認證機制（Bearer Token in Headers）
- 完整 GPU 監控支援（nvidia-ml-py3）
  - GPU 使用率、溫度、功率監控
  - 時鐘頻率、風扇轉速監控
  - 進程數量統計
- 獨立心跳線程（每 60 秒自動更新）
- .env 配置檔案支援
- 詳細的錯誤提示和日誌

### Changed
- API 端點從 `/aigen/api/` 改為 `/ai/api/`
- 認證方式從 URL 參數改為 Headers
- Worker ID 生成邏輯優化

### Fixed
- 修正 pynvml 返回 bytes 類型無法 JSON 序列化的問題
- 改進連線測試的錯誤處理
- 優化 GPU 監控的異常處理

### Security
- 移除 URL 中的敏感資訊
- 使用環境變數存放 Token
- 增強 Token 驗證機制

### Dependencies
- 新增 `python-dotenv` (必須)
- 新增 `nvidia-ml-py3` (可選，用於詳細 GPU 監控)

### Breaking Changes
- **必須設定** `INFINITETALK_WORKER_TOKEN` 環境變數
- API 端點變更，需要後端配合更新
- 需要在 Admin 後台創建 Worker 並獲取 Token

### Migration
1. 安裝新依賴: `pip install python-dotenv nvidia-ml-py3`
2. 複製配置: `cp .env.example .env`
3. 在 Admin 後台創建 Worker
4. 填入 Token 到 .env
5. 啟動 Worker: `python worker.py`

---

## [7.1] - 2025-10-09

### Added
- 支援品質參數（fast/balanced/high/ultra）
- 支援字典覆蓋預設參數
- 清理備份檔案功能

### Changed
- 優化 .gitignore
- 改進任務處理流程

---

## [7.0] - 2025-10-08

### Added
- 初始版本
- InfiniteTalk 模型整合
- LoRA 加速支援
- 基本任務處理流程
