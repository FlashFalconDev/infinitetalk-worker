# Changelog

All notable changes to InfiniteTalk Worker will be documented in this file.

## [7.3] - 2025-10-10

### Added
- 主備 API 自動切換機制
  - 主 API: www.flashfalcon.info
  - 備用 API: host.flashfalcon.info
  - 連續失敗 2 次自動切換到備用 API
  - 主 API 恢復後自動切回
- 統一請求處理方法 `_make_request()`
  - 自動處理失敗重試
  - 自動切換 API
  - 應用於所有端點
- 詳細的 API 切換日誌
- 失敗計數器機制

### Changed
- API 端點動態生成
- 改進錯誤處理邏輯
- 優化日誌輸出

### Fixed
- 提高 API 可用性
- 改善故障恢復能力

---

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
- API 端點從 `/aigen/api/` 改為 `/ai/api/`
- 認證方式從 URL 參數改為 Headers
- Worker ID 生成邏輯優化
- 日誌輸出格式改進

### Fixed
- 修正 pynvml 返回 bytes 類型無法 JSON 序列化的問題
- 改進連線測試的錯誤處理
- 優化 GPU 監控的異常處理
- 修正虛擬環境自動檢測邏輯

### Security
- 移除 URL 中的敏感資訊
- 使用環境變數存放 Token
- 增強 Token 驗證機制
- 添加 .gitignore 保護配置檔案

### Dependencies
- 新增 `python-dotenv>=1.0.0` (必須)
- 新增 `nvidia-ml-py3>=7.352.0` (可選)

### Breaking Changes
- **必須設定** `INFINITETALK_WORKER_TOKEN` 環境變數
- API 端點變更，需要後端配合更新
- 需要在 Admin 後台創建 Worker 並獲取 Token

---

## [7.1] - 2025-10-09

### Added
- 支援品質參數（fast/balanced/high/ultra）
- 支援字典覆蓋預設參數
- 清理備份檔案功能

### Changed
- 優化 .gitignore
- 改進任務處理流程
- 更新文檔

### Fixed
- 修正品質參數解析問題

---

## [7.0] - 2025-10-08

### Added
- 初始版本發布
- InfiniteTalk 模型整合
- LoRA 加速支援
- 基本任務處理流程
- 圖片和音頻下載
- 影片生成和上傳
- 結果回報機制
- 臨時檔案清理

---

## 未來計劃

### [7.4] - 計劃中
- [ ] 支援多模型切換
- [ ] 任務優先級處理
- [ ] 性能統計和報告
- [ ] Web UI 管理介面

### [7.5] - 計劃中
- [ ] 分散式任務調度
- [ ] 任務快取機制
- [ ] 模型版本管理
- [ ] A/B 測試支援

---

## 支援

如有問題，請：
1. 查看 [README.md](README.md)
2. 提交 [GitHub Issue](https://github.com/FlashFalconDev/infinitetalk-worker/issues)
3. 聯繫技術支援: support@flashfalcon.info
