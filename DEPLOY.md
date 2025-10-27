# InfiniteTalk 快速部署

## 1. 克隆專案
```bash
cd /workspace
git clone <your-repo-url> InfiniteTalk
cd InfiniteTalk
```

## 2. 安裝依賴
```bash
pip install -r requirements.txt
```

## 3. 下載模型
```bash
bash download_models.sh
```

## 4. 啟動 Worker
```bash
nohup python worker.py > worker.log 2>&1 &
tail -f worker.log
```

## 檢查狀態
```bash
ps aux | grep worker.py
```
