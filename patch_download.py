import re

with open('worker.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到 download_file 方法並替換
old_method = r'''    def download_file\(self, url, save_path\):
        """下載檔案"""
        try:
            logger\.info\(f"📥 下載: \{url\}"\)
            response = requests\.get\(url, timeout=300, stream=True\)
            
            if response\.status_code == 200:
                with open\(save_path, 'wb'\) as f:
                    for chunk in response\.iter_content\(chunk_size=8192\):
                        f\.write\(chunk\)
                logger\.info\(f"✅ 下載完成: \{save_path\}"\)
                return True
            else:
                logger\.error\(f"下載失敗: \{response\.status_code\}"\)
                return False
                
        except Exception as e:
            logger\.error\(f"下載錯誤: \{e\}"\)
            return False'''

new_method = '''    def download_file(self, url, save_path, max_retries=3):
        """下載檔案（帶重試機制）"""
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"📥 下載 (嘗試 {attempt}/{max_retries}): {url}")
                
                # 增加超時到 10 分鐘，添加進度顯示
                response = requests.get(url, timeout=600, stream=True)
                
                if response.status_code == 200:
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    with open(save_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                
                                # 每下載 10MB 顯示一次進度
                                if downloaded % (10 * 1024 * 1024) == 0 and total_size > 0:
                                    progress = (downloaded / total_size) * 100
                                    logger.info(f"   下載中: {downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB ({progress:.1f}%)")
                    
                    logger.info(f"✅ 下載完成: {save_path} ({downloaded / 1024 / 1024:.2f}MB)")
                    return True
                else:
                    logger.error(f"下載失敗: HTTP {response.status_code}")
                    if attempt < max_retries:
                        logger.info(f"   等待 5 秒後重試...")
                        time.sleep(5)
                        continue
                    return False
                    
            except requests.exceptions.Timeout as e:
                logger.error(f"下載超時: {e}")
                if attempt < max_retries:
                    logger.info(f"   等待 10 秒後重試...")
                    time.sleep(10)
                    continue
                return False
                
            except Exception as e:
                logger.error(f"下載錯誤: {e}")
                if attempt < max_retries:
                    logger.info(f"   等待 5 秒後重試...")
                    time.sleep(5)
                    continue
                return False
        
        return False'''

content = re.sub(old_method, new_method, content, flags=re.DOTALL)

with open('worker.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ download_file 已更新")
