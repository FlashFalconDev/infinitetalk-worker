import re

with open('worker.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 完整替換 download_file 方法
pattern = r'    def download_file\(self, url, save_path.*?\n(?=    def |\Z)'

replacement = '''    def download_file(self, url, save_path, max_retries=3):
        """下載檔案（S3 使用代理）"""
        # S3 URL 使用後端代理
        if 's3.amazonaws.com' in url or 's3-ap-northeast-1' in url:
            return self._download_via_proxy(url, save_path, max_retries)
        
        # 其他 URL 直接下載
        return self._download_direct(url, save_path, max_retries)
    
    def _download_via_proxy(self, url, save_path, max_retries):
        """通過後端代理下載"""
        import urllib.parse
        proxy_url = f"{self.current_base}/aigen/api/proxy_s3/?url={urllib.parse.quote(url)}"
        
        logger.info(f"📥 代理下載: {url}")
        
        for attempt in range(1, max_retries + 1):
            try:
                headers = self._get_auth_headers()
                response = requests.get(proxy_url, headers=headers, timeout=120, stream=True)
                
                if response.status_code == 200:
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    with open(save_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                    
                    logger.info(f"✅ 下載完成: {save_path} ({downloaded / 1024 / 1024:.2f}MB)")
                    return True
                else:
                    logger.error(f"代理失敗: HTTP {response.status_code}")
                    if attempt < max_retries:
                        time.sleep(3)
                        continue
                    return False
                    
            except Exception as e:
                logger.error(f"代理錯誤 ({attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(3)
                    continue
                return False
        
        return False
    
    def _download_direct(self, url, save_path, max_retries):
        """直接下載"""
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"📥 下載: {url}")
                response = requests.get(url, timeout=300, stream=True)
                
                if response.status_code == 200:
                    with open(save_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    logger.info(f"✅ 下載完成: {save_path}")
                    return True
                else:
                    logger.error(f"下載失敗: HTTP {response.status_code}")
                    if attempt < max_retries:
                        time.sleep(3)
                        continue
                    return False
                    
            except Exception as e:
                logger.error(f"下載錯誤 ({attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(3)
                    continue
                return False
        
        return False

'''

content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open('worker.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ 已修改為代理模式")
