# 找到 worker.py 中的 upload_video 方法並替換

import re

with open('worker.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到並替換 upload_video 方法
old_upload = r'''    def upload_video\(self, video_path, task_id\):
        """上傳影片"""
        try:
            logger\.info\(f"📤 上傳影片: {video_path}"\)
            
            with open\(video_path, "rb"\) as f:
                files = \{"file": \(f"{task_id}\.mp4", f, "video/mp4"\)\}
                
                endpoints = self\._get_api_endpoints\(\)
                response = requests\.post\(endpoints\['upload'\], files=files, timeout=600\)'''

new_upload = '''    def upload_video(self, video_path, task_id):
        """上傳影片"""
        try:
            logger.info(f"📤 上傳影片: {video_path}")
            
            with open(video_path, "rb") as f:
                files = {"file": (f"{task_id}.mp4", f, "video/mp4")}
                
                # ✅ 添加認證 Headers
                headers = {'Authorization': f'Bearer {self.worker_token}'}
                
                endpoints = self._get_api_endpoints()
                response = requests.post(endpoints['upload'], files=files, headers=headers, timeout=600)'''

content = re.sub(old_upload, new_upload, content)

with open('worker.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ upload_video 已修正")
