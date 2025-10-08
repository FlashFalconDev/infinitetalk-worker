"""
優化版 Worker - 使用常駐模型服務
"""
import requests
import json
import os
import uuid
import time
import logging
from model_service import get_model_service

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastWorker:
    def __init__(self):
        self.base_url = "https://host.flashfalcon.info"
        self.task_api = f"{self.base_url}/aigen/api/pending_task/?model_code=InfiniteTalk_S2V"
        self.result_api = f"{self.base_url}/aigen/api/task_result/"
        self.upload_api = f"{self.base_url}/api/save_file/"
        
        self.temp_dir = "temp_downloads"
        self.output_dir = "outputs"
        
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 啟動時載入模型 (只一次)
        logger.info("🔧 初始化模型服務...")
        self.model = get_model_service()
        logger.info("✅ Worker 就緒,開始監聽任務")
    
    def fetch_task(self):
        try:
            response = requests.get(self.task_api, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if result.get("success") and result["data"]["Generation_Video_task"]:
                    return result["data"]["Generation_Video_task"]
            return []
        except Exception as e:
            logger.error(f"獲取任務錯誤: {e}")
            return []
    
    def download_file(self, url, save_path):
        try:
            response = requests.get(url, timeout=300, stream=True)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            return False
        except Exception as e:
            logger.error(f"下載錯誤: {e}")
            return False
    
    def upload_video(self, video_path, task_id):
        try:
            with open(video_path, "rb") as f:
                files = {"file": (f"{task_id}.mp4", f, "video/mp4")}
                response = requests.post(self.upload_api, files=files, timeout=600)
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("ok"):
                        return result["data"]["url"]
            return None
        except Exception as e:
            logger.error(f"上傳錯誤: {e}")
            return None
    
    def report_result(self, task_pk, video_url):
        try:
            data = {
                "video_generation_image_audio_pk": task_pk,
                "video_url": video_url
            }
            response = requests.post(self.result_api, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("success", False)
            return False
        except Exception as e:
            logger.error(f"回報錯誤: {e}")
            return False
    
    def cleanup(self, task_id):
        files = [
            os.path.join(self.temp_dir, f"{task_id}_image.jpg"),
            os.path.join(self.temp_dir, f"{task_id}_audio.wav"),
            os.path.join(self.output_dir, f"{task_id}.mp4")
        ]
        for f in files:
            if os.path.exists(f):
                os.remove(f)
    
    def process_task(self, task):
        task_id = str(uuid.uuid4())
        task_pk = task["video_generation_image_audio_pk"]
        
        logger.info("=" * 70)
        logger.info(f"📋 任務 PK: {task_pk}")
        logger.info(f"💬 Prompt: {task['prompt']}")
        logger.info("=" * 70)
        
        try:
            # 下載
            image_path = os.path.join(self.temp_dir, f"{task_id}_image.jpg")
            audio_path = os.path.join(self.temp_dir, f"{task_id}_audio.wav")
            
            logger.info("📥 下載素材...")
            if not self.download_file(task["image_model_url"], image_path):
                raise Exception("下載圖片失敗")
            if not self.download_file(task["sound_model_url"], audio_path):
                raise Exception("下載音頻失敗")
            
            # 生成 (使用常駐模型,超快!)
            output_path = os.path.join(self.output_dir, task_id)
            video_file = self.model.generate(
                image_path=image_path,
                audio_path=audio_path,
                prompt=task["prompt"],
                output_path=output_path,
                resolution="480",
                sample_steps=8
            )
            
            # 上傳
            logger.info("📤 上傳影片...")
            video_url = self.upload_video(video_file, task_id)
            if not video_url:
                raise Exception("上傳失敗")
            
            logger.info(f"🔗 影片 URL: {video_url}")
            
            # 回報
            if not self.report_result(task_pk, video_url):
                raise Exception("回報失敗")
            
            # 清理
            self.cleanup(task_id)
            
            logger.info("✅ 任務完成!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 任務失敗: {e}")
            self.cleanup(task_id)
            return False
    
    def run(self, poll_interval=30):
        logger.info(f"🔄 開始輪詢 (間隔 {poll_interval} 秒)")
        
        while True:
            try:
                tasks = self.fetch_task()
                if tasks:
                    for task in tasks:
                        self.process_task(task)
                else:
                    logger.info(f"⏳ 無任務,等待 {poll_interval} 秒...")
                    time.sleep(poll_interval)
            except KeyboardInterrupt:
                logger.info("👋 收到停止信號")
                break
            except Exception as e:
                logger.error(f"運行錯誤: {e}")
                time.sleep(poll_interval)

if __name__ == "__main__":
    worker = FastWorker()
    worker.run()
