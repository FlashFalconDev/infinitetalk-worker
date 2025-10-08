import requests
import json
import os
import uuid
import time
import logging
import socket

from model_service import get_model_service

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InfiniteTalkWorker:
    def __init__(self):
        self.base_url = "https://host.flashfalcon.info"
        self.task_api = f"{self.base_url}/aigen/api/pending_task/?model_code=InfiniteTalk_S2V"
        self.result_api = f"{self.base_url}/aigen/api/task_result/"
        self.upload_api = f"{self.base_url}/api/save_file/"
        
        self.temp_dir = "temp_downloads"
        self.output_dir = "outputs"
        
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.worker_id = self._generate_worker_id()
        
        logger.info("=" * 70)
        logger.info("🚀 初始化 Worker - 載入模型 (只執行一次)")
        logger.info(f"🆔 Worker ID: {self.worker_id}")
        logger.info("=" * 70)
        self.model_service = get_model_service()
        logger.info("=" * 70)
        logger.info("✅ Worker 準備就緒!")
        logger.info("=" * 70)
    
    def _generate_worker_id(self):
        if os.getenv('WORKER_ID'):
            return os.getenv('WORKER_ID')
        
        hostname = socket.gethostname()
        short_uuid = str(uuid.uuid4())[:8]
        return f"worker-{hostname}-{short_uuid}"
    
    def fetch_task(self):
        try:
            logger.info("查詢待處理任務...")
            url = f"{self.task_api}&worker_id={self.worker_id}"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("success") and result["data"]["Generation_Video_task"]:
                    tasks = result["data"]["Generation_Video_task"]
                    logger.info(f"獲取到 {len(tasks)} 個任務")
                    return tasks
                else:
                    logger.info("目前沒有待處理任務")
                    return []
            else:
                logger.error(f"獲取任務失敗: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"獲取任務錯誤: {e}")
            return []
    
    def download_file(self, url, save_path):
        try:
            logger.info(f"下載: {url}")
            response = requests.get(url, timeout=300, stream=True)
            
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"✅ 下載完成: {save_path}")
                return True
            else:
                logger.error(f"下載失敗: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"下載錯誤: {e}")
            return False
    
    def generate_video(self, image_path, audio_path, prompt, task_id, quality='balanced'):
        try:
            output_path = os.path.join(self.output_dir, f"{task_id}_output")
            
            start_time = time.time()
            
            final_path = self.model_service.generate(
                image_path=image_path,
                audio_path=audio_path,
                prompt=prompt,
                output_path=output_path,
                quality=quality
            )
            
            generation_time = int((time.time() - start_time) / 60)
            
            return final_path, generation_time
                
        except Exception as e:
            logger.error(f"生成失敗: {e}")
            return None, 0
    
    def upload_video(self, video_path, task_id):
        try:
            logger.info(f"📤 上傳影片: {video_path}")
            
            with open(video_path, "rb") as f:
                files = {"file": (f"{task_id}.mp4", f, "video/mp4")}
                
                response = requests.post(self.upload_api, files=files, timeout=600)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get("ok"):
                        video_url = result["data"]["url"]
                        logger.info(f"✅ 上傳成功: {video_url}")
                        return video_url
                    else:
                        raise Exception(f"上傳失敗: {result}")
                else:
                    raise Exception(f"HTTP 錯誤: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"上傳錯誤: {e}")
            return None
    
    def report_result(self, task_pk, video_url, quality, generation_time):
        try:
            logger.info(f"📮 回報結果 - 任務PK: {task_pk}")
            
            data = {
                "video_generation_image_audio_pk": task_pk,
                "video_url": video_url,
                "worker_id": self.worker_id,
                "quality": quality,
                "generation_time": generation_time
            }
            
            response = requests.post(self.result_api, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("success"):
                    logger.info("✅ 結果回報成功")
                    logger.info(f"   Worker: {self.worker_id}")
                    logger.info(f"   Quality: {quality}")
                    logger.info(f"   Time: {generation_time} 分鐘")
                    return True
                else:
                    logger.error(f"❌ 回報失敗: {result.get('error')}")
                    return False
            else:
                logger.error(f"❌ HTTP 錯誤: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"回報錯誤: {e}")
            return False
    
    def cleanup(self, task_id):
        try:
            files_to_delete = [
                os.path.join(self.temp_dir, f"{task_id}_image.jpg"),
                os.path.join(self.temp_dir, f"{task_id}_audio.wav"),
                os.path.join(self.output_dir, f"{task_id}_output.mp4")
            ]
            
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"🗑️  已刪除: {file_path}")
                    
        except Exception as e:
            logger.error(f"清理錯誤: {e}")
    
    def process_task(self, task):
        task_id = str(uuid.uuid4())
        task_pk = task["video_generation_image_audio_pk"]
        quality = task.get("quality", "balanced")
        
        logger.info("=" * 70)
        logger.info(f"🎬 處理任務 PK: {task_pk}")
        logger.info(f"📝 Prompt: {task['prompt']}")
        logger.info(f"🎨 Quality: {quality}")
        logger.info("=" * 70)
        
        try:
            image_path = os.path.join(self.temp_dir, f"{task_id}_image.jpg")
            audio_path = os.path.join(self.temp_dir, f"{task_id}_audio.wav")
            
            if not self.download_file(task["image_model_url"], image_path):
                raise Exception("下載圖片失敗")
            
            if not self.download_file(task["sound_model_url"], audio_path):
                raise Exception("下載音頻失敗")
            
            video_path, generation_time = self.generate_video(
                image_path, audio_path, task["prompt"], task_id, quality
            )
            
            if not video_path:
                raise Exception("生成影片失敗")
            
            video_url = self.upload_video(video_path, task_id)
            
            if not video_url:
                raise Exception("上傳影片失敗")
            
            if not self.report_result(task_pk, video_url, quality, generation_time):
                raise Exception("回報結果失敗")
            
            self.cleanup(task_id)
            
            logger.info("=" * 70)
            logger.info("✅ 任務完成!")
            logger.info(f"   品質: {quality}")
            logger.info(f"   耗時: {generation_time} 分鐘")
            logger.info("=" * 70)
            return True
            
        except Exception as e:
            logger.error(f"❌ 任務失敗: {e}")
            self.cleanup(task_id)
            return False
    
    def run(self, poll_interval=30):
        logger.info("=" * 70)
        logger.info("🤖 InfiniteTalk Worker 運行中...")
        logger.info(f"🆔 Worker ID: {self.worker_id}")
        logger.info(f"⏱️  輪詢間隔: {poll_interval} 秒")
        logger.info("=" * 70)
        
        while True:
            try:
                tasks = self.fetch_task()
                
                if tasks:
                    for task in tasks:
                        self.process_task(task)
                else:
                    logger.info(f"💤 等待 {poll_interval} 秒...")
                    time.sleep(poll_interval)
                    
            except KeyboardInterrupt:
                logger.info("⛔ 收到中斷信號，停止 worker")
                break
            except Exception as e:
                logger.error(f"運行錯誤: {e}")
                time.sleep(poll_interval)

if __name__ == "__main__":
    worker = InfiniteTalkWorker()
    worker.run(poll_interval=30)
