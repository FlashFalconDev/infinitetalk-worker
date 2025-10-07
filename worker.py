import requests
import json
import os
import subprocess
import uuid
from datetime import datetime
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InfiniteTalkWorker:
    def __init__(self):
        self.base_url = "https://host.flashfalcon.info"
        self.task_api = f"{self.base_url}/aigen/api/pending_task/?model_code=InfiniteTalk_S2V"
        self.result_api = f"{self.base_url}/aigen/api/task_result/"
        self.upload_api = f"{self.base_url}/api/save_file/"
        
        self.python_path = "/workspace/infinitetalk-env/bin/python"
        self.work_dir = "/workspace/InfiniteTalk"
        self.temp_dir = "temp_downloads"
        self.output_dir = "outputs"
        
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def fetch_task(self):
        """獲取待處理任務"""
        try:
            logger.info("查詢待處理任務...")
            response = requests.get(self.task_api, timeout=30)
            
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
        """下載文件"""
        try:
            logger.info(f"下載: {url}")
            response = requests.get(url, timeout=300, stream=True)
            
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"下載完成: {save_path}")
                return True
            else:
                logger.error(f"下載失敗: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"下載錯誤: {e}")
            return False
    
    def generate_video(self, image_path, audio_path, prompt, task_id):
        """生成影片"""
        try:
            logger.info(f"開始生成影片 - 任務 {task_id}")
            
            # 創建輸入 JSON
            input_json = {
                "prompt": prompt,
                "cond_video": image_path,
                "cond_audio": {
                    "person1": audio_path
                }
            }
            
            input_json_path = os.path.join(self.temp_dir, f"{task_id}_input.json")
            with open(input_json_path, "w") as f:
                json.dump(input_json, f, indent=4)
            
            output_path = os.path.join(self.output_dir, task_id)
            
            # 構建命令 (使用 LoRA 加速)
            cmd = [
                self.python_path,
                "generate_infinitetalk.py",
                "--ckpt_dir", "weights/Wan2.1-I2V-14B-480P",
                "--wav2vec_dir", "weights/chinese-wav2vec2-base",
                "--infinitetalk_dir", "weights/InfiniteTalk/single/infinitetalk.safetensors",
                "--input_json", input_json_path,
                "--size", "infinitetalk-480",
                "--sample_steps", "8",
                "--mode", "streaming",
                "--motion_frame", "9",
                "--num_persistent_param_in_dit", "0",
                "--save_file", output_path,
                # LoRA 加速
                "--lora_dir", "weights/Wan2.1_I2V_14B_FusionX_LoRA.safetensors",
                "--lora_scale", "1.0",
                "--sample_text_guide_scale", "1.0",
                "--sample_audio_guide_scale", "2.0",
                "--sample_shift", "2"
            ]
            
            logger.info("執行生成...")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.work_dir
            )
            
            # 顯示進度
            for line in process.stdout:
                print(line, end='')
            
            process.wait()
            
            if process.returncode != 0:
                raise Exception(f"生成失敗: {process.returncode}")
            
            # 重命名輸出文件
            generated_file = f"{output_path}.mp4"
            final_output = os.path.join(self.output_dir, f"{task_id}_output.mp4")
            
            if os.path.exists(generated_file):
                os.rename(generated_file, final_output)
                logger.info(f"生成完成: {final_output}")
                return final_output
            else:
                raise Exception("找不到生成的影片")
                
        except Exception as e:
            logger.error(f"生成失敗: {e}")
            return None
    
    def upload_video(self, video_path, task_id):
        """上傳影片到服務器"""
        try:
            logger.info(f"上傳影片: {video_path}")
            
            with open(video_path, "rb") as f:
                files = {"file": (f"{task_id}.mp4", f, "video/mp4")}
                
                response = requests.post(self.upload_api, files=files, timeout=600)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get("ok"):
                        video_url = result["data"]["url"]
                        logger.info(f"上傳成功: {video_url}")
                        return video_url
                    else:
                        raise Exception(f"上傳失敗: {result}")
                else:
                    raise Exception(f"HTTP 錯誤: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"上傳錯誤: {e}")
            return None
    
    def report_result(self, task_pk, video_url):
        """回報任務結果"""
        try:
            logger.info(f"回報結果 - 任務PK: {task_pk}")
            
            data = {
                "video_generation_image_audio_pk": task_pk,
                "video_url": video_url
            }
            
            response = requests.post(self.result_api, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("success"):
                    logger.info("✅ 結果回報成功")
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
        """清理臨時文件"""
        try:
            files_to_delete = [
                os.path.join(self.temp_dir, f"{task_id}_image.jpg"),
                os.path.join(self.temp_dir, f"{task_id}_audio.wav"),
                os.path.join(self.temp_dir, f"{task_id}_input.json"),
                os.path.join(self.output_dir, f"{task_id}_output.mp4")
            ]
            
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"已刪除: {file_path}")
                    
        except Exception as e:
            logger.error(f"清理錯誤: {e}")
    
    def process_task(self, task):
        """處理單個任務"""
        task_id = str(uuid.uuid4())
        task_pk = task["video_generation_image_audio_pk"]
        
        logger.info("=" * 60)
        logger.info(f"處理任務 PK: {task_pk}")
        logger.info(f"Prompt: {task['prompt']}")
        logger.info("=" * 60)
        
        try:
            # 1. 下載圖片和音頻
            image_path = os.path.join(self.temp_dir, f"{task_id}_image.jpg")
            audio_path = os.path.join(self.temp_dir, f"{task_id}_audio.wav")
            
            if not self.download_file(task["image_model_url"], image_path):
                raise Exception("下載圖片失敗")
            
            if not self.download_file(task["sound_model_url"], audio_path):
                raise Exception("下載音頻失敗")
            
            # 2. 生成影片
            video_path = self.generate_video(image_path, audio_path, task["prompt"], task_id)
            
            if not video_path:
                raise Exception("生成影片失敗")
            
            # 3. 上傳影片
            video_url = self.upload_video(video_path, task_id)
            
            if not video_url:
                raise Exception("上傳影片失敗")
            
            # 4. 回報結果
            if not self.report_result(task_pk, video_url):
                raise Exception("回報結果失敗")
            
            # 5. 清理文件
            self.cleanup(task_id)
            
            logger.info("✅ 任務完成!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 任務失敗: {e}")
            self.cleanup(task_id)
            return False
    
    def run(self, poll_interval=30):
        """持續運行 worker"""
        logger.info("InfiniteTalk Worker 啟動")
        logger.info(f"輪詢間隔: {poll_interval} 秒")
        
        while True:
            try:
                # 獲取任務
                tasks = self.fetch_task()
                
                if tasks:
                    for task in tasks:
                        self.process_task(task)
                else:
                    logger.info(f"等待 {poll_interval} 秒...")
                    time.sleep(poll_interval)
                    
            except KeyboardInterrupt:
                logger.info("收到中斷信號，停止 worker")
                break
            except Exception as e:
                logger.error(f"運行錯誤: {e}")
                time.sleep(poll_interval)

if __name__ == "__main__":
    worker = InfiniteTalkWorker()
    worker.run(poll_interval=30)  # 每 30 秒檢查一次
