import requests
import json
import os
import subprocess
import uuid
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InfiniteTalkProcessor:
    def __init__(self, upload_url="https://host.flashfacon.info/api/save_file/"):
        self.upload_url = upload_url
        self.python_path = "/workspace/infinitetalk-env/bin/python"
        self.upload_dir = "temp_uploads"
        self.output_dir = "outputs"
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_and_upload(self, audio_path, input_path, input_type, resolution="480", sample_steps=40):
        """處理任務並上傳到你的 API"""
        task_id = str(uuid.uuid4())
        
        try:
            logger.info(f"開始處理任務 {task_id}")
            
            # 1. 創建輸入 JSON
            input_json = [{
                "audio": audio_path,
                "input": input_path,
                "type": input_type
            }]
            
            input_json_path = os.path.join(self.upload_dir, f"{task_id}_input.json")
            with open(input_json_path, "w") as f:
                json.dump(input_json, f)
            
            output_path = os.path.join(self.output_dir, task_id)
            
            # 2. 生成影片
            cmd = [
                self.python_path,
                "generate_infinitetalk.py",
                "--ckpt_dir", "weights/Wan2.1-I2V-14B-480P",
                "--wav2vec_dir", "weights/chinese-wav2vec2-base",
                "--infinitetalk_dir", "weights/InfiniteTalk/single/infinitetalk.safetensors",
                "--input_json", input_json_path,
                "--size", f"infinitetalk-{resolution}",
                "--sample_steps", str(sample_steps),
                "--mode", "streaming",
                "--motion_frame", "9",
                "--num_persistent_param_in_dit", "0",
                "--save_file", output_path
            ]
            
            logger.info("開始生成影片...")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd="/workspace/InfiniteTalk"
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"生成失敗: {stderr}")
            
            # 3. 找到生成的文件
            generated_file = f"{output_path}.mp4"
            final_output = os.path.join(self.output_dir, f"{task_id}_output.mp4")
            
            if os.path.exists(generated_file):
                os.rename(generated_file, final_output)
            else:
                raise Exception("找不到生成的影片")
            
            logger.info(f"影片生成完成: {final_output}")
            
            # 4. 上傳到你的 API
            upload_result = self.upload_to_api(final_output, task_id)
            
            logger.info(f"任務 {task_id} 完成: {upload_result}")
            return {
                "task_id": task_id,
                "status": "success",
                "upload_result": upload_result
            }
            
        except Exception as e:
            logger.error(f"任務 {task_id} 失敗: {e}")
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(e)
            }
    
    def upload_to_api(self, video_path, task_id):
        """上傳影片到 https://host.flashfacon.info/api/save_file/"""
        try:
            logger.info(f"開始上傳影片到 {self.upload_url}")
            
            with open(video_path, "rb") as f:
                files = {
                    "file": (f"{task_id}.mp4", f, "video/mp4")
                }
                
                # 可以添加額外的 metadata
                data = {
                    "task_id": task_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                response = requests.post(
                    self.upload_url,
                    files=files,
                    data=data,
                    timeout=600  # 10分鐘超時
                )
                
                if response.status_code == 200:
                    logger.info(f"上傳成功: {response.text}")
                    return response.json() if response.headers.get('content-type') == 'application/json' else response.text
                else:
                    raise Exception(f"上傳失敗: {response.status_code} - {response.text}")
                    
        except Exception as e:
            logger.error(f"上傳失敗: {e}")
            raise

# 測試用
if __name__ == "__main__":
    processor = InfiniteTalkProcessor()
    
    # 測試處理
    result = processor.process_and_upload(
        audio_path="examples/single/1.wav",
        input_path="examples/single/ref_image.png",
        input_type="image",
        resolution="480",
        sample_steps=20  # 測試用較少步數
    )
    
    print(json.dumps(result, indent=2))
