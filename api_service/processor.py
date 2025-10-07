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
    def __init__(self, upload_url="https://host.flashfalcon.info/api/save_file/"):
        self.upload_url = upload_url
        self.python_path = "/workspace/infinitetalk-env/bin/python"
        self.upload_dir = "temp_uploads"
        self.output_dir = "outputs"
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_and_upload(self, audio_path, input_path, 
                          prompt="", resolution="480", use_lora=True, sample_steps=None):
        task_id = str(uuid.uuid4())
        
        if sample_steps is None:
            sample_steps = 8 if use_lora else 40
        
        try:
            logger.info(f"任務 {task_id}, LoRA: {use_lora}, Steps: {sample_steps}")
            
            input_json = {
                "prompt": prompt or "A person speaking or singing",
                "cond_video": input_path,
                "cond_audio": {
                    "person1": audio_path
                }
            }
            
            input_json_path = os.path.join(self.upload_dir, f"{task_id}_input.json")
            with open(input_json_path, "w") as f:
                json.dump(input_json, f, indent=4)
            
            output_path = os.path.join(self.output_dir, task_id)
            
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
            
            if use_lora:
                lora_path = "weights/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"
                
                if not os.path.exists(lora_path):
                    logger.info("下載 LoRA...")
                    subprocess.run([
                        "wget",
                        "https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX/resolve/main/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors",
                        "-O", lora_path
                    ], check=True)
                
                cmd.extend([
                    "--lora_dir", lora_path,
                    "--lora_scale", "1.0",
                    "--sample_text_guide_scale", "1.0",
                    "--sample_audio_guide_scale", "2.0",
                    "--sample_shift", "2"
                ])
            
            logger.info("開始生成...")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd="/workspace/InfiniteTalk"
            )
            
            for line in process.stdout:
                print(line, end='')
            
            process.wait()
            
            if process.returncode != 0:
                raise Exception(f"生成失敗: {process.returncode}")
            
            generated_file = f"{output_path}.mp4"
            final_output = os.path.join(self.output_dir, f"{task_id}_output.mp4")
            
            if os.path.exists(generated_file):
                os.rename(generated_file, final_output)
            else:
                raise Exception("找不到生成的影片")
            
            logger.info(f"生成完成: {final_output}")
            
            # 上傳並解析結果
            video_url = self.upload_to_api(final_output, task_id)
            
            # 打印 URL
            print("\n" + "="*50)
            print(f"上傳成功! 影片 URL:")
            print(video_url)
            print("="*50 + "\n")
            
            return {
                "task_id": task_id,
                "status": "success",
                "video_url": video_url
            }
            
        except Exception as e:
            logger.error(f"失敗: {e}")
            return {"task_id": task_id, "status": "failed", "error": str(e)}
    
    def upload_to_api(self, video_path, task_id):
        try:
            logger.info(f"上傳到 {self.upload_url}")
            
            with open(video_path, "rb") as f:
                files = {"file": (f"{task_id}.mp4", f, "video/mp4")}
                data = {"task_id": task_id}
                
                response = requests.post(self.upload_url, files=files, data=data, timeout=600)
                
                if response.status_code == 200:
                    logger.info("上傳成功")
                    
                    # 解析 JSON 回應
                    result = response.json()
                    
                    if result.get("ok"):
                        return result["data"]["url"]
                    else:
                        raise Exception(f"API 返回失敗: {result}")
                else:
                    raise Exception(f"上傳失敗: {response.status_code} - {response.text}")
                    
        except Exception as e:
            logger.error(f"上傳錯誤: {e}")
            raise

if __name__ == "__main__":
    processor = InfiniteTalkProcessor()
    
    result = processor.process_and_upload(
        audio_path="examples/single/1.wav",
        input_path="examples/single/ref_image.png",
        prompt="A woman singing into a microphone in a recording studio",
        use_lora=True
    )
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
