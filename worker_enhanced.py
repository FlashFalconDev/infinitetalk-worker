"""
InfiniteTalk Worker - 增強版
支援完整的 GPU 監控資訊
"""
import requests
import json
import os
import uuid
import time
import logging
import socket
import threading
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from model_service import get_model_service

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InfiniteTalkWorker:
    def __init__(self):
        # ===== 配置讀取 =====
        self.base_url = os.getenv('INFINITETALK_API_BASE', 'https://host.flashfalcon.info')
        self.worker_token = os.getenv('INFINITETALK_WORKER_TOKEN')
        
        if not self.worker_token:
            logger.error("=" * 70)
            logger.error("❌ 缺少環境變數: INFINITETALK_WORKER_TOKEN")
            logger.error("")
            logger.error("請執行以下步驟:")
            logger.error("1. 複製範例: cp .env.example .env")
            logger.error("2. 編輯檔案: nano .env")
            logger.error("3. 填入從 Admin 後台複製的 Token")
            logger.error("=" * 70)
            raise ValueError("Missing INFINITETALK_WORKER_TOKEN")
        
        # API 端點
        self.heartbeat_api = f"{self.base_url}/ai/api/worker/heartbeat"
        self.task_api = f"{self.base_url}/aigen/api/pending_task/"
        self.result_api = f"{self.base_url}/aigen/api/task_result/"
        self.upload_api = f"{self.base_url}/api/save_file/"
        
        # Worker 資訊
        self.worker_id = self._generate_worker_id()
        self.worker_version = "7.2"  # 增強版本號
        
        # 目錄設定
        self.temp_dir = "temp_downloads"
        self.output_dir = "outputs"
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # ✅ 檢查 GPU 監控能力
        self.gpu_monitoring_available = self._check_gpu_monitoring()
        
        # 初始化
        logger.info("=" * 70)
        logger.info("🚀 初始化 InfiniteTalk Worker v7.2 (增強版)")
        logger.info(f"🆔 Worker ID: {self.worker_id}")
        logger.info(f"🔑 Token: {self.worker_token[:10]}...{self.worker_token[-10:]}")
        logger.info(f"🌐 API Base: {self.base_url}")
        logger.info(f"📊 GPU 監控: {'✅ 已啟用' if self.gpu_monitoring_available else '⚠️  基本模式'}")
        logger.info("=" * 70)
        
        # 測試連線
        if not self._test_connection():
            raise ConnectionError("❌ 無法連接到伺服器，請檢查 Token 是否正確")
        
        # 載入模型
        logger.info("📥 載入模型（只執行一次）...")
        self.model_service = get_model_service()
        
        # 啟動心跳線程
        self._start_heartbeat_thread()
        
        logger.info("=" * 70)
        logger.info("✅ Worker 準備就緒!")
        logger.info("=" * 70)
    
    def _generate_worker_id(self):
        """生成 Worker ID"""
        if os.getenv('WORKER_ID'):
            return os.getenv('WORKER_ID')
        
        hostname = socket.gethostname()
        short_uuid = str(uuid.uuid4())[:8]
        return f"{hostname}-{short_uuid}"
    
    def _check_gpu_monitoring(self):
        """檢查是否可以使用詳細的 GPU 監控"""
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            pynvml.nvmlShutdown()
            logger.info(f"✅ GPU 監控已啟用 (偵測到 {device_count} 個 GPU)")
            return True
        except ImportError:
            logger.warning("⚠️  未安裝 nvidia-ml-py3，使用基本 GPU 監控")
            logger.info("   安裝方式: pip install nvidia-ml-py3")
            return False
        except Exception as e:
            logger.warning(f"⚠️  無法初始化 GPU 監控: {e}")
            return False
    
    def _get_auth_headers(self):
        """獲取認證 Headers"""
        return {
            'Authorization': f'Bearer {self.worker_token}',
            'Content-Type': 'application/json'
        }
    
    def _get_system_info(self):
        """獲取系統資訊（首次連線時發送）"""
        info = {
            'hostname': socket.gethostname(),
            'version': self.worker_version
        }
        
        # GPU 基本資訊
        try:
            import torch
            if torch.cuda.is_available():
                gpu_props = torch.cuda.get_device_properties(0)
                info['gpu_info'] = {
                    'name': torch.cuda.get_device_name(0),
                    'count': torch.cuda.device_count(),
                    'total_memory_gb': round(gpu_props.total_memory / 1024**3, 2),
                    'cuda_version': torch.version.cuda,
                    'pytorch_version': torch.__version__
                }
                
                # ✅ 如果有詳細監控，加入驅動版本
                if self.gpu_monitoring_available:
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        driver_version = pynvml.nvmlSystemGetDriverVersion()
                        info['gpu_info']['driver_version'] = driver_version
                        pynvml.nvmlShutdown()
                    except:
                        pass
                
        except Exception as e:
            logger.warning(f"無法獲取 GPU 資訊: {e}")
        
        return info
    
    def _get_gpu_stats(self):
        """✅ 增強版：獲取詳細的 GPU 狀態"""
        stats = {}
        
        try:
            import torch
            if not torch.cuda.is_available():
                return stats
            
            # 基本記憶體資訊（PyTorch）
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            stats['gpu_memory_used'] = round(allocated, 2)
            stats['gpu_memory_reserved'] = round(reserved, 2)
            stats['gpu_memory_total'] = round(total, 2)
            stats['gpu_memory_utilization'] = round((allocated / total) * 100, 2)
            
            # ✅ 詳細資訊（nvidia-ml-py3）
            if self.gpu_monitoring_available:
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    
                    # GPU 使用率
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    stats['gpu_utilization'] = util.gpu
                    stats['gpu_memory_controller_utilization'] = util.memory
                    
                    # GPU 溫度
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    stats['gpu_temperature'] = temp
                    
                    # GPU 功率
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                        power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                        stats['gpu_power_usage'] = round(power, 2)
                        stats['gpu_power_limit'] = round(power_limit, 2)
                        stats['gpu_power_utilization'] = round((power / power_limit) * 100, 2)
                    except:
                        pass
                    
                    # GPU 時鐘頻率
                    try:
                        clock_graphics = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                        clock_memory = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                        stats['gpu_clock_graphics_mhz'] = clock_graphics
                        stats['gpu_clock_memory_mhz'] = clock_memory
                    except:
                        pass
                    
                    # GPU 風扇轉速
                    try:
                        fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                        stats['gpu_fan_speed'] = fan_speed
                    except:
                        pass
                    
                    # GPU 進程資訊
                    try:
                        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                        stats['gpu_process_count'] = len(processes)
                    except:
                        pass
                    
                    pynvml.nvmlShutdown()
                    
                except Exception as e:
                    logger.debug(f"詳細 GPU 監控失敗: {e}")
            
        except Exception as e:
            logger.debug(f"獲取 GPU 狀態失敗: {e}")
        
        return stats
    
    def _test_connection(self):
        """測試連線"""
        try:
            logger.info("🔌 測試連線...")
            
            headers = self._get_auth_headers()
            data = {
                'worker_id': self.worker_id,
                'status': 'online',
                'timestamp': datetime.now().isoformat(),
                **self._get_system_info()
            }
            
            response = requests.post(
                self.heartbeat_api,
                json=data,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    logger.info("✅ 連線成功")
                    if 'data' in result and 'worker_name' in result['data']:
                        logger.info(f"   後端識別為: {result['data']['worker_name']}")
                    return True
            elif response.status_code == 401:
                logger.error("❌ Token 無效或已停用")
                return False
            else:
                logger.error(f"❌ 連線失敗: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            logger.error(f"❌ 無法連接到 {self.base_url}")
            return False
        except Exception as e:
            logger.error(f"❌ 連線測試失敗: {e}")
            return False
    
    def _send_heartbeat(self):
        """發送心跳（包含詳細 GPU 資訊）"""
        try:
            headers = self._get_auth_headers()
            
            # ✅ 收集所有資訊
            data = {
                'worker_id': self.worker_id,
                'status': 'online',
                'timestamp': datetime.now().isoformat(),
                'version': self.worker_version,
                **self._get_gpu_stats()  # 包含所有 GPU 狀態
            }
            
            # 調試：顯示發送的資料（只在第一次）
            if not hasattr(self, '_first_heartbeat_logged'):
                logger.info("📊 首次心跳資料:")
                for key, value in data.items():
                    if key not in ['worker_id', 'status', 'timestamp']:
                        logger.info(f"   {key}: {value}")
                self._first_heartbeat_logged = True
            
            response = requests.post(
                self.heartbeat_api,
                json=data,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.debug(f"💓 心跳發送成功")
                
                # 顯示關鍵資訊
                gpu_stats = self._get_gpu_stats()
                if gpu_stats:
                    logger.debug(
                        f"   GPU: {gpu_stats.get('gpu_memory_used', 0):.1f}GB / "
                        f"{gpu_stats.get('gpu_utilization', 0):.0f}% / "
                        f"{gpu_stats.get('gpu_temperature', 0):.0f}°C"
                    )
                
                return True
            elif response.status_code == 401:
                logger.error(f"❌ Token 已失效")
                return False
            else:
                logger.warning(f"⚠️  心跳失敗: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️  心跳錯誤: {e}")
            return False
    
    def _heartbeat_loop(self):
        """心跳循環"""
        while True:
            try:
                self._send_heartbeat()
                time.sleep(60)
            except Exception as e:
                logger.error(f"心跳循環錯誤: {e}")
                time.sleep(60)
    
    def _start_heartbeat_thread(self):
        """啟動心跳線程"""
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="HeartbeatThread"
        )
        heartbeat_thread.start()
        logger.info("💓 心跳線程已啟動（每 60 秒）")
    
    def fetch_task(self):
        """獲取待處理任務"""
        try:
            logger.info("查詢待處理任務...")
            
            params = {'model_code': 'InfiniteTalk_S2V'}
            headers = self._get_auth_headers()
            
            response = requests.get(
                self.task_api,
                params=params,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 401:
                logger.error("❌ Token 已失效")
                return []
            
            if response.status_code == 403:
                result = response.json()
                logger.error(f"❌ {result.get('error', 'Permission denied')}")
                return []
            
            if response.status_code == 200:
                result = response.json()
                
                if result is None or not isinstance(result, dict):
                    return []
                
                if not result.get("success"):
                    return []
                
                if "data" not in result:
                    return []
                
                data = result["data"]
                
                if data is None or not isinstance(data, dict):
                    return []
                
                if "Generation_Video_task" not in data:
                    return []
                
                tasks = data["Generation_Video_task"]
                
                if tasks is None or not isinstance(tasks, list):
                    return []
                
                if len(tasks) > 0:
                    logger.info(f"✅ 獲取到 {len(tasks)} 個任務")
                    return tasks
                
                return []
            else:
                logger.error(f"獲取任務失敗: HTTP {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"獲取任務錯誤: {e}")
            return []
    
    def download_file(self, url, save_path):
        """下載檔案"""
        try:
            logger.info(f"📥 下載: {url}")
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
        """生成影片"""
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
            import traceback
            logger.error(traceback.format_exc())
            return None, 0
    
    def upload_video(self, video_path, task_id):
        """上傳影片"""
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
        """回報結果"""
        try:
            logger.info(f"📮 回報結果 - 任務PK: {task_pk}")
            
            data = {
                "video_generation_image_audio_pk": task_pk,
                "video_url": video_url,
                "worker_id": self.worker_id,
                "quality": quality,
                "generation_time": generation_time
            }
            
            headers = self._get_auth_headers()
            
            response = requests.post(
                self.result_api,
                json=data,
                headers=headers,
                timeout=30
            )
            
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
        """清理臨時檔案"""
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
        """處理任務"""
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
        """運行 Worker"""
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
                import traceback
                logger.error(traceback.format_exc())
                time.sleep(poll_interval)

if __name__ == "__main__":
    worker = InfiniteTalkWorker()
    worker.run(poll_interval=30)
