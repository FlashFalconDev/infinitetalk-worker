"""
InfiniteTalk 模型服務 - 最終優化版 v7.1
支援字串預設配置 + 字典自訂配置
"""
import torch
import os
import sys
import logging
import soundfile as sf
import numpy as np
import subprocess
from types import SimpleNamespace

sys.path.insert(0, '/workspace/InfiniteTalk')

import wan
from src.audio_analysis.wav2vec2 import Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor
from wan.utils.multitalk_utils import save_video_ffmpeg
from wan.configs import WAN_CONFIGS
from generate_infinitetalk import audio_prepare_single, get_embedding

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

QUALITY_PRESETS = {
    'ultra_fast': {
        'resolution': '480',
        'sampling_steps': 3,
        'motion_frame': 3,
        'mode': 'streaming',
        'est_time': '12分',
        'description': '⚡⚡⚡ 超快速 480P (3步+3幀 12分) - 極限速度測試'
    },
    'turbo': {
        'resolution': '480',
        'sampling_steps': 4,
        'motion_frame': 4,
        'mode': 'streaming',
        'est_time': '18分',
        'description': '⚡⚡ 極速 480P (4步+4幀 18分) - 快速預覽'
    },
    'fast': {
        'resolution': '480',
        'sampling_steps': 6,
        'motion_frame': 7,
        'mode': 'streaming',
        'est_time': '35分',
        'description': '⚡ 快速 480P (6步+7幀 35分) - 測試推薦 ★'
    },
    'balanced': {
        'resolution': '480',
        'sampling_steps': 7,
        'motion_frame': 8,
        'mode': 'streaming',
        'est_time': '50分',
        'description': '⭐⭐⭐⭐⭐ 日常推薦 480P (7步+8幀 50分) - 最佳平衡 ★★★'
    },
    'high': {
        'resolution': '480',
        'sampling_steps': 8,
        'motion_frame': 9,
        'mode': 'streaming',
        'est_time': '70分',
        'description': '⭐⭐⭐⭐ 高品質 480P (8步+9幀 70分) - 細節豐富'
    },
    'ultra': {
        'resolution': '720',
        'sampling_steps': 8,
        'motion_frame': 9,
        'mode': 'streaming',
        'est_time': '120分',
        'description': '⭐⭐ 極致品質 720P (8步+9幀 120分) - 原生高清'
    }
}

class InfiniteTalkModelService:
    def __init__(self, 
                 ckpt_dir='weights/Wan2.1-I2V-14B-480P',
                 wav2vec_dir='weights/chinese-wav2vec2-base',
                 infinitetalk_dir='weights/InfiniteTalk/single/infinitetalk.safetensors',
                 lora_dir='weights/Wan2.1_I2V_14B_FusionX_LoRA.safetensors',
                 device='cuda'):
        
        self.ckpt_dir = ckpt_dir
        self.wav2vec_dir = wav2vec_dir
        self.infinitetalk_dir = infinitetalk_dir
        self.lora_dir = lora_dir
        self.device = device
        
        self.loaded = False
        self.wan_i2v = None
        self.wav2vec_feature_extractor = None
        self.audio_encoder = None
        
        lora_exists = os.path.exists(lora_dir)
        logger.info("=" * 80)
        logger.info(f"🎨 LoRA 配置:")
        logger.info(f"   路徑: {lora_dir}")
        if lora_exists:
            size_mb = os.path.getsize(lora_dir) / (1024*1024)
            logger.info(f"   狀態: ✅ 已載入")
            logger.info(f"   大小: {size_mb:.1f} MB")
        else:
            logger.info(f"   狀態: ⚪ 未使用（選用功能，不影響基礎生成）")
        logger.info("=" * 80)
        
        logger.info("📊 品質方案 (6 檔精選 - 最終優化版 v7.1)")
        logger.info("")
        logger.info("   ⚡ 快速測試 (降低 steps/frames):")
        logger.info("   ├─ Ultra Fast - 12分  (3步+3幀)           極限速度")
        logger.info("   ├─ Turbo      - 18分  (4步+4幀)           快速預覽")
        logger.info("   └─ Fast       - 35分  (6步+7幀)           測試推薦 ★")
        logger.info("")
        logger.info("   ⭐ 日常使用:")
        logger.info("   └─ Balanced   - 50分  (7步+8幀)           品質速度最佳 ★★★")
        logger.info("")
        logger.info("   🎨 高品質:")
        logger.info("   ├─ High       - 70分  (8步+9幀)           細節豐富")
        logger.info("   └─ Ultra      - 120分 (720P 8步+9幀)      原生高清")
        logger.info("")
        logger.info("   💡 支援自訂配置: 傳入字典即可覆蓋預設參數")
        logger.info("=" * 80)
    
    def load_models(self):
        """載入模型（啟動時調用）"""
        if self.loaded:
            return
        
        logger.info("=" * 80)
        logger.info("🚀 載入模型（最終優化版 v7.1）...")
        logger.info("=" * 80)
        
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            logger.info("📥 載入 wav2vec2...")
            self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.wav2vec_dir
            )
            # Force eager attention to support output_attentions=True
            self.audio_encoder = Wav2Vec2Model.from_pretrained(
                self.wav2vec_dir,
                attn_implementation='eager'
            ).to(self.device)
            logger.info("✅ wav2vec2 完成")
            
            logger.info("📥 載入 InfiniteTalk...")
            logger.info(f"   LoRA: {os.path.basename(self.lora_dir)}")
            logger.info(f"   LoRA Scale: 1.0")
            logger.info(f"   Text Guide: 1.0 (使用 LoRA)")
            logger.info(f"   Audio Guide: 2.0 (使用 LoRA)")
            logger.info(f"   解析度: 480P / 720P")
            logger.info(f"   加速方式: 降低 steps/frames（穩定可靠）")
            logger.info(f"   VRAM 管理: 啟用 (num_persistent=0)")
            
            cfg = WAN_CONFIGS['infinitetalk-14B']

            # 確保 LoRA 檔案存在（必須使用）
            # LoRA 下載來源: https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX
            # 檔案: Wan2.1_I2V_14B_FusionX_LoRA.safetensors (353.9 MB)
            if not os.path.exists(self.lora_dir):
                raise FileNotFoundError(
                    f"❌ LoRA 檔案不存在: {self.lora_dir}\n"
                    f"請先下載 LoRA 檔案到 weights/ 目錄\n"
                    f"下載來源: https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX\n"
                    f"檔案名稱: Wan2.1_I2V_14B_FusionX_LoRA.safetensors (353.9 MB)"
                )

            self.wan_i2v = wan.InfiniteTalkPipeline(
                config=cfg,
                checkpoint_dir=self.ckpt_dir,
                device_id=0,
                rank=0,
                lora_dir=[self.lora_dir],
                lora_scales=[1.0],
                infinitetalk_dir=self.infinitetalk_dir
            )
            
            self.wan_i2v.vram_management = True
            self.wan_i2v.enable_vram_management(num_persistent_param_in_dit=0)
            
            logger.info("✅ InfiniteTalk 完成")
            self.loaded = True
            
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                free = total - allocated
                logger.info(f"📊 GPU 顯存: {allocated:.1f}GB 使用 / {free:.1f}GB 可用 / {total:.1f}GB 總量")
            
            logger.info("=" * 80)
            logger.info("🎉 模型已常駐！ (最終優化版 v7.1)")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ 載入失敗: {e}")
            raise
    
    def extend_audio(self, audio_path, motion_frame=9):
        """延長音頻"""
        try:
            audio_data, sample_rate = sf.read(audio_path)
            min_duration = (motion_frame / 8.0) + 0.5
            min_samples = int(min_duration * sample_rate)
            current_duration = len(audio_data) / sample_rate
            
            logger.info(f"🎵 音頻: {current_duration:.2f}秒")
            
            if len(audio_data) < min_samples:
                repeat_times = int(np.ceil(min_samples / len(audio_data)))
                audio_data = np.tile(audio_data, repeat_times)[:min_samples]
                sf.write(audio_path, audio_data, sample_rate)
                logger.info(f"✅ 延長至 {len(audio_data)/sample_rate:.2f}秒")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 音頻處理失敗: {e}")
            return False
    
    def _parse_quality_config(self, quality):
        """
        解析品質配置
        支援兩種格式:
        1. 字串: 'fast', 'balanced' 等預設配置
        2. 字典: 自訂配置參數
        
        返回: (preset_dict, config_type, config_name)
        """
        # 判斷是字典還是字串
        if isinstance(quality, dict):
            logger.info("🔧 使用自訂配置（字典格式）")
            
            # 必要參數檢查
            required_params = ['sampling_steps', 'motion_frame']
            missing_params = [p for p in required_params if p not in quality]
            
            if missing_params:
                raise ValueError(f"❌ 自訂配置缺少必要參數: {missing_params}")
            
            # 設定預設值
            preset = {
                'resolution': quality.get('resolution', '480'),
                'sampling_steps': quality['sampling_steps'],
                'motion_frame': quality['motion_frame'],
                'mode': quality.get('mode', 'streaming'),
                'est_time': quality.get('est_time', '未知'),
                'description': quality.get('description', '自訂配置')
            }
            
            return preset, 'custom', 'custom'
            
        elif isinstance(quality, str):
            logger.info(f"📋 使用預設配置: '{quality}'")
            
            if quality not in QUALITY_PRESETS:
                logger.warning(f"⚠️  未知品質 '{quality}'，使用 balanced")
                quality = 'balanced'
            
            return QUALITY_PRESETS[quality], 'preset', quality
            
        else:
            raise TypeError(f"❌ quality 參數類型錯誤: {type(quality)}，應為 str 或 dict")
    
    def generate(self, 
                 image_path, 
                 audio_path, 
                 prompt,
                 output_path,
                 quality='balanced',
                 **kwargs):
        """
        統一生成接口
        
        參數:
            quality: str 或 dict
                - str: 預設配置名稱 ('ultra_fast', 'turbo', 'fast', 'balanced', 'high', 'ultra')
                - dict: 自訂配置，必須包含:
                    - sampling_steps (必要): int, 採樣步數
                    - motion_frame (必要): int, 動作幀數
                    - resolution (可選): str, '480' 或 '720', 預設 '480'
                    - mode (可選): str, 預設 'streaming'
                    - est_time (可選): str, 預估時間說明
                    - description (可選): str, 配置描述
        
        範例:
            # 使用預設配置
            generate(..., quality='fast')
            
            # 使用自訂配置
            generate(..., quality={
                'sampling_steps': 5,
                'motion_frame': 6,
                'resolution': '480',
                'description': '自訂快速配置'
            })
        """
        
        if not self.loaded:
            raise Exception("模型未載入！請確保已調用 load_models()")
        
        # 解析配置
        try:
            preset, config_type, config_name = self._parse_quality_config(quality)
        except (ValueError, TypeError) as e:
            logger.error(str(e))
            raise
        
        # 提取參數
        resolution = preset['resolution']
        sample_steps = preset['sampling_steps']
        motion_frame = preset['motion_frame']
        mode = preset['mode']
        est_time = preset.get('est_time', '未知')
        description = preset.get('description', '自訂配置')
        
        # 顯示配置資訊
        logger.info("=" * 80)
        if config_type == 'custom':
            logger.info(f"🎨 自訂配置")
        else:
            logger.info(f"🎨 {config_name.upper()}")
        logger.info(f"   {description}")
        logger.info(f"   參數: {resolution}P, {sample_steps}步, {motion_frame}幀, mode={mode}")
        logger.info(f"   預估時間: {est_time}")
        logger.info("=" * 80)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        try:
            if not self.extend_audio(audio_path, motion_frame):
                raise Exception("音頻處理失敗")
            
            temp_dir = f"temp_gen_{os.path.basename(output_path)}"
            os.makedirs(temp_dir, exist_ok=True)
            
            logger.info("🎵 處理音頻...")
            human_speech = audio_prepare_single(audio_path)
            sum_audio = os.path.join(temp_dir, 'audio.wav')
            sf.write(sum_audio, human_speech, 16000)
            
            audio_duration = len(human_speech) / 16000
            
            logger.info("🎤 提取特徵...")
            audio_embedding = get_embedding(
                human_speech, 
                self.wav2vec_feature_extractor, 
                self.audio_encoder,
                device='cuda'
            )
            
            actual_audio_frames = audio_embedding.shape[0]
            max_frames = actual_audio_frames + 20
            
            logger.info(f"📊 音頻: {audio_duration:.2f}秒, {actual_audio_frames}幀")
            logger.info(f"📊 最大幀數: {max_frames} (約 {max_frames/25:.1f}秒)")
            
            emb_path = os.path.join(temp_dir, 'audio_emb.pt')
            torch.save(audio_embedding, emb_path)
            
            input_clip = {
                'prompt': prompt,
                'cond_video': image_path,
                'cond_audio': {'person1': emb_path},
                'video_audio': sum_audio
            }
            
            # 簡潔穩定的配置
            extra_args = SimpleNamespace(
                use_teacache=False,
                teacache_thresh=1.0,
                use_apg=False,
                audio_mode='localfile',
                scene_seg=False,
                size=1.0
            )
            
            logger.info(f"🚀 開始生成 ({resolution}P, {mode} 模式)...")
            
            import time
            start_time = time.time()
            
            video = self.wan_i2v.generate_infinitetalk(
                input_clip,
                size_buckget=f'infinitetalk-{resolution}',
                motion_frame=motion_frame,
                shift=2,
                sampling_steps=sample_steps,
                text_guide_scale=1.0,
                audio_guide_scale=2.0,
                seed=0,
                offload_model=False,
                max_frames_num=max_frames,
                color_correction_strength=0.0,
                extra_args=extra_args
            )
            
            elapsed_time = time.time() - start_time
            
            if hasattr(video, 'shape'):
                actual_frames = video.shape[2] if len(video.shape) > 2 else 0
                logger.info(f"📹 生成: {actual_frames}幀 ({actual_frames/25:.1f}秒)")
            
            logger.info(f"⏱️  實際耗時: {elapsed_time/60:.1f}分鐘 (預估: {est_time})")
            
            logger.info("💾 保存...")
            save_video_ffmpeg(video, output_path, [sum_audio], high_quality_save=False)
            
            final_path = f"{output_path}.mp4"
            
            if os.path.exists(final_path):
                result = subprocess.run(
                    ['ffprobe', '-v', 'error', '-show_entries', 
                     'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', 
                     final_path],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    video_duration = float(result.stdout.strip())
                    logger.info(f"✅ 視頻: {video_duration:.2f}秒 / 音頻: {audio_duration:.2f}秒")
            
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            logger.info(f"✅ 完成: {final_path}")
            if config_type == 'custom':
                logger.info(f"📊 性能統計: 自訂配置 ({sample_steps}步+{motion_frame}幀)，{elapsed_time/60:.1f}分鐘")
            else:
                logger.info(f"📊 性能統計: {config_name} 模式，{elapsed_time/60:.1f}分鐘")
            
            return final_path
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"❌ 顯存不足!")
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                logger.error(f"   當前使用: {allocated:.1f}GB")
            logger.error("💡 解決方案:")
            logger.error("   1. 使用更低品質: ultra_fast/turbo/fast")
            logger.error("   2. 降低解析度: ultra → balanced (720P → 480P)")
            logger.error("   3. 降低自訂參數: sampling_steps 和 motion_frame")
            logger.error("   4. 檢查其他進程: nvidia-smi")
            torch.cuda.empty_cache()
            raise
        except Exception as e:
            logger.error(f"❌ 生成失敗: {e}")
            import traceback
            traceback.print_exc()
            raise

_model_instance = None

def get_model_service():
    global _model_instance
    if _model_instance is None:
        _model_instance = InfiniteTalkModelService()
        _model_instance.load_models()
    return _model_instance

if __name__ == "__main__":
    service = get_model_service()
    print("✅ 服務就緒 - 最終優化版 v7.1")
    print("")
    print("📊 使用方式:")
    print("")
    print("1️⃣  預設配置（字串）:")
    print("   service.generate(..., quality='fast')")
    print("")
    print("2️⃣  自訂配置（字典）:")
    print("   service.generate(..., quality={")
    print("       'sampling_steps': 5,")
    print("       'motion_frame': 6,")
    print("       'resolution': '480',  # 可選")
    print("       'description': '我的配置'  # 可選")
    print("   })")
    print("")
    print("📋 預設配置:")
    print("   ultra_fast / turbo / fast / balanced / high / ultra")
