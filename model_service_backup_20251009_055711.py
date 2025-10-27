"""
InfiniteTalk 模型服務 - 最終優化版
根據實測結果精心調整
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

# 設定環境變數優化顯存
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

QUALITY_PRESETS = {
    'turbo': {
        'resolution': '480',
        'sampling_steps': 4,
        'motion_frame': 4,
        'use_teacache': True,          # ← 啟用 TeaCache 加速
        'teacache_thresh': 0.3,        # ← 閾值 0.3
        'description': '⚡ 極速測試 480P (4步+4幀+TeaCache 15分) - 快速預覽'
    },
    'fast': {
        'resolution': '480',
        'sampling_steps': 6,
        'motion_frame': 7,
        'use_teacache': True,          # ← 啟用 TeaCache 加速
        'teacache_thresh': 0.3,
        'description': '⚡⚡ 快速生成 480P (6步+7幀+TeaCache 28分) - 測試可用'
    },
    'balanced': {
        'resolution': '480',
        'sampling_steps': 7,
        'motion_frame': 8,
        'use_teacache': False,         # ← 品質優先，不用 TeaCache
        'teacache_thresh': 1.0,
        'description': '⭐⭐⭐⭐⭐ 日常推薦 480P (7步+8幀 50分) - 品質與速度最佳平衡'
    },
    'high': {
        'resolution': '480',
        'sampling_steps': 8,
        'motion_frame': 9,
        'use_teacache': False,
        'teacache_thresh': 1.0,
        'description': '⭐⭐⭐⭐ 高品質 480P (8步+9幀 70分) - 細節豐富，手部清晰'
    },
    'ultra': {
        'resolution': '720',
        'sampling_steps': 8,
        'motion_frame': 9,
        'use_teacache': False,
        'teacache_thresh': 1.0,
        'description': '⭐⭐ 極致品質 720P (8步+9幀 120分) - 原生高清，需充足顯存'
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
        
        # 檢查 LoRA
        lora_exists = os.path.exists(lora_dir)
        logger.info("=" * 70)
        logger.info(f"🎨 LoRA 配置:")
        logger.info(f"   路徑: {lora_dir}")
        logger.info(f"   狀態: {'✅ 存在' if lora_exists else '❌ 不存在'}")
        if lora_exists:
            size_mb = os.path.getsize(lora_dir) / (1024*1024)
            logger.info(f"   大小: {size_mb:.1f} MB")
        logger.info("=" * 70)
        
        logger.info("📊 品質方案 (5檔精選):")
        logger.info("")
        logger.info("   ⚡ 快速測試:")
        logger.info("   ├─ Turbo    - 15分  (4步+4幀+TeaCache)  極速預覽")
        logger.info("   └─ Fast     - 28分  (6步+7幀+TeaCache)  測試可用")
        logger.info("")
        logger.info("   ⭐ 日常使用:")
        logger.info("   └─ Balanced - 50分  (7步+8幀)           品質速度平衡 ★推薦★")
        logger.info("")
        logger.info("   🎨 高品質:")
        logger.info("   ├─ High     - 70分  (8步+9幀)           細節豐富")
        logger.info("   └─ Ultra    - 120分 (720P 8步+9幀)      原生高清")
        logger.info("")
        logger.info("   💡 TeaCache: Turbo/Fast 啟用，可節省 30% 時間")
        logger.info("   💡 同一模型支援 480P/720P 切換")
        logger.info("=" * 70)
    
    def load_models(self):
        """載入模型（啟動時調用）"""
        if self.loaded:
            return
        
        logger.info("=" * 70)
        logger.info("🚀 載入模型（顯存優化模式）...")
        logger.info("=" * 70)
        
        try:
            # 預先清理顯存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            logger.info("📥 wav2vec2...")
            self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.wav2vec_dir
            )
            self.audio_encoder = Wav2Vec2Model.from_pretrained(
                self.wav2vec_dir
            ).to(self.device)
            logger.info("✅ wav2vec2")
            
            logger.info("📥 InfiniteTalk...")
            logger.info(f"   LoRA: {os.path.basename(self.lora_dir)}")
            logger.info(f"   Scale: 1.0")
            logger.info(f"   解析度: 480P / 720P")
            logger.info(f"   VRAM 管理: 啟用 (num_persistent=0)")
            cfg = WAN_CONFIGS['infinitetalk-14B']
            
            self.wan_i2v = wan.InfiniteTalkPipeline(
                config=cfg,
                checkpoint_dir=self.ckpt_dir,
                device_id=0,
                rank=0,
                lora_dir=[self.lora_dir],
                lora_scales=[1.0],
                infinitetalk_dir=self.infinitetalk_dir
            )
            
            # 啟用最激進的顯存管理
            self.wan_i2v.vram_management = True
            self.wan_i2v.enable_vram_management(num_persistent_param_in_dit=0)
            
            logger.info("✅ InfiniteTalk")
            self.loaded = True
            
            # 顯示顯存狀態
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                free = total - allocated
                logger.info(f"📊 GPU 顯存: {allocated:.1f}GB 使用中 / {free:.1f}GB 可用 / {total:.1f}GB 總量")
            
            logger.info("=" * 70)
            logger.info("🎉 模型已常駐!")
            logger.info("=" * 70)
            
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
    
    def generate(self, 
                 image_path, 
                 audio_path, 
                 prompt,
                 output_path,
                 quality='balanced',
                 **kwargs):
        """統一生成接口"""
        
        if not self.loaded:
            raise Exception("模型未載入！請確保已調用 load_models()")
        
        logger.info(f"🔍 品質: '{quality}'")
        
        if quality not in QUALITY_PRESETS:
            logger.warning(f"⚠️  未知品質 '{quality}'，使用 balanced")
            quality = 'balanced'
        
        preset = QUALITY_PRESETS[quality]
        
        # 提取參數
        resolution = preset['resolution']
        sample_steps = preset['sampling_steps']
        motion_frame = preset['motion_frame']
        use_teacache = preset['use_teacache']
        teacache_thresh = preset['teacache_thresh']
        
        logger.info(f"🎨 {quality}")
        logger.info(f"   {preset['description']}")
        logger.info(f"   參數: {resolution}P, {sample_steps}步, {motion_frame}幀" + 
                   (", TeaCache" if use_teacache else ""))
        
        # 生成前清理顯存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        try:
            # 音頻處理
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
            
            emb_path = os.path.join(temp_dir, 'audio_emb.pt')
            torch.save(audio_embedding, emb_path)
            
            input_clip = {
                'prompt': prompt,
                'cond_video': image_path,
                'cond_audio': {'person1': emb_path},
                'video_audio': sum_audio
            }
            
            extra_args = SimpleNamespace(
                use_teacache=use_teacache,
                teacache_thresh=teacache_thresh,
                use_apg=False,
                audio_mode='localfile',
                scene_seg=False,
                size=1.0
            )
            
            logger.info(f"🚀 開始生成 ({resolution}P)...")
            
            # 生成
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
            
            if hasattr(video, 'shape'):
                actual_frames = video.shape[2] if len(video.shape) > 2 else 0
                logger.info(f"📹 生成: {actual_frames}幀")
            
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
            
            # 清理
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            # 生成後清理顯存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            logger.info(f"✅ 完成: {final_path}")
            return final_path
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"❌ 顯存不足!")
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                logger.error(f"   當前使用: {allocated:.1f}GB")
            logger.error("💡 解決方案:")
            logger.error("   1. 使用較低品質 (turbo/fast/balanced)")
            logger.error("   2. 執行: nvidia-smi 查看其他進程")
            logger.error("   3. 停止不必要的 GPU 進程")
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
    print("✅ 服務就緒 - 5 檔精選配置 (TeaCache 加速 + 顯存優化)")
