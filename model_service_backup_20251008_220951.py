"""
InfiniteTalk 模型服務 - 穩定版（FFmpeg 增強型升頻）
品質方案：
- Standard: 480P (快速) - 70分鐘
- Premium: 480P (高品質) - 120分鐘  
- Ultra: 720P (Premium + 增強升頻) - 135分鐘
- Supreme: 1080P (Premium + 兩段增強升頻) - 155分鐘
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

QUALITY_PRESETS = {
    'standard': {
        'resolution': '480',
        'sampling_steps': 6,
        'motion_frame': 7,
        'use_teacache': False,
        'teacache_thresh': 1.0,
        'upscale': False,
        'description': '標準品質 480P (快速)'
    },
    'premium': {
        'resolution': '480',
        'sampling_steps': 8,
        'motion_frame': 9,
        'use_teacache': False,
        'teacache_thresh': 1.0,
        'upscale': False,
        'description': '高品質 480P'
    },
    'ultra': {
        'resolution': '480',
        'sampling_steps': 8,
        'motion_frame': 9,
        'use_teacache': False,
        'teacache_thresh': 1.0,
        'upscale': True,
        'target_resolution': '720',
        'upscale_stages': 1,
        'ffmpeg_preset': 'veryslow',
        'description': '超高品質 720P (Premium + 增強升頻)'
    },
    'supreme': {
        'resolution': '480',
        'sampling_steps': 8,
        'motion_frame': 9,
        'use_teacache': False,
        'teacache_thresh': 1.0,
        'upscale': True,
        'target_resolution': '1080',
        'upscale_stages': 2,
        'ffmpeg_preset': 'veryslow',
        'description': '極致品質 1080P (Premium + 兩段增強升頻)'
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
        
        self._check_ffmpeg()
        
    def _check_ffmpeg(self):
        """檢查 FFmpeg 是否可用"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info("=" * 70)
                logger.info("📊 升頻方案: FFmpeg 增強型")
                logger.info("   使用多重濾鏡組合（Lanczos + 銳化 + 降噪）")
                logger.info("   品質接近 AI 升頻，但更穩定可靠")
                logger.info("=" * 70)
                return True
        except Exception as e:
            logger.error(f"❌ FFmpeg 不可用: {e}")
            raise
    
    def upscale_video_ffmpeg(self, input_path, output_path, target_height=720, preset='veryslow'):
        """使用 FFmpeg 增強型升頻"""
        try:
            logger.info(f"🔍 使用 FFmpeg 增強型升頻到 {target_height}P...")
            logger.info(f"   預設: {preset} (品質優先)")
            
            # 多重濾鏡組合提升品質
            filters = [
                f'scale=-1:{target_height}:flags=lanczos',  # Lanczos 升頻
                'unsharp=5:5:1.0:5:5:0.0',                   # 銳化
                'hqdn3d=1.5:1.5:6:6'                         # 降噪
            ]
            
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-vf', ','.join(filters),
                '-c:v', 'libx264',
                '-preset', preset,
                '-crf', '16',  # 低 CRF = 高品質
                '-c:a', 'copy',
                '-y',
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0 and os.path.exists(output_path):
                logger.info(f"✅ FFmpeg 升頻完成")
                return True
            else:
                logger.error(f"❌ FFmpeg 升頻失敗: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ FFmpeg 升頻錯誤: {e}")
            return False
    
    def upscale_video_two_stage(self, input_path, output_path, preset='veryslow'):
        """兩段式升頻：480P → 720P → 1080P"""
        try:
            logger.info(f"📊 兩段式升頻: 480P → 720P → 1080P")
            
            # 第一段：480P → 720P (高品質)
            temp_720p = input_path.replace('.mp4', '_temp_720p.mp4')
            logger.info(f"   🎬 第一段: 480P → 720P (增強品質)")
            
            if not self.upscale_video_ffmpeg(input_path, temp_720p, 720, preset):
                logger.error("   ❌ 第一段升頻失敗")
                return False
            
            logger.info(f"   ✅ 第一段完成")
            
            # 第二段：720P → 1080P (輕度放大)
            logger.info(f"   🎬 第二段: 720P → 1080P (輕度放大)")
            
            if not self.upscale_video_ffmpeg(temp_720p, output_path, 1080, 'medium'):
                logger.error("   ❌ 第二段失敗")
                if os.path.exists(temp_720p):
                    os.remove(temp_720p)
                return False
            
            # 清理
            if os.path.exists(temp_720p):
                os.remove(temp_720p)
            
            logger.info(f"   ✅ 第二段完成")
            logger.info(f"✅ 兩段式升頻完成！")
            return True
                
        except Exception as e:
            logger.error(f"❌ 兩段式升頻錯誤: {e}")
            return False
    
    def load_models(self):
        if self.loaded:
            return
        
        logger.info("=" * 70)
        logger.info("🚀 載入模型...")
        logger.info("=" * 70)
        
        try:
            logger.info("📥 wav2vec2...")
            self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.wav2vec_dir
            )
            self.audio_encoder = Wav2Vec2Model.from_pretrained(
                self.wav2vec_dir
            ).to(self.device)
            logger.info("✅ wav2vec2")
            
            logger.info("📥 InfiniteTalk...")
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
            
            self.wan_i2v.vram_management = True
            self.wan_i2v.enable_vram_management(num_persistent_param_in_dit=0)
            
            logger.info("✅ InfiniteTalk")
            self.loaded = True
            logger.info("=" * 70)
            logger.info("🎉 模型已常駐!")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"❌ 失敗: {e}")
            raise
    
    def extend_audio(self, audio_path, motion_frame=9):
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
                logger.info(f"✅ 已延長音頻至 {len(audio_data)/sample_rate:.2f}秒")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 音頻處理失敗: {e}")
            return False
    
    def generate(self, 
                 image_path, 
                 audio_path, 
                 prompt,
                 output_path,
                 quality='standard',
                 resolution=None,
                 sample_steps=None,
                 motion_frame=None):
        if not self.loaded:
            self.load_models()
        
        logger.info(f"🔍 收到品質: '{quality}'")
        
        if quality in QUALITY_PRESETS:
            preset = QUALITY_PRESETS[quality]
            resolution = resolution or preset['resolution']
            sample_steps = sample_steps or preset['sampling_steps']
            motion_frame = motion_frame or preset['motion_frame']
            use_teacache = preset['use_teacache']
            teacache_thresh = preset['teacache_thresh']
            need_upscale = preset.get('upscale', False)
            target_resolution = preset.get('target_resolution', '480')
            upscale_stages = preset.get('upscale_stages', 1)
            ffmpeg_preset = preset.get('ffmpeg_preset', 'slow')
            
            logger.info(f"🎨 使用品質: {quality}")
            logger.info(f"   說明: {preset['description']}")
            logger.info(f"   生成解析度: {resolution}P")
            logger.info(f"   採樣步數: {sample_steps}")
            logger.info(f"   動作幀: {motion_frame}")
            if need_upscale:
                logger.info(f"   ✨ 將升頻至: {target_resolution}P")
                logger.info(f"   📊 升頻階段: {upscale_stages} 段")
        else:
            logger.warning(f"⚠️ 品質 '{quality}' 不在預設中，使用 standard")
            preset = QUALITY_PRESETS['standard']
            resolution = preset['resolution']
            sample_steps = preset['sampling_steps']
            motion_frame = preset['motion_frame']
            use_teacache = preset['use_teacache']
            teacache_thresh = preset['teacache_thresh']
            need_upscale = False
            ffmpeg_preset = 'slow'
        
        logger.info(f"🎬 {output_path}")
        
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
            
            logger.info(f"📊 音頻時長: {audio_duration:.2f}秒")
            logger.info(f"📊 Audio embedding 幀數: {actual_audio_frames}")
            logger.info(f"📊 設定最大幀數: {max_frames}")
            
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
            
            logger.info(f"🚀 開始生成 (steps={sample_steps}, motion={motion_frame})...")
            
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
                actual_duration = actual_frames / 8.0
                logger.info(f"📹 生成: {actual_frames} 幀, {actual_duration:.2f}秒")
            
            logger.info("💾 保存 480P...")
            save_video_ffmpeg(video, output_path, [sum_audio], high_quality_save=False)
            
            video_480p_path = f"{output_path}.mp4"
            final_path = video_480p_path
            
            # 升頻處理
            if need_upscale and os.path.exists(video_480p_path):
                logger.info(f"✨ 開始升頻到 {target_resolution}P...")
                
                video_final_path = f"{output_path}_{target_resolution}p.mp4"
                
                if upscale_stages == 2:
                    # 兩段式
                    if self.upscale_video_two_stage(video_480p_path, video_final_path, ffmpeg_preset):
                        final_path = video_final_path
                        logger.info(f"✅ 兩段式升頻完成: {video_final_path}")
                    else:
                        logger.warning("⚠️  升頻失敗，使用 480P 原檔")
                else:
                    # 單段
                    if self.upscale_video_ffmpeg(video_480p_path, video_final_path, 
                                                  int(target_resolution), ffmpeg_preset):
                        final_path = video_final_path
                        logger.info(f"✅ 升頻完成: {video_final_path}")
                    else:
                        logger.warning("⚠️  升頻失敗，使用 480P 原檔")
            
            # 驗證
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
                    logger.info(f"✅ 最終視頻: {video_duration:.2f}秒 / 音頻: {audio_duration:.2f}秒")
                    
                    if abs(video_duration - audio_duration) > 2:
                        logger.warning(f"⚠️  長度不符! 差距: {abs(video_duration - audio_duration):.2f}秒")
            
            # 清理
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            logger.info(f"✅ 完成: {final_path}")
            return final_path
            
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
    print("✅ 服務已就緒（使用 FFmpeg 增強型升頻）")
