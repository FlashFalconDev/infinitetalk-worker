"""
InfiniteTalk 模型服務 - 修復長視頻生成
"""
import torch
import os
import sys
import logging
import soundfile as sf
import numpy as np
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
        
    def load_models(self):
        """載入模型"""
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
        """延長音頻以滿足 motion_frame 需求"""
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
                 resolution='480',
                 sample_steps=8,
                 motion_frame=9):
        """生成影片"""
        if not self.loaded:
            raise Exception("模型未載入")
        
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
            
            # 關鍵修改：根據 audio embedding 的實際長度設置 max_frames
            actual_audio_frames = audio_embedding.shape[0]
            max_frames = actual_audio_frames + 20  # 加一些緩衝
            
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
                use_teacache=False,
                teacache_thresh=1.0,
                use_apg=False,
                audio_mode='localfile',
                scene_seg=False
            )
            
            logger.info(f"🚀 開始生成...")
            
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
                max_frames_num=max_frames,  # 使用正確的幀數
                color_correction_strength=0.0,
                extra_args=extra_args
            )
            
            if hasattr(video, 'shape'):
                actual_frames = video.shape[2] if len(video.shape) > 2 else 0
                actual_duration = actual_frames / 8.0
                logger.info(f"📹 生成: {actual_frames} 幀, {actual_duration:.2f}秒")
            
            logger.info("💾 保存...")
            save_video_ffmpeg(video, output_path, [sum_audio], high_quality_save=False)
            
            final_path = f"{output_path}.mp4"
            
            if os.path.exists(final_path):
                import subprocess
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
            
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            logger.info(f"✅ {final_path}")
            return final_path
            
        except Exception as e:
            logger.error(f"❌ {e}")
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
    result = service.generate(
        image_path="examples/single/ref_image.png",
        audio_path="examples/single/1.wav",
        prompt="A woman singing",
        output_path="test_service_output"
    )
    print(f"✅ 完成: {result}")
