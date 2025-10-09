"""
InfiniteTalk 模型服務 - 支援多解析度 AI 升頻
品質方案：
- Standard: 480P (快速) - 70分鐘
- Premium: 480P (高品質) - 120分鐘
- Ultra: 720P (Premium + AI升頻) - 135分鐘
- Supreme: 1080P (Premium + 兩段AI升頻) - 155分鐘
升頻優先級：Real-ESRGAN > FFmpeg > Topaz (可選)
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
        'sampling_steps': 6,        # 快速參數
        'motion_frame': 7,          # 快速參數
        'use_teacache': False,
        'teacache_thresh': 1.0,
        'upscale': False,
        'description': '標準品質 480P (快速)'
    },
    'premium': {
        'resolution': '480',
        'sampling_steps': 8,        # 高品質參數
        'motion_frame': 9,          # 高品質參數
        'use_teacache': False,
        'teacache_thresh': 1.0,
        'upscale': False,
        'description': '高品質 480P'
    },
    'ultra': {
        'resolution': '480',
        'sampling_steps': 8,        # Premium 參數
        'motion_frame': 9,          # Premium 參數
        'use_teacache': False,
        'teacache_thresh': 1.0,
        'upscale': True,
        'target_resolution': '720',
        'upscale_stages': 1,        # 單段升頻
        'description': '超高品質 720P (Premium + AI升頻)'
    },
    'supreme': {
        'resolution': '480',
        'sampling_steps': 8,        # Premium 參數
        'motion_frame': 9,          # Premium 參數
        'use_teacache': False,
        'teacache_thresh': 1.0,
        'upscale': True,
        'target_resolution': '1080',
        'upscale_stages': 2,        # 兩段式升頻
        'description': '極致品質 1080P (Premium + 兩段AI升頻)'
    }
}

class InfiniteTalkModelService:
    def __init__(self, 
                 ckpt_dir='weights/Wan2.1-I2V-14B-480P',
                 wav2vec_dir='weights/chinese-wav2vec2-base',
                 infinitetalk_dir='weights/InfiniteTalk/single/infinitetalk.safetensors',
                 lora_dir='weights/Wan2.1_I2V_14B_FusionX_LoRA.safetensors',
                 device='cuda',
                 realesrgan_model='weights/realesr-animevideov3.pth',
                 topaz_path='/usr/local/bin/tvai-cli'):
        
        self.ckpt_dir = ckpt_dir
        self.wav2vec_dir = wav2vec_dir
        self.infinitetalk_dir = infinitetalk_dir
        self.lora_dir = lora_dir
        self.device = device
        self.realesrgan_model = realesrgan_model
        self.topaz_path = topaz_path
        
        self.loaded = False
        self.wan_i2v = None
        self.wav2vec_feature_extractor = None
        self.audio_encoder = None
        
        # 檢查各種升頻工具的可用性
        self.realesrgan_available = self._check_realesrgan()
        self.ffmpeg_available = self._check_ffmpeg()
        self.topaz_available = self._check_topaz()
        
        # 顯示可用的升頻方案
        self._display_upscale_options()
        
    def _check_realesrgan(self):
        """檢查 Real-ESRGAN 是否可用"""
        try:
            import cv2
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            if os.path.exists(self.realesrgan_model):
                logger.info(f"✅ Real-ESRGAN 可用 (主力升頻方案)")
                return True
            else:
                logger.warning(f"⚠️  Real-ESRGAN 模型不存在: {self.realesrgan_model}")
                logger.info(f"   請下載: wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth -P weights/")
                return False
        except ImportError as e:
            logger.warning(f"⚠️  Real-ESRGAN 未安裝: {e}")
            logger.info(f"   請安裝: pip install realesrgan basicsr facexlib gfpgan opencv-python")
            return False
    
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
                logger.info(f"✅ FFmpeg 可用 (備用升頻方案)")
                return True
        except Exception as e:
            logger.error(f"❌ FFmpeg 不可用: {e}")
        return False
    
    def _check_topaz(self):
        """檢查 Topaz Video AI 是否可用（可選）"""
        try:
            result = subprocess.run(
                [self.topaz_path, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info(f"✅ Topaz Video AI 可用 (可選高級方案)")
                return True
        except Exception:
            pass
        return False
    
    def _display_upscale_options(self):
        """顯示可用的升頻選項"""
        logger.info("=" * 70)
        logger.info("📊 升頻方案狀態:")
        
        options = []
        if self.realesrgan_available:
            options.append("Real-ESRGAN (AI增強) ⭐⭐⭐⭐⭐")
        if self.ffmpeg_available:
            options.append("FFmpeg (快速穩定) ⭐⭐⭐⭐")
        if self.topaz_available:
            options.append("Topaz (頂級品質) ⭐⭐⭐⭐⭐")
        
        if options:
            for opt in options:
                logger.info(f"   ✅ {opt}")
        else:
            logger.error("   ❌ 沒有可用的升頻方案！")
        
        logger.info("=" * 70)
    
    def upscale_video_realesrgan(self, input_path, output_path, target_height=720):
        """使用 Real-ESRGAN AI 升頻"""
        try:
            logger.info(f"🔍 使用 Real-ESRGAN AI 升頻到 {target_height}P...")
            
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            import cv2
            
            # 計算縮放倍數
            probe_cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=height',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                input_path
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            current_height = int(result.stdout.strip())
            scale = target_height / current_height
            
            logger.info(f"   原高度: {current_height}P → 目標: {target_height}P (縮放: {scale:.2f}x)")
            
            # 初始化 Real-ESRGAN
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                           num_block=23, num_grow_ch=32, scale=2)
            
            upsampler = RealESRGANer(
                scale=2,
                model_path=self.realesrgan_model,
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True,
                device=self.device
            )
            
            # 提取視頻幀
            temp_dir = f"temp_realesrgan_{os.path.basename(input_path)}"
            os.makedirs(temp_dir, exist_ok=True)
            frames_dir = os.path.join(temp_dir, 'frames')
            output_frames_dir = os.path.join(temp_dir, 'output_frames')
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(output_frames_dir, exist_ok=True)
            
            # 提取幀
            logger.info(f"   📹 提取視頻幀...")
            extract_cmd = [
                'ffmpeg', '-i', input_path,
                '-qscale:v', '1', '-qmin', '1', '-qmax', '1',
                '-vsync', '0',
                f'{frames_dir}/frame_%08d.png'
            ]
            subprocess.run(extract_cmd, capture_output=True, timeout=120)
            
            # 處理每一幀
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
            total_frames = len(frame_files)
            logger.info(f"   🎬 開始處理 {total_frames} 幀...")
            
            for i, frame_file in enumerate(frame_files, 1):
                if i % 10 == 0 or i == total_frames:
                    logger.info(f"   處理進度: {i}/{total_frames} ({i*100//total_frames}%)")
                
                frame_path = os.path.join(frames_dir, frame_file)
                output_frame_path = os.path.join(output_frames_dir, frame_file)
                
                img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
                output, _ = upsampler.enhance(img, outscale=scale)
                cv2.imwrite(output_frame_path, output)
            
            logger.info(f"   ✅ AI 升頻完成")
            
            # 重新組合視頻
            logger.info(f"   🎞️  重組視頻...")
            
            fps_cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                input_path
            ]
            fps_result = subprocess.run(fps_cmd, capture_output=True, text=True)
            fps = fps_result.stdout.strip()
            
            combine_cmd = [
                'ffmpeg',
                '-framerate', fps,
                '-i', f'{output_frames_dir}/frame_%08d.png',
                '-i', input_path,
                '-map', '0:v',
                '-map', '1:a?',
                '-c:v', 'libx264',
                '-preset', 'slow',
                '-crf', '18',
                '-c:a', 'copy',
                '-pix_fmt', 'yuv420p',
                '-y',
                output_path
            ]
            
            result = subprocess.run(combine_cmd, capture_output=True, 
                                   text=True, timeout=300)
            
            # 清理臨時檔案
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            if result.returncode == 0 and os.path.exists(output_path):
                logger.info(f"✅ Real-ESRGAN 升頻完成")
                return True
            else:
                logger.error(f"❌ 視頻組合失敗: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Real-ESRGAN 升頻錯誤: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def upscale_video_ffmpeg(self, input_path, output_path, target_height=720):
        """使用 FFmpeg Lanczos 升頻（備用方案）"""
        try:
            logger.info(f"🔍 使用 FFmpeg 升頻到 {target_height}P...")
            
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-vf', f'scale=-1:{target_height}:flags=lanczos',
                '-c:v', 'libx264',
                '-preset', 'slow',
                '-crf', '18',
                '-c:a', 'copy',
                '-y',
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
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
    
    def upscale_video_topaz(self, input_path, output_path, target_height=720):
        """使用 Topaz Video AI 升頻（可選方案）"""
        try:
            logger.info(f"🔍 使用 Topaz AI 升頻到 {target_height}P...")
            
            probe_cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=height',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                input_path
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            current_height = int(result.stdout.strip())
            scale = target_height / current_height
            
            cmd = [
                self.topaz_path,
                input_path,
                '--output', output_path,
                '--model', 'prob-3',
                '--scale', str(scale),
                '--device', '0',
                '--vcodec', 'h264',
                '--quality', 'high'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=900
            )
            
            if result.returncode == 0 and os.path.exists(output_path):
                logger.info(f"✅ Topaz 升頻完成")
                return True
            else:
                logger.error(f"❌ Topaz 升頻失敗: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Topaz 升頻錯誤: {e}")
            return False
    
    def upscale_video_single_stage(self, input_path, output_path, target_height):
        """單段升頻（優先級：Real-ESRGAN > FFmpeg > Topaz）"""
        logger.info(f"📊 單段升頻: 480P → {target_height}P")
        
        # 優先使用 Real-ESRGAN
        if self.realesrgan_available:
            if self.upscale_video_realesrgan(input_path, output_path, target_height):
                return True
            logger.warning("⚠️  Real-ESRGAN 失敗，嘗試備用方案...")
        
        # 備用：FFmpeg
        if self.ffmpeg_available:
            if self.upscale_video_ffmpeg(input_path, output_path, target_height):
                return True
            logger.warning("⚠️  FFmpeg 失敗，嘗試 Topaz...")
        
        # 最後：Topaz（如果有的話）
        if self.topaz_available:
            if self.upscale_video_topaz(input_path, output_path, target_height):
                return True
        
        logger.error("❌ 所有升頻方案都失敗了")
        return False
    
    def upscale_video_two_stage(self, input_path, output_path):
        """兩段式升頻：480P → 720P (AI) → 1080P (FFmpeg)"""
        try:
            logger.info(f"📊 兩段式升頻: 480P → 720P → 1080P")
            
            # 第一段：480P → 720P (AI 增強)
            temp_720p = input_path.replace('.mp4', '_temp_720p.mp4')
            logger.info(f"   🎬 第一段: 480P → 720P (AI增強)")
            
            # 優先使用 Real-ESRGAN
            stage1_success = False
            if self.realesrgan_available:
                stage1_success = self.upscale_video_realesrgan(input_path, temp_720p, 720)
            
            if not stage1_success and self.topaz_available:
                logger.warning("   ⚠️  Real-ESRGAN 失敗，嘗試 Topaz...")
                stage1_success = self.upscale_video_topaz(input_path, temp_720p, 720)
            
            if not stage1_success and self.ffmpeg_available:
                logger.warning("   ⚠️  AI方案失敗，降級使用 FFmpeg...")
                stage1_success = self.upscale_video_ffmpeg(input_path, temp_720p, 720)
            
            if not stage1_success:
                logger.error("   ❌ 第一段升頻失敗")
                return False
            
            logger.info(f"   ✅ 第一段完成")
            
            # 第二段：720P → 1080P (輕度放大)
            logger.info(f"   🎬 第二段: 720P → 1080P (輕度放大)")
            
            cmd = [
                'ffmpeg',
                '-i', temp_720p,
                '-vf', 'scale=-1:1080:flags=lanczos',
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '20',
                '-c:a', 'copy',
                '-y',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, 
                                   text=True, timeout=300)
            
            # 清理臨時檔案
            if os.path.exists(temp_720p):
                os.remove(temp_720p)
            
            if result.returncode == 0 and os.path.exists(output_path):
                logger.info(f"   ✅ 第二段完成")
                logger.info(f"✅ 兩段式升頻完成！")
                return True
            else:
                logger.error(f"   ❌ 第二段失敗: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 兩段式升頻錯誤: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def upscale_video(self, input_path, output_path, target_height=720, stages=1):
        """智能升頻入口"""
        if stages == 1:
            return self.upscale_video_single_stage(input_path, output_path, target_height)
        elif stages == 2:
            return self.upscale_video_two_stage(input_path, output_path)
        else:
            logger.error(f"❌ 不支援的升頻階段數: {stages}")
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
            logger.info(f"✅ 找到品質預設")
            preset = QUALITY_PRESETS[quality]
            resolution = resolution or preset['resolution']
            sample_steps = sample_steps or preset['sampling_steps']
            motion_frame = motion_frame or preset['motion_frame']
            use_teacache = preset['use_teacache']
            teacache_thresh = preset['teacache_thresh']
            need_upscale = preset.get('upscale', False)
            target_resolution = preset.get('target_resolution', '480')
            upscale_stages = preset.get('upscale_stages', 1)
            
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
            quality = 'standard'
            preset = QUALITY_PRESETS['standard']
            resolution = preset['resolution']
            sample_steps = preset['sampling_steps']
            motion_frame = preset['motion_frame']
            use_teacache = preset['use_teacache']
            teacache_thresh = preset['teacache_thresh']
            need_upscale = False
        
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
                
                if upscale_stages == 2:
                    # 兩段式升頻
                    video_final_path = f"{output_path}_{target_resolution}p.mp4"
                    if self.upscale_video_two_stage(video_480p_path, video_final_path):
                        final_path = video_final_path
                        logger.info(f"✅ 兩段式升頻完成: {video_final_path}")
                    else:
                        logger.warning("⚠️  升頻失敗，使用 480P 原檔")
                else:
                    # 單段升頻
                    video_final_path = f"{output_path}_{target_resolution}p.mp4"
                    if self.upscale_video_single_stage(video_480p_path, video_final_path, 
                                                       int(target_resolution)):
                        final_path = video_final_path
                        logger.info(f"✅ 升頻完成: {video_final_path}")
                    else:
                        logger.warning("⚠️  升頻失敗，使用 480P 原檔")
            
            # 驗證最終視頻
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
            
            # 清理臨時檔案
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
    
    print("\n" + "="*70)
    print("測試 Standard (480P - 快速)")
    print("="*70)
    result = service.generate(
        image_path="examples/single/ref_image.png",
        audio_path="examples/single/1.wav",
        prompt="A woman speaking",
        output_path="test_standard",
        quality="standard"
    )
    print(f"✅ Standard 完成: {result}")
