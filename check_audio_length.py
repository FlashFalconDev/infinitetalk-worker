# 在 model_service.py 中添加日誌
import sys
sys.path.insert(0, '/workspace/InfiniteTalk')

# 讀取 model_service.py
with open('model_service.py', 'r') as f:
    content = f.read()

# 在 audio_embedding 生成後添加日誌
old_code = '''            emb_path = os.path.join(temp_dir, 'audio_emb.pt')
            torch.save(audio_embedding, emb_path)'''

new_code = '''            emb_path = os.path.join(temp_dir, 'audio_emb.pt')
            torch.save(audio_embedding, emb_path)
            
            # 調試：檢查 audio embedding 長度
            logger.info(f"🔍 Audio embedding shape: {audio_embedding.shape}")
            logger.info(f"🔍 Audio embedding 幀數: {audio_embedding.shape[0] if len(audio_embedding.shape) > 0 else 'unknown'}")'''

content = content.replace(old_code, new_code)

with open('model_service.py', 'w') as f:
    f.write(content)

print("✅ 已添加 audio embedding 調試輸出")
