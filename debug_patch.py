import sys
sys.path.insert(0, '/workspace/InfiniteTalk')

# 讀取文件
with open('wan/multitalk.py', 'r') as f:
    lines = f.readlines()

# 在第 797 行後添加調試輸出（if arrive_last_frame: break 這一行之前）
# 找到 "if arrive_last_frame: break" 這一行
for i, line in enumerate(lines):
    if 'if arrive_last_frame: break' in line and i > 700:
        # 在這一行前插入日誌
        indent = len(line) - len(line.lstrip())
        debug_line = ' ' * indent + 'print(f"DEBUG: Loop iteration - audio_start_idx={audio_start_idx}, audio_end_idx={audio_end_idx}, arrive_last_frame={arrive_last_frame}, gen_video_list_len={len(gen_video_list)}")\n'
        lines.insert(i, debug_line)
        print(f"✅ 在第 {i+1} 行添加了調試輸出")
        break

# 寫回文件
with open('wan/multitalk.py', 'w') as f:
    f.writelines(lines)

print("✅ 調試代碼已添加")
