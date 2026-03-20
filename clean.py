import os
import argparse

def process_subtitles(subtitle_name, audio_subdir):
    """
    在当前工作目录下处理字幕和删除音频
    """
    target_indices = []
    
    # 1. 解析 SRT 文件
    if not os.path.exists(subtitle_name):
        print(f"❌ 错误：在当前目录下找不到字幕文件 '{subtitle_name}'")
        return

    try:
        with open(subtitle_name, 'r', encoding='utf-8') as f:
            # 以双换行符分割条目
            content = f.read().strip().split('\n\n')
            
            for block in content:
                lines = block.split('\n')
                if len(lines) >= 3:
                    index = lines[0].strip()       # 第1行：序号 x
                    text = "".join(lines[2:])      # 合并可能换行的字幕内容
                    
                    # 逻辑：第一个中文冒号之后包含 '栞'
                    if '：' in text:
                        after_colon = text.split('：', 1)[1]
                        if '栞' in after_colon:
                            target_indices.append(index)
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return

    if not target_indices:
        print("💡 未发现匹配条件的条目。")
        return

    print(f"🔍 找到符合条件的序号: {', '.join(target_indices)}")

    # 2. 删除对应的音频文件
    audio_path = os.path.join('.', audio_subdir)
    if not os.path.exists(audio_path):
        print(f"❌ 错误：音频目录 '{audio_subdir}' 不存在")
        return

    deleted_count = 0
    for x in target_indices:
        # 格式化为 000x.wav
        file_name = f"{(int(x)+1):04d}.wav"
        file_to_delete = os.path.join(audio_path, file_name)
        
        if os.path.exists(file_to_delete):
            try:
                os.remove(file_to_delete)
                print(f"🗑️  已删除: {file_name}")
                deleted_count += 1
            except Exception as e:
                print(f"⚠️  无法删除 {file_name}: {e}")
        else:
            print(f"❓ 跳过: {file_name} (文件不存在)")

    print(f"\n✅ 任务完成！共找到 {len(target_indices)} 处匹配，成功删除 {deleted_count} 个文件。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据SRT字幕内容批量删除对应编号的音频文件")
    
    # 添加参数
    parser.add_argument('--dir', type=str, required=True, help='指定工作目录路径')
    parser.add_argument('--srt', type=str, default='input.srt', help='字幕文件名 (默认: input.srt)')
    parser.add_argument('--audio_dir', type=str, default='output_audio', help='音频文件夹名 (默认: output_audio)')

    args = parser.parse_args()

    # 切换工作目录
    try:
        os.chdir(args.dir)
        print(f"📂 已切换工作目录至: {os.getcwd()}")
        
        # 执行逻辑
        process_subtitles(args.srt, args.audio_dir)
        
    except FileNotFoundError:
        print(f"❌ 错误：指定的工作目录 '{args.dir}' 不存在")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")