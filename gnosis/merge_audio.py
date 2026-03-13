import soundfile as sf
import numpy as np
import os

def generate_precise_master_audio(timeline_payload, output_master_path):
    """
    根据精确时间轴，直接在内存中无损拼接音频并插入静音，完全避开 FFmpeg 的时间戳漂移。
    """
    sample_rate = timeline_payload["sample_rate"]
    pause_samples = timeline_payload["pause_samples"]
    audio_dir = timeline_payload["audio_dir"]
    segments = timeline_payload["segments"]
    
    # 1. 生成绝对精准的静音数组 (全0)
    silence_data = np.zeros(pause_samples, dtype=np.float32)
    
    audio_arrays = []
    
    # 2. 遍历读取并拼装
    for i, segment in enumerate(segments):
        wav_path = os.path.join(audio_dir, segment["file_name"])
        
        # 读取分段音频数据
        data, sr = sf.read(wav_path, dtype='float32')
        if sr != sample_rate:
            raise ValueError(f"采样率不匹配: {wav_path} ({sr}Hz != {sample_rate}Hz)")
            
        audio_arrays.append(data)
        
        # 如果不是最后一条，插入静音数组
        if i < len(segments) - 1:
            audio_arrays.append(silence_data)
            
    # 3. 内存极速拼接
    final_audio = np.concatenate(audio_arrays)
    
    # 4. 校验总采样数是否与我们预期的一致
    expected_samples = timeline_payload["total_samples"]
    if len(final_audio) != expected_samples:
        print(f"警告：拼接后样本数 ({len(final_audio)}) 与预期 ({expected_samples}) 不符！")
        
    # 5. 导出母带 WAV
    sf.write(output_master_path, final_audio, sample_rate, subtype='PCM_16')
    print(f"✅ 精确母带生成完毕: {output_master_path}")
    print(f"总采样数: {len(final_audio)} (理论时长: {len(final_audio)/sample_rate:.2f}秒)")
    return True