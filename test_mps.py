import re
from rapidfuzz import fuzz
from faster_whisper import WhisperModel

def clean_text(text):
    if not text: return ""
    # 保留汉字、英文、数字、假名
    pattern = re.compile(r'[^\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fa5a-zA-Z0-9]')
    return re.sub(pattern, '', text)

def split_into_chunks(text):
    # 按标点切分，确保每段话都是一个独立的语义锚点
    chunks = re.split(r'[。！？；.!?;]', text)
    return [clean_text(c) for c in chunks if len(clean_text(c)) > 0]

def analyze_calibration(audio_path, full_expected_text):
    # 1. 转录 (M2 Pro 建议使用 cpu)
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, initial_prompt="简体中文")
    transcribed_all = clean_text("".join([s.text for s in segments]))

    # 2. 将预期文本切分为片段
    expected_chunks = split_into_chunks(full_expected_text)
    
    report = []
    missing_count = 0

    for i, chunk in enumerate(expected_chunks):
        # 使用 partial_ratio 在整段识别结果中寻找该片段
        score = fuzz.partial_ratio(chunk, transcribed_all)
        
        # 判定阈值：Tiny模型下，45-50 是一个比较稳健的界限
        is_missing = score < 50
        if is_missing:
            missing_count += 1
            
        report.append({
            "chunk_index": i,
            "text": chunk[:15] + "...", # 仅用于显示
            "score": round(score, 2),
            "status": "MISSING" if is_missing else "OK"
        })

    # 3. 总体结论
    is_fully_aligned = missing_count == 0
    return is_fully_aligned, report

# --- 测试你的例子 ---
expected_text = "冬天的平日早晨让我感到非常忧郁。首先是起床就很冷。昨天晚上我钻进被窝，花了好多时间才达到最舒适的温度，却非得离开被窝不可。因为我必须要去上学。虽然学业成绩不差，但我并不是喜欢念书。"

# 假设 transcribed_all 只有前四句的内容
# 运行后，report 的最后一个元素 score 会极低（接近 0-20），从而被标记为 MISSING
a, b = analyze_calibration("./0000.wav", expected_text)
print(a)
print(b)