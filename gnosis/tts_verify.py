import re
from rapidfuzz import fuzz
from faster_whisper import WhisperModel

class LNAligner:
    def __init__(self):
        # M2 Pro 上 tiny 模型 int8 极快
        self.model = WhisperModel("tiny", device="cpu", compute_type="int8")
        # 日式文本清洗规则：保留汉字、英文、数字、平假名、片假名
        self.clean_pat = re.compile(r'[^\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fa5a-zA-Z0-9]')

    def clean(self, text):
        return self.clean_pat.sub('', text)

    def get_chunks(self, text):
        """
        专门针对轻小说优化的切分逻辑
        1. 识别并提取对话框内容 「...」 或 『...』
        2. 对非对话文本按标点切分
        """
        # 匹配各种日式括号内容
        pattern = r'([「『].*?[」』]|[^。！？；\n!?;]+[。！？；\n!?;]?)'
        raw_chunks = re.findall(pattern, text)
        
        final_chunks = []
        for c in raw_chunks:
            cleaned = self.clean(c)
            final_chunks.append({"raw": c.strip(), "clean": cleaned})
        return final_chunks

    def verify(self, audio_path, full_text):
        # 1. 转录
        segments, _ = self.model.transcribe(audio_path, initial_prompt="简体中文")
        transcribed_all = self.clean("".join([s.text for s in segments]))

        # 2. 获取优化后的切片
        chunks = self.get_chunks(full_text)
        
        fail = False
        for item in chunks:
            target = item["clean"]
            
            # 使用更健壮的算法组合
            # score1: 顺序匹配 (适合长句)
            score1 = fuzz.partial_ratio(target, transcribed_all)
            # score2: 集合匹配 (适合乱序或识别丢词)
            score2 = fuzz.token_set_ratio(target, transcribed_all)
            
            final_score = max(score1, score2)
            
            threshold = 48
            
            is_missing = final_score < threshold
            if is_missing:
                fail = True
            
            
        return {
            "ok": not fail,
            "audio": audio_path,
            "script": item['clean'],
            "asr": transcribed_all
        }
