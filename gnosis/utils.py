import re


def remove_code_fences_regex(text):
    # 匹配开头和结尾的 ``` 行（包括可能带有的语言标识符，如 ```python）
    text = text.strip()
    text = re.sub(r"^```.*?\n", "", text)
    text = re.sub(r"\n```.*?$", "", text)
    return text.strip()
