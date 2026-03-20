import json
with open("./script.json", "r") as f:
    content = json.load(f)
    speakers = {item['speaker'] for item in content['script'] if 'speaker' in item}
    print(speakers)