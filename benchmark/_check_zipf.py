import re
from wordfreq import zipf_frequency

tests = ['file', 'USB', '软件', '钢琴', 'LocalSend', 'Syncthing',
         'v2rayN', 'MIDI', 'Korg', '开心', '孤独', '羁绊', '服务配置', 'Docker Compose']
print(f"{'词':<14} {'zh':>6} {'en':>6}  选用")
print("-" * 45)
for w in tests:
    z = zipf_frequency(w, 'zh')
    e = zipf_frequency(w, 'en')
    is_en = bool(re.match(r'^[a-zA-Z]', w))
    pick = 'en' if is_en else 'zh'
    use = e if is_en else z
    print(f"{w:<14} {z:>6.2f} {e:>6.2f}  ->{pick}={use:.2f}")
