from wordfreq import zipf_frequency

tests = ['file', 'USB', '软件', '电脑', '手机', '钢琴', '容器配置',
         'LocalSend', 'Syncthing', 'v2rayN', 'MIDI', 'Korg',
         '开心', '难过', '孤独', '羁绊', '愧疚', '服务配置', 'Docker Compose',
         '引入软件', '发现钢琴', '电脑扫描手机']

print(f"{'词':<14} {'zh':>6} {'en':>6} {'max':>6}")
print("-" * 40)
for w in tests:
    z = zipf_frequency(w, 'zh')
    e = zipf_frequency(w, 'en')
    m = max(z, e)
    print(f"{w:<14} {z:>6.2f} {e:>6.2f} {m:>6.2f}")
