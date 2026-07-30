import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from memento.concept.concept_graph import ConceptGraph

cg = ConceptGraph()
total_docs = 145
print(f"{'词':<14} {'zipf':>5} {'df':>3} {'idf_n':>6} {'pen':>5} {'energy':>7}")
print("-" * 55)
for kw, df in [("file",1),("USB",2),("MIDI",4),("Korg D1",3),
               ("LocalSend",2),("Syncthing",2),("v2rayN",2),
               ("容器配置",2),("钢琴",5),("软件",3),("电脑",5),
               ("开心",3),("难过",2),("孤独",1),("Docker Compose",1),
               ("引入软件",1),("电脑扫描手机",1),("发现钢琴",1)]:
    zipf = cg._zipf_frequency(kw)
    idf = math.log((total_docs + 1) / (df + 1)) + 1
    idf_n = min(1.0, idf / (math.log(total_docs + 1) + 1))
    is_emo = cg.is_emotion_word(kw)
    if is_emo:
        pen = 1.0
    elif zipf < 2.0:
        pen = 1.0
    elif zipf < 4.0:
        pen = 1.0 - 0.5 * (zipf - 2.0) / 2.0
    else:
        pen = max(0.1, 0.5 - 0.2 * (zipf - 4.0))
    energy = cg.initial_energy(kw, df, total_docs, is_emotion=is_emo)
    flag = "情感" if is_emo else ("过滤!" if energy < 0.5 else "")
    print(f"{kw:<14} {zipf:>5.2f} {df:>3} {idf_n:>6.3f} {pen:>5.2f} {energy:>7.4f} {flag}")
