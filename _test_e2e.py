"""端到端测试：build -> save -> 新实例 load -> query_with_concepts 不报错。
用 statistical 关键词法避开 keyatten 模型下载。"""
import shutil
import tempfile
import traceback
from pathlib import Path

try:
    from memento import Memento

    texts = [
        "Docker 容器启动后立即退出，应先检查日志和启动命令。",
        "Compose 可以通过 env_file 为服务加载环境变量。",
        "手机传文件到电脑用 LocalSend 软件很方便。",
        "钢琴连电脑需要 MIDI to USB 线。",
    ]

    mem = Memento(embedding_model="tfidf-svd")
    for i, t in enumerate(texts):
        mem.add_node(t, node_id=f"mem_{i:03d}")
    mem.build_index()

    info = mem.build_concept_graph(
        top_k=5,
        keyword_method="statistical",  # 避开 keyatten 模型
        max_concepts=50,
        min_concept_energy=0.0,
    )
    print("build_concept_graph:", info["concepts"], "concepts,",
          info["event_concept_edges"], "event-concept edges")

    # 查询确认概念图可用
    r1 = mem.query_with_concepts("容器配置", k=3)
    print("query before save: ", len(r1), "results")

    d = tempfile.mkdtemp()
    mem.save(d)
    print("saved files:", sorted(Path(d).iterdir(), key=lambda p: p.name))

    # 新实例 load
    mem2 = Memento(embedding_model="tfidf-svd")
    mem2.load(d)
    print("loaded: stats =", mem2.stats["total_nodes"], "nodes,",
          len(mem2.concept_graph.concepts), "concepts")

    # 关键：load 后 query_with_concepts 必须不报错
    r2 = mem2.query_with_concepts("容器配置", k=3)
    print("query after load: ", len(r2), "results")

    # 结果应与 save 前一致
    ids1 = [x["id"] for x in r1]
    ids2 = [x["id"] for x in r2]
    print("same result ids:", ids1 == ids2, ids1, ids2)

    shutil.rmtree(d, ignore_errors=True)
    print("E2E OK")
except Exception:
    traceback.print_exc()
    raise
