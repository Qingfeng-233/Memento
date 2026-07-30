"""
Memento 命令行接口

一次性进程模型：每次调用从磁盘 load，操作完（写操作）自动 save。
适合脚本化、cron、被其他程序调用。

入口:
    python -m memento.cli <command> [options]

子命令:
    add            存入一条记忆
    import         从 jsonl 批量导入
    build          构建向量索引 / 概念图 / 关键词边
    query          向量 + 扩散检索
    query-concepts 概念图检索
    query-rag      纯 RAG 检索（对照）
    get            按 id 取节点
    stats          系统状态
    link           连接两个节点
    link-concepts  连接两个关键词概念
    activate       激活一组节点（情境共现建边）
    mark-important 调整节点重要性
    sleep          触发睡眠巩固周期
    clock          推进时钟步
    save           显式落盘
    serve          启动 HTTP 服务（需 fastapi/uvicorn）
    mcp            启动 MCP 服务（需 mcp）

全局选项:
    --store PATH           存储目录（默认 data/memento_store / $MEMENTO_STORE）
    --embedding-model NAME 新建 store 时用的 embedding 后端
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Optional

from memento import store as store_mod


# ─── 工具函数 ──────────────────────────────────────────────


def _print_json(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _read_text(args) -> str:
    """从 --text / --file / stdin 三选一拿到文本。"""
    if getattr(args, "text", None):
        return args.text
    if getattr(args, "file", None):
        with open(args.file, "r", encoding="utf-8") as f:
            return f.read().strip()
    if getattr(args, "stdin", False) or sys.stdin.isatty() is False:
        if not sys.stdin.isatty():
            return sys.stdin.read().strip()
    raise SystemExit("错误: 请通过 --text、--file 或管道 stdin 提供文本")


def _parse_tags(s: Optional[str]) -> list:
    if not s:
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


def _parse_ids(s: str) -> list:
    return [t.strip() for t in s.split(",") if t.strip()]


# ─── 子命令实现 ────────────────────────────────────────────


def cmd_add(args):
    mem = store_mod.load_or_create(args.store, args.embedding_model)
    text = _read_text(args)
    node_id = mem.add_node(
        text=text,
        node_id=args.id,
        importance=args.importance,
        tags=_parse_tags(args.tags),
        source=args.source,
        created_at=args.created_at,
    )
    if args.build:
        count = mem.build_index()
        if not args.no_save:
            store_mod.save(mem, args.store)
        if args.json:
            _print_json({"id": node_id, "index_built": True, "added": count})
        else:
            print(f"已存入 {node_id}，并构建索引（{count} 个节点）")
        return
    if not args.no_save:
        store_mod.save(mem, args.store)
    if args.json:
        _print_json(
            {"id": node_id, "index_built": False, "hint": "运行 build 子命令后才能查询"}
        )
    else:
        print(
            f"已暂存 {node_id}（尚未构建索引，运行 "
            f"`python -m memento.cli build` 后才可查询）"
        )


def cmd_import(args):
    mem = store_mod.load_or_create(args.store, args.embedding_model)
    count = 0
    with open(args.file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            mem.add_node(
                text=m["text"],
                node_id=m.get("id"),
                importance=m.get("importance", 0.5),
                tags=m.get("tags", []),
                source=m.get("source", "import"),
                created_at=m.get("created_at"),
            )
            count += 1
    if args.build:
        built = mem.build_index()
        if not args.no_save:
            store_mod.save(mem, args.store)
        if args.json:
            _print_json({"imported": count, "index_built": True, "added": built})
        else:
            print(f"导入 {count} 条，构建索引 {built} 个节点")
        return
    if not args.no_save:
        store_mod.save(mem, args.store)
    if args.json:
        _print_json({"imported": count, "index_built": False})
    else:
        print(f"导入 {count} 条（尚未构建索引）")


def cmd_build(args):
    mem = store_mod.load_or_create(args.store, args.embedding_model)
    result = {"index": None, "concept_graph": None, "keyword_edges": None}

    if not args.skip_index:
        result["index"] = mem.build_index()

    if args.concepts:
        info = mem.build_concept_graph(
            top_k=args.top_k,
            keyword_method=args.keyword_method,
            max_concepts=args.max_concepts,
            min_concept_energy=args.min_concept_energy,
            keyword_sim_threshold=args.keyword_sim_threshold,
            keyword_temperature=args.keyword_temperature,
            keyword_model=args.keyword_model,
            keyword_device=args.keyword_device,
            keyword_dtype=args.keyword_dtype,
            dedup_concepts=args.dedup_concepts,
            dedup_threshold=args.dedup_threshold,
        )
        result["concept_graph"] = info

    if args.keyword_edges:
        info = mem.build_keyword_edges(
            top_k=args.ke_top_k,
            keyword_model=args.keyword_model,
            keyword_device=args.keyword_device,
            keyword_dtype=args.keyword_dtype,
            semantic_filter=args.ke_semantic_filter,
            min_cos_sim=args.ke_min_cos_sim,
        )
        result["keyword_edges"] = info

    if not args.no_save:
        store_mod.save(mem, args.store)

    if args.json:
        _print_json(result)
    else:
        if result["index"] is not None:
            print(f"向量索引: 新增 {result['index']} 个节点")
        if result["concept_graph"] is not None:
            cg = result["concept_graph"]
            print(
                f"概念图: {cg['concepts']} 概念, "
                f"{cg['event_concept_edges']} 事件-概念边, "
                f"{cg['concept_edges']} 概念边"
            )
        if result["keyword_edges"] is not None:
            ke = result["keyword_edges"]
            print(
                f"关键词边: 新增 {ke['edges_added']} 条, 拒绝 {ke['edges_rejected']} 条"
            )
        if not any(result.values()):
            print(
                "无待构建内容（没有 pending 节点，且未指定 --concepts/--keyword-edges）"
            )


def cmd_query(args):
    mem = store_mod.load_or_create(args.store, args.embedding_model)
    text = _read_text(args)
    if args.rag_only:
        results = mem.query_rag_only(text, k=args.k)
    else:
        results = mem.query(text, k=args.k, seed_k=args.seed_k)
    if args.json:
        _print_json(results)
    else:
        if not results:
            print("无匹配结果")
            return
        for i, r in enumerate(results, 1):
            print(
                f"{i}. [{r['id']}] s={r['score']:.4f} "
                f"w={r['importance']:.2f} v={r['vitality']:.2f} "
                f"e={r['edges']}"
            )
            print(f"   {r['text'][:120]}")


def cmd_query_concepts(args):
    mem = store_mod.load_or_create(args.store, args.embedding_model)
    text = _read_text(args)
    out = mem.query_with_concepts(
        text,
        k=args.k,
        seed_k=args.seed_k,
        concept_k=args.concept_k,
        concept_hops=args.concept_hops,
        concept_weight=args.concept_weight,
        debug=args.debug,
    )
    if args.debug and not args.json:
        # debug 返回的是 dict 结构，无论 --json 与否都打印 JSON 最清晰
        _print_json(out)
    elif args.json:
        _print_json(out)
    else:
        results = out
        if not results:
            print("无匹配结果")
            return
        for i, r in enumerate(results, 1):
            print(
                f"{i}. [{r['id']}] final={r['score']:.4f} "
                f"rag={r['rag_score']:.4f} concept={r['concept_score']:.4f}"
            )
            print(f"   {r['text'][:120]}")


def cmd_get(args):
    mem = store_mod.load_or_create(args.store, args.embedding_model)
    node = mem.get_node(args.id)
    if node is None:
        if args.json:
            _print_json({"error": "not found", "id": args.id})
        else:
            print(f"未找到节点 {args.id}")
        raise SystemExit(1)
    if args.json:
        _print_json(store_mod.node_to_dict(node))
    else:
        d = store_mod.node_to_dict(node)
        print(f"id:          {d['id']}")
        print(f"status:      {d['status']}")
        print(f"importance:  {d['importance']}")
        print(f"vitality:    {d['vitality']}")
        print(f"edges:       {d['edge_count']}")
        print(f"tags:        {d['tags']}")
        print(f"source:      {d['source']}")
        print(f"created_at:  {d['created_at']}")
        print(f"text:")
        print(d["text"])


def cmd_stats(args):
    mem = store_mod.load_or_create(args.store, args.embedding_model)
    s = mem.stats
    if args.json:
        _print_json(s)
    else:
        print("Memento 系统状态:")
        for k, v in s.items():
            print(f"  {k}: {v}")


def cmd_link(args):
    mem = store_mod.load_or_create(args.store, args.embedding_model)
    mem.link(args.node_a, args.node_b, weight=args.weight)
    if not args.no_save:
        store_mod.save(mem, args.store)
    if args.json:
        _print_json(
            {
                "ok": True,
                "node_a": args.node_a,
                "node_b": args.node_b,
                "weight": args.weight,
            }
        )
    else:
        print(f"已连接 {args.node_a} <-> {args.node_b} (w={args.weight})")


def cmd_link_concepts(args):
    mem = store_mod.load_or_create(args.store, args.embedding_model)
    mem.link_concepts(args.source, args.target, weight=args.weight)
    if not args.no_save:
        store_mod.save(mem, args.store)
    if args.json:
        _print_json(
            {
                "ok": True,
                "source": args.source,
                "target": args.target,
                "weight": args.weight,
            }
        )
    else:
        print(f"已连接概念 {args.source} <-> {args.target} (w={args.weight})")


def cmd_activate(args):
    mem = store_mod.load_or_create(args.store, args.embedding_model)
    ids = _parse_ids(args.node_ids)
    mem.activate(ids)
    if not args.no_save:
        store_mod.save(mem, args.store)
    if args.json:
        _print_json({"ok": True, "activated": ids})
    else:
        print(f"已激活 {len(ids)} 个节点: {ids}")


def cmd_mark_important(args):
    mem = store_mod.load_or_create(args.store, args.embedding_model)
    mem.mark_important(args.id, importance=args.importance)
    if not args.no_save:
        store_mod.save(mem, args.store)
    if args.json:
        _print_json({"ok": True, "id": args.id, "importance": args.importance})
    else:
        print(f"已标记 {args.id} 重要性 = {args.importance}")


def cmd_sleep(args):
    mem = store_mod.load_or_create(args.store, args.embedding_model)
    report = mem.trigger_sleep()
    if not args.no_save:
        store_mod.save(mem, args.store)
    d = asdict(report)
    if args.json:
        _print_json(d)
    else:
        print(report.summary())


def cmd_clock(args):
    mem = store_mod.load_or_create(args.store, args.embedding_model)
    for _ in range(args.steps):
        mem.clock_step()
    if not args.no_save:
        store_mod.save(mem, args.store)
    if args.json:
        _print_json({"ok": True, "steps": args.steps, "clock_step": mem._clock_step})
    else:
        print(f"推进 {args.steps} 步，当前 clock_step={mem._clock_step}")


def cmd_save(args):
    mem = store_mod.load_or_create(args.store, args.embedding_model)
    path = store_mod.save(mem, args.store)
    if args.json:
        _print_json({"ok": True, "path": path})
    else:
        print(f"已保存到 {path}")


def cmd_serve(args):
    try:
        from memento.server import run_server
    except ImportError as e:
        raise SystemExit(
            f"启动 HTTP 服务需要可选依赖: {e}\n"
            "请安装: pip install -r requirements-server.txt"
        )
    run_server(
        store=args.store,
        embedding_model=args.embedding_model,
        host=args.host,
        port=args.port,
        no_autosave=args.no_autosave,
    )


def cmd_mcp(args):
    try:
        from memento.mcp_server import run_mcp
    except ImportError as e:
        raise SystemExit(
            f"启动 MCP 服务需要可选依赖: {e}\n"
            "请安装: pip install -r requirements-server.txt"
        )
    run_mcp(
        store=args.store,
        embedding_model=args.embedding_model,
        no_autosave=args.no_autosave,
    )


# ─── 参数解析 ──────────────────────────────────────────────


def _add_text_input(p, help_text="记忆文本"):
    p.add_argument("--text", "-t", help=help_text)
    p.add_argument("--file", "-f", help="从文件读取文本")
    p.add_argument(
        "--stdin", action="store_true", help="从 stdin 读取文本（管道模式自动启用）"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m memento.cli",
        description="Memento 双系统联想记忆 — 命令行",
    )
    p.add_argument(
        "--store",
        default=None,
        help="存储目录（默认 data/memento_store / $MEMENTO_STORE）",
    )
    p.add_argument(
        "--embedding-model",
        default=None,
        help="新建 store 时的 embedding 后端 "
        "(默认 tfidf-svd / $MEMENTO_EMBEDDING_MODEL)",
    )

    sub = p.add_subparsers(dest="command", required=True)

    # add
    sp = sub.add_parser("add", help="存入一条记忆")
    _add_text_input(sp)
    sp.add_argument("--id", default=None, help="节点 id（自动生成则留空）")
    sp.add_argument("--importance", type=float, default=0.5)
    sp.add_argument("--tags", default=None, help="逗号分隔的标签")
    sp.add_argument("--source", default="cli")
    sp.add_argument("--created-at", default=None)
    sp.add_argument("--build", action="store_true", help="存入后立即构建向量索引")
    sp.add_argument(
        "--no-save", action="store_true", help="不自动落盘（默认写操作后自动 save）"
    )
    sp.add_argument("--json", action="store_true", help="JSON 输出")
    sp.set_defaults(func=cmd_add)

    # import
    sp = sub.add_parser("import", help="从 jsonl 批量导入")
    sp.add_argument(
        "--file",
        "-f",
        required=True,
        help="jsonl 路径，每行 {text, id?, importance?, tags?, ...}",
    )
    sp.add_argument("--build", action="store_true", help="导入后立即构建向量索引")
    sp.add_argument("--no-save", action="store_true")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_import)

    # build
    sp = sub.add_parser("build", help="构建向量索引 / 概念图 / 关键词边")
    sp.add_argument(
        "--skip-index",
        action="store_true",
        help="跳过 build_index（仅用于已有索引时补建图）",
    )
    sp.add_argument(
        "--concepts", action="store_true", help="同时构建概念图 build_concept_graph"
    )
    sp.add_argument(
        "--keyword-edges",
        action="store_true",
        help="同时构建关键词边 build_keyword_edges",
    )
    # 概念图参数
    sp.add_argument("--top-k", type=int, default=8)
    sp.add_argument(
        "--keyword-method", default="keyatten", choices=["keyatten", "statistical"]
    )
    sp.add_argument("--max-concepts", type=int, default=300)
    sp.add_argument("--min-concept-energy", type=float, default=0.5)
    sp.add_argument("--keyword-sim-threshold", type=float, default=0.65)
    sp.add_argument("--keyword-temperature", type=float, default=0.08)
    sp.add_argument("--dedup-concepts", action="store_true")
    sp.add_argument("--dedup-threshold", type=float, default=0.90)
    # 共享关键词模型参数
    sp.add_argument("--keyword-model", default="models/Qwen3-Embedding-0.6B")
    sp.add_argument("--keyword-device", default=None)
    sp.add_argument("--keyword-dtype", default="float16")
    # keyword_edges 专用
    sp.add_argument("--ke-top-k", type=int, default=5)
    sp.add_argument("--ke-semantic-filter", action="store_true")
    sp.add_argument("--ke-min-cos-sim", type=float, default=0.30)
    sp.add_argument("--no-save", action="store_true")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_build)

    # query
    sp = sub.add_parser("query", help="向量 + 扩散检索")
    _add_text_input(sp, help_text="查询文本")
    sp.add_argument("--k", type=int, default=10)
    sp.add_argument("--seed-k", type=int, default=20)
    sp.add_argument("--rag-only", action="store_true", help="只做纯 RAG 检索（不扩散）")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_query)

    # query-concepts
    sp = sub.add_parser("query-concepts", help="概念图检索")
    _add_text_input(sp, help_text="查询文本")
    sp.add_argument("--k", type=int, default=10)
    sp.add_argument("--seed-k", type=int, default=20)
    sp.add_argument("--concept-k", type=int, default=8)
    sp.add_argument("--concept-hops", type=int, default=2)
    sp.add_argument("--concept-weight", type=float, default=0.35)
    sp.add_argument(
        "--debug", action="store_true", help="返回 seed/activated concepts 等调试信息"
    )
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_query_concepts)

    # get
    sp = sub.add_parser("get", help="按 id 取节点")
    sp.add_argument("id", help="节点 id")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_get)

    # stats
    sp = sub.add_parser("stats", help="系统状态")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_stats)

    # link
    sp = sub.add_parser("link", help="连接两个节点")
    sp.add_argument("node_a")
    sp.add_argument("node_b")
    sp.add_argument("--weight", type=float, default=0.8)
    sp.add_argument("--no-save", action="store_true")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_link)

    # link-concepts
    sp = sub.add_parser("link-concepts", help="连接两个关键词概念")
    sp.add_argument("source")
    sp.add_argument("target")
    sp.add_argument("--weight", type=float, default=0.8)
    sp.add_argument("--no-save", action="store_true")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_link_concepts)

    # activate
    sp = sub.add_parser("activate", help="激活一组节点（情境共现建边）")
    sp.add_argument("node_ids", help="逗号分隔的节点 id 列表")
    sp.add_argument("--no-save", action="store_true")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_activate)

    # mark-important
    sp = sub.add_parser("mark-important", help="调整节点重要性")
    sp.add_argument("id")
    sp.add_argument("--importance", type=float, default=1.0)
    sp.add_argument("--no-save", action="store_true")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_mark_important)

    # sleep
    sp = sub.add_parser("sleep", help="触发睡眠巩固周期")
    sp.add_argument("--no-save", action="store_true")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_sleep)

    # clock
    sp = sub.add_parser("clock", help="推进时钟步")
    sp.add_argument("--steps", type=int, default=1)
    sp.add_argument("--no-save", action="store_true")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_clock)

    # save
    sp = sub.add_parser("save", help="显式落盘")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_save)

    # serve
    sp = sub.add_parser("serve", help="启动 HTTP 服务（需 fastapi/uvicorn）")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--no-autosave", action="store_true", help="禁用写操作后自动落盘")
    sp.set_defaults(func=cmd_serve)

    # mcp
    sp = sub.add_parser("mcp", help="启动 MCP 服务（需 mcp）")
    sp.add_argument("--no-autosave", action="store_true", help="禁用写操作后自动落盘")
    sp.set_defaults(func=cmd_mcp)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
