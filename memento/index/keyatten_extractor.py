"""
keyatten 惊奇关键词提取器 — Memento 封装

基于 keyatten 库的注意力权重关键词提取，支持 IDF 惊奇度加权。
比旧的 TF-IDF 统计法和 Qwen3-token 法更精准：
  - 用 BERT 注意力权重定位关键实体（不是简单的词频统计）
  - IDF 惊奇度让罕见实体（如"机电一体化"）排在常见词前面
  - 零水词：不会提取"今天"、"发现"、"异常"之类的通用词
"""

from __future__ import annotations
import hashlib
import re
from pathlib import Path

from keyatten import KeyAttenExtractor


class MemoryKeywordExtractor:
    """Memento 关键词提取封装

    用法:
        ext = MemoryKeywordExtractor()
        ext.update_idf(all_memory_texts)   # 建立惊奇度基线
        keywords = ext.extract("一段记忆文本")  # => ["钢琴", "USB", ...]
    """

    # 方案 B：phrase-merge 断点词典。
    # 这些词虽然是内容词（动词/副词/形容词），但语义太泛/太碎，
    # 拼进 n-gram 只会产生垃圾短语（如"引入软件确实"、"发现钢琴"）。
    # 碰到它们就截断合并窗口，不允许跨过它们拼接。
    PHRASE_MERGE_STOP_TOKENS: frozenset = frozenset({
        # 泛动词/引导词
        "引入", "发现", "解决", "需要", "进行", "开始", "觉得", "发现",
        "认为", "感觉", "知道", "理解", "看到", "听到", "想起", "提到",
        # 副词/语气词（即使 jieba 标为形容词/副词）
        "确实", "真的", "其实", "当然", "肯定", "应该", "可能", "也许",
        "非常", "特别", "比较", "稍微", "有点", "一点", "一下",
        # 连接/转折
        "然后", "但是", "不过", "而且", "因为", "所以", "虽然", "如果",
        # 泛指
        "这个", "那个", "这种", "那种", "什么", "怎么", "为什么",
    })

    def __init__(
        self,
        model_path: str = "models/Qwen3-Embedding-0.6B",
        bio_model_path: str | None = None,
        device: str = "cuda",
        dtype: str | None = "float16",
        default_top_k: int = 5,
        method: str = "fusion_attn",
        cache_enabled: bool = True,
        cache_dir: str | Path = "data/keyatten_cache",
        phrase_merge_enabled: bool = True,
        phrase_merge_top_k: int = 3,
        phrase_backend: str = "jieba_pos",
    ) -> None:
        self._default_top_k = default_top_k
        self._method = method
        self._phrase_merge_enabled = phrase_merge_enabled
        self._phrase_merge_top_k = phrase_merge_top_k
        self._phrase_backend = phrase_backend
        self._idf_lookup: dict[str, float] | None = None
        self._idf_seen_text_hashes: set[str] = set()
        self._idf_needs_full_rebuild = False
        resolved_model_path = self._resolve_model_path(model_path)
        self._extractor = KeyAttenExtractor(
            model=resolved_model_path,
            language="zh",
            device=device,
            candidate_scoring="bio" if bio_model_path else "word",
            dtype=dtype,
            bio_model_path=bio_model_path,
            cache_enabled=cache_enabled,
            cache_dir=cache_dir,
        )

    def extract(self, text: str, top_k: int | None = None) -> list[str]:
        """提取记忆关键锚点。有 IDF 时自动启用惊奇度加权。"""
        method = f"{self._method}_idf" if self._idf_lookup else self._method
        limit = top_k or self._default_top_k
        keywords = self._extractor.extract_keywords(
            text, method=method,
            top_k=limit,
            idf_lookup=self._idf_lookup,
        )
        if not self._phrase_merge_enabled:
            return self._final_filter(keywords)
        merged = self._merge_adjacent_phrases(text, keywords)
        # 合并短语先过滤垃圾（含 stop token / 子串冗余）
        merged = self._suppress_junk_phrases(merged, keywords)
        combined = list(dict.fromkeys(merged + keywords))[:limit]
        return self._final_filter(combined)

    def _final_filter(self, keywords: list[str]) -> list[str]:
        """最终过滤：子串抑制（NMS）。

        如果关键词 A 是关键词 B 的子串（如 A="软件"，B="引入软件"），
        说明 B 是 A 的冗余扩展 → 丢弃 B，保留更简洁的 A。
        这能拦住 keyatten 因 IDF 虚高而抽出的垃圾 span。
        """
        if len(keywords) < 2:
            return keywords
        result = []
        kw_set = set(keywords)
        for kw in keywords:
            # 如果有更短的关键词是当前词的子串 → 丢弃当前词
            is_redundant = any(
                other != kw and other in kw and len(other) >= 2
                for other in kw_set
            )
            if not is_redundant:
                result.append(kw)
        return result

    def _suppress_junk_phrases(
        self, merged: list[str], raw_keywords: list[str]
    ) -> list[str]:
        """过滤垃圾合并短语。

        1. 含 stop token 的短语直接丢弃（方案 B 的兜底）
        2. 如果短语 A 包含原始关键词 B，且 A 不是 B，
           说明 A 是 B 的冗余扩展 → 丢弃 A（子串抑制）
        """
        result = []
        raw_set = set(raw_keywords)
        for phrase in merged:
            # 含 stop token → 丢
            if any(st in phrase for st in self.PHRASE_MERGE_STOP_TOKENS):
                continue
            # 被原始关键词子串覆盖 → 丢
            if any(rk in phrase and rk != phrase and len(rk) >= 2
                   for rk in raw_set):
                continue
            result.append(phrase)
        return result

    def update_idf(self, corpus: list[str]) -> int:
        """喂入记忆库文本，增量更新 IDF 惊奇度基线。

        调用方可以继续传全量文本；本封装只会把未见过的文本追加进 IDF
        统计，避免重复扫描旧文章和重复计数。
        """
        if self._idf_needs_full_rebuild:
            self._idf_lookup = self._extractor.fit_idf(corpus)
            self._idf_seen_text_hashes = {self._text_hash(text) for text in corpus}
            self._idf_needs_full_rebuild = False
            return len(self._idf_lookup)

        new_texts: list[str] = []
        for text in corpus:
            text_hash = self._text_hash(text)
            if text_hash in self._idf_seen_text_hashes:
                continue
            self._idf_seen_text_hashes.add(text_hash)
            new_texts.append(text)

        if new_texts or self._idf_lookup is None:
            self._idf_lookup = self._extractor.update_idf(new_texts)
        return len(self._idf_lookup)

    def get_idf(self) -> dict[str, float] | None:
        """导出 IDF（用于持久化到磁盘）。"""
        return dict(self._idf_lookup) if self._idf_lookup else None

    def set_idf(self, idf_lookup: dict[str, float]) -> None:
        """从持久化中恢复 IDF。"""
        self._idf_lookup = dict(idf_lookup)
        self._extractor.idf_lookup = dict(idf_lookup)
        self._idf_needs_full_rebuild = True

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _resolve_model_path(model_path: str) -> str:
        path = Path(model_path)
        if path.is_absolute() or path.exists():
            return str(path)
        project_root = Path(__file__).resolve().parents[2]
        project_path = project_root / path
        if project_path.exists():
            return str(project_path)
        return model_path

    def _merge_adjacent_phrases(self, text: str, keywords: list[str]) -> list[str]:
        """把相邻关键词恢复成短语候选。

        KeyAtten 的 attention 负责打分，但候选生成可能把“Docker Compose”
        切成“千金 / 小姐”。优先用 NLP 词性边界合并；失败时退回保守
        的字符邻近规则。
        """
        if len(keywords) < 2:
            return []
        if self._phrase_backend == "auto":
            phrases = self._merge_phrases_by_spacy(text, keywords)
            if phrases:
                return phrases[:self._phrase_merge_top_k]
            phrases = self._merge_phrases_by_jieba_pos(text, keywords)
            if phrases:
                return phrases[:self._phrase_merge_top_k]
        if self._phrase_backend == "spacy":
            phrases = self._merge_phrases_by_spacy(text, keywords)
            if phrases:
                return phrases[:self._phrase_merge_top_k]
        if self._phrase_backend == "jieba_pos":
            phrases = self._merge_phrases_by_jieba_pos(text, keywords)
            # jieba_pos 返回空说明没有好的短语可合并，
            # fallback 字符邻近规则只会造垃圾（如"引入软件"），不执行。
            return phrases[:self._phrase_merge_top_k]
        if self._phrase_backend == "none":
            return []
        spans = []
        for kw in keywords:
            if len(kw) < 2:
                continue
            start = text.find(kw)
            if start < 0:
                continue
            spans.append((start, start + len(kw), kw))
        spans.sort(key=lambda item: (item[0], item[1]))

        phrases: list[str] = []
        for i, (start_a, end_a, kw_a) in enumerate(spans):
            for start_b, end_b, kw_b in spans[i + 1:]:
                if start_b < end_a:
                    continue
                gap = text[end_a:start_b]
                if len(gap) > 2:
                    break
                if gap in {
                    "是", "的", "了", "和", "与", "及", "或",
                    "这个", "那个", "一种", "一个", "一些",
                }:
                    continue
                if re.search(r"[，。！？；：,.!?;:\n\r]", gap):
                    continue
                phrase = text[start_a:end_b].strip()
                if self._is_good_merged_phrase(phrase, kw_a, kw_b):
                    phrases.append(phrase)
        phrases = list(dict.fromkeys(phrases))
        phrases.sort(key=lambda phrase: (-len(phrase), text.find(phrase)))
        return phrases[:self._phrase_merge_top_k]

    def _merge_phrases_by_jieba_pos(self, text: str, keywords: list[str]) -> list[str]:
        try:
            import jieba.posseg as pseg
        except ImportError:
            return []

        keyword_set = set(keywords)
        tokens = [(w.word, w.flag) for w in pseg.cut(text)]
        phrases: list[str] = []
        window: list[str] = []
        for word, flag in tokens:
            # 方案 B：碰到断点词就截断窗口，不允许跨过它拼接 n-gram
            if word in self.PHRASE_MERGE_STOP_TOKENS:
                phrases.extend(self._phrases_from_token_window(window, keyword_set))
                window = []
                continue
            if self._is_content_token(word, flag):
                window.append(word)
                continue
            phrases.extend(self._phrases_from_token_window(window, keyword_set))
            window = []
        phrases.extend(self._phrases_from_token_window(window, keyword_set))

        phrases = list(dict.fromkeys(phrases))
        phrases.sort(key=lambda phrase: (-self._keyword_coverage(phrase, keyword_set), -len(phrase), text.find(phrase)))
        return phrases[:self._phrase_merge_top_k]

    def _merge_phrases_by_spacy(self, text: str, keywords: list[str]) -> list[str]:
        nlp = self._load_spacy_pipeline(text)
        if nlp is None:
            return []

        keyword_set = set(keywords)
        try:
            doc = nlp(text)
        except Exception:
            return []

        phrases: list[str] = []
        try:
            noun_chunks = list(doc.noun_chunks)
        except (AttributeError, NotImplementedError, ValueError):
            noun_chunks = []
        for chunk in noun_chunks:
            phrase = self._normalize_phrase_text(chunk.text)
            if self._is_good_multilingual_phrase(phrase, keyword_set):
                phrases.append(phrase)

        window: list[str] = []
        for token in doc:
            if self._is_spacy_content_token(token):
                window.append(token.text)
                continue
            phrases.extend(self._phrases_from_spacy_window(window, keyword_set))
            window = []
        phrases.extend(self._phrases_from_spacy_window(window, keyword_set))

        phrases = list(dict.fromkeys(phrases))
        phrases.sort(key=lambda phrase: (-self._keyword_coverage(phrase, keyword_set), -len(phrase), text.find(phrase)))
        return phrases[:self._phrase_merge_top_k]

    @classmethod
    def _load_spacy_pipeline(cls, text: str):
        try:
            import spacy
        except ImportError:
            return None

        model_names = (
            ("zh_core_web_sm", "zh_core_web_trf")
            if cls._looks_chinese(text)
            else ("en_core_web_sm", "xx_sent_ud_sm")
        )
        for model_name in model_names:
            try:
                return spacy.load(model_name)
            except Exception:
                continue
        try:
            return spacy.blank("zh" if cls._looks_chinese(text) else "en")
        except Exception:
            return None

    @staticmethod
    def _looks_chinese(text: str) -> bool:
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        return chinese_chars >= max(2, len(text) // 5)

    @staticmethod
    def _is_spacy_content_token(token) -> bool:
        text = token.text.strip()
        if not text or token.is_space or token.is_punct:
            return False
        if token.is_stop:
            return False
        pos = getattr(token, "pos_", "")
        if pos:
            return pos in {"NOUN", "PROPN", "VERB", "ADJ", "X"}
        return bool(re.search(r"[\w\u4e00-\u9fff]", text))

    def _phrases_from_spacy_window(self, words: list[str], keywords: set[str]) -> list[str]:
        phrases: list[str] = []
        max_ngram = min(4, len(words))
        for size in range(2, max_ngram + 1):
            for start in range(0, len(words) - size + 1):
                phrase = self._normalize_phrase_text(" ".join(words[start:start + size]))
                if self._is_good_multilingual_phrase(phrase, keywords):
                    phrases.append(phrase)
        return phrases

    @staticmethod
    def _normalize_phrase_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text.strip())
        if re.fullmatch(r"[\u4e00-\u9fff ]+", text):
            return text.replace(" ", "")
        return text

    def _phrases_from_token_window(self, words: list[str], keywords: set[str]) -> list[str]:
        phrases: list[str] = []
        max_ngram = min(4, len(words))
        for size in range(2, max_ngram + 1):
            for start in range(0, len(words) - size + 1):
                phrase = "".join(words[start:start + size])
                if self._is_good_nlp_phrase(phrase, keywords):
                    phrases.append(phrase)
        return phrases

    @staticmethod
    def _is_content_token(word: str, flag: str) -> bool:
        if not re.fullmatch(r"[\u4e00-\u9fff]+", word):
            return False
        if flag.startswith(("r", "c", "u", "p", "m", "q", "x", "f")):
            return False
        if len(word) == 1 and not flag.startswith(("a", "n")):
            return False
        return flag.startswith(("n", "v", "a", "i", "l", "j"))

    @staticmethod
    def _keyword_coverage(phrase: str, keywords: set[str]) -> int:
        return sum(1 for kw in keywords if len(kw) >= 2 and kw in phrase)

    def _is_good_nlp_phrase(self, phrase: str, keywords: set[str]) -> bool:
        if not re.fullmatch(r"[\u4e00-\u9fff]{3,12}", phrase):
            return False
        if len(set(phrase)) <= 1:
            return False
        return self._keyword_coverage(phrase, keywords) >= 2

    def _is_good_multilingual_phrase(self, phrase: str, keywords: set[str]) -> bool:
        if not 3 <= len(phrase) <= 80:
            return False
        if len(set(phrase)) <= 1:
            return False
        return self._keyword_coverage(phrase, keywords) >= 2

    @staticmethod
    def _is_good_merged_phrase(phrase: str, left: str, right: str) -> bool:
        if phrase in {left, right}:
            return False
        if not re.fullmatch(r"[\u4e00-\u9fff]{3,10}", phrase):
            return False
        if len(set(phrase)) <= 1:
            return False
        bad_prefixes = ("为什么", "这个", "那个", "就是", "其实", "如果", "因为", "所以")
        bad_suffixes = ("什么", "如何", "怎么", "时候", "一点", "一样")
        return not (phrase.startswith(bad_prefixes) or phrase.endswith(bad_suffixes))

    def build_keyword_edges(
        self,
        node_ids: list[str],
        node_keywords: list[list[str]],
        min_overlap: int = 1,
    ) -> list[tuple[str, str, float, list[str]]]:
        """基于关键词重叠建边

        两个记忆共享越多的惊奇关键词，边权重越大。

        Args:
            node_ids: 节点 ID 列表
            node_keywords: 每个节点的关键词列表（与 node_ids 一一对应）
            min_overlap: 最少共享关键词数

        Returns:
            [(src, tgt, weight, shared_keywords), ...]
        """
        edges = []
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                kws_i = set(node_keywords[i])
                kws_j = set(node_keywords[j])
                shared = kws_i & kws_j
                if len(shared) >= min_overlap:
                    weight = len(shared) / max(len(kws_i), len(kws_j))
                    edges.append((
                        node_ids[i], node_ids[j],
                        round(weight, 4),
                        list(shared),
                    ))
        return edges
