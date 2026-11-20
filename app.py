# ═══ 多策略 RAG 文件問答（線上版）· Streamlit + Groq + ChromaDB + LINE Bot 投放 ═══
# 本機執行：streamlit run app.py
# 部署：GitHub → Streamlit Community Cloud / Render / Hugging Face Spaces（見 requirements.txt）
# Key 來源：側欄貼上；Groq Key 也可由部署者放在 st.secrets / 環境變數 GROQ_API_KEY（所有訪客共用，請留意額度）
# LINE：僅接受使用者在側欄貼上的 Channel Access Token（不讀取伺服器端設定，避免訪客濫用部署者帳號群發）
from __future__ import annotations

import sys

try:  # 部分雲端主機的 sqlite3 版本過舊，ChromaDB 需要 >= 3.35
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import csv
import io
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime

import chromadb
import numpy as np
import streamlit as st
from docx import Document
from docx.shared import Pt, RGBColor
from groq import Groq
from linebot.v3.messaging import ApiClient, BroadcastRequest, Configuration, MessagingApi, TextMessage
from linebot.v3.messaging.exceptions import ApiException
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

# ── 設定 ──
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")   # 可換 multilingual-e5-base、BAAI/bge-m3
MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3-32b"]
CHUNK, OVERLAP, CHILD, CHILD_OVERLAP, MAX_CTX = 500, 100, 200, 40, 7000
MAX_LIVE_DOCS = 20            # 伺服器同時保留的文件上限（超過時釋放最舊的，避免記憶體無限成長）
MAX_UPLOAD_MB = 30
LINE_LIMIT, LINE_HOST = 500, "https://api.line.me"
ACCENT = "#10b981"
DEFAULT_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_report.pdf")   # 預設文件（與 app.py 同資料夾）
DEFAULT_NAME = "114年國家永續發展年報.pdf"
DEFAULT_Q = "此份年報新增要點？"

STRATEGIES = {
    "hybrid": ("⚡ 混合搜尋", "語意 + 關鍵字，RRF 融合排名（推薦）"),
    "semantic": ("🔍 語意搜尋", "向量相似度檢索"),
    "tfidf": ("📊 關鍵字", "字元 n-gram，適合專有名詞、條號、數字"),
    "rerank": ("🎯 重新排序", "LLM 對候選片段批次評分後重排"),
    "multi_query": ("🔄 多查詢", "LLM 改寫 3 個角度問題聯合檢索"),
    "compress": ("✂️ 壓縮", "LLM 抽取每段最相關的句子"),
    "parent_child": ("📂 父子文檔", "小片段定位，回傳完整大片段"),
    "hyde": ("💡 HyDE", "先生成假設答案再檢索"),
}
_SENT = re.compile(r"(?<=[。！？!?；;])|\n+")


def secret(name: str) -> str:
    try:
        v = st.secrets.get(name, "")
    except Exception:  # 沒有 secrets.toml
        v = ""
    return str(v or os.getenv(name, "")).strip()


def split_text(text, size, overlap):
    sents = [s.strip() for s in _SENT.split(re.sub(r"[ \t　\xa0]+", " ", text)) if s and s.strip()]
    chunks, buf, fresh = [], "", 0
    for p in (s[i:i + size] for s in sents for i in range(0, len(s), size)):
        if fresh and len(buf) + len(p) + 1 > size:
            chunks.append(buf)
            buf, fresh = (buf[-overlap:].lstrip() if overlap else ""), 0
        buf = f"{buf} {p}" if buf else p
        fresh += len(p)
    return chunks + ([buf.strip()] if fresh else [])


@dataclass
class Doc:
    sid: str
    name: str
    chunks: list
    labels: list
    n_child: int = 0
    tfidf: object = field(default=None, repr=False)
    tm: object = field(default=None, repr=False)
    shared: bool = False   # 全站共用的預設文件：不隨個別使用者上傳而釋放


class LLM:
    def __init__(self, key, model):
        key = (key or "").strip()
        self.client = Groq(api_key=key) if key else None
        self.model = model or MODELS[0]

    def __call__(self, prompt, max_tokens=300, temp=0.3, system=None, raise_err=False):
        if not self.client:
            return None
        kw = dict(model=self.model, temperature=temp, max_completion_tokens=max_tokens,
                  messages=([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}])
        if "gpt-oss" in self.model:  # 推理模型：推理 token 也計入上限
            kw.update(max_completion_tokens=max(max_tokens * 3, 1024), reasoning_effort="low", include_reasoning=False)
        try:
            try:
                r = self.client.chat.completions.create(**kw)
            except Exception as e:
                if "reasoning" not in str(e).lower():
                    raise
                kw.pop("reasoning_effort", None)
                kw.pop("include_reasoning", None)
                r = self.client.chat.completions.create(**kw)
            return re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S).strip()
        except Exception:
            if raise_err:
                raise
            return None


@st.cache_data(ttl=600, show_spinner=False)
def list_models(key):
    try:
        ids = [m.id for m in Groq(api_key=key.strip()).models.list().data if getattr(m, "active", True) is not False]
    except Exception:
        return []
    return sorted(i for i in ids if not any(s in i.lower() for s in ("whisper", "tts", "orpheus", "guard", "playai")))


class RAG:
    """全站共用：嵌入模型 + 向量庫。文件本身存在各使用者的 session_state，以 sid 隔離。"""

    def __init__(self):
        self._embedder, self._lock = None, threading.Lock()
        self.e5 = "e5" in EMBED_MODEL.lower()
        self.db = chromadb.EphemeralClient()
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", EMBED_MODEL.split("/")[-1])[:40].strip("_")
        self.col = self.db.get_or_create_collection(f"p_{slug}", metadata={"hnsw:space": "cosine"})
        self.ccol = self.db.get_or_create_collection(f"c_{slug}", metadata={"hnsw:space": "cosine"})
        self.live: OrderedDict[str, Doc] = OrderedDict()

    def embedder(self):
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    import torch
                    from sentence_transformers import SentenceTransformer
                    m = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
                    m.max_seq_length = min(512, m.max_seq_length or 512)
                    self._embedder = m
        return self._embedder

    # ── 文件載入 ──
    def load(self, name, data: bytes, shared=False):
        t0 = time.time()
        ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
        if ext not in ("pdf", "docx"):
            raise ValueError("僅支援 PDF / DOCX")
        segs = self._pdf(data) if ext == "pdf" else self._docx(data)
        pairs = [(c, lab) for lab, txt in segs for c in split_text(txt, CHUNK, OVERLAP)]
        if not pairs:
            raise ValueError("無可擷取文字（可能是掃描檔，需先 OCR）")
        doc = Doc(uuid.uuid4().hex, name, [p[0] for p in pairs], [p[1] for p in pairs], shared=shared)
        try:
            self._add(self.col, [f"{doc.sid}_{i}" for i in range(len(pairs))], doc.chunks,
                      [{"sid": doc.sid, "idx": i} for i in range(len(pairs))])
            kids = [(f"{doc.sid}_{p}_{c}", t, p) for p, par in enumerate(doc.chunks)
                    for c, t in enumerate(split_text(par, CHILD, CHILD_OVERLAP))]
            doc.n_child = len(kids)
            self._add(self.ccol, [k[0] for k in kids], [k[1] for k in kids], [{"sid": doc.sid, "idx": k[2]} for k in kids])
            doc.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, max_features=50000)
            doc.tm = doc.tfidf.fit_transform(doc.chunks)
        except Exception:
            self.drop(doc)
            raise
        if not shared:
            self.live[doc.sid] = doc
        while len(self.live) > MAX_LIVE_DOCS:  # 釋放最舊的文件
            self.drop(self.live[next(iter(self.live))])
        return doc, f"✓ {doc.name} · {len(doc.chunks)} 片段 · {time.time() - t0:.1f}s"

    def drop(self, doc):
        if doc is None or doc.shared:
            return
        self.live.pop(doc.sid, None)
        for c in (self.col, self.ccol):
            try:
                c.delete(where={"sid": doc.sid})
            except Exception:
                pass

    @staticmethod
    def _pdf(data):
        return [(f"第 {i} 頁", t) for i, p in enumerate(PdfReader(io.BytesIO(data)).pages, 1) if (t := p.extract_text() or "").strip()]

    @staticmethod
    def _docx(data):
        from docx.oxml.table import CT_Tbl
        from docx.oxml.text.paragraph import CT_P
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        doc, segs, head, buf, nt = Document(io.BytesIO(data)), [], "前言", [], 0
        flush = lambda: (segs.append((head, "\n".join(buf))), buf.clear()) if buf else None
        for el in doc.element.body.iterchildren():
            if isinstance(el, CT_P):
                para = Paragraph(el, doc)
                txt = para.text.strip()
                if not txt:
                    continue
                style = (para.style.name if para.style is not None else "") or ""
                if style.lower().startswith(("heading", "title")) or style.startswith(("標題", "标题")):
                    flush()
                    head = txt[:40]
                buf.append(txt)
            elif isinstance(el, CT_Tbl):
                nt += 1
                rows = []
                for row in Table(el, doc).rows:
                    cells = []
                    for c in row.cells:
                        t = re.sub(r"\s+", " ", c.text).strip()
                        if t and (not cells or cells[-1] != t):
                            cells.append(t)
                    if cells:
                        rows.append(" | ".join(cells))
                if rows:
                    flush()
                    segs.append((f"表格 {nt}（{head}）", "\n".join(rows)))
        flush()
        return segs

    # ── 向量與檢索 ──
    def _enc(self, texts, query=False):
        if self.e5:
            texts = [("query: " if query else "passage: ") + t for t in texts]
        return self.embedder().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False).tolist()

    def _add(self, col, ids, docs, metas, b=500):
        for i in range(0, len(docs), b):
            col.add(ids=ids[i:i + b], documents=docs[i:i + b], metadatas=metas[i:i + b], embeddings=self._enc(docs[i:i + b]))

    def vec(self, doc, qs, k, child=False):
        total = doc.n_child if child else len(doc.chunks)
        if not total:
            return [[] for _ in qs]
        res = (self.ccol if child else self.col).query(query_embeddings=self._enc(qs, True), n_results=max(1, min(k, total)),
                                                       where={"sid": doc.sid}, include=["metadatas"])
        return [[int(m["idx"]) for m in ms] for ms in res["metadatas"]]

    @staticmethod
    def kw(doc, q, k):
        s = (doc.tm @ doc.tfidf.transform([q]).T).toarray().ravel()
        return [int(i) for i in np.argsort(-s)[:k] if s[i] > 0]

    @staticmethod
    def rrf(lists, k):
        sc = {}
        for lst in lists:
            for r, i in enumerate(dict.fromkeys(lst)):
                sc[i] = sc.get(i, 0) + 1 / (61 + r)
        return sorted(sc, key=sc.get, reverse=True)[:k]

    def retrieve(self, doc, q, strategy, k, llm):  # 回傳 [(片段索引, 文字)]
        hybrid = lambda n: self.rrf([self.vec(doc, [q], n * 2)[0], self.kw(doc, q, n * 2)], n)
        if strategy == "semantic":
            idx = self.vec(doc, [q], k)[0]
        elif strategy == "tfidf":
            idx = self.kw(doc, q, k)
        elif strategy == "parent_child":
            idx = list(dict.fromkeys(self.vec(doc, [q], k * 4, True)[0]))[:k]
        elif strategy == "multi_query":
            r = llm(f"將以下問題改寫成 3 個角度不同的繁體中文問題，每行一題，不加編號：\n{q}", 200, 0.7) or ""
            extra = [e for e in (l.strip("-•*·0123456789.、)） ") for l in r.splitlines()) if len(e) > 3][:3]
            idx = self.rrf(self.vec(doc, [q] + extra, k * 2), k)
        elif strategy == "hyde":
            h = llm(f"請針對以下問題寫一段約 100 字、語氣像文件原文的假設性答案（繁體中文）：\n{q}", 250, 0.7)
            idx = self.rrf(self.vec(doc, [q, h] if h else [q], k * 2), k)
        elif strategy == "rerank":
            cands = hybrid(min(max(k * 3, 6), 12))
            idx = cands[:k]
            if llm.client and len(cands) > 1:
                lst = "\n\n".join(f"[{n}] {doc.chunks[i][:400]}" for n, i in enumerate(cands, 1))
                r = llm(f"問題：{q}\n\n候選片段：\n{lst}\n\n評估每段與問題的相關度（0-10），只輸出「編號:分數」每行一個。",
                        15 * len(cands), 0) or ""
                sc = {int(a): float(b) for a, b in re.findall(r"\[?(\d+)\]?\s*[:：]\s*(\d+(?:\.\d+)?)", r)}
                idx = [cands[n] for n in sorted(range(len(cands)), key=lambda n: (-sc.get(n + 1, -1), n))[:k]]
        else:  # hybrid、compress
            idx = hybrid(k)
        hits = [(i, doc.chunks[i]) for i in idx]
        if strategy == "compress" and llm.client and hits:
            def comp(h):
                r = llm(f"從以下文本逐字擷取與問題「{q}」最相關的 1-3 句，不改寫不解釋；完全無關請輸出「無」。\n\n{h[1]}", 250, 0)
                return (h[0], r if r and r != "無" else h[1][:300])
            with ThreadPoolExecutor(min(4, len(hits))) as ex:
                hits = list(ex.map(comp, hits))
        return hits

    def answer(self, doc, q, strategy, k, model, key):
        """回傳 (回答, 參考片段, 耗時說明)。耗時說明非空 = 模型確實產生了回答。"""
        llm = LLM(key, model)
        t0 = time.time()
        hits = self.retrieve(doc, q, strategy, int(k), llm)
        t1 = time.time()
        blocks, used = [], 0
        for n, (i, txt) in enumerate(hits, 1):
            b = f"[片段{n}｜{doc.labels[i]}]\n{txt}"
            if used and used + len(b) > MAX_CTX:
                break
            blocks.append(b)
            used += len(b)
        ctx = "\n\n".join(blocks)
        src = f"📄 {doc.name} · {STRATEGIES[strategy][0]} · {len(blocks)} 片段 · 檢索 {t1 - t0:.2f}s\n{'─' * 40}\n\n{ctx}"
        if not hits:
            return "找不到相關片段，請換個問法或策略。", src, ""
        if not llm.client:
            return "⚠ 尚未設定 Groq API Key（檢索結果請看「檢索片段」）", src, ""
        ans = llm(f"請只根據上下文回答。若上下文沒有相關資訊，請說「文件中未提及」，不要編造。\n"
                  f"每個重點後以 [片段n] 標註依據，使用繁體中文條列整理。\n\n上下文：\n{ctx}\n\n問題：{q}",
                  1024, 0.2, "你是嚴謹的文件問答助手，只依據提供的上下文作答。", True)
        if not ans:
            return "（模型未回傳內容）", src, ""
        return ans, src, f"{llm.model} · 檢索 {t1 - t0:.1f}s · 生成 {time.time() - t1:.1f}s"


@st.cache_resource(show_spinner="⏳ 載入嵌入模型（首次需下載，約 470 MB）…")
def get_rag() -> RAG:
    r = RAG()
    r.embedder()
    return r


@st.cache_resource(show_spinner="⏳ 載入預設文件並建立索引…")
def get_default_doc():
    """啟動時載入一次預設文件，所有訪客共用（唯讀）。找不到檔案則回傳 None。"""
    if not os.path.exists(DEFAULT_PDF):
        return None
    try:
        with open(DEFAULT_PDF, "rb") as f:
            doc, msg = get_rag().load(DEFAULT_NAME, f.read(), shared=True)
        return doc, msg
    except Exception:  # noqa: BLE001
        return None


# ── 匯出：Word / CSV（回傳 bytes 供下載）──
def _md_runs(par, text):  # 將 **粗體** 轉成 Word 粗體
    for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if part:
            par.add_run(part).bold = bool(i % 2)


def export_csv(records) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(records[0]))
    w.writeheader()
    w.writerows(records)
    return buf.getvalue().encode("utf-8-sig")  # utf-8-sig：Excel 開啟中文不亂碼


def export_docx(records, with_src=True) -> bytes:
    doc = Document()
    st_ = doc.styles["Normal"]
    st_.font.name, st_.font.size = "Microsoft JhengHei", Pt(11)
    st_.element.rPr.rFonts.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia", "Microsoft JhengHei")
    doc.add_heading("RAG 文件問答紀錄", 0)
    meta = doc.add_paragraph(f"文件：{records[-1]['文件']}　·　匯出時間：{datetime.now():%Y-%m-%d %H:%M}　·　共 {len(records)} 題")
    meta.runs[0].font.color.rgb = RGBColor(0x6B, 0x72, 0x80)
    for n, r in enumerate(records, 1):
        doc.add_heading(f"Q{n}．{r['問題']}", level=1)
        info = doc.add_paragraph(f"{r['時間']}　·　{r['策略']}　·　{r['模型']}")
        info.runs[0].font.size, info.runs[0].font.color.rgb = Pt(9), RGBColor(0x6B, 0x72, 0x80)
        for line in r["回答"].splitlines():
            line = line.rstrip()
            if not line.strip():
                continue
            if m := re.match(r"^\s*#{1,6}\s+(.*)", line):
                doc.add_heading(m.group(1).replace("**", ""), level=2)
                continue
            if m := re.match(r"^(\s*)[-*•]\s+(.*)", line):
                par = doc.add_paragraph(style="List Bullet 2" if len(m.group(1)) >= 2 else "List Bullet")
                _md_runs(par, m.group(2))
            elif m := re.match(r"^\s*\d+[.)]\s+(.*)", line):
                par = doc.add_paragraph(style="List Number")
                _md_runs(par, m.group(1))
            else:
                _md_runs(doc.add_paragraph(), line)
        if with_src and r["參考片段"]:
            h = doc.add_paragraph()
            run = h.add_run("參考片段")
            run.bold, run.font.size = True, Pt(9)
            for blk in r["參考片段"].split("\n\n"):
                p = doc.add_paragraph(blk.strip())
                p.paragraph_format.left_indent = Pt(12)
                for run in p.runs:
                    run.font.size, run.font.color.rgb = Pt(8.5), RGBColor(0x55, 0x5B, 0x66)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ── LINE Bot 投放 ──
def plain(md: str) -> str:
    """LINE 不渲染 Markdown：移除粗體標記，標題轉成一般行。"""
    md = re.sub(r"\*\*(.+?)\*\*", r"\1", md)
    return re.sub(r"^\s*#{1,6}\s+", "", md, flags=re.M)


def line_text(q, ans, limit=LINE_LIMIT):
    """組成 LINE 訊息：問題 + 回答（去除 Markdown 與 [片段n] 標註），總長不超過 limit 字。"""
    body = re.sub(r"[ \t]*\[片段[^\]]*\]", "", plain(ans))
    body = re.sub(r"[ \t]+(?=\n|$)", "", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    q = " ".join(q.split())
    msg = f"❓ {q[:59] + '…' if len(q) > 60 else q}\n\n{body}"
    if len(msg) > limit:
        cut = msg[:limit - 1]
        end = max(cut.rfind(c) for c in "\n。！？；")  # 盡量在句尾截斷
        msg = (cut[:end + 1] if end > limit * 0.6 else cut).rstrip() + "…"
    return msg


def line_send(token, text, host=LINE_HOST):
    """Broadcast 給所有加入此官方帳號的好友（會消耗每月訊息額度）。"""
    with ApiClient(Configuration(access_token=token, host=host)) as c:
        MessagingApi(c).broadcast(BroadcastRequest(messages=[TextMessage(text=text[:LINE_LIMIT])]))


def line_check(token, host=LINE_HOST):
    """驗證 Token，回傳官方帳號名稱。"""
    with ApiClient(Configuration(access_token=token, host=host)) as c:
        return MessagingApi(c).get_bot_info().display_name


def line_err(e):
    if isinstance(e, ApiException):
        hint = {400: "請求格式錯誤", 401: "Token 無效或已過期", 403: "權限不足", 429: "超過訊息額度或請求過於頻繁"}.get(e.status, "")
        return f"HTTP {e.status} {hint}".strip()
    return f"{type(e).__name__}: {e}"


# ═══════════════ 介面 ═══════════════
st.set_page_config(page_title="114年國家永續發展年報 智慧問答", page_icon="📚", layout="wide")
st.markdown(f"""<style>
.block-container{{padding-top:2rem}}
div[data-testid="stSidebar"] .stButton>button{{width:100%}}
.stButton>button[kind="primary"]{{background:{ACCENT};border-color:{ACCENT}}}
</style>""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("doc", None)
ss.setdefault("doc_sig", None)
ss.setdefault("msgs", [])      # [{q, ans, src, timing, strategy, model, time, ok, line}]
ss.setdefault("line_name", None)

try:
    rag = get_rag()
except Exception as e:  # noqa: BLE001
    st.error(f"嵌入模型載入失敗：{type(e).__name__}: {e}")
    st.stop()

if ss.doc is None and not ss.get("default_tried"):   # 新訪客：自動套用預設文件
    ss.default_tried = True
    if (dd := get_default_doc()) is not None:
        ss.doc, ss.doc_msg, ss.doc_sig = dd[0], dd[1] + "（預設）", (DEFAULT_NAME, 0)

with st.sidebar:
    st.title("📚 智慧問答")
    st.caption("Groq × ChromaDB × Streamlit")

    st.subheader("🔑 Groq API Key")
    gkey = st.text_input("Groq API Key", type="password", placeholder="gsk_…", label_visibility="collapsed")
    gkey = gkey.strip() or secret("GROQ_API_KEY")
    models = list_models(gkey) if gkey else []
    if gkey and models:
        st.success(f"Key 有效 · {len(models)} 個模型")
    elif gkey:
        st.error("Key 無效或無法取得模型")
    else:
        st.warning("尚未設定 Key（仍可檢索，但無法生成回答）")
    options = models or MODELS
    model = st.selectbox("模型", options, index=options.index(MODELS[0]) if MODELS[0] in options else 0)

    st.subheader("📂 文件")
    up = st.file_uploader("上傳 PDF / DOCX", type=["pdf", "docx"], label_visibility="collapsed")
    if ss.doc is not None and ss.doc.shared:
        st.caption("目前使用預設文件；上傳新檔即可改問其他文件。")
    if up is not None:
        sig = (up.name, up.size)
        if sig != ss.doc_sig:
            if up.size > MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"檔案超過 {MAX_UPLOAD_MB} MB")
            else:
                try:
                    with st.spinner("解析並建立索引中…"):
                        new, msg = rag.load(up.name, up.getvalue())
                    rag.drop(ss.doc)  # 新文件成功後才釋放舊文件
                    ss.doc, ss.doc_sig, ss.doc_msg = new, sig, msg
                    ss.msgs = []
                except ValueError as e:
                    st.error(f"✗ {e}")
                except Exception as e:  # noqa: BLE001
                    st.error(f"✗ 載入失敗：{type(e).__name__}: {e}")
    if ss.doc is not None and (ss.doc.shared or ss.doc.sid in rag.live):
        st.success(ss.get("doc_msg", ss.doc.name))
    elif ss.doc is not None:  # 伺服器已因容量釋放此文件
        st.warning("文件已被伺服器釋放，請重新上傳")
        ss.doc, ss.doc_sig = None, None
    else:
        st.caption("尚未載入文件")

    st.subheader("檢索策略")
    skey = st.selectbox("檢索策略", list(STRATEGIES), format_func=lambda s: STRATEGIES[s][0], label_visibility="collapsed")
    st.caption(STRATEGIES[skey][1])
    topk = st.slider("片段數量 Top-K", 1, 10, 4)

    st.subheader("💬 LINE Bot 投放")
    ltok = st.text_input("LINE Channel Access Token", type="password", placeholder="Channel Access Token",
                         label_visibility="collapsed").strip()
    if st.button("驗證 Token", disabled=not ltok):
        try:
            ss.line_name = line_check(ltok)
            st.success(f"Token 有效 · {ss.line_name or 'LINE 官方帳號'}")
        except Exception as e:  # noqa: BLE001
            ss.line_name = None
            st.error(f"驗證失敗：{line_err(e)}")
    elif not ltok:
        st.caption("貼上 Token 後，可把回答（≤500 字）群發給官方帳號的所有好友。Token 只在本次連線使用，不會儲存。")
    auto_line = st.checkbox(f"回答後自動投放（≤{LINE_LIMIT} 字）", disabled=not ltok,
                            help="每次回答都會 Broadcast 給所有好友，並消耗每月訊息額度")
    if auto_line:
        st.warning("已啟用：每次回答都會群發給所有好友！")

    st.subheader("匯出問答紀錄")
    if ss.msgs:
        recs = [{"時間": m["time"], "文件": ss.doc.name if ss.doc else "", "問題": m["q"], "回答": m["ans"],
                 "策略": STRATEGIES[m["strategy"]][0], "模型": m["model"],
                 "參考片段": m["src"].split("─" * 40, 1)[-1].strip()} for m in ss.msgs]
        with_src = st.checkbox("Word 包含參考片段", value=True)
        c1, c2 = st.columns(2)
        stamp = f"RAG問答紀錄_{datetime.now():%Y%m%d_%H%M%S}"
        c1.download_button("📄 Word", export_docx(recs, with_src), f"{stamp}.docx",
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        c2.download_button("📊 CSV", export_csv(recs), f"{stamp}.csv", "text/csv")
    else:
        st.caption("目前沒有問答紀錄可匯出")

    if st.button("🗑 清除對話"):
        ss.msgs = []
        st.rerun()


def send_line_ui(token, text, key):
    """投放並回傳狀態字串。"""
    try:
        line_send(token, text)
        return f"✓ 已投放到 LINE（{len(text)} 字）· {datetime.now():%H:%M:%S}"
    except Exception as e:  # noqa: BLE001
        return f"✗ LINE 投放失敗：{line_err(e)}"


st.header("📚 114年國家永續發展年報 智慧問答")
if ss.doc is None:
    st.info("請先在左側上傳 PDF 或 DOCX 文件，再開始提問。例如：這份文件的重點是什麼？")
else:
    st.caption(f"📄 目前文件：{ss.doc.name}")

for i, m in enumerate(ss.msgs):
    with st.chat_message("user"):
        st.write(m["q"])
    with st.chat_message("assistant"):
        st.markdown(m["ans"])
        if m["timing"]:
            st.caption(m["timing"])
        with st.expander("🔎 檢索片段"):
            st.text(m["src"])
        if m["ok"]:
            if m.get("line"):
                st.caption(m["line"])
            with st.expander("💬 投放到 LINE"):
                txt = st.text_area("訊息內容（可編輯）", line_text(m["q"], m["ans"]), max_chars=LINE_LIMIT, key=f"lt_{i}", height=180)
                st.caption(f"{len(txt)} / {LINE_LIMIT} 字")
                sure = st.checkbox("我確認要群發給所有好友", key=f"ok_{i}")
                if st.button("📤 投放", key=f"sb_{i}", disabled=not (ltok and sure and txt.strip())):
                    with st.spinner("投放中…"):
                        m["line"] = send_line_ui(ltok, txt, i)
                    st.rerun()
                if not ltok:
                    st.caption("請先在側欄貼上 LINE Token")

ss.setdefault("q_in", DEFAULT_Q)
with st.form("ask", clear_on_submit=False):
    qc, bc = st.columns([6, 1], vertical_alignment="bottom")
    q_text = qc.text_input("問題", key="q_in", label_visibility="collapsed", placeholder="輸入問題，Enter 送出")
    go = bc.form_submit_button("送出", type="primary", disabled=ss.doc is None, use_container_width=True)

if go and (q := q_text.strip()):
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        with st.spinner("思考中…"):
            try:
                ans, src, timing = rag.answer(ss.doc, q, skey, topk, model, gkey)
                err = None
            except Exception as e:  # noqa: BLE001
                ans, src, timing, err = f"✗ {type(e).__name__}: {e}", "", "", e
    m = {"q": q, "ans": ans, "src": src, "timing": timing, "strategy": skey, "model": model or MODELS[0],
         "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "ok": bool(timing) and err is None, "line": ""}
    if m["ok"] and auto_line and ltok:
        m["line"] = send_line_ui(ltok, line_text(q, ans), len(ss.msgs))
    ss.msgs.append(m)
    st.rerun()
