# streamlit_app.py
import os, io, base64, re, json
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# safe import for pipeline.py
try:
    from pipeline import build_or_load_index
except Exception as e:
    import importlib.util, pathlib
    st.warning(f"Direct import failed: {e}. Falling back to path-based loader.")
    p = pathlib.Path(__file__).with_name("pipeline.py")
    spec = importlib.util.spec_from_file_location("pipeline", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    build_or_load_index = mod.build_or_load_index

# --- Runtime deps ---
try:
    import faiss  # pip install faiss-cpu
except Exception:
    st.error("FAISS is required. Please add `faiss-cpu` to requirements.txt.")
    st.stop()

from sentence_transformers import SentenceTransformer

EMB_MODEL = "clip-ViT-B-32"   # for query encoding when image is uploaded (kept)
EMB_DIM = 512
REQUEST_TIMEOUT = 7
MAX_ROWS = 8000

REQUIRED_COLS = ["title", "brand", "price", "features", "description", "image_url"]
ALIASES = {
    "product_title": "title", "name": "title", "title": "title",
    "brand_name": "brand", "brand": "brand",
    "img_url": "image_url","imageurl":"image_url","imageurlhighres":"image_url","image_link":"image_url","images":"image_url",
    "feature":"features","features":"features",
    "text":"description","description":"description",
    # Henry CSV 예
    "product name":"title","selling price":"price","about product":"description",
    "product specification":"features","technical details":"features","shipping weight":"features",
    "product dimensions":"features","variants":"features","image":"image_url","text_blob":"description",
}
URL_RE = re.compile(r"https?://[^\s|,;]+", re.IGNORECASE)

def _first_url(s: str) -> str:
    if not isinstance(s, str): return ""
    m = URL_RE.search(s);  return m.group(0) if m else ""

@st.cache_resource
def load_clip(device: Optional[str] = None):
    return SentenceTransformer(EMB_MODEL, device=device)

def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    plan = {}
    for c in list(df.columns):
        key = str(c).lower().strip()
        if key in ALIASES and ALIASES[key] not in df.columns:
            plan[c] = ALIASES[key]
    if plan: df = df.rename(columns=plan)

    merge_cols = [c for c in ["features","product specification","technical details","variants","product dimensions","shipping weight"] if c in df.columns]
    if merge_cols:
        def _merge(row):
            parts = []
            for c in merge_cols:
                v = row.get(c, "")
                if isinstance(v, (list, tuple)):
                    v = " ; ".join([str(x) for x in v if str(x).strip()])
                parts.append(str(v))
            return " ; ".join([p for p in parts if p and p.lower() not in ("nan","none","null")])
        df["features"] = df.apply(_merge, axis=1)

    if "description" in df.columns: df["description"] = df["description"].fillna("").astype(str)
    if "image_url" in df.columns:   df["image_url"] = df["image_url"].astype(str).apply(_first_url)
    for c in ["title","brand","price","features","description","image_url"]:
        if c in df.columns: df[c] = df[c].fillna("").astype(str)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.warning(f"Missing columns {missing}. The app will still work with available columns, but retrieval quality may be lower.")

    text_cols = [c for c in ["title","description","features","brand","price"] if c in df.columns]
    if text_cols:
        def _has_any(row): return any(isinstance(row.get(c,""), str) and row.get(c,"").strip() for c in text_cols)
        df = df[df.apply(_has_any, axis=1)].copy()
    return df

def read_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"): return pd.read_parquet(path)
    if path.endswith(".csv") or path.endswith(".gz"): return pd.read_csv(path, compression="infer")
    try: return pd.read_parquet(path)
    except Exception: return pd.read_csv(path, compression="infer")

# --------- Artifacts I/O ----------
def artifacts_exist(dirpath: str = "artifacts") -> bool:
    return (os.path.exists(os.path.join(dirpath, "index.faiss"))
            and os.path.exists(os.path.join(dirpath, "metas.json")))

def load_artifacts(dirpath: str = "artifacts") -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    idx = faiss.read_index(os.path.join(dirpath, "index.faiss"))
    with open(os.path.join(dirpath, "metas.json"), "r", encoding="utf-8") as f:
        metas = json.load(f)
    return idx, metas

# ---- Query helpers ----
def _download_image(url: str) -> Optional[bytes]:
    import requests
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT); r.raise_for_status(); return r.content
    except Exception:
        return None

def _encode_query(model: SentenceTransformer, text: str, img: Optional[Image.Image]) -> np.ndarray:
    tv = model.encode([text], normalize_embeddings=True).astype("float32") if text else None
    iv = model.encode([img],  normalize_embeddings=True).astype("float32") if img is not None else None
    if tv is not None and iv is not None: return ((tv + iv) / 2.0).astype("float32")
    return (tv if tv is not None else iv).astype("float32")

def _search(index: faiss.IndexFlatIP, qvec: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    D, I = index.search(qvec, k);  return D[0], I[0]

def _render_card(meta: Dict[str,Any], score: float):
    title = (meta.get("title") or meta.get("product_title") or "").strip() or "(no title)"
    brand = (meta.get("brand") or "").strip()
    price = (meta.get("price") or "").strip()
    st.markdown(f"**🔎 {title}**  \n{(' · '.join([x for x in [brand, price] if x]))}")
    st.caption(f"`score={score:.3f}`")
    if "_image_b64" in meta:
        try: st.image(Image.open(io.BytesIO(base64.b64decode(meta["_image_b64"]))), width=220)
        except Exception: pass
    short = (str(meta.get("features") or meta.get("description") or "")).strip()[:300]
    if short: st.caption(short + ("..." if len(short)==300 else ""))

@st.cache_resource(show_spinner=False)
def _session_llm():
    api_key = (os.getenv("OPENAI_API_KEY")
               or (st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None)
               or st.session_state.get("OPENAI_API_KEY"))
    if not api_key: return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception:
        return None

def _answer_with_openai(query: str, snippets: List[Dict[str,Any]]) -> str:
    client = _session_llm()
    ctx = "\n\n".join(
        f"[{i+1}] {s.get('title','')} ({s.get('brand','')}) {s.get('price','')}\n"
        f"Features: {s.get('features','')}\nDesc: {s.get('description','')}"
        for i, s in enumerate(snippets)
    )
    if client is None:
        return f"(No LLM key · showing retrieved summary)\nQuestion: {query}\n\n{ctx}"
    try:
        model_name = (os.getenv("OPENAI_MODEL")
                      or (st.secrets.get("OPENAI_MODEL") if "OPENAI_MODEL" in st.secrets else "gpt-4o-mini")
                      or st.session_state.get("OPENAI_MODEL_BOX") or "gpt-4o-mini")
        msgs = [
            {"role":"system","content":"You are a helpful e-commerce multimodal assistant. Use retrieved facts faithfully. Answer in the user's language."},
            {"role":"user","content": f"Question:\n{query}\n\nRetrieved Context:\n{ctx}"}
        ]
        resp = client.chat_completions.create(model=model_name, messages=msgs, temperature=0.2) \
            if hasattr(client, "chat_completions") else \
            client.chat.completions.create(model=model_name, messages=msgs, temperature=0.2)
        return resp.choices[0].message.content
    except Exception as e:
        return f"(OpenAI call failed: {e})\nFallback summary:\n{ctx}"

# ----------------- UI -----------------
st.set_page_config(page_title="Multimodal RAG Chatbot", page_icon="🛍️", layout="wide")
st.title("🛍️ Multimodal RAG Chatbot (MVP)")

st.sidebar.header("Settings")
topk = st.sidebar.slider("Top-K results", 1, 10, 5)
device = st.sidebar.selectbox("Embedding Device", ["auto", "cpu"], index=0)

st.sidebar.subheader("OpenAI (optional)")
openai_key_input = st.sidebar.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY",""))
if openai_key_input: st.session_state["OPENAI_API_KEY"] = openai_key_input
st.sidebar.text_input("OPENAI_MODEL", value=os.getenv("OPENAI_MODEL","gpt-4o-mini"), key="OPENAI_MODEL_BOX")

# Quick prompts
st.caption("Quick prompts:")
c1, c2, c3 = st.columns(3)
if "qtext" not in st.session_state: st.session_state["qtext"] = ""
if c1.button("Specs: Galaxy S21"): st.session_state["qtext"] = "What are the key specs of Samsung Galaxy S21?"
if c2.button("Compare: Echo Dot vs Nest Mini"): st.session_state["qtext"] = "Compare Amazon Echo Dot and Google Nest Mini for sound and smart home."
if c3.button("Find similar to my photo"): st.session_state["qtext"] = "Identify this product and list similar alternatives."

# ===== Index 준비 (자동) =====
def ensure_index_ready() -> bool:
    if "INDEX" in st.session_state and "METAS" in st.session_state:
        return True
    # 1) artifacts 가 있으면 재사용
    try:
        if artifacts_exist("artifacts"):
            idx, metas = load_artifacts("artifacts")
            st.session_state["INDEX"] = idx; st.session_state["METAS"] = metas
            st.success(f"Loaded prebuilt index: {len(metas)} documents");  return True
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")

    # 2) 기본 데이터셋으로 자동 빌드 (parquet 우선)
    try:
        src = "clean_data.parquet" if os.path.exists("clean_data.parquet") else "clean_data.csv"
        with st.spinner(f"Preparing index from bundled dataset ({src})..."):
            df = read_any(src)
            df = coerce_schema(df)
            idx, metas = build_or_load_index(
                df,
                device=None if device == "auto" else device,
                use_images=False,   # 지금 파이프라인은 텍스트 전용 MiniLM
                limit=MAX_ROWS,
                outdir="artifacts",
                batch_size=64,
                force_rebuild=False
            )
            st.session_state["INDEX"] = idx; st.session_state["METAS"] = metas
            st.success(f"Index ready: {len(metas)} documents");  return True
    except Exception as e:
        st.error(f"Could not prepare index automatically: {e}")
    return False

index_loaded = ensure_index_ready()

st.divider()
qcol, icol = st.columns([2, 1])
with qcol:
    qtext = st.text_input("💬 Ask a question", key="qtext",
                          placeholder="e.g., Tell me the specs of Galaxy S21 / Compare Echo Dot vs Nest Mini")
with icol:
    qimg = st.file_uploader("📷 Upload an image (optional)", type=["png","jpg","jpeg","webp"])

send = st.button("Send", type="primary")
if send:
    if not index_loaded:
        st.warning("Index is not ready. Please refresh the app.")
    elif not qtext and not qimg:
        st.warning("Enter a question or upload an image.")
    else:
        img = None
        if qimg:
            try:
                img = Image.open(qimg).convert("RGB")
                st.image(img, caption="Query Image", width=240)
            except Exception:
                st.info("Could not read the uploaded image; continuing with text only.")

        # 질의 임베딩은 CLIP 텍스트(or 이미지)로 생성해서 FAISS에 검색
        model = load_clip(None if device == "auto" else "cpu")
        qvec  = _encode_query(model, qtext or "", img)
        D, I  = _search(st.session_state["INDEX"], qvec, k=topk)

        snippets = []
        for score, idx in zip(D, I):
            if idx == -1: continue
            meta = dict(st.session_state["METAS"][idx])
            _render_card(meta, float(score))
            snippets.append({
                "title": meta.get("title"),
                "brand": meta.get("brand"),
                "price": meta.get("price"),
                "features": meta.get("features"),
                "description": meta.get("description")
            })
        ans = _answer_with_openai(qtext or "(image query)", snippets)
        st.chat_message("assistant").write(ans)
