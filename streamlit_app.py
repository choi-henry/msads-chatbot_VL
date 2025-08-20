# streamlit_app.py
import os, io, base64, re, json
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# --- Runtime deps ---
try:
    import faiss  # pip install faiss-cpu
except Exception:
    st.error("FAISS is required. Please add `faiss-cpu` to requirements.txt.")
    st.stop()

from sentence_transformers import SentenceTransformer

EMB_MODEL = "clip-ViT-B-32"
EMB_DIM = 512
REQUEST_TIMEOUT = 7
MAX_ROWS = 8000  # hard cap for Streamlit Cloud memory

# ===== CSV schema (canonical) =====
REQUIRED_COLS = ["title", "brand", "price", "features", "description", "image_url"]

# aliases (lowercased keys) -> canonical
ALIASES = {
    # common
    "product_title": "title",
    "name": "title",
    "title": "title",
    "brand_name": "brand",
    "brand": "brand",
    "img_url": "image_url",
    "imageurl": "image_url",
    "imageurlhighres": "image_url",
    "image_link": "image_url",
    "images": "image_url",
    "feature": "features",
    "features": "features",
    "text": "description",
    "description": "description",

    # CSV 예: Uniq Id, Product Name, ... Image, Variants, text_blob ...
    "product name": "title",
    "selling price": "price",
    "about product": "description",
    "product specification": "features",
    "technical details": "features",
    "shipping weight": "features",
    "product dimensions": "features",
    "variants": "features",
    "image": "image_url",
    "product url": "product_url",
    "text_blob": "description",
}

URL_RE = re.compile(r"https?://[^\s|,;]+", re.IGNORECASE)

def _first_url(s: str) -> str:
    if not isinstance(s, str):
        return ""
    m = URL_RE.search(s)
    return m.group(0) if m else ""

# ----------------- Caching -----------------
@st.cache_resource
def load_clip(device: Optional[str] = None):
    # device=None -> auto
    return SentenceTransformer(EMB_MODEL, device=device)

# schema fixer
def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    # 1) rename common/known aliases -> canonical
    plan = {}
    for c in list(df.columns):
        key = str(c).lower().strip()
        if key in ALIASES and ALIASES[key] not in df.columns:
            plan[c] = ALIASES[key]
    if plan:
        df = df.rename(columns=plan)

    # 2) compose features from multiple columns
    merge_cols = []
    for c in ["features", "product specification", "technical details", "variants",
              "product dimensions", "shipping weight"]:
        if c in df.columns:
            merge_cols.append(c)
    if merge_cols:
        def _merge_feats(row):
            parts = []
            for c in merge_cols:
                v = row.get(c, "")
                if isinstance(v, (list, tuple)):
                    v = " ; ".join([str(x) for x in v if str(x).strip()])
                v = str(v).strip()
                if v and v.lower() not in ("nan", "none", "null"):
                    parts.append(v)
            return " ; ".join(parts)
        df["features"] = df.apply(_merge_feats, axis=1)

    # 3) prefer About Product, else text_blob (이미 ALIASES로 description에 매핑됨)
    if "description" in df.columns:
        df["description"] = df["description"].fillna("").astype(str)

    # 4) clean image_url
    if "image_url" in df.columns:
        df["image_url"] = df["image_url"].astype(str).apply(_first_url)

    # 5) tidy types and fill
    for c in ["title", "brand", "price", "features", "description", "image_url"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    # 6) warn if missing important columns (just informational)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.warning(
            f"Missing columns {missing}. The app will still work with available columns, "
            "but retrieval quality may be lower."
        )

    # 7) drop rows with no usable text at all
    text_cols = [c for c in ["title", "description", "features", "brand", "price"] if c in df.columns]
    def _has_any_text(row):
        return any(isinstance(row.get(c, ""), str) and row.get(c, "").strip() for c in text_cols)
    if text_cols:
        df = df[df.apply(_has_any_text, axis=1)].copy()

    return df

def read_any(path_or_file) -> pd.DataFrame:
    """Accepts parquet/csv/csv.gz; file-like or path."""
    name = getattr(path_or_file, "name", "")
    if isinstance(path_or_file, str) and not name:
        name = path_or_file
    if name.endswith(".parquet"):
        return pd.read_parquet(path_or_file)
    if name.endswith(".gz") or name.endswith(".csv"):
        return pd.read_csv(path_or_file, compression="infer")
    # try parquet then csv
    try:
        return pd.read_parquet(path_or_file)
    except Exception:
        return pd.read_csv(path_or_file, compression="infer")

# --------- Artifacts I/O (from pipeline.py output) ----------
def artifacts_exist(dirpath: str = "artifacts") -> bool:
    return (
        os.path.exists(os.path.join(dirpath, "index.faiss")) and
        os.path.exists(os.path.join(dirpath, "metas.json"))
    )

def load_artifacts(dirpath: str = "artifacts") -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    idx = faiss.read_index(os.path.join(dirpath, "index.faiss"))
    with open(os.path.join(dirpath, "metas.json"), "r", encoding="utf-8") as f:
        metas = json.load(f)
    return idx, metas

def save_uploaded_artifacts(index_file, metas_file, dirpath: str = "artifacts") -> Tuple[bool, str]:
    """Allow users to upload index.faiss & metas.json from pipeline and use immediately."""
    try:
        os.makedirs(dirpath, exist_ok=True)
        if index_file is not None:
            with open(os.path.join(dirpath, "index.faiss"), "wb") as f:
                f.write(index_file.read())
        if metas_file is not None:
            metas_bytes = metas_file.read()
            # Try to parse to ensure it's valid JSON
            json.loads(metas_bytes.decode("utf-8"))
            with open(os.path.join(dirpath, "metas.json"), "wb") as f:
                f.write(metas_bytes)
        return True, "Uploaded artifacts saved."
    except Exception as e:
        return False, f"Failed to save artifacts: {e}"

# Cache index and metadata (build on-the-fly)
@st.cache_resource(show_spinner=False)
def build_index(df: pd.DataFrame,
                text_cols=("title","brand","price","features","description"),
                image_col: Optional[str]="image_url",
                limit: Optional[int]=None,
                device: Optional[str]=None):
    model = load_clip(device)
    idx = faiss.IndexFlatIP(EMB_DIM)  # cosine via L2-normalized vectors
    metas: List[Dict[str,Any]] = []
    vectors = []

    # limit rows for memory
    if limit:
        df = df.head(limit)

    # ---- batch encode texts for speed
    texts: List[str] = []
    rows: List[pd.Series] = []
    for _, row in df.iterrows():
        parts = [str(row[c]) for c in text_cols if c in df.columns and pd.notna(row[c])]
        texts.append(" | ".join(parts) if parts else "")
        rows.append(row)

    if not texts:
        raise ValueError("No usable text fields found after preprocessing.")

    text_vecs = model.encode(texts, normalize_embeddings=True, batch_size=64).astype("float32")

    for i, row in enumerate(rows):
        meta = row.to_dict()
        vec = text_vecs[i:i+1]  # default to text vector

        # If an image is available, combine text+image embeddings
        if image_col and image_col in df.columns and pd.notna(row.get(image_col, None)):
            url = str(row[image_col]).strip()
            if url:
                img_bytes = _download_image(url)
                if img_bytes:
                    meta["_image_b64"] = base64.b64encode(img_bytes).decode("utf-8")
                    try:
                        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        iv = model.encode([pil], normalize_embeddings=True).astype("float32")
                        vec = (vec + iv) / 2.0
                    except Exception:
                        pass

        metas.append(meta)
        vectors.append(vec[0])

    X = np.vstack(vectors).astype("float32")
    idx.add(X)
    return idx, metas

@st.cache_resource(show_spinner=False)
def _session_llm():
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or (st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None)
        or st.session_state.get("OPENAI_API_KEY")
    )
    if not api_key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception:
        return None

# ----------------- Helpers -----------------
def _download_image(url: str) -> Optional[bytes]:
    import requests
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.content
    except Exception:
        return None

def _encode_query(model: SentenceTransformer, text: str, img: Optional[Image.Image]) -> np.ndarray:
    tv = model.encode([text], normalize_embeddings=True).astype("float32") if text else None
    iv = model.encode([img], normalize_embeddings=True).astype("float32") if img is not None else None
    if tv is not None and iv is not None:
        return ((tv + iv) / 2.0).astype("float32")
    return (tv if tv is not None else iv).astype("float32")

def _search(index: faiss.IndexFlatIP, qvec: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    D, I = index.search(qvec, k)
    return D[0], I[0]

def _render_card(meta: Dict[str,Any], score: float):
    title = (meta.get("title") or meta.get("product_title") or "").strip() or "(no title)"
    brand = (meta.get("brand") or "").strip()
    price = (meta.get("price") or "").strip()
    st.markdown(f"**🔎 {title}**  \n{(' · '.join([x for x in [brand, price] if x]))}")
    st.caption(f"`score={score:.3f}`")
    if "_image_b64" in meta:
        try:
            st.image(Image.open(io.BytesIO(base64.b64decode(meta["_image_b64"]))), width=220)
        except Exception:
            pass
    short = (str(meta.get("features") or meta.get("description") or "")).strip()[:300]
    if short:
        st.caption(short + ("..." if len(short)==300 else ""))

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
        model_name = (
            os.getenv("OPENAI_MODEL")
            or (st.secrets.get("OPENAI_MODEL") if "OPENAI_MODEL" in st.secrets else "gpt-4o-mini")
            or st.session_state.get("OPENAI_MODEL_BOX")
            or "gpt-4o-mini"
        )
        msgs = [
            {"role":"system","content":"You are a helpful e-commerce multimodal assistant. Use retrieved facts faithfully. Answer in the user's language."},
            {"role":"user","content": f"Question:\n{query}\n\nRetrieved Context:\n{ctx}"}
        ]
        # support both old/new openai python client namespaces
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
limit = st.sidebar.number_input("Number of samples to index", 50, MAX_ROWS, 2000, step=50)
topk = st.sidebar.slider("Top-K results", 1, 10, 5)
device = st.sidebar.selectbox("Embedding Device", ["auto","cpu"], index=0)

# --- Pipeline integration: load prebuilt artifacts ---
st.sidebar.subheader("Index Source")
use_prebuilt = st.sidebar.checkbox("Use prebuilt artifacts (artifacts/index.faiss + metas.json)", value=True)

if use_prebuilt:
    st.sidebar.caption("If artifacts are not in repo, upload them below.")
    up_idx = st.sidebar.file_uploader("Upload index.faiss", type=["faiss"], accept_multiple_files=False)
    up_meta = st.sidebar.file_uploader("Upload metas.json", type=["json"], accept_multiple_files=False)
    if st.sidebar.button("Save uploaded artifacts"):
        ok, msg = save_uploaded_artifacts(up_idx, up_meta, dirpath="artifacts")
        (st.sidebar.success if ok else st.sidebar.error)(msg)

st.sidebar.subheader("OpenAI (optional)")
openai_key_input = st.sidebar.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY",""))
if openai_key_input:
    st.session_state["OPENAI_API_KEY"] = openai_key_input
st.sidebar.text_input("OPENAI_MODEL", value=os.getenv("OPENAI_MODEL","gpt-4o-mini"), key="OPENAI_MODEL_BOX")

# Quick prompts
st.caption("Quick prompts:")
c1, c2, c3 = st.columns(3)
if "qtext" not in st.session_state:
    st.session_state["qtext"] = ""
if c1.button("Specs: Galaxy S21"):
    st.session_state["qtext"] = "What are the key specs of Samsung Galaxy S21?"
if c2.button("Compare: Echo Dot vs Nest Mini"):
    st.session_state["qtext"] = "Compare Amazon Echo Dot and Google Nest Mini for sound and smart home."
if c3.button("Find similar to my photo"):
    st.session_state["qtext"] = "Identify this product and list similar alternatives."

# ===== Load/Build Index =====
index_loaded = False
if use_prebuilt and artifacts_exist("artifacts"):
    try:
        index, metas = load_artifacts("artifacts")
        st.session_state["INDEX"] = index
        st.session_state["METAS"] = metas
        index_loaded = True
        st.success(f"Loaded prebuilt index: {len(metas)} documents")
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")

# Fallback: build from bundled dataset or upload
uploaded_df = None
if not index_loaded:
    st.info("No prebuilt artifacts loaded. You can upload a dataset to build an index here.")
    uf = st.file_uploader(
        "Upload dataset (.parquet / .csv / .csv.gz). Columns auto-mapped to: "
        "title, brand, price, features, description, image_url",
        type=["parquet", "csv", "gz"]
    )
    if uf:
        try:
            uploaded_df = read_any(uf)
            uploaded_df = coerce_schema(uploaded_df)
            st.success(f"Loaded {len(uploaded_df):,} rows. Columns: {list(uploaded_df.columns)}")
            st.dataframe(uploaded_df.head(5))
        except Exception as e:
            st.error(f"Failed to read/parse file: {e}")

    build = st.button("🔨 Build / Reset Index", type="primary")
    if build:
        if uploaded_df is None or uploaded_df.empty:
            st.error("Please upload a dataset first.")
        else:
            with st.spinner("Building index... (creating CLIP embeddings)"):
                index, metas = build_index(uploaded_df, limit=limit,
                                           device=None if device=="auto" else "cpu")
                st.session_state["INDEX"] = index
                st.session_state["METAS"] = metas
            st.success(f"Index built: {len(metas)} documents")
            index_loaded = True

st.divider()
qcol, icol = st.columns([2,1])
with qcol:
    qtext = st.text_input("💬 Ask a question", key="qtext",
                          placeholder="e.g., Tell me the specs of Galaxy S21 / Compare Echo Dot vs Nest Mini")
with icol:
    qimg = st.file_uploader("📷 Upload an image (optional)", type=["png","jpg","jpeg","webp"])

send = st.button("Send", type="primary")
if send:
    if "INDEX" not in st.session_state:
        st.warning("Please load artifacts or build the index first.")
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

        model = load_clip(None if device=="auto" else "cpu")
        qvec = _encode_query(model, qtext or "", img)
        D, I = _search(st.session_state["INDEX"], qvec, k=topk)

        snippets = []
        for score, idx in zip(D, I):
            if idx == -1:
                continue
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
