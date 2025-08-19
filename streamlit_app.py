# streamlit_app.py
import os, io, base64, hashlib
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# --- Runtime deps ---
try:
    import faiss                   # pip install faiss-cpu
except Exception as e:
    st.stop()

from sentence_transformers import SentenceTransformer

EMB_MODEL = "clip-ViT-B-32"
EMB_DIM = 512

@st.cache_resource
def load_clip(device: Optional[str] = None):
    # device=None -> auto
    return SentenceTransformer(EMB_MODEL, device=device)

# Cache index and metadata
@st.cache_resource(show_spinner=False)
def build_index(df: pd.DataFrame,
                text_cols=("title","brand","price","features","description"),
                image_col: Optional[str]="image_url",
                limit: Optional[int]=None,
                device: Optional[str]=None):
    model = load_clip(device)
    idx = faiss.IndexFlatIP(EMB_DIM)  # cosine similarity via L2-normalized vectors
    normed = True
    metas: List[Dict[str,Any]] = []
    vectors = []

    def _emb_text(text: str) -> np.ndarray:
        return model.encode([text], normalize_embeddings=normed).astype("float32")

    def _emb_img(img: Image.Image) -> np.ndarray:
        return model.encode([img], normalize_embeddings=normed).astype("float32")

    if limit:
        df = df.head(limit)

    for _, row in df.iterrows():
        meta = row.to_dict()
        parts = [str(row[c]) for c in text_cols if c in df.columns and pd.notna(row[c])]
        text = " | ".join(parts) if parts else ""
        vec = None

        # If an image is available, combine text+image embeddings
        img_b64 = None
        if image_col and image_col in df.columns and pd.notna(row.get(image_col, None)):
            url = str(row[image_col])
            img_bytes = _download_image(url)
            if img_bytes:
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                meta["_image_b64"] = img_b64
                try:
                    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    iv = _emb_img(pil)
                    tv = _emb_text(text) if text else iv
                    vec = (tv + iv) / 2.0
                except Exception:
                    pass

        if vec is None:
            vec = _emb_text(text if text else str(row.get("title","")))

        metas.append(meta)
        vectors.append(vec[0])

    X = np.vstack(vectors).astype("float32")
    idx.add(X)
    return idx, metas

@st.cache_resource(show_spinner=False)
def _session_llm():
    api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception:
        return None

def _download_image(url: str) -> Optional[bytes]:
    import requests
    try:
        r = requests.get(url, timeout=7)
        if r.status_code == 200:
            return r.content
    except Exception:
        return None
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
    title = meta.get("title") or meta.get("product_title") or "(no title)"
    brand = meta.get("brand") or ""
    price = meta.get("price") or ""
    st.markdown(f"**🔎 {title}**  \n{brand} · {price}  \n`score={score:.3f}`")
    if "_image_b64" in meta:
        st.image(Image.open(io.BytesIO(base64.b64decode(meta["_image_b64"]))), width=220)
    short = (str(meta.get("features") or meta.get("description") or ""))[:300]
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
        msgs = [
            {"role":"system","content":"You are a helpful e-commerce multimodal assistant. Use retrieved facts faithfully. Answer in the user's language."},
            {"role":"user","content": f"Question:\n{query}\n\nRetrieved Context:\n{ctx}"}
        ]
        resp = client.chat.completions.create(model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
                                              messages=msgs, temperature=0.2)
        return resp.choices[0].message.content
    except Exception as e:
        return f"(OpenAI call failed: {e})\nFallback summary:\n{ctx}"

st.set_page_config(page_title="Multimodal RAG Chatbot", page_icon="🛍️", layout="wide")
st.title("🛍️ Multimodal RAG Chatbot (MVP)")

st.sidebar.header("Settings")
csv_choice = st.sidebar.selectbox("Data Source", ["Upload CSV", "Use sample schema"], index=0)
limit = st.sidebar.number_input("Number of samples to index", 50, 5000, 500, step=50)
topk = st.sidebar.slider("Top-K results", 1, 10, 5)
device = st.sidebar.selectbox("Embedding Device", ["auto","cpu"], index=0)

st.sidebar.subheader("OpenAI (optional)")
openai_key_input = st.sidebar.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY",""))
if openai_key_input:
    st.session_state["OPENAI_API_KEY"] = openai_key_input
st.sidebar.text_input("OPENAI_MODEL", value=os.getenv("OPENAI_MODEL","gpt-4o-mini"), key="OPENAI_MODEL_BOX")

# Load data
uploaded_df = None
if csv_choice == "Upload CSV":
    uf = st.file_uploader("Upload CSV (recommended columns: title, brand, price, features, description, image_url)", type=["csv"])
    if uf:
        uploaded_df = pd.read_csv(uf)

build = st.button("🔨 Build / Reset Index", type="primary")
if build:
    if csv_choice == "Use repo dataset (data/clean_data.csv.gz)":
        try:
            with st.spinner("Loading repo dataset and building index..."):
                df_repo = pd.read_csv("data/clean_data.csv.gz")  # <-- 압축 CSV 자동 인식
                index, metas = build_index(df_repo, limit=limit,
                                           device=None if device=="auto" else "cpu")
                st.session_state["INDEX"] = index
                st.session_state["METAS"] = metas
            st.success(f"Index built from repo dataset: {len(metas)} documents")
        except FileNotFoundError:
            st.error("`data/clean_data.csv.gz` not found. Make sure the file exists in the repo.")
        except Exception as e:
            st.error(f"Failed to load repo dataset: {e}")

    else:  # "Upload CSV"
        if uploaded_df is None or uploaded_df.empty:
            st.error("Please upload a CSV file first.")
        else:
            with st.spinner("Building index from uploaded CSV... (CLIP embeddings)"):
                index, metas = build_index(uploaded_df, limit=limit,
                                           device=None if device=="auto" else "cpu")
                st.session_state["INDEX"] = index
                st.session_state["METAS"] = metas
            st.success(f"Index built: {len(metas)} documents")

st.divider()
qcol, icol = st.columns([2,1])
with qcol:
    qtext = st.text_input("💬 Ask a question", placeholder="e.g., Tell me the specs of Galaxy S21 / Compare Echo Dot vs Nest Mini")
with icol:
    qimg = st.file_uploader("📷 Upload an image (optional)", type=["png","jpg","jpeg","webp"])

send = st.button("Send", type="primary")
if send:
    if "INDEX" not in st.session_state:
        st.warning("Please build the index first.")
    elif not qtext and not qimg:
        st.warning("Enter a question or upload an image.")
    else:
        img = None
        if qimg:
            try:
                img = Image.open(qimg).convert("RGB")
                st.image(img, caption="Query Image", width=240)
            except Exception:
                pass

        model = load_clip(None if device=="auto" else "cpu")
        qvec = _encode_query(model, qtext or "", img)
        D, I = _search(st.session_state["INDEX"], qvec, k=topk)

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
