# streamlit_app.py
# Multimodal RAG Chatbot (CLIP-only, unified encoder for index & query)
# - No dataset upload UI
# - On app start, auto: load artifacts/ OR build from local clean_data.parquet/csv
# - Queries can be text and/or image; both encoded by CLIP and averaged

import os, io, base64, json
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# ---------- Runtime deps ----------
try:
    import faiss  # pip install faiss-cpu
except Exception as e:
    st.error("FAISS is required. Please add `faiss-cpu` to requirements.txt.")
    st.stop()

from sentence_transformers import SentenceTransformer

# ---------- Import pipeline (must be in same repo) ----------
# The pipeline should also be CLIP-based and produce artifacts in 'artifacts/'
try:
    from pipeline import (
        build_or_load_index,            # (df, device, use_images, limit, outdir, ...)
        artifacts_exist, load_artifacts,# artifacts I/O
        coerce_schema, read_any,         # data utilities
        encode_query,
    )
except Exception as e:
    st.error(f"Failed to import pipeline: {e}")
    st.stop()

# ---------- Constants ----------
EMB_MODEL = "clip-ViT-B-32"   # must match pipeline encoder
REQUEST_TIMEOUT = 7
MAX_ROWS = 8000               # memory cap for Streamlit/Cloud

# ---------- Caches ----------
@st.cache_resource
def load_clip(device: Optional[str] = None) -> SentenceTransformer:
    # device: "cpu" | "cuda" | "mps" | None(auto)
    return SentenceTransformer(EMB_MODEL, device=device)

@st.cache_data(show_spinner=False)
def _read_bundled_dataset() -> pd.DataFrame:
    """
    Load local dataset (parquet preferred; fallback to csv) and coerce schema.
    """
    src = "clean_data.parquet" if os.path.exists("clean_data.parquet") else (
        "clean_data.csv" if os.path.exists("clean_data.csv") else None
    )
    if not src:
        raise FileNotFoundError("No bundled dataset found (clean_data.parquet or clean_data.csv).")
    df = read_any(src)
    df = coerce_schema(df)
    # cap rows for memory safety
    if len(df) > MAX_ROWS:
        df = df.head(MAX_ROWS).copy()
    return df, src

# ---------- CLIP helpers ----------
def _encode_query_clip(model: SentenceTransformer, text: str, img: Optional[Image.Image]) -> np.ndarray:
    """
    Encode text/image with CLIP (L2-normalized) and return (1, d) float32.
    If both exist, average the two embeddings.
    """
    tv = model.encode([text], normalize_embeddings=True).astype("float32") if text else None
    iv = None
    if img is not None:
        iv = model.encode([img], normalize_embeddings=True).astype("float32")
    if tv is not None and iv is not None:
        return ((tv + iv) / 2.0).astype("float32")
    return (tv if tv is not None else iv).astype("float32")

def _search(index: faiss.IndexFlatIP, qvec: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    FAISS search with a single query vector qvec shape (1, d).
    Returns: (scores_1d, indices_1d)
    """
    if qvec is None:
        raise ValueError("Empty query vector.")
    # Ensure 2D shape
    if qvec.ndim == 1:
        qvec = qvec.reshape(1, -1)
    D, I = index.search(qvec, k)
    return D[0], I[0]

def _render_card(meta: Dict[str, Any], score: float):
    title = (meta.get("title") or meta.get("product_title") or "").strip() or "(no title)"
    brand = (meta.get("brand") or "").strip()
    price = (meta.get("price") or "").strip()
    header = " · ".join([x for x in [brand, price] if x])
    st.markdown(f"**🔎 {title}**  \n{header}")
    st.caption(f"`score={score:.3f}`")

    # Show image if available
    if meta.get("_image_b64"):
        try:
            st.image(Image.open(io.BytesIO(base64.b64decode(meta["_image_b64"]))), width=220)
        except Exception:
            pass
    elif meta.get("image_url"):
        # Let Streamlit fetch by URL; if some 403, it's fine to skip
        try:
            st.image(meta["image_url"], width=220)
        except Exception:
            pass

    short = (str(meta.get("features") or meta.get("description") or "")).strip()
    if short:
        short = short[:300]
        st.caption(short + ("..." if len(short) == 300 else ""))

# ---------- OpenAI (optional) ----------
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
        )
        msgs = [
            {"role":"system","content":"You are a helpful e-commerce multimodal assistant. Use retrieved facts faithfully. Answer in the user's language."},
            {"role":"user","content": f"Question:\n{query}\n\nRetrieved Context:\n{ctx}"}
        ]
        resp = client.chat.completions.create(model=model_name, messages=msgs, temperature=0.2)
        return resp.choices[0].message.content
    except Exception as e:
        return f"(OpenAI call failed: {e})\nFallback summary:\n{ctx}"

# ---------- UI ----------
st.set_page_config(page_title="Multimodal RAG Chatbot", page_icon="🛍️", layout="wide")
st.title("🛍️ Multimodal RAG Chatbot (CLIP unified)")

st.sidebar.header("Settings")
topk   = st.sidebar.slider("Top-K results", 1, 12, 5)
device = st.sidebar.selectbox("Embedding Device", ["auto","cpu","cuda","mps"], index=0)

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

# ===== Ensure index is ready (no upload UI) =====
@st.cache_resource(show_spinner=True)
def _ensure_index_ready(device_choice: str):
    """
    Try artifacts/ first; otherwise build from local clean_data.parquet/csv automatically.
    Returns: (index, metas, src_path)
    """
    # 1) Artifacts present
    if artifacts_exist("artifacts"):
        try:
            idx, metas = load_artifacts("artifacts")
            return idx, metas, "artifacts/"
        except Exception as e:
            # fall through to build
            st.warning(f"Artifacts exist but failed to load: {e}. Rebuilding from dataset...")

    # 2) Build from bundled dataset
    df, src = _read_bundled_dataset()
    dev = None if device_choice == "auto" else device_choice
    with st.spinner(f"Preparing index from bundled dataset ({src})..."):
        idx, metas = build_or_load_index(
            df,
            device=dev,
            use_images=True,     # CLIP can fuse text+image when available during build
            limit=MAX_ROWS,
            outdir="artifacts",
            batch_size=64,
            force_rebuild=False
        )
    return idx, metas, src

try:
    index, metas, src_info = _ensure_index_ready(device)
    st.session_state["INDEX"] = index
    st.session_state["METAS"] = metas
    st.success(f"Index ready ({len(metas)} documents) — source: {src_info}")
except Exception as e:
    st.error(f"Failed to prepare index: {e}")
    st.stop()

st.divider()
qcol, icol = st.columns([2, 1])
with qcol:
    qtext = st.text_input(
        "💬 Ask a question",
        key="qtext",
        placeholder="e.g., Tell me the specs of Galaxy S21 / Compare Echo Dot vs Nest Mini",
    )
with icol:
    qimg = st.file_uploader("📷 Upload an image (optional)",
                            type=["png", "jpg", "jpeg", "webp"])

# ===== Query =====
send = st.button("Send", type="primary")
if send:
    if "INDEX" not in st.session_state:
        st.warning("Index not ready. Please refresh the app.")
    elif not qtext and not qimg:
        st.warning("Enter a question or upload an image.")
    else:
        # Render query image (if any)
        img = None
        if qimg:
            try:
                img = Image.open(qimg).convert("RGB")
                st.image(img, caption="Query Image", width=240)
            except Exception:
                st.info("Could not read the uploaded image; continuing with text only.")

        # Encode with CLIP (same model as index)
        model = load_clip(None if device == "auto" else device)
        qvec = _encode_query_clip(model, qtext or "", img)

        # FAISS search
        D, I = _search(st.session_state["INDEX"], qvec, k=topk)

        # Render results & build snippets for LLM
        snippets: List[Dict[str, Any]] = []
        for score, idx_ in zip(D, I):
            if idx_ == -1:
                continue
            meta = dict(st.session_state["METAS"][idx_])
            _render_card(meta, float(score))
            snippets.append({
                "title": meta.get("title"),
                "brand": meta.get("brand"),
                "price": meta.get("price"),
                "features": meta.get("features"),
                "description": meta.get("description")
            })

        # Optional LLM answer
        ans = _answer_with_openai(qtext or "(image query)", snippets)
        st.chat_message("assistant").write(ans)
