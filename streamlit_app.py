import os, io, json
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st

# ---- Safe import pipeline (Cloud/Local 모두 동작) ----
try:
    from pipeline import build_or_load_index, coerce_schema, read_any, encode_query
except Exception as e:
    import importlib.util, pathlib
    st.warning(f"Direct import failed: {e}. Falling back to path-based loader.")
    p = pathlib.Path(__file__).with_name("pipeline.py")
    spec = importlib.util.spec_from_file_location("pipeline", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    build_or_load_index = mod.build_or_load_index
    coerce_schema = mod.coerce_schema
    read_any = mod.read_any
    encode_query = mod.encode_query

# ---- Runtime deps ----
try:
    import faiss  # pip install faiss-cpu
except Exception:
    st.error("FAISS is required. Please add `faiss-cpu` to requirements.txt.")
    st.stop()

REQUEST_TIMEOUT = 7
MAX_ROWS = 8000  # safety cap

# ---- UI header ----
st.set_page_config(page_title="Multimodal RAG Chatbot", page_icon="🛍️", layout="wide")
st.title("🛍️ Multimodal RAG Chatbot (MVP)")

# ---- Sidebar (간단) ----
st.sidebar.header("Settings")
topk = st.sidebar.slider("Top-K results", 1, 10, 5)
device = st.sidebar.selectbox("Embedding Device", ["auto", "cpu"], index=0)

st.sidebar.subheader("OpenAI (optional)")
openai_key_input = st.sidebar.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY",""))
if openai_key_input:
    st.session_state["OPENAI_API_KEY"] = openai_key_input
st.sidebar.text_input("OPENAI_MODEL", value=os.getenv("OPENAI_MODEL","gpt-4o-mini"), key="OPENAI_MODEL_BOX")

# ---- Quick prompts ----
st.caption("Quick prompts:")
c1, c2, c3 = st.columns(3)
if "qtext" not in st.session_state:
    st.session_state["qtext"] = ""
if c1.button("Specs: Galaxy S21"):
    st.session_state["qtext"] = "What are the key specs of Samsung Galaxy S21?"
if c2.button("Compare: Echo Dot vs Nest Mini"):
    st.session_state["qtext"] = "Compare Amazon Echo Dot and Google Nest Mini for sound and smart home."
if c3.button("Find similar to my photo"):
    st.session_state["qtext"] = "Find alternatives for a compact smart speaker with strong bass."

# ---- Helpers ----
def artifacts_exist(dirpath: str = "artifacts") -> bool:
    return os.path.exists(os.path.join(dirpath, "index.faiss")) and os.path.exists(os.path.join(dirpath, "metas.json"))

def load_artifacts(dirpath: str = "artifacts") -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    idx = faiss.read_index(os.path.join(dirpath, "index.faiss"))
    with open(os.path.join(dirpath, "metas.json"), "r", encoding="utf-8") as f:
        metas = json.load(f)
    return idx, metas

def _search(index: faiss.Index, qvec: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    D, I = index.search(qvec, k)
    return D[0], I[0]

def _render_card(meta: Dict[str,Any], score: float):
    title = (meta.get("title") or "(no title)").strip()
    brand = (meta.get("brand") or "").strip()
    price = (meta.get("price") or "").strip()
    subtitle = " · ".join([x for x in [brand, price] if x])
    st.markdown(f"**🔎 {title}**  \n{subtitle}")
    st.caption(f"`score={score:.3f}`")
    short = (str(meta.get("features") or meta.get("description") or "")).strip()[:300]
    if short:
        st.caption(short + ("..." if len(short)==300 else ""))

def _answer_with_openai(query: str, snippets: List[Dict[str,Any]]) -> str:
    api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
    if not api_key:
        ctx = "\n\n".join(f"[{i+1}] {s.get('title','')} ({s.get('brand','')}) {s.get('price','')}\n"
                          f"Features: {s.get('features','')}\nDesc: {s.get('description','')}"
                          for i, s in enumerate(snippets))
        return f"(No LLM key · showing retrieved summary)\nQuestion: {query}\n\n{ctx}"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        model_name = os.getenv("OPENAI_MODEL") or st.session_state.get("OPENAI_MODEL_BOX") or "gpt-4o-mini"
        ctx = "\n\n".join(f"[{i+1}] {s.get('title','')} ({s.get('brand','')}) {s.get('price','')}\n"
                          f"Features: {s.get('features','')}\nDesc: {s.get('description','')}"
                          for i, s in enumerate(snippets))
        msgs = [
            {"role":"system","content":"You are a helpful e-commerce assistant. Use retrieved facts faithfully."},
            {"role":"user","content": f"Question:\n{query}\n\nRetrieved Context:\n{ctx}"},
        ]
        # client compatibility
        resp = client.chat_completions.create(model=model_name, messages=msgs, temperature=0.2) \
            if hasattr(client, "chat_completions") else \
            client.chat.completions.create(model=model_name, messages=msgs, temperature=0.2)
        return resp.choices[0].message.content
    except Exception as e:
        return f"(OpenAI call failed: {e})"

# ---- Ensure index (no upload UI; local file만) ----
def ensure_index_ready() -> bool:
    if "INDEX" in st.session_state and "METAS" in st.session_state:
        return True

    # 1) artifacts 재사용
    try:
        if artifacts_exist("artifacts"):
            idx, metas = load_artifacts("artifacts")
            st.session_state["INDEX"] = idx
            st.session_state["METAS"] = metas
            st.success(f"Loaded prebuilt index: {len(metas)} documents")
            return True
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")

    # 2) 로컬 데이터로 자동 빌드 (parquet 우선)
    try:
        src = "clean_data.parquet" if os.path.exists("clean_data.parquet") else "clean_data.csv"
        with st.spinner(f"Preparing index from local {src}..."):
            df = read_any(src)
            df = coerce_schema(df)
            idx, metas = build_or_load_index(
                df,
                device=None if device == "auto" else device,  # "auto" -> None
                use_images=False,          # MiniLM 텍스트 전용
                limit=MAX_ROWS,
                outdir="artifacts",
                force_rebuild=False
            )
            st.session_state["INDEX"] = idx
            st.session_state["METAS"] = metas
            st.success(f"Index ready: {len(metas)} documents")
            return True
    except Exception as e:
        st.error(f"Could not prepare index automatically: {e}")

    return False

index_ready = ensure_index_ready()

st.divider()
qtext = st.text_input("💬 Ask a question", key="qtext",
                      placeholder="e.g., Tell me the specs of Galaxy S21 / Compare Echo Dot vs Nest Mini")

if st.button("Send", type="primary"):
    if not index_ready:
        st.warning("Index is not ready. Please refresh the app.")
    elif not qtext:
        st.warning("Enter a question.")
    else:
        # Encode query with MiniLM
        qvec = encode_query(qtext, device=None if device == "auto" else device)
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
                "description": meta.get("description"),
            })

        ans = _answer_with_openai(qtext, snippets)
        st.chat_message("assistant").write(ans)
