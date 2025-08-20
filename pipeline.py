# pipeline.py — Build & use a FAISS index (text-only, MiniLM) from product data.
# Input : clean_data.parquet (or clean_data.csv/.csv.gz)
# Output: artifacts/index.faiss, artifacts/metas.json, artifacts/manifest.json

import os, re, json, hashlib
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import faiss                                  # pip install faiss-cpu
from sentence_transformers import SentenceTransformer

# ====== Models (midterm-style selector) ======
MODELS = {
    "2": {"model_name": "sentence-transformers/paraphrase-MiniLM-L6-v2", "model_type": "encoder-only"},
}
DEFAULT_MODEL_KEY = "2"  # MiniLM

# ====== Canonical schema & alias ======
CANON = ["title", "price", "features", "description", "image_url", "brand"]
ALIASES = {
    "product_title":"title","name":"title","title":"title",
    "brand_name":"brand","brand":"brand",
    "price_usd":"price","list_price":"price","current_price":"price","sale_price":"price","amount":"price","price":"price",
    "feature":"features","features":"features",
    "product_description":"description","short_description":"description","long_description":"description","text":"description","description":"description",
    "imageurl":"image_url","imageurlhighres":"image_url","image_link":"image_url","images":"image_url","img_url":"image_url","image":"image_url",
    # Henry CSV 호환
    "product name":"title","selling price":"price","about product":"description",
    "product specification":"features","technical details":"features","variants":"features",
    "product dimensions":"features","shipping weight":"features","text_blob":"description",
}
URL_RE = re.compile(r"https?://[^\s|,;]+", re.IGNORECASE)

def _first_url(s: Any) -> str:
    s = "" if s is None else str(s)
    m = URL_RE.search(s)
    return m.group(0) if m else ""

def _to_str(x: Any) -> str:
    # JSON-safe 문자열화 (Series/ndarray/list 모두 대응)
    try:
        if x is None:
            return ""
        if isinstance(x, (list, tuple, np.ndarray)):
            return " ; ".join([_to_str(xx) for xx in x])
        return str(x)
    except Exception:
        return ""

def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    # rename aliases
    plan = {}
    for c in list(df.columns):
        key = str(c).lower().strip()
        if key in ALIASES and ALIASES[key] not in df.columns:
            plan[c] = ALIASES[key]
    if plan:
        df = df.rename(columns=plan)

    # merge several feature-like cols
    join_cols = [c for c in ["features","product specification","technical details","variants","product dimensions","shipping weight"] if c in df.columns]
    if join_cols:
        def _merge(row):
            parts = []
            for c in join_cols:
                parts.append(_to_str(row.get(c, "")))
            return " ; ".join([p for p in parts if p and p.lower() not in ("nan","none","null")])
        df["features"] = df.apply(_merge, axis=1)

    # clean description / image_url / brand / title / price
    for c in ["title","brand","price","features","description"]:
        if c in df.columns:
            df[c] = df[c].map(_to_str)
    if "image_url" in df.columns:
        df["image_url"] = df["image_url"].map(_first_url)

    # drop rows with no text signal
    text_cols = [c for c in ["title","description","features","brand","price"] if c in df.columns]
    if text_cols:
        def _has_text(row):
            return any(_to_str(row.get(c,"")).strip() for c in text_cols)
        df = df[df.apply(_has_text, axis=1)].copy()

    return df

def read_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv") or path.endswith(".gz"):
        return pd.read_csv(path, compression="infer")
    # fallback
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path, compression="infer")

def _fingerprint_df(df: pd.DataFrame, cols: List[str]) -> str:
    # 작은 샘플을 문자열로 해시
    safe_cols = [c for c in cols if c in df.columns]
    sample = df[safe_cols].head(200).astype(str).to_csv(index=False)
    h = hashlib.sha1()
    h.update(sample.encode("utf-8"))
    return h.hexdigest()

def load_encoder(model_key: str = DEFAULT_MODEL_KEY, device: Optional[str] = None) -> SentenceTransformer:
    name = MODELS[model_key]["model_name"]
    return SentenceTransformer(name, device=device)

def embed_text_rows(
    df: pd.DataFrame,
    model: SentenceTransformer,
    text_cols: List[str],
    limit: Optional[int] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if limit:
        df = df.head(limit)

    texts, metas = [], []
    for _, row in df.iterrows():
        parts = [_to_str(row.get(c, "")) for c in text_cols if _to_str(row.get(c, "")).strip()]
        text = " | ".join(parts) if parts else ""
        if not text:
            continue
        texts.append(text)
        metas.append({
            "title": _to_str(row.get("title","")),
            "brand": _to_str(row.get("brand","")),
            "price": _to_str(row.get("price","")),
            "features": _to_str(row.get("features","")),
            "description": _to_str(row.get("description","")),
            "image_url": _to_str(row.get("image_url","")),
        })

    if not texts:
        raise ValueError("No usable text to embed.")

    X = model.encode(texts, normalize_embeddings=True, batch_size=64)
    X = np.asarray(X, dtype="float32")
    return X, metas

def build_faiss_index(X: np.ndarray) -> faiss.IndexFlatIP:
    d = X.shape[1]
    idx = faiss.IndexFlatIP(d)
    idx.add(X.astype("float32"))
    return idx

def save_artifacts(index: faiss.Index, metas: List[Dict[str, Any]], outdir: str, manifest: Dict[str, Any]):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "index.faiss"))
    # 안전 저장: default=str
    (out / "metas.json").write_text(json.dumps(metas, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, default=str), encoding="utf-8")

def load_artifacts(outdir: str) -> Tuple[faiss.Index, List[Dict[str, Any]], Dict[str, Any]]:
    out = Path(outdir)
    idx = faiss.read_index(str(out / "index.faiss"))
    metas = json.loads((out / "metas.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8")) if (out / "manifest.json").exists() else {}
    return idx, metas, manifest

def build_or_load_index(
    df: pd.DataFrame,
    device: Optional[str] = None,
    model_key: str = DEFAULT_MODEL_KEY,
    use_images: bool = False,         # kept for signature compatibility (ignored)
    limit: Optional[int] = None,
    outdir: str = "artifacts",
    batch_size: int = 64,             # not used directly (SentenceTransformer handles)
    force_rebuild: bool = False,
) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Build (or reuse) a FAISS index from df using MiniLM sentence embeddings.
    Returns: (faiss.Index, metas)
    """
    text_cols = [c for c in ["title","brand","price","features","description"] if c in df.columns]
    if not text_cols:
        raise ValueError("No text columns found to build index.")

    fp = _fingerprint_df(df, text_cols)
    model_name = MODELS[model_key]["model_name"]
    manifest_new = {"fingerprint": fp, "model": model_name, "limit": int(limit) if limit else None}

    # Reuse if possible
    try:
        idx_old, metas_old, man_old = load_artifacts(outdir)
        if (not force_rebuild
            and man_old.get("fingerprint") == fp
            and man_old.get("model") == model_name
            and (man_old.get("limit") == (int(limit) if limit else None))):
            return idx_old, metas_old
    except Exception:
        pass

    # Build fresh
    encoder = load_encoder(model_key, device=device)
    X, metas = embed_text_rows(df, encoder, text_cols, limit=limit)
    index = build_faiss_index(X)
    save_artifacts(index, metas, outdir, manifest_new)
    return index, metas

# ---- Query helper for the app ----
def encode_query(text: str, device: Optional[str] = None, model_key: str = DEFAULT_MODEL_KEY) -> np.ndarray:
    encoder = load_encoder(model_key, device=device)
    q = encoder.encode([text], normalize_embeddings=True)
    return np.asarray(q, dtype="float32")

