# pipeline.py — Build & use a FAISS index with CLIP (text & optional image)
# Input : clean_data.parquet (or clean_data.csv/.csv.gz)
# Output: artifacts/index.faiss, artifacts/metas.json, artifacts/manifest.json

import os, io, re, json, base64, hashlib
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import faiss                                  # pip install faiss-cpu
from sentence_transformers import SentenceTransformer

try:
    import requests
except Exception:
    requests = None

REQUEST_TIMEOUT = 7

# ====== CLIP model (unified) ======
EMB_MODEL = "clip-ViT-B-32"   # 👈 통일
EMB_DIM   = 512

# ====== Canonical schema & alias ======
CANON = ["title", "brand", "price", "features", "description", "image_url"]
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

def _first_url(s: str) -> str:
    if not isinstance(s, str): return ""
    m = URL_RE.search(s)
    return m.group(0) if m else ""

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
                v = row.get(c, "")
                if isinstance(v, (list, tuple)):
                    v = " ; ".join([str(x) for x in v if str(x).strip()])
                v = str(v).strip()
                if v and v.lower() not in ("nan","none","null"):
                    parts.append(v)
            return " ; ".join(parts)
        df["features"] = df.apply(_merge, axis=1)

    # clean description / image_url
    if "description" in df.columns:
        df["description"] = df["description"].fillna("").astype(str)
    if "image_url" in df.columns:
        df["image_url"] = df["image_url"].astype(str).apply(_first_url)

    # fill types
    for c in CANON:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    # drop rows with no text signal
    text_cols = [c for c in ["title","description","features","brand","price"] if c in df.columns]
    if text_cols:
        def _has_text(row):
            return any(str(row.get(c,"")).strip() for c in text_cols)
        df = df[df.apply(_has_text, axis=1)].copy()

    return df

def read_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"): return pd.read_parquet(path)
    if path.endswith(".csv") or path.endswith(".gz"): return pd.read_csv(path, compression="infer")
    try: return pd.read_parquet(path)
    except Exception: return pd.read_csv(path, compression="infer")

def _fingerprint_df(df: pd.DataFrame, cols: List[str]) -> str:
    sample = df[cols].head(200).to_csv(index=False)
    h = hashlib.sha1(); h.update(sample.encode("utf-8"))
    return h.hexdigest()

def load_encoder(device: Optional[str] = None) -> SentenceTransformer:
    return SentenceTransformer(EMB_MODEL, device=device)

def _download_image(url: str) -> Optional[bytes]:
    if not (requests and url): return None
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT); r.raise_for_status()
        return r.content
    except Exception:
        return None

def embed_rows(
    df: pd.DataFrame,
    model: SentenceTransformer,
    use_images: bool = True,
    limit: Optional[int] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if limit: df = df.head(limit)

    text_cols = [c for c in ["title","brand","price","features","description"] if c in df.columns]
    texts, rows = [], []
    for _, row in df.iterrows():
        parts = [str(row[c]) for c in text_cols if str(row.get(c,"")).strip()]
        texts.append(" | ".join(parts) if parts else "")
        rows.append(row)
    if not texts:
        raise ValueError("No usable text to embed.")

    tv = model.encode(texts, normalize_embeddings=True, batch_size=64).astype("float32")

    metas, vecs = [], []
    for i, row in enumerate(rows):
        meta = {
            "title": row.get("title",""),
            "brand": row.get("brand",""),
            "price": row.get("price",""),
            "features": row.get("features",""),
            "description": row.get("description",""),
            "image_url": row.get("image_url",""),
        }
        vec = tv[i:i+1]

        if use_images:
            # image_b64 우선, 없으면 image_url
            b64 = str(row.get("image_b64","")).strip()
            raw = None
            if b64:
                try: raw = base64.b64decode(b64)
                except Exception: raw = None
            elif meta["image_url"]:
                raw = _download_image(meta["image_url"])

            if raw:
                try:
                    pil = Image.open(io.BytesIO(raw)).convert("RGB")
                    iv = model.encode([pil], normalize_embeddings=True).astype("float32")
                    meta["_image_b64"] = base64.b64encode(raw).decode("utf-8")
                    vec = (vec + iv) / 2.0
                except Exception:
                    pass

        metas.append(meta); vecs.append(vec[0])

    X = np.vstack(vecs).astype("float32")
    assert X.shape[1] == EMB_DIM, f"Expected {EMB_DIM}D, got {X.shape[1]}"
    return X, metas

def build_faiss_index(X: np.ndarray) -> faiss.IndexFlatIP:
    assert X.shape[1] == EMB_DIM, f"Index dim must be {EMB_DIM}"
    idx = faiss.IndexFlatIP(EMB_DIM)
    idx.add(X)
    return idx

def save_artifacts(index: faiss.Index, metas: List[Dict[str, Any]], outdir: str, manifest: Dict[str, Any]):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out/"index.faiss"))
    (out/"metas.json").write_text(json.dumps(metas, ensure_ascii=False), encoding="utf-8")
    (out/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

def load_artifacts(outdir: str) -> Tuple[faiss.Index, List[Dict[str, Any]], Dict[str, Any]]:
    out = Path(outdir)
    idx = faiss.read_index(str(out/"index.faiss"))
    metas = json.loads((out/"metas.json").read_text(encoding="utf-8"))
    manifest = json.loads((out/"manifest.json").read_text(encoding="utf-8")) if (out/"manifest.json").exists() else {}
    return idx, metas, manifest

def build_or_load_index(
    df: pd.DataFrame,
    device: Optional[str] = None,
    use_images: bool = True,
    limit: Optional[int] = None,
    outdir: str = "artifacts",
    batch_size: int = 64,     # kept for signature
    force_rebuild: bool = False,
) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    text_cols = [c for c in ["title","brand","price","features","description"] if c in df.columns]
    if not text_cols:
        raise ValueError("No text columns found to build index.")

    fp = _fingerprint_df(df, text_cols)
    manifest_new = {"fingerprint": fp, "model": EMB_MODEL, "dim": EMB_DIM, "limit": limit, "use_images": use_images}

    # Reuse if possible
    try:
        idx_old, metas_old, man_old = load_artifacts(outdir)
        if (not force_rebuild and
            man_old.get("fingerprint") == fp and
            man_old.get("model") == EMB_MODEL and
            man_old.get("dim") == EMB_DIM and
            man_old.get("limit") == limit and
            man_old.get("use_images") == use_images):
            return idx_old, metas_old
    except Exception:
        pass

    encoder = load_encoder(device=device)
    X, metas = embed_rows(df, encoder, use_images=use_images, limit=limit)
    index = build_faiss_index(X)
    save_artifacts(index, metas, outdir, manifest_new)
    return index, metas

# ---- Query helper (text and optional image path) ----
def encode_query(text: str = "", image_path: Optional[str] = None, device: Optional[str] = None) -> np.ndarray:
    model = load_encoder(device=device)
    tv = model.encode([text], normalize_embeddings=True).astype("float32") if text else None
    iv = None
    if image_path:
        with open(image_path, "rb") as f:
            pil = Image.open(io.BytesIO(f.read())).convert("RGB")
        iv = model.encode([pil], normalize_embeddings=True).astype("float32")
    if tv is not None and iv is not None:
        return ((tv + iv) / 2.0).astype("float32")
    return (tv if tv is not None else iv).astype("float32")

