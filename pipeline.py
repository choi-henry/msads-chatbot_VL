import os, io, re, json, base64, argparse
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Runtime deps
import faiss                           # pip install faiss-cpu
from sentence_transformers import SentenceTransformer

try:
    import requests
except Exception:
    requests = None  # we guard usage below

EMB_MODEL = "clip-ViT-B-32"
EMB_DIM = 512
REQUEST_TIMEOUT = 8

# ---------- Schema mapping ----------
CANON = ["title", "brand", "price", "features", "description", "image_url"]

ALIASES = {
    # generic
    "product_title":"title", "name":"title", "title":"title",
    "brand_name":"brand", "brand":"brand",
    "price_usd":"price","list_price":"price","current_price":"price","sale_price":"price","amount":"price","price":"price",
    "feature":"features","features":"features",
    "product_description":"description","short_description":"description","long_description":"description","text":"description","description":"description",
    "imageurl":"image_url","imageurlhighres":"image_url","image_link":"image_url","images":"image_url","img_url":"image_url","image":"image_url",

    # Henry CSV 스키마 대응
    "product name":"title",
    "selling price":"price",
    "about product":"description",
    "product specification":"features",
    "technical details":"features",
    "variants":"features",
    "product dimensions":"features",
    "shipping weight":"features",
    "text_blob":"description",
}

URL_RE = re.compile(r"https?://[^\s|,;]+", re.IGNORECASE)

def _first_url(s: str) -> str:
    if not isinstance(s, str): return ""
    m = URL_RE.search(s)
    return m.group(0) if m else ""

def parse_meta_cell(val: Any) -> Dict[str, Any]:
    """metadata 컬럼이 dict/JSON/str 섞여 있을 때 dict로 파싱"""
    if isinstance(val, dict):
        return val
    if isinstance(val, str) and val.strip():
        try:
            return json.loads(val)
        except Exception:
            out = {}
            for part in re.split(r"[|;]\s*", val):
                if ":" in part:
                    k, v = part.split(":", 1)
                    out[k.strip()] = v.strip()
            return out
    return {}

def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    # 1) rename → canonical
    plan = {}
    for c in list(df.columns):
        key = c.lower().strip()
        if key in ALIASES and ALIASES[key] not in df.columns:
            plan[c] = ALIASES[key]
    if plan:
        df = df.rename(columns=plan)

    # 2) features 여러 열 합치기
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

    # 3) description 보정
    if "description" in df.columns:
        df["description"] = df["description"].fillna("").astype(str)

    # 4) image_url 정제 (혼합 문자열에서 첫 URL 추출)
    if "image_url" in df.columns:
        df["image_url"] = df["image_url"].astype(str).apply(_first_url)

    # 5) dtype 정리
    for c in ["title","brand","price","features","description","image_url","image_b64"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    # 6) 텍스트 신호가 전혀 없는 행 제거
    text_cols = [c for c in ["title","description","features","brand","price"] if c in df.columns]
    if text_cols:
        def _has_text(row):
            return any(isinstance(row.get(c,""), str) and row.get(c,"").strip() for c in text_cols)
        df = df[df.apply(_has_text, axis=1)].copy()

    return df

# ---------- Embedding helpers ----------
def load_clip(device: Optional[str] = None) -> SentenceTransformer:
    return SentenceTransformer(EMB_MODEL, device=device)

def _download_image(url: str) -> Optional[bytes]:
    if not url or not requests:
        return None
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.content
    except Exception:
        return None

def embed_rows(df: pd.DataFrame, model: SentenceTransformer, use_images: bool,
               limit: Optional[int]=None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if limit:
        df = df.head(limit)

    text_cols = [c for c in ["title","brand","price","features","description"] if c in df.columns]
    texts, rows = [], []
    for _, row in df.iterrows():
        parts = [str(row[c]) for c in text_cols if str(row.get(c,"")).strip()]
        texts.append(" | ".join(parts) if parts else "")
        rows.append(row)
    if not texts:
        raise ValueError("No usable text after preprocessing; check your columns.")

    text_vecs = model.encode(texts, normalize_embeddings=True, batch_size=64).astype("float32")

    metas: List[Dict[str, Any]] = []
    vecs: List[np.ndarray] = []

    for i, row in enumerate(rows):
        meta = {
            "id": str(row.get("Uniq Id","")) or str(i),
            "title": row.get("title",""),
            "brand": row.get("brand",""),
            "price": row.get("price",""),
            "features": row.get("features",""),
            "description": row.get("description",""),
            "image_url": row.get("image_url",""),
        }
        vec = text_vecs[i:i+1]

        if use_images:
            b64 = str(row.get("image_b64","")).strip()
            if b64:
                try:
                    raw = base64.b64decode(b64)
                    meta["_image_b64"] = b64
                    pil = Image.open(io.BytesIO(raw)).convert("RGB")
                    iv = model.encode([pil], normalize_embeddings=True).astype("float32")
                    vec = (vec + iv) / 2.0
                except Exception:
                    pass
            else:
                url = str(row.get("image_url","")).strip()
                if url:
                    raw = _download_image(url)
                    if raw:
                        meta["_image_b64"] = base64.b64encode(raw).decode("utf-8")
                        try:
                            pil = Image.open(io.BytesIO(raw)).convert("RGB")
                            iv = model.encode([pil], normalize_embeddings=True).astype("float32")
                            vec = (vec + iv) / 2.0
                        except Exception:
                            pass

        metas.append(meta)
        vecs.append(vec[0])

    X = np.vstack(vecs).astype("float32")
    return X, metas

# ---------- Index I/O ----------
def build_faiss(X: np.ndarray) -> faiss.IndexFlatIP:
    idx = faiss.IndexFlatIP(EMB_DIM)
    idx.add(X.astype("float32"))
    return idx

def save_artifacts(index: faiss.Index, metas: List[Dict[str, Any]], outdir: str):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out/"index.faiss"))
    (out/"metas.json").write_text(json.dumps(metas, ensure_ascii=False), encoding="utf-8")
    (out/"manifest.json").write_text(json.dumps({"emb_model": EMB_MODEL, "emb_dim": EMB_DIM}, ensure_ascii=False), encoding="utf-8")

def load_artifacts(outdir: str) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    out = Path(outdir)
    index = faiss.read_index(str(out/"index.faiss"))
    metas = json.loads((out/"metas.json").read_text(encoding="utf-8"))
    return index, metas

# ---------- Query ----------
def encode_query(model: SentenceTransformer, text: str = "", image_path: Optional[str] = None) -> np.ndarray:
    tv = model.encode([text], normalize_embeddings=True).astype("float32") if text else None
    iv = None
    if image_path:
        with open(image_path, "rb") as f:
            pil = Image.open(io.BytesIO(f.read())).convert("RGB")
        iv = model.encode([pil], normalize_embeddings=True).astype("float32")
    if tv is not None and iv is not None:
        return ((tv + iv) / 2.0).astype("float32")
    return (tv if tv is not None else iv).astype("float32")

def search(index: faiss.IndexFlatIP, qvec: np.ndarray, k: int = 5):
    D, I = index.search(qvec, k)
    return D[0].tolist(), I[0].tolist()

# ---------- CLI ----------
def read_any(path: str) -> pd.DataFrame:
    p = str(path)
    if p.endswith(".parquet"):
        return pd.read_parquet(p)
    if p.endswith(".gz") or p.endswith(".csv"):
        return pd.read_csv(p, compression="infer")
    # try auto
    try:
        return pd.read_parquet(p)
    except Exception:
        return pd.read_csv(p, compression="infer")

def main():
    ap = argparse.ArgumentParser(description="Build & query a multimodal FAISS index.")
    ap.add_argument("--data", default="clean_data.parquet", help="Input parquet/csv path")
    ap.add_argument("--outdir", default="artifacts", help="Where to save index & metas")
    ap.add_argument("--limit", type=int, default=None, help="Head limit")
    ap.add_argument("--device", default=None, choices=[None,"cpu","cuda","mps"], help="Encoder device")
    ap.add_argument("--with-images", action="store_true", help="Combine image embeddings if available")
    ap.add_argument("--query", default=None, help="Quick test query after build")
    ap.add_argument("--image", default=None, help="Optional query image path")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    print(f"[pipeline] loading: {args.data}")
    df = read_any(args.data)
    print(f"[pipeline] rows: {len(df)}")

    df = coerce_schema(df)
    print(f"[pipeline] columns -> {list(df.columns)}")

    model = load_clip(args.device)
    X, metas = embed_rows(df, model, use_images=args.with_images, limit=args.limit)
    index = build_faiss(X)
    save_artifacts(index, metas, args.outdir)
    print(f"[pipeline] saved artifacts to {args.outdir} (docs={len(metas)})")

    if args.query or args.image:
        print(f"[pipeline] quick search: q='{args.query}' img='{args.image}'")
        qvec = encode_query(model, args.query or "", args.image)
        D, I = search(index, qvec, k=args.topk)
        for rank, (d, i) in enumerate(zip(D, I), start=1):
            m = metas[i] if 0 <= i < len(metas) else {}
            print(f"{rank:2d}. score={d:.3f} | {m.get('title','(no title)')} | {m.get('brand','')} | {m.get('price','')}")
            if m.get('image_url'): print("    img:", m.get('image_url'))

if __name__ == "__main__":
    main()
