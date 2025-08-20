import os, io, re, json, base64, argparse, hashlib
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

# Runtime deps
import faiss                           # pip install faiss-cpu
from sentence_transformers import SentenceTransformer

try:
    import requests
except Exception:
    requests = None  # guarded usage

# ---------------- Models ----------------
# text encoder는 MiniLM(L6) 기본값, image encoder는 CLIP 고정
MODELS = {
    "text": {
        "model_name": "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "dim": 384
    },
    "image_text": {
        "model_name": "clip-ViT-B-32",
        "dim": 512
    }
}

TEXT_EMB_MODEL = MODELS["text"]["model_name"]           # ← 미드텀에서 쓰던 MiniLM
CLIP_EMB_MODEL = MODELS["image_text"]["model_name"]     # 이미지 겸용(CLIP)
EMB_DIM = MODELS["image_text"]["dim"]                   # 파이널 인덱스는 CLIP 차원(512)로 맞춤
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
    # rename
    plan = {}
    for c in list(df.columns):
        key = str(c).lower().strip()
        if key in ALIASES and ALIASES[key] not in df.columns:
            plan[c] = ALIASES[key]
    if plan:
        df = df.rename(columns=plan)

    # features merge
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

    if "description" in df.columns:
        df["description"] = df["description"].fillna("").astype(str)

    if "image_url" in df.columns:
        df["image_url"] = df["image_url"].astype(str).apply(_first_url)

    for c in ["title","brand","price","features","description","image_url","image_b64"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    text_cols = [c for c in ["title","description","features","brand","price"] if c in df.columns]
    if text_cols:
        def _has_text(row):
            return any(isinstance(row.get(c,""), str) and row.get(c,"").strip() for c in text_cols)
        df = df[df.apply(_has_text, axis=1)].copy()
    return df

# ---------- Embedding helpers ----------
def load_text_encoder(device: Optional[str] = None) -> SentenceTransformer:
    """
    텍스트 전용 임베더(MiniLM). CPU에서도 매우 빠름.
    """
    return SentenceTransformer(TEXT_EMB_MODEL, device=device)

def load_clip(device: Optional[str] = None) -> SentenceTransformer:
    """
    CLIP 임베더(텍스트+이미지). 여기서는 이미지 전용 + 텍스트 대체용으로 사용 가능.
    """
    return SentenceTransformer(CLIP_EMB_MODEL, device=device)

def _download_image(url: str) -> Optional[bytes]:
    if not url or not requests:
        return None
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.content
    except Exception:
        return None

def embed_rows(
    df: pd.DataFrame,
    text_model: SentenceTransformer,
    image_model: Optional[SentenceTransformer],
    use_images: bool,
    limit: Optional[int]=None
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
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

    # ▶ 텍스트: MiniLM
    text_vecs = text_model.encode(texts, normalize_embeddings=True, batch_size=128).astype("float32")

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
        # MiniLM 차원(384)을 CLIP 차원(512)에 맞추기 위해 패딩(간단한 방식).
        t = text_vecs[i:i+1]
        if t.shape[1] != EMB_DIM:
            pad = np.zeros((1, EMB_DIM - t.shape[1]), dtype="float32")
            t = np.concatenate([t, pad], axis=1)

        vec = t  # 기본은 텍스트

        # ▶ 이미지 있으면 CLIP으로 임베딩 후 평균 결합
        if use_images and image_model is not None:
            b64 = str(row.get("image_b64","")).strip()
            raw = None
            if b64:
                try:
                    raw = base64.b64decode(b64)
                except Exception:
                    raw = None
            if raw is None:
                url = str(row.get("image_url","")).strip()
                if url:
                    raw = _download_image(url)

            if raw:
                try:
                    pil = Image.open(io.BytesIO(raw)).convert("RGB")
                    iv = image_model.encode([pil], normalize_embeddings=True).astype("float32")
                    meta["_image_b64"] = base64.b64encode(raw).decode("utf-8")
                    # iv는 512차원(CLIP). t도 위에서 512로 패딩했으니 평균 가능
                    vec = (vec + iv) / 2.0
                except (UnidentifiedImageError, Exception):
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
    (out/"manifest.json").write_text(json.dumps({
        "text_model": TEXT_EMB_MODEL,
        "image_model": CLIP_EMB_MODEL,
        "emb_dim": EMB_DIM
    }, ensure_ascii=False), encoding="utf-8")

def load_artifacts(outdir: str) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    out = Path(outdir)
    index = faiss.read_index(str(out/"index.faiss"))
    metas = json.loads((out/"metas.json").read_text(encoding="utf-8"))
    return index, metas

# ---------- Query ----------
def encode_query(
    text_model: SentenceTransformer,
    image_model: Optional[SentenceTransformer],
    text: str = "",
    image_path: Optional[str] = None
) -> np.ndarray:
    tv = text_model.encode([text], normalize_embeddings=True).astype("float32") if text else None
    if tv is not None and tv.shape[1] != EMB_DIM:
        pad = np.zeros((1, EMB_DIM - tv.shape[1]), dtype="float32")
        tv = np.concatenate([tv, pad], axis=1)

    iv = None
    if image_path and image_model is not None:
        with open(image_path, "rb") as f:
            pil = Image.open(io.BytesIO(f.read())).convert("RGB")
        iv = image_model.encode([pil], normalize_embeddings=True).astype("float32")

    if tv is not None and iv is not None:
        return ((tv + iv) / 2.0).astype("float32")
    return (tv if tv is not None else iv).astype("float32")

def search(index: faiss.IndexFlatIP, qvec: np.ndarray, k: int = 5):
    D, I = index.search(qvec, k)
    return D[0].tolist(), I[0].tolist()

# ---------- File I/O ----------
def read_any(path: str) -> pd.DataFrame:
    p = str(path)
    if os.path.exists(p) and p.endswith(".parquet"):
        return pd.read_parquet(p)
    if os.path.exists(p) and (p.endswith(".csv") or p.endswith(".gz")):
        return pd.read_csv(p, compression="infer")
    # try parquet → csv 자동
    try:
        return pd.read_parquet(p)
    except Exception:
        return pd.read_csv(p, compression="infer")

# ---------- Build-or-load (used by Streamlit) ----------
def build_or_load_index(
    df: Optional[pd.DataFrame] = None,
    device: Optional[str] = None,
    use_images: bool = True,
    limit: Optional[int] = None,
    outdir: str = "artifacts",
    batch_size: int = 128,
    force_rebuild: bool = False
):
    """
    df가 없으면 clean_data.parquet → clean_data.csv 순으로 로드해서 인덱스 생성/재사용
    """
    # 1) 데이터 준비
    if df is None:
        src = "clean_data.parquet" if os.path.exists("clean_data.parquet") else (
              "clean_data.csv" if os.path.exists("clean_data.csv") else None)
        if src is None:
            raise FileNotFoundError("Neither clean_data.parquet nor clean_data.csv exists.")
        df = read_any(src)
    df = coerce_schema(df)

    # 2) 재사용 체크 (fingerprint)
    fp_src = json.dumps({
        "rows": int(len(df) if limit is None else min(limit, len(df))),
        "cols": sorted(list(df.columns)),
        "text_model": TEXT_EMB_MODEL,
        "image_model": CLIP_EMB_MODEL,
        "use_images": bool(use_images)
    }, sort_keys=True)
    fp = hashlib.md5(fp_src.encode("utf-8")).hexdigest()
    fp_path = Path(outdir) / "manifest.json"

    if not force_rebuild and fp_path.exists():
        try:
            mani = json.loads(fp_path.read_text(encoding="utf-8"))
            if mani.get("fingerprint") == fp and (Path(outdir)/"index.faiss").exists() and (Path(outdir)/"metas.json").exists():
                return load_artifacts(outdir)
        except Exception:
            pass

    # 3) 임베딩 및 인덱스 빌드
    text_model = load_text_encoder(device)
    image_model = load_clip(device) if use_images else None
    X, metas = embed_rows(df, text_model, image_model, use_images=use_images, limit=limit)
    index = build_faiss(X)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    save_artifacts(index, metas, outdir)

    # manifest에 fingerprint 저장
    mani = {
        "text_model": TEXT_EMB_MODEL,
        "image_model": CLIP_EMB_MODEL,
        "emb_dim": EMB_DIM,
        "fingerprint": fp
    }
    (Path(outdir)/"manifest.json").write_text(json.dumps(mani, ensure_ascii=False), encoding="utf-8")
    return index, metas

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Build & query a multimodal FAISS index (MiniLM + CLIP).")
    ap.add_argument("--data", default=None, help="Input parquet/csv path (default: clean_data.parquet → clean_data.csv)")
    ap.add_argument("--outdir", default="artifacts", help="Where to save index & metas")
    ap.add_argument("--limit", type=int, default=None, help="Head limit")
    ap.add_argument("--device", default=None, choices=[None,"cpu","cuda","mps"], help="Encoder device")
    ap.add_argument("--with-images", action="store_true", help="Combine image embeddings if available")
    ap.add_argument("--query", default=None, help="Quick test query after build")
    ap.add_argument("--image", default=None, help="Optional query image path")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--force", action="store_true", help="Force rebuild even if artifacts look reusable")
    args = ap.parse_args()

    if args.data is None:
        args.data = "clean_data.parquet" if os.path.exists("clean_data.parquet") else "clean_data.csv"

    print(f"[pipeline] loading: {args.data}")
    df = read_any(args.data)
    print(f"[pipeline] rows: {len(df)}")
    df = coerce_schema(df)

    text_model = load_text_encoder(args.device)
    image_model = load_clip(args.device) if args.with_images else None
    X, metas = embed_rows(df, text_model, image_model, use_images=args.with_images, limit=args.limit)
    index = build_faiss(X)
    save_artifacts(index, metas, args.outdir)
    print(f"[pipeline] saved artifacts to {args.outdir} (docs={len(metas)})")

    if args.query or args.image:
        print(f"[pipeline] quick search: q='{args.query}' img='{args.image}'")
        qvec = encode_query(text_model, image_model, text=args.query or "", image_path=args.image)
        D, I = search(index, qvec, k=args.topk)
        for rank, (d, i) in enumerate(zip(D, I), start=1):
            m = metas[i] if 0 <= i < len(metas) else {}
            print(f"{rank:2d}. score={d:.3f} | {m.get('title','(no title)')} | {m.get('brand','')} | {m.get('price','')}")
            if m.get('image_url'): print("    img:", m.get('image_url'))

if __name__ == "__main__":
    main()

