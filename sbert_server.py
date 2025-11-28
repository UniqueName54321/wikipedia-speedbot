import argparse
import time
from typing import List

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn


app = FastAPI()

model: SentenceTransformer | None = None
device: str = "cpu"


class EmbedRequest(BaseModel):
    texts: List[str]
    normalize: bool = True


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]


def pick_device(preferred: str) -> str:
    """
    preferred: 'auto', 'cuda', 'cpu', 'xpu', 'mps'
    """
    preferred = preferred.lower()

    def has_cuda() -> bool:
        return torch.cuda.is_available()

    def has_xpu() -> bool:
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    def has_mps() -> bool:
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if preferred == "cuda":
        if has_cuda():
            return "cuda"
        print("[SBERT server] Requested cuda, but no CUDA device found; falling back to auto.")
        preferred = "auto"

    if preferred == "xpu":
        if has_xpu():
            return "xpu"
        print("[SBERT server] Requested xpu, but no Intel XPU device found; falling back to auto.")
        preferred = "auto"

    if preferred == "mps":
        if has_mps():
            return "mps"
        print("[SBERT server] Requested mps, but MPS not available; falling back to auto.")
        preferred = "auto"

    if preferred == "cpu":
        return "cpu"

    # auto mode
    if has_cuda():
        return "cuda"      # Nvidia or AMD ROCm build
    if has_xpu():
        return "xpu"       # Intel
    if has_mps():
        return "mps"       # Apple Silicon
    return "cpu"


@app.get("/health")
def health():
    return {"status": "ok", "device": device}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if model is None:
        # This was your error before — model stayed None in the uvicorn worker
        raise RuntimeError("Model not loaded on server.")

    if not req.texts:
        return {"embeddings": []}

    with torch.no_grad():
        emb = model.encode(
            req.texts,
            batch_size=64,
            convert_to_tensor=True,
            normalize_embeddings=req.normalize,
            show_progress_bar=False,
        )

    # Always return CPU lists for JSON
    emb = emb.cpu().tolist()
    return {"embeddings": emb}


def main():
    global model, device

    parser = argparse.ArgumentParser(description="SBERT HTTP embedding server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto / cuda / cpu / xpu / mps",
    )

    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"[SBERT server] Using device: {device}")

    t0 = time.time()
    model = SentenceTransformer(args.model_name, device=device)
    t1 = time.time()
    print(f"[SBERT server] Model '{args.model_name}' loaded in {t1 - t0:.3f}s")

    # IMPORTANT: use the in-memory app, NOT the string "sbert_server:app"
    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
