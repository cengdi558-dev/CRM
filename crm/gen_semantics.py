"""Generate and cache the three-level semantic descriptions used by SSA.

The paper prompts a Llama-family instruction model with three levels —
coarse (global appearance + occlusion ratio), medium (dominant AUs + bilateral
consistency), fine (AU intensities + local texture) — and aligns the resulting
phrases with visual tokens. Calling an LLM inside the training loop is not
practical, so this script produces the descriptions **once** and stores both the
text and its embedding:

    python -m crm.gen_semantics --data datasets/RAF-DB/train --out sem_train.npz
    python -m crm.gen_semantics --data faces --out sem.npz --provider openai \
        --base-url https://api.deepseek.com/v1 --model deepseek-chat

Providers:
    template   offline, deterministic sentences built from the AU statistics
    openai     any OpenAI-compatible ``/chat/completions`` endpoint
Embedders:
    hash       offline hashing of character n-grams (default)
    st         sentence-transformers, if installed
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Callable

import numpy as np

from .regions import AU_REGIONS

LEVELS = ("coarse", "medium", "fine")


# ---------------------------------------------------------------- AU features
def au_features(image_path: str, size: int = 112) -> tuple[np.ndarray, float, float]:
    """Eight regional responses, occlusion ratio and bilateral consistency.

    Mirrors :meth:`crm.coc.COCModule.au_responses` in numpy so this script needs
    no torch.
    """
    from PIL import Image

    from .regions import _TEMPLATE

    with Image.open(image_path) as im:
        g = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32)

    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    grad = np.sqrt(gx * gx + gy * gy)

    dark = g < np.quantile(g, 0.2)
    flat = grad < np.quantile(grad, 0.2)
    occ = dark | flat

    resp, ratios = [], []
    for x0, y0, x1, y1 in _TEMPLATE:
        ys, ye = int(y0 * size), int(y1 * size)
        xs, xe = int(x0 * size), int(x1 * size)
        sub_g = grad[ys:ye, xs:xe]
        sub_o = occ[ys:ye, xs:xe]
        visible = ~sub_o
        ratios.append(float(visible.mean()) if visible.size else 1.0)
        resp.append(float(sub_g[visible].mean()) if visible.any() else 0.0)

    a = np.asarray(resp, dtype=np.float64)
    # same problem as in COCModule: a fixed 0/1 scale saturates tanh, so the
    # responses are standardised before the squashing
    a = (a - a.mean()) / (a.std() + 1e-6)
    a = np.tanh(a)
    consistency = 1.0 - float(np.mean(np.abs(a[[0, 2, 5, 7]] - a[[1, 3, 6, 7]])) / 2.0)
    return a, float(1.0 - np.mean(ratios)), consistency


# ------------------------------------------------------------------- prompts
def build_prompts(a: np.ndarray, occlusion: float, consistency: float) -> dict[str, str]:
    order = np.argsort(-np.abs(a))[:3]
    dominant = ", ".join(AU_REGIONS[i] for i in order)
    strongest = ", ".join(
        f"{AU_REGIONS[i]}={a[i]:+.2f}" for i in np.argsort(-np.abs(a))[:4]
    )
    return {
        "coarse": (
            "Describe this face in one sentence: global appearance and how much of it "
            f"is occluded (occlusion ratio {occlusion:.2f})."
        ),
        "medium": (
            "Given that description, name the dominant facial action regions and whether "
            f"the two sides are consistent (dominant: {dominant}; bilateral consistency {consistency:.2f})."
        ),
        "fine": (
            "Refine into local detail: per-region intensities and texture, not a restatement "
            f"({strongest})."
        ),
    }


def template_descriptions(a: np.ndarray, occlusion: float, consistency: float) -> dict[str, str]:
    """Offline stand-in for the LLM answer, phrased like the paper's levels."""
    order = np.argsort(-np.abs(a))
    top = order[:3]
    mild = int(np.sum(np.abs(a) < 0.15))
    return {
        "coarse": (
            f"A face filling the frame with {occlusion:.0%} of the AU regions occluded; "
            f"{mild} of 8 regions stay near neutral."
        ),
        "medium": (
            "Co-activation centres on "
            + ", ".join(AU_REGIONS[i] for i in top)
            + f"; the two sides agree at {consistency:.2f}."
        ),
        "fine": (
            "Local texture is strongest at "
            + ", ".join(f"{AU_REGIONS[i]} ({a[i]:+.2f})" for i in order[:4])
            + "."
        ),
    }


# ------------------------------------------------------------------ providers
def openai_provider(base_url: str, model: str, api_key_env: str) -> Callable:
    def describe(prompts: dict[str, str]) -> dict[str, str]:
        import json as _json
        import urllib.request

        key = os.environ.get(api_key_env, "")
        if not key:
            raise SystemExit(f"set {api_key_env} to use --provider openai")
        out = {}
        for level in LEVELS:
            body = _json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content": "You describe facial expressions concisely."},
                    {"role": "user", "content": prompts[level]},
                ],
                "temperature": 0.2,
                "max_tokens": 120,
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url.rstrip('/')}/chat/completions",
                data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
            out[level] = data["choices"][0]["message"]["content"].strip()
        return out

    return describe


# ------------------------------------------------------------------ embedders
class HashEmbedder:
    """Deterministic character n-gram hashing; no dependencies, no network."""

    def __init__(self, dim: int = 4096, ngram: int = 3) -> None:
        self.dim = dim
        self.ngram = ngram

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        clean = re.sub(r"\s+", " ", text.lower()).strip()
        for i in range(max(len(clean) - self.ngram + 1, 1)):
            gram = clean[i:i + self.ngram]
            h = hash(gram) if False else _stable_hash(gram)
            vec[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


def _stable_hash(text: str) -> int:
    """FNV-1a, so embeddings do not change between runs (``hash()`` is salted)."""
    h = 0x811C9DC5
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def st_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Callable:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def encode(text: str) -> np.ndarray:
        return model.encode([text], normalize_embeddings=True)[0].astype(np.float32)

    return encode


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder of face images")
    ap.add_argument("--out", required=True, help="cache .npz")
    ap.add_argument("--provider", choices=["template", "openai"], default="template")
    ap.add_argument("--base-url", default="https://api.deepseek.com/v1")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--embedder", choices=["hash", "st"], default="hash")
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    paths = sorted(
        os.path.join(root, f)
        for root, _, files in os.walk(args.data)
        for f in files if f.lower().endswith(exts)
    )
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no images under {args.data}")

    if args.provider == "template":
        def describe(a, occ, cons, prompts):  # noqa: ARG001 - uniform signature
            return template_descriptions(a, occ, cons)
    else:
        _llm = openai_provider(args.base_url, args.model, args.api_key_env)

        def describe(a, occ, cons, prompts):  # noqa: ARG001
            return _llm(prompts)

    encode = HashEmbedder(args.dim).encode if args.embedder == "hash" else st_embedder()

    embeddings, texts, used = [], [], []
    for i, path in enumerate(paths, 1):
        a, occ, cons = au_features(path)
        prompts = build_prompts(a, occ, cons)
        described = describe(a, occ, cons, prompts)
        embeddings.append(np.stack([encode(described[lvl]) for lvl in LEVELS]))
        texts.append(described)
        used.append(path)
        if i % 50 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)}")

    emb = np.asarray(embeddings, dtype=np.float32)
    np.savez_compressed(args.out, paths=np.asarray(used), emb=emb)
    with open(os.path.splitext(args.out)[0] + ".jsonl", "w", encoding="utf-8") as fh:
        for path, text in zip(used, texts):
            fh.write(json.dumps({"path": path, **text}, ensure_ascii=False) + "\n")

    print(f"saved {args.out}: {emb.shape[0]} images x {emb.shape[1]} levels x {emb.shape[2]} dims")
    print(f"transcript: {os.path.splitext(args.out)[0]}.jsonl")
    print("sample:", texts[0]["coarse"][:90])


if __name__ == "__main__":
    main()
