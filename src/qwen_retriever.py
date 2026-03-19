from __future__ import annotations
import torch # type: ignore
import io
import logging
import os
from typing import Any, Dict, List, Optional
import faiss # type: ignore
from PIL import Image # type: ignore

import numpy as np

logger = logging.getLogger(__name__)

MODEL_MAP = {
    "Qwen-VL-Chat (7B)":      "Qwen/Qwen-VL-Chat",
    "Qwen2-VL-2B-Instruct":   "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen2-VL-7B-Instruct":   "Qwen/Qwen2-VL-7B-Instruct",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_faiss():
    try:
        
        return faiss
    except ImportError:
        raise ImportError("Install faiss-cpu: pip install faiss-cpu")


def _load_pil_image(path_or_bytes) -> "PIL.Image.Image": # type: ignore
    
    if isinstance(path_or_bytes, (str, os.PathLike)):
        return Image.open(path_or_bytes).convert("RGB")
    return Image.open(io.BytesIO(path_or_bytes)).convert("RGB")


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return vec
    return vec / norm


class _FallbackEmbedder:

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.dim = 384
            logger.info("Fallback embedder loaded: all-MiniLM-L6-v2 (384-dim)")
        except ImportError:
            raise ImportError(
                "sentence-transformers not found. "
                "Install with: pip install sentence-transformers"
            )

    @property
    def embedding_dim(self) -> int:
        return self.dim

    def encode_text(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], normalize_embeddings=True)[0]
        return vec.astype(np.float32)

    def encode_image_text(self, text: str, image=None) -> np.ndarray:
        # Fallback has no vision — prefix text to distinguish image chunks
        prefix = "[IMAGE] " if image is not None else ""
        return self.encode_text(prefix + text)

class _QwenEmbedder:

    def __init__(self, model_name: str, device: str = "cpu"):
        # Lazy imports — only load when actually needed
        try:
            from transformers import AutoTokenizer, AutoModel    # type: ignore
        except ImportError as e:
            raise ImportError(f"Could not import torch/transformers: {e}")

        logger.info(f"Loading Qwen model: {model_name} on device={device}")

        # Resolve device safely
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available — falling back to CPU.")
            device = "cpu"

        self.device = torch.device(device)
        self.torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if "cuda" in str(self.device) else torch.float32,
        ).to(self.device).eval()

        self.dim = self.model.config.hidden_size
        logger.info(f"Qwen loaded — hidden_size={self.dim}, device={self.device}")

    @property
    def embedding_dim(self) -> int:
        return self.dim

    def _mean_pool(
        self,
        hidden_states: "torch.Tensor",
        attention_mask: "torch.Tensor",
    ) -> np.ndarray:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1e-9)
        vec = (summed / count).squeeze(0)
        return vec.detach().cpu().float().numpy()

    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with self.torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden = outputs.last_hidden_state           # (1, seq_len, dim)
        vec = self._mean_pool(hidden, inputs["attention_mask"])
        return _normalize(vec.astype(np.float32))

    def encode_image_text(self, text: str, image=None) -> np.ndarray:
        
        if image is not None and hasattr(self.model, "visual"):
            try:
                pil_img = _load_pil_image(image)

                if hasattr(self.tokenizer, "from_list_format"):
                    # Qwen-VL style tokenizer
                    query = self.tokenizer.from_list_format([
                        {"image": pil_img},
                        {"text": text},
                    ])
                    inputs = self.tokenizer(
                        query, return_tensors="pt", truncation=True
                    ).to(self.device)
                else:
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    ).to(self.device)

                with self.torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)

                hidden = outputs.last_hidden_state
                vec = self._mean_pool(hidden, inputs["attention_mask"])
                return _normalize(vec.astype(np.float32))

            except Exception as e:
                logger.warning(
                    f"Vision encoding failed ({e}) — falling back to text-only."
                )

        return self.encode_text(text)


class QwenRetriever:
    
    def __init__(
        self,
        model_variant: str = "Qwen2-VL-2B-Instruct",
        device: str = "cpu",
    ):
        hf_name = MODEL_MAP.get(model_variant, model_variant)
        self.using_fallback = False

        try:
            self.embedder = _QwenEmbedder(hf_name, device)
            logger.info(f"QwenRetriever: using Qwen embedder ({hf_name})")
        except Exception as e:
            logger.warning(
                f"Could not load Qwen ({e}). Switching to fallback embedder."
            )
            self.embedder = _FallbackEmbedder()
            self.using_fallback = True

        self.faiss = _get_faiss()
        self.index: Any = None
        self.chunks: List[Dict[str, Any]] = []


    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        
        if not chunks:
            logger.warning("build_index called with empty chunk list.")
            return

        dim = self.embedder.embedding_dim
        self.index = self.faiss.IndexFlatIP(dim)
        self.chunks = chunks

        logger.info(f"Building FAISS index: {len(chunks)} chunks, dim={dim}")

        vecs: List[np.ndarray] = []
        for i, chunk in enumerate(chunks):
            try:
                if chunk.get("has_image") and chunk.get("image_path"):
                    vec = self.embedder.encode_image_text(
                        chunk["text"], chunk["image_path"]
                    )
                else:
                    vec = self.embedder.encode_text(chunk["text"])
            except Exception as e:
                logger.warning(f"Encoding failed for chunk {i}: {e} — using zeros.")
                vec = np.zeros(dim, dtype=np.float32)

            # Ensure unit vector before insertion
            vec = _normalize(vec.astype(np.float32))
            vecs.append(vec)

        matrix = np.stack(vecs, axis=0).astype(np.float32)

        self.faiss.normalize_L2(matrix)
        self.index.add(matrix)

        logger.info(
            f"FAISS index built: {self.index.ntotal} vectors "
            f"({'fallback' if self.using_fallback else 'Qwen'})"
        )

    def retrieve(
        self,
        query_text: str = "",
        query_image: Optional[bytes] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        
        if self.index is None or self.index.ntotal == 0:
            logger.warning("retrieve() called before build_index().")
            return []

        # Encode query
        try:
            if query_image is not None:
                q_vec = self.embedder.encode_image_text(
                    query_text or "", query_image
                )
            else:
                q_vec = self.embedder.encode_text(
                    query_text if query_text.strip() else "document"
                )
        except Exception as e:
            logger.error(f"Query encoding failed: {e}")
            return []

        # Normalize query vector
        q_vec = _normalize(q_vec.astype(np.float32)).reshape(1, -1)
        self.faiss.normalize_L2(q_vec)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_vec, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        # Sort descending by score (FAISS already does this, but be explicit)
        results.sort(key=lambda x: x["score"], reverse=True)

        logger.info(
            f"Retrieved {len(results)} results. "
            f"Top score: {results[0]['score']:.4f}" if results else "No results."
        )
        return results


    def info(self) -> Dict[str, Any]:

        return {
            "embedder": "Fallback (all-MiniLM-L6-v2)" if self.using_fallback else "Qwen-VL",
            "embedding_dim": self.embedder.embedding_dim,
            "indexed_chunks": self.index.ntotal if self.index else 0,
            "index_type": "IndexFlatIP (cosine)",
        }