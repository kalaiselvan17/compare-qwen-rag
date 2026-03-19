from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List
from sentence_transformers import SentenceTransformer  # type: ignore
import numpy as np

logger = logging.getLogger(__name__)


class _BM25:

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.corpus: List[List[str]] = []
        self.idf: Dict[str, float]   = {}
        self.avgdl: float            = 0.0

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, documents: List[str]):
        self.corpus = [self._tokenize(d) for d in documents]
        n = len(self.corpus)
        df: Dict[str, int] = defaultdict(int)
        total_len = 0
        for doc in self.corpus:
            total_len += len(doc)
            for term in set(doc):
                df[term] += 1
        self.avgdl = total_len / max(n, 1)
        self.idf = {
            term: math.log((n - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

    def score(self, query: str, doc_idx: int) -> float:
        q_terms = self._tokenize(query)
        doc = self.corpus[doc_idx]
        dl  = len(doc)
        tf_map = Counter(doc)
        score = 0.0
        for term in q_terms:
            if term not in self.idf:
                continue
            tf = tf_map.get(term, 0)
            numerator   = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
            score += self.idf[term] * numerator / max(denominator, 1e-9)
        return score

    def search(self, query: str, top_k: int = 5) -> List[tuple]:
        scores = [(i, self.score(query, i)) for i in range(len(self.corpus))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class _DenseRetriever:

    def __init__(self):
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self._use_dense = True
            logger.info("Dense retriever: all-MiniLM-L6-v2")
        except ImportError:
            self._use_dense = False

        self._texts: List[str] = []
        self._vecs: np.ndarray = np.empty((0,))

    def fit(self, texts: List[str]):
        self._texts = texts
        if self._use_dense:
            self._vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=32)

    def search(self, query: str, top_k: int = 5) -> List[tuple]:
        if not self._use_dense or len(self._texts) == 0:
            return []
        q_vec = self.model.encode([query], normalize_embeddings=True)[0]
        sims  = self._vecs @ q_vec
        idxs  = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in idxs]


class TraditionalRAGRetriever:
    
    def __init__(self, prefer_dense: bool = True):
        self.bm25   = _BM25()
        self.dense  = _DenseRetriever() if prefer_dense else None
        self.chunks: List[Dict[str, Any]] = []

    def build_index(self, chunks: List[Dict[str, Any]]):
        # Only index text chunks (traditional RAG ignores image-only chunks)
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self.bm25.fit(texts)
        if self.dense:
            self.dense.fit(texts)
        logger.info(f"Traditional RAG index built for {len(chunks)} chunks.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []

        # Try dense first
        if self.dense and self.dense._use_dense and len(self.dense._texts) > 0:
            hits = self.dense.search(query, top_k=top_k)
        else:
            hits = self.bm25.search(query, top_k=top_k)
            # Normalise BM25 scores to [0,1]
            max_score = max(s for _, s in hits) if hits else 1.0
            hits = [(i, s / max(max_score, 1e-9)) for i, s in hits]

        results = []
        for idx, score in hits:
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results

