import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import torch
# faiss.set_num_threads(os.cpu_count())
# -----------------------------------------------------------------------------
# PaperReranker: builds FAISS index (IndexFlatIP) using normalized embeddings,
# returns distances (inner product scores) and supports cross-encoder reranking.
# -----------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
class PaperReranker:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        base_path: str = "vector_stores",
        name: str = "default",
    ):
        print("🔍 Loading embedding and reranker models...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # Check for GPU
        self.embedder = SentenceTransformer(embedding_model_name, device=self.device)
        self.reranker = CrossEncoder(reranker_model_name, device=self.device)

        # storage path
        self.base_path = Path(base_path)
        self.name = name
        self.store_dir = self.base_path / self.name
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.index = None
        self.df = None
        self.embeddings = None
        self.dim = None
        self._index_type = None  # "IP" expected

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        # avoid division by zero
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return arr / norms
    

    def build_vector_store(self, chunk_df: pd.DataFrame, batch_size: int = 64):
        """
        chunk_df must have a 'content' column.
        This function computes normalized embeddings and builds a FAISS IndexFlatIP index.
        """
        self.df = chunk_df.reset_index(drop=True).copy()
        texts = self.df["content"].astype(str).tolist()
        print(f"⚙️ Encoding {len(texts)} chunks in batches of {batch_size}...")
        all_embs = []

        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            batch_texts = texts[start:end]

            # Encode the current batch
            batch_emb = self.embedder.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=batch_size,
            )
            all_embs.append(batch_emb)

            print(f"Encoded batch {start // batch_size + 1} "
                f"({end if end < len(texts) else len(texts)} / {len(texts)} texts)")
            
        # normalize embeddings for cosine similarity via inner product
        emb = self._normalize(np.vstack(all_embs))
        self.embeddings = emb.astype("float32")
        self.dim = self.embeddings.shape[1]

        # use Inner Product index for normalized embeddings (cosine similarity)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.embeddings)
        self._index_type = "IP"
        print(f"✅ Vector store ready: {len(self.embeddings)} vectors (dim={self.dim})")
        self.save_vector_store()

    def save_vector_store(self):
        print("💾 Saving vector store to disk...")
        faiss.write_index(self.index, str(self.store_dir / "index.faiss"))
        with open(self.store_dir/"data.pkl", "wb") as f:
            pickle.dump(self.df, f)
        print("✅ Saved FAISS index and metadata.")

    def load_vector_store(self) -> bool:
        index_path = self.store_dir / "index.faiss"
        meta_path = self.store_dir / "data.pkl"

        if index_path.exists() and meta_path.exists():
            print(f"📂 Loading existing vector store from {self.store_dir}")
            self.index = faiss.read_index(str(index_path))
            with open(meta_path, "rb") as f:
                self.df = pickle.load(f)
            self.dim = self.index.d
            print("✅ Vector store loaded successfully.")
            return True
        else:
            print(f"⚠️ No existing vector store found in {self.store_dir}")
            return False


    def retrieve_candidates(self, query: str, top_k: int ) -> pd.DataFrame:
        """
        Returns a DataFrame of the top_k candidates with an added 'score' column (inner product).
        Higher score => more similar (because we use normalized embeddings + IP).
        """
        print(f"Retrieving top {top_k} candidates for query: {query}")
        if self.index is None:
            raise ValueError("Index not built. Call build_vector_store(...) first.")

        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        q_emb = self._normalize(q_emb).astype("float32")
        distances, indices = self.index.search(q_emb, top_k)  # distances are inner products
        # distances shape (1, top_k), indices shape (1, top_k)
        idx_list = indices[0].tolist()
        score_list = distances[0].tolist()

        candidates = self.df.iloc[idx_list].copy().reset_index(drop=True)
        candidates["score"] = score_list  # higher is better
        # sort by score desc
        candidates = candidates.sort_values(by="score", ascending=False).reset_index(drop=True)
        return candidates

    def rerank(self, query: str, candidates: pd.DataFrame, top_k: int) -> pd.DataFrame:
        """
        Rerank candidate rows using a CrossEncoder. Returns top_k with added 'rerank_score'.
        """
        print(f"Reranking top {top_k} candidates...")
        if candidates is None or len(candidates) == 0:
            return candidates

        # Prepare (query, doc) pairs for cross-encoder
        pairs = [(query, str(text)) for text in candidates["content"].tolist()]
        rerank_scores = self.reranker.predict(pairs)  # higher = more relevant

        reranked = candidates.copy()
        reranked["rerank_score"] = rerank_scores
        reranked = reranked.sort_values(by="rerank_score", ascending=False).reset_index(drop=True)
        return reranked.head(top_k)
