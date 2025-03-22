"""Embedding cache management for RISCORE."""

from typing import List, Optional, Dict, Any
from pathlib import Path
import numpy as np
import pickle
import hashlib
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from ..utils.validation import EmbeddingConfig


class EmbeddingCache:
    """Manage embedding cache to avoid recomputation."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize embedding cache.
        
        Args:
            cache_dir: Directory to store cached embeddings
        """
        if cache_dir is None:
            cache_dir = Path("cache/embeddings")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._memory_cache: Dict[str, np.ndarray] = {}
    
    def _get_cache_key(self, texts: List[str], model_name: str) -> str:
        """Generate cache key from texts and model name."""
        # Create deterministic hash
        text_hash = hashlib.md5("".join(texts).encode()).hexdigest()
        model_hash = hashlib.md5(model_name.encode()).hexdigest()
        return f"{model_hash}_{text_hash}"
    
    def get(
        self,
        texts: List[str],
        model_name: str,
    ) -> Optional[np.ndarray]:
        """
        Get cached embeddings.
        
        Args:
            texts: List of texts
            model_name: Embedding model name
            
        Returns:
            Cached embeddings or None if not found
        """
        cache_key = self._get_cache_key(texts, model_name)
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embeddings = pickle.load(f)
                
                # Store in memory cache
                self._memory_cache[cache_key] = embeddings
                return embeddings
            except Exception as e:
                print(f"Failed to load cache {cache_file}: {e}")
                return None
        
        return None
    
    def set(
        self,
        texts: List[str],
        model_name: str,
        embeddings: np.ndarray,
    ):
        """
        Save embeddings to cache.
        
        Args:
            texts: List of texts
            model_name: Embedding model name
            embeddings: Computed embeddings
        """
        cache_key = self._get_cache_key(texts, model_name)
        
        # Save to memory cache
        self._memory_cache[cache_key] = embeddings
        
        # Save to disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
            
            # Also save metadata
            metadata_file = self.cache_dir / f"{cache_key}.json"
            metadata = {
                "model_name": model_name,
                "num_texts": len(texts),
                "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else embeddings.shape[0],
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
        except Exception as e:
            print(f"Failed to save cache {cache_file}: {e}")
    
    def clear(self):
        """Clear all caches."""
        self._memory_cache.clear()
        
        # Clear disk cache
        for file in self.cache_dir.glob("*.pkl"):
            file.unlink()
        for file in self.cache_dir.glob("*.json"):
            file.unlink()


class EmbeddingManager:
    """Manage embeddings with caching and similarity search."""
    
    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        cache: Optional[EmbeddingCache] = None,
    ):
        """
        Initialize embedding manager.
        
        Args:
            config: Embedding configuration
            cache: Embedding cache (creates new if None)
        """
        self.config = config or EmbeddingConfig()
        self.cache = cache or EmbeddingCache(self.config.cache_dir)
        self.model: Optional[SentenceTransformer] = None
    
    def _load_model(self):
        """Load embedding model."""
        if self.model is None:
            self.model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
            )
    
    def encode(
        self,
        texts: List[str],
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            use_cache: Whether to use cache
            
        Returns:
            Embeddings array of shape (len(texts), embedding_dim)
        """
        # Try cache first
        if use_cache:
            cached = self.cache.get(texts, self.config.model_name)
            if cached is not None:
                return cached
        
        # Load model if needed
        self._load_model()
        
        # Encode texts
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
        )
        
        # Cache if requested
        if use_cache:
            self.cache.set(texts, self.config.model_name, embeddings)
        
        return embeddings
    
    def compute_similarity(
        self,
        query_embeddings: np.ndarray,
        candidate_embeddings: np.ndarray,
        metric: str = "cosine",
    ) -> np.ndarray:
        """
        Compute similarity between query and candidates.
        
        Args:
            query_embeddings: Query embeddings (n_queries, dim)
            candidate_embeddings: Candidate embeddings (n_candidates, dim)
            metric: Similarity metric ('cosine', 'euclidean', 'dot')
            
        Returns:
            Similarity matrix (n_queries, n_candidates)
        """
        if metric == "cosine":
            return cosine_similarity(query_embeddings, candidate_embeddings)
        elif metric == "euclidean":
            # Convert distance to similarity (inverse)
            distances = euclidean_distances(query_embeddings, candidate_embeddings)
            return 1.0 / (1.0 + distances)
        elif metric == "dot":
            return np.dot(query_embeddings, candidate_embeddings.T)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def find_similar(
        self,
        query_text: str,
        candidate_texts: List[str],
        top_k: int = 5,
        threshold: float = 0.4,
        metric: str = "cosine",
        filter_duplicates: bool = True,
    ) -> List[tuple[int, float]]:
        """
        Find most similar texts to query.
        
        Args:
            query_text: Query text
            candidate_texts: List of candidate texts
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            metric: Similarity metric
            filter_duplicates: Filter out duplicate/identical texts
            
        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        # Encode query and candidates separately for better caching
        query_embedding = self.encode([query_text])
        candidate_embeddings = self.encode(candidate_texts)
        
        # Compute similarities
        similarities = self.compute_similarity(
            query_embedding,
            candidate_embeddings,
            metric=metric
        )[0]  # Get first (and only) query result
        
        # Filter by threshold
        valid_indices = np.where(similarities >= threshold)[0]
        
        # Filter duplicates if requested
        if filter_duplicates:
            # Remove exact matches or very high similarity (likely duplicates)
            duplicate_threshold = 0.99
            valid_indices = valid_indices[similarities[valid_indices] < duplicate_threshold]
        
        # Sort by similarity (descending)
        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
        
        # Take top-k
        top_indices = sorted_indices[:top_k]
        
        # Return as list of (index, score) tuples
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def batch_find_similar(
        self,
        query_texts: List[str],
        candidate_texts: List[str],
        top_k: int = 5,
        threshold: float = 0.4,
        metric: str = "cosine",
        filter_duplicates: bool = True,
    ) -> List[List[tuple[int, float]]]:
        """
        Find similar texts for multiple queries.
        
        Args:
            query_texts: List of query texts
            candidate_texts: List of candidate texts
            top_k: Number of results per query
            threshold: Minimum similarity threshold
            metric: Similarity metric
            filter_duplicates: Filter duplicates
            
        Returns:
            List of results, one per query
        """
        # Encode all at once for efficiency
        query_embeddings = self.encode(query_texts)
        candidate_embeddings = self.encode(candidate_texts)
        
        # Compute all similarities
        similarities = self.compute_similarity(
            query_embeddings,
            candidate_embeddings,
            metric=metric
        )
        
        # Process each query
        results = []
        for i, query_sims in enumerate(similarities):
            # Filter by threshold
            valid_indices = np.where(query_sims >= threshold)[0]
            
            # Filter duplicates
            if filter_duplicates:
                duplicate_threshold = 0.99
                valid_indices = valid_indices[query_sims[valid_indices] < duplicate_threshold]
            
            # Sort and take top-k
            sorted_indices = valid_indices[np.argsort(query_sims[valid_indices])[::-1]]
            top_indices = sorted_indices[:top_k]
            
            query_results = [(int(idx), float(query_sims[idx])) for idx in top_indices]
            results.append(query_results)
        
        return results
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        filepath: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Save embeddings to file.
        
        Args:
            embeddings: Embeddings array
            filepath: Path to save to
            metadata: Optional metadata to save
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        np.save(filepath, embeddings)
        
        # Save metadata if provided
        if metadata:
            metadata_path = filepath.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def load_embeddings(
        self,
        filepath: Path,
    ) -> tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Load embeddings from file.
        
        Args:
            filepath: Path to load from
            
        Returns:
            Tuple of (embeddings, metadata)
        """
        filepath = Path(filepath)
        
        # Load embeddings
        embeddings = np.load(filepath)
        
        # Load metadata if exists
        metadata_path = filepath.with_suffix('.json')
        metadata = None
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return embeddings, metadata
