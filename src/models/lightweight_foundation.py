"""
Lightweight Foundation Model Manager for Decentralized Recommendations
Uses all-MiniLM-L6-v2 (80MB) for CPU-compatible, efficient content embeddings
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LightweightFoundationModel:
    """
    Lightweight foundation model for content understanding.

    Uses all-MiniLM-L6-v2:
    - 80MB model size (vs 1.3GB for e5-large-v2)
    - 384d embeddings (vs 1024d)
    - CPU-compatible
    - Still captures semantic relationships well
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize lightweight foundation model.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cpu', 'cuda', or None for auto)
            cache_dir: Directory to cache model
        """
        self.model_name = model_name

        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # Load model
        logger.info(f"Loading foundation model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir)

        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"Foundation model loaded: {self.embedding_dim}d embeddings")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress
            normalize: Whether to L2 normalize embeddings

        Returns:
            Embeddings as numpy array (N, embedding_dim)
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        # Encode
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )

        # Return single embedding if single input
        if single_input:
            return embeddings[0]

        return embeddings

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get model parameters for federated learning.

        Returns:
            Dictionary of parameter tensors
        """
        return {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }

    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """
        Set model parameters from federated update.

        Args:
            parameters: Dictionary of parameter tensors
        """
        state_dict = self.model.state_dict()

        for name, param in parameters.items():
            if name in state_dict:
                state_dict[name] = param.to(self.device)

        self.model.load_state_dict(state_dict)
        logger.debug(f"Updated {len(parameters)} parameters")

    def save(self, save_path: str):
        """Save model to disk."""
        Path(save_path).mkdir(parents=True, exist_ok=True)
        self.model.save(save_path)
        logger.info(f"Foundation model saved to {save_path}")

    def load(self, load_path: str):
        """Load model from disk."""
        self.model = SentenceTransformer(load_path, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Foundation model loaded from {load_path}")


class EmbeddingCache:
    """
    Cache for post embeddings to avoid recomputation.
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of embeddings to cache
        """
        self.cache: Dict[str, np.ndarray] = {}
        self.max_size = max_size
        self.access_count: Dict[str, int] = {}

    def get(self, key: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None

    def put(self, key: str, embedding: np.ndarray):
        """Add embedding to cache."""
        # Evict least accessed item if cache is full
        if len(self.cache) >= self.max_size:
            least_accessed = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[least_accessed]
            del self.access_count[least_accessed]

        self.cache[key] = embedding
        self.access_count[key] = 1

    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_count.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


def test_lightweight_foundation():
    """Test lightweight foundation model."""
    print("🧪 Testing Lightweight Foundation Model")
    print("=" * 60)

    # Initialize model
    model = LightweightFoundationModel()

    print(f"✅ Model loaded: {model.model_name}")
    print(f"   Embedding dimension: {model.embedding_dim}")
    print(f"   Device: {model.device}")

    # Test encoding
    texts = [
        "I love action movies with great special effects",
        "Romantic comedies are my favorite genre",
        "Documentary films about nature are fascinating"
    ]

    print(f"\n📝 Encoding {len(texts)} texts...")
    embeddings = model.encode(texts)

    print(f"✅ Embeddings shape: {embeddings.shape}")
    print(f"   First embedding norm: {np.linalg.norm(embeddings[0]):.4f}")

    # Test similarity
    print(f"\n🔍 Computing similarities...")
    from sklearn.metrics.pairwise import cosine_similarity

    similarities = cosine_similarity(embeddings)
    print("   Similarity matrix:")
    for i, text in enumerate(texts):
        print(f"   {i}: {text[:40]}...")
        print(f"      Similarities: {similarities[i]}")

    # Test caching
    print(f"\n💾 Testing embedding cache...")
    cache = EmbeddingCache(max_size=2)

    cache.put("post_1", embeddings[0])
    cache.put("post_2", embeddings[1])
    print(f"   Cache size: {cache.size()}")

    cached = cache.get("post_1")
    print(f"   Retrieved embedding: {cached is not None}")
    print(f"   Matches original: {np.allclose(cached, embeddings[0])}")

    # Test eviction
    cache.put("post_3", embeddings[2])
    print(f"   After adding 3rd item (max=2), cache size: {cache.size()}")

    print(f"\n✅ All tests passed!")

    return model, embeddings


if __name__ == "__main__":
    test_lightweight_foundation()
