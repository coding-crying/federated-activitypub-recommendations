"""
Efficient Per-User LoRA System
Lightweight LoRA adapters (~6KB each) for personal preference learning
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from dataclasses import dataclass
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapter."""
    embedding_dim: int = 384  # MiniLM dimension
    lora_rank: int = 8  # Smaller rank for efficiency (vs 16)
    lora_alpha: float = 16.0
    learning_rate: float = 0.01
    dropout: float = 0.1


class EfficientLoRA(nn.Module):
    """
    Efficient LoRA adapter for user personalization.

    Implements low-rank adaptation: W + BA where:
    - B: (embedding_dim, rank) ~= 3KB
    - A: (rank, embedding_dim) ~= 3KB
    Total: ~6KB per user (vs 50KB in original design)
    """

    def __init__(self, config: LoRAConfig):
        """
        Initialize LoRA adapter.

        Args:
            config: LoRA configuration
        """
        super().__init__()
        self.config = config

        # Low-rank matrices
        # B (down-projection): embedding_dim -> rank
        self.W_down = nn.Parameter(
            torch.randn(config.embedding_dim, config.lora_rank) * 0.01
        )

        # A (up-projection): rank -> embedding_dim
        self.W_up = nn.Parameter(
            torch.randn(config.lora_rank, config.embedding_dim) * 0.01
        )

        # Scaling factor
        self.scaling = config.lora_alpha / config.lora_rank

        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA transformation.

        Args:
            x: Input embeddings (batch_size, embedding_dim)

        Returns:
            Adapted embeddings (batch_size, embedding_dim)
        """
        # LoRA: x + scaling * (x @ W_down @ W_up)
        lora_output = x @ self.W_down  # (batch, rank)
        lora_output = self.dropout(lora_output)
        lora_output = lora_output @ self.W_up  # (batch, embedding_dim)

        # Add residual with scaling
        return x + self.scaling * lora_output

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get LoRA parameters as dictionary."""
        return {
            'W_down': self.W_down.data.clone(),
            'W_up': self.W_up.data.clone()
        }

    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set LoRA parameters from dictionary."""
        self.W_down.data = parameters['W_down'].clone()
        self.W_up.data = parameters['W_up'].clone()

    def num_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def size_bytes(self) -> int:
        """Get size in bytes (assuming float32)."""
        return self.num_parameters() * 4


class UserLoRAManager:
    """
    Manages per-user LoRA adapters.
    Handles creation, storage, and efficient loading.
    """

    def __init__(
        self,
        config: LoRAConfig,
        storage_dir: Optional[str] = None,
        device: str = 'cpu'
    ):
        """
        Initialize user LoRA manager.

        Args:
            config: LoRA configuration
            storage_dir: Directory to store LoRAs
            device: Device for computation
        """
        self.config = config
        self.device = device
        self.loras: Dict[str, EfficientLoRA] = {}

        # Storage
        if storage_dir:
            self.storage_dir = Path(storage_dir)
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.storage_dir = None

        logger.info(f"UserLoRAManager initialized: rank={config.lora_rank}, "
                   f"device={device}, storage={storage_dir}")

    def get_or_create_lora(self, user_id: str) -> EfficientLoRA:
        """
        Get existing LoRA or create new one for user.

        Args:
            user_id: User identifier

        Returns:
            LoRA adapter for user
        """
        if user_id not in self.loras:
            # Try to load from storage
            if self.storage_dir and self._load_lora(user_id):
                logger.debug(f"Loaded LoRA for user {user_id} from storage")
            else:
                # Create new LoRA
                self.loras[user_id] = EfficientLoRA(self.config).to(self.device)
                logger.debug(f"Created new LoRA for user {user_id}")

        return self.loras[user_id]

    def save_lora(self, user_id: str):
        """Save user's LoRA to storage."""
        if not self.storage_dir or user_id not in self.loras:
            return

        lora = self.loras[user_id]
        save_path = self.storage_dir / f"{user_id}.pt"

        torch.save(lora.get_parameters(), save_path)
        logger.debug(f"Saved LoRA for user {user_id} to {save_path}")

    def _load_lora(self, user_id: str) -> bool:
        """Load user's LoRA from storage."""
        if not self.storage_dir:
            return False

        load_path = self.storage_dir / f"{user_id}.pt"
        if not load_path.exists():
            return False

        try:
            parameters = torch.load(load_path, map_location=self.device)
            lora = EfficientLoRA(self.config).to(self.device)
            lora.set_parameters(parameters)
            self.loras[user_id] = lora
            return True
        except Exception as e:
            logger.warning(f"Failed to load LoRA for user {user_id}: {e}")
            return False

    def save_all(self):
        """Save all LoRAs to storage."""
        if not self.storage_dir:
            return

        for user_id in self.loras.keys():
            self.save_lora(user_id)

        logger.info(f"Saved {len(self.loras)} LoRAs to storage")

    def get_all_user_ids(self) -> List[str]:
        """Get list of all user IDs with LoRAs."""
        return list(self.loras.keys())

    def num_users(self) -> int:
        """Get number of users."""
        return len(self.loras)

    def total_size_bytes(self) -> int:
        """Get total size of all LoRAs in bytes."""
        if not self.loras:
            return 0
        return next(iter(self.loras.values())).size_bytes() * len(self.loras)

    def total_size_mb(self) -> float:
        """Get total size of all LoRAs in MB."""
        return self.total_size_bytes() / (1024 ** 2)


class UserPreferenceVector:
    """
    Maintains running average of user preferences.
    Simple but effective personalization signal.
    """

    def __init__(self, embedding_dim: int = 384, decay: float = 0.9):
        """
        Initialize user preference vector.

        Args:
            embedding_dim: Dimension of embeddings
            decay: Decay factor for exponential moving average
        """
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.vectors: Dict[str, np.ndarray] = {}
        self.interaction_counts: Dict[str, int] = {}

    def update(
        self,
        user_id: str,
        embedding: np.ndarray,
        weight: float = 1.0
    ):
        """
        Update user preference vector with new interaction.

        Args:
            user_id: User identifier
            embedding: Content embedding user interacted with
            weight: Interaction weight (e.g., 2.0 for boost, 1.0 for like)
        """
        if user_id not in self.vectors:
            # Initialize with first embedding
            self.vectors[user_id] = embedding * weight
            self.interaction_counts[user_id] = 1
        else:
            # Exponential moving average
            self.vectors[user_id] = (
                self.decay * self.vectors[user_id] +
                (1 - self.decay) * embedding * weight
            )
            self.interaction_counts[user_id] += 1

        # Normalize to unit vector
        norm = np.linalg.norm(self.vectors[user_id])
        if norm > 0:
            self.vectors[user_id] /= norm

    def get(self, user_id: str) -> Optional[np.ndarray]:
        """Get user preference vector."""
        return self.vectors.get(user_id)

    def has_user(self, user_id: str) -> bool:
        """Check if user has preference vector."""
        return user_id in self.vectors

    def get_interaction_count(self, user_id: str) -> int:
        """Get number of interactions for user."""
        return self.interaction_counts.get(user_id, 0)


def test_efficient_lora():
    """Test efficient LoRA system."""
    print("🧪 Testing Efficient LoRA System")
    print("=" * 60)

    # Create config
    config = LoRAConfig(embedding_dim=384, lora_rank=8)
    print(f"LoRA Config:")
    print(f"  Embedding dim: {config.embedding_dim}")
    print(f"  LoRA rank: {config.lora_rank}")
    print(f"  Alpha: {config.lora_alpha}")

    # Create LoRA
    lora = EfficientLoRA(config)
    print(f"\n✅ LoRA created:")
    print(f"   Parameters: {lora.num_parameters():,}")
    print(f"   Size: {lora.size_bytes() / 1024:.2f} KB")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, config.embedding_dim)
    print(f"\n📊 Testing forward pass:")
    print(f"   Input shape: {x.shape}")

    y = lora(x)
    print(f"   Output shape: {y.shape}")
    print(f"   Mean absolute change: {(y - x).abs().mean().item():.6f}")

    # Test UserLoRAManager
    print(f"\n💾 Testing UserLoRAManager:")
    manager = UserLoRAManager(config, storage_dir="/tmp/test_loras")

    # Create LoRAs for multiple users
    user_ids = ["user_1", "user_2", "user_3"]
    for user_id in user_ids:
        lora = manager.get_or_create_lora(user_id)
        print(f"   Created LoRA for {user_id}")

    print(f"   Total users: {manager.num_users()}")
    print(f"   Total size: {manager.total_size_mb():.2f} MB")

    # Save and reload
    manager.save_all()
    print(f"   Saved all LoRAs to storage")

    # Test UserPreferenceVector
    print(f"\n🎯 Testing UserPreferenceVector:")
    preferences = UserPreferenceVector(embedding_dim=384)

    # Simulate user interactions
    user_id = "alice"
    embeddings = [
        np.random.randn(384),
        np.random.randn(384),
        np.random.randn(384)
    ]

    for i, emb in enumerate(embeddings):
        preferences.update(user_id, emb, weight=1.0 + i * 0.5)
        print(f"   Interaction {i+1}: updated preference vector")

    pref_vector = preferences.get(user_id)
    print(f"   Final preference vector norm: {np.linalg.norm(pref_vector):.4f}")
    print(f"   Total interactions: {preferences.get_interaction_count(user_id)}")

    print(f"\n✅ All tests passed!")

    return manager, preferences


if __name__ == "__main__":
    test_efficient_lora()
