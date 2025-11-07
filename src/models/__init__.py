"""Models module for federated recommendations."""

from .lightweight_foundation import LightweightFoundationModel, EmbeddingCache
from .efficient_lora import EfficientLoRA, LoRAConfig, UserLoRAManager, UserPreferenceVector
from .decentralized_recommender import DecentralizedRecommender, Post, Interaction

__all__ = [
    'LightweightFoundationModel',
    'EmbeddingCache',
    'EfficientLoRA',
    'LoRAConfig',
    'UserLoRAManager',
    'UserPreferenceVector',
    'DecentralizedRecommender',
    'Post',
    'Interaction'
]
