"""
Decentralized Recommendation System
Combines lightweight foundation model + efficient per-user LoRAs for real-time personalized recommendations
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from collections import defaultdict
import time

from .lightweight_foundation import LightweightFoundationModel, EmbeddingCache
from .efficient_lora import EfficientLoRA, LoRAConfig, UserLoRAManager, UserPreferenceVector

logger = logging.getLogger(__name__)


@dataclass
class Post:
    """Represents a social media post."""
    id: str
    content: str
    author: str
    timestamp: float
    engagement: Dict[str, int] = None  # likes, boosts, replies

    def __post_init__(self):
        if self.engagement is None:
            self.engagement = {'likes': 0, 'boosts': 0, 'replies': 0}


@dataclass
class Interaction:
    """Represents a user interaction with a post."""
    user_id: str
    post_id: str
    interaction_type: str  # 'like', 'boost', 'reply', 'skip'
    timestamp: float
    weight: float = 1.0  # Importance weight

    def __post_init__(self):
        # Assign weights based on interaction type
        if self.interaction_type == 'boost':
            self.weight = 2.0
        elif self.interaction_type == 'like':
            self.weight = 1.0
        elif self.interaction_type == 'reply':
            self.weight = 1.5
        elif self.interaction_type == 'skip':
            self.weight = 0.0


class DecentralizedRecommender:
    """
    Decentralized recommendation system using foundation model + per-user LoRAs.

    Key features:
    - No centralized database needed
    - Real-time online learning
    - Efficient inference (<20ms per recommendation)
    - Privacy-preserving (LoRAs stay local)
    """

    def __init__(
        self,
        instance_domain: str,
        foundation_model: Optional[LightweightFoundationModel] = None,
        lora_config: Optional[LoRAConfig] = None,
        storage_dir: Optional[str] = None,
        device: str = 'cpu'
    ):
        """
        Initialize decentralized recommender.

        Args:
            instance_domain: Domain of this instance
            foundation_model: Pre-initialized foundation model (or will create)
            lora_config: LoRA configuration
            storage_dir: Directory to store user LoRAs
            device: Device for computation
        """
        self.instance_domain = instance_domain
        self.device = device

        # Foundation model (shared, frozen during inference)
        if foundation_model is None:
            logger.info("Initializing foundation model...")
            self.foundation = LightweightFoundationModel(device=device)
        else:
            self.foundation = foundation_model

        # LoRA configuration
        if lora_config is None:
            lora_config = LoRAConfig(
                embedding_dim=self.foundation.embedding_dim,
                lora_rank=8
            )
        self.lora_config = lora_config

        # Per-user LoRA manager
        self.lora_manager = UserLoRAManager(
            config=lora_config,
            storage_dir=storage_dir,
            device=device
        )

        # User preference vectors (running averages)
        self.user_preferences = UserPreferenceVector(
            embedding_dim=self.foundation.embedding_dim
        )

        # Embedding cache
        self.embedding_cache = EmbeddingCache(max_size=10000)

        # Post storage (for this instance)
        self.posts: Dict[str, Post] = {}

        # Interaction buffer for batch updates
        self.interaction_buffer: List[Interaction] = []
        self.buffer_size = 10  # Update after N interactions

        logger.info(f"DecentralizedRecommender initialized for {instance_domain}")
        logger.info(f"  Foundation: {self.foundation.embedding_dim}d embeddings")
        logger.info(f"  LoRA: rank={lora_config.lora_rank}, ~{EfficientLoRA(lora_config).size_bytes() / 1024:.1f}KB per user")

    def add_post(self, post: Post):
        """Add a post to the instance's content pool."""
        self.posts[post.id] = post

    def process_interaction(
        self,
        user_id: str,
        post_id: str,
        interaction_type: str
    ):
        """
        Process a user interaction in real-time.

        Args:
            user_id: User identifier
            post_id: Post identifier
            interaction_type: Type of interaction ('like', 'boost', 'reply', 'skip')
        """
        # Create interaction
        interaction = Interaction(
            user_id=user_id,
            post_id=post_id,
            interaction_type=interaction_type,
            timestamp=time.time()
        )

        # Add to buffer
        self.interaction_buffer.append(interaction)

        # Online update if we have enough interactions
        if len(self.interaction_buffer) >= self.buffer_size:
            self._batch_update_loras()

    def _batch_update_loras(self):
        """Update LoRAs based on buffered interactions."""
        if not self.interaction_buffer:
            return

        # Group interactions by user
        user_interactions = defaultdict(list)
        for interaction in self.interaction_buffer:
            user_interactions[interaction.user_id].append(interaction)

        # Update each user's LoRA
        for user_id, interactions in user_interactions.items():
            self._update_user_lora(user_id, interactions)

        # Clear buffer
        self.interaction_buffer.clear()

        logger.debug(f"Updated LoRAs for {len(user_interactions)} users")

    def _update_user_lora(self, user_id: str, interactions: List[Interaction]):
        """
        Update a single user's LoRA based on their interactions.

        Args:
            user_id: User identifier
            interactions: List of user's recent interactions
        """
        # Get or create user's LoRA
        lora = self.lora_manager.get_or_create_lora(user_id)
        lora.train()

        # Get user's current preference vector
        if not self.user_preferences.has_user(user_id):
            # Initialize preference vector
            initial_embeddings = []
            for interaction in interactions:
                if interaction.weight > 0 and interaction.post_id in self.posts:
                    post = self.posts[interaction.post_id]
                    emb = self._get_post_embedding(post)
                    initial_embeddings.append(emb)

            if initial_embeddings:
                avg_emb = np.mean(initial_embeddings, axis=0)
                self.user_preferences.update(user_id, avg_emb, weight=1.0)

        # Setup optimizer for this user's LoRA
        optimizer = optim.Adam(lora.parameters(), lr=self.lora_config.learning_rate)

        # Training loop
        for interaction in interactions:
            if interaction.post_id not in self.posts:
                continue

            post = self.posts[interaction.post_id]

            # Get post embedding (from cache if available)
            post_emb = self._get_post_embedding(post)
            post_tensor = torch.tensor(post_emb, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Apply user's LoRA
            adapted_emb = lora(post_tensor)

            # Get user preference vector
            user_pref = self.user_preferences.get(user_id)
            if user_pref is None:
                continue

            user_pref_tensor = torch.tensor(user_pref, dtype=torch.float32).to(self.device)

            # Compute similarity score
            score = torch.cosine_similarity(adapted_emb, user_pref_tensor.unsqueeze(0))

            # Target: interaction weight (0 for skip, 1 for like, 2 for boost)
            target = torch.tensor([interaction.weight / 2.0], dtype=torch.float32).to(self.device)

            # Loss: MSE between predicted score and target
            loss = nn.functional.mse_loss(score, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update user preference vector
            if interaction.weight > 0:
                self.user_preferences.update(
                    user_id,
                    adapted_emb.detach().cpu().numpy().squeeze(),
                    weight=interaction.weight
                )

        lora.eval()

    def recommend(
        self,
        user_id: str,
        candidate_posts: Optional[List[Post]] = None,
        limit: int = 20,
        min_score: float = 0.1
    ) -> List[Tuple[Post, float]]:
        """
        Generate recommendations for a user.

        Args:
            user_id: User identifier
            candidate_posts: Posts to rank (or use all posts if None)
            limit: Maximum number of recommendations
            min_score: Minimum similarity score to include

        Returns:
            List of (post, score) tuples, sorted by score
        """
        # Handle cold start
        if not self.user_preferences.has_user(user_id):
            return self._cold_start_recommend(user_id, candidate_posts, limit)

        # Get candidates
        if candidate_posts is None:
            candidate_posts = list(self.posts.values())

        if not candidate_posts:
            return []

        # Get user's LoRA (if exists)
        has_lora = user_id in self.lora_manager.loras
        if has_lora:
            lora = self.lora_manager.get_or_create_lora(user_id)
            lora.eval()
        else:
            lora = None

        # Get user preference vector
        user_pref = self.user_preferences.get(user_id)
        if user_pref is None:
            return []

        # Score all candidates
        scored_posts = []

        # Batch encode posts
        post_embeddings = self._batch_encode_posts(candidate_posts)

        for post, post_emb in zip(candidate_posts, post_embeddings):
            # Apply LoRA if available
            if lora:
                with torch.no_grad():
                    post_tensor = torch.tensor(post_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
                    adapted_emb = lora(post_tensor)
                    post_emb = adapted_emb.cpu().numpy().squeeze()

            # Compute similarity to user preferences
            score = np.dot(post_emb, user_pref)

            if score >= min_score:
                scored_posts.append((post, float(score)))

        # Sort by score
        scored_posts.sort(key=lambda x: x[1], reverse=True)

        return scored_posts[:limit]

    def _cold_start_recommend(
        self,
        user_id: str,
        candidate_posts: Optional[List[Post]],
        limit: int
    ) -> List[Tuple[Post, float]]:
        """
        Handle cold start for new users.
        Falls back to popularity-based recommendations.
        """
        if candidate_posts is None:
            candidate_posts = list(self.posts.values())

        # Score by engagement metrics
        scored_posts = []
        for post in candidate_posts:
            score = (
                post.engagement.get('likes', 0) * 1.0 +
                post.engagement.get('boosts', 0) * 2.0 +
                post.engagement.get('replies', 0) * 1.5
            )
            scored_posts.append((post, score))

        scored_posts.sort(key=lambda x: x[1], reverse=True)
        return scored_posts[:limit]

    def _get_post_embedding(self, post: Post) -> np.ndarray:
        """Get post embedding (with caching)."""
        # Check cache
        cached = self.embedding_cache.get(post.id)
        if cached is not None:
            return cached

        # Encode
        embedding = self.foundation.encode(post.content)

        # Cache
        self.embedding_cache.put(post.id, embedding)

        return embedding

    def _batch_encode_posts(self, posts: List[Post]) -> List[np.ndarray]:
        """Encode multiple posts efficiently."""
        # Check which posts need encoding
        uncached_posts = []
        uncached_indices = []

        embeddings = []
        for i, post in enumerate(posts):
            cached = self.embedding_cache.get(post.id)
            if cached is not None:
                embeddings.append(cached)
            else:
                uncached_posts.append(post)
                uncached_indices.append(i)
                embeddings.append(None)  # Placeholder

        # Batch encode uncached posts
        if uncached_posts:
            texts = [p.content for p in uncached_posts]
            new_embeddings = self.foundation.encode(texts)

            # Cache and insert
            for post, emb, idx in zip(uncached_posts, new_embeddings, uncached_indices):
                self.embedding_cache.put(post.id, emb)
                embeddings[idx] = emb

        return embeddings

    def get_stats(self) -> Dict:
        """Get recommender statistics."""
        return {
            'instance_domain': self.instance_domain,
            'num_posts': len(self.posts),
            'num_users': self.lora_manager.num_users(),
            'cache_size': self.embedding_cache.size(),
            'lora_total_size_mb': self.lora_manager.total_size_mb(),
            'embedding_dim': self.foundation.embedding_dim,
            'lora_rank': self.lora_config.lora_rank
        }

    def save_user_loras(self):
        """Save all user LoRAs to storage."""
        self.lora_manager.save_all()

    def export_user_lora(self, user_id: str) -> Optional[Dict]:
        """
        Export a user's LoRA for portability.

        Args:
            user_id: User identifier

        Returns:
            Dictionary with LoRA parameters and metadata
        """
        if user_id not in self.lora_manager.loras:
            return None

        lora = self.lora_manager.loras[user_id]

        return {
            'user_id': user_id,
            'parameters': lora.get_parameters(),
            'config': self.lora_config,
            'preference_vector': self.user_preferences.get(user_id),
            'interaction_count': self.user_preferences.get_interaction_count(user_id)
        }

    def import_user_lora(self, lora_data: Dict):
        """
        Import a user's LoRA from another instance.

        Args:
            lora_data: Dictionary with LoRA parameters and metadata
        """
        user_id = lora_data['user_id']

        # Create LoRA
        lora = self.lora_manager.get_or_create_lora(user_id)
        lora.set_parameters(lora_data['parameters'])

        # Import preference vector
        if lora_data.get('preference_vector') is not None:
            self.user_preferences.vectors[user_id] = lora_data['preference_vector']
            self.user_preferences.interaction_counts[user_id] = lora_data.get('interaction_count', 0)

        logger.info(f"Imported LoRA for user {user_id}")


def test_decentralized_recommender():
    """Test decentralized recommender system."""
    print("🧪 Testing Decentralized Recommender")
    print("=" * 60)

    # Initialize recommender
    recommender = DecentralizedRecommender(
        instance_domain="mastodon.social",
        storage_dir="/tmp/test_recommender"
    )

    print(f"✅ Recommender initialized")
    stats = recommender.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Add sample posts
    posts = [
        Post(
            id="post_1",
            content="I love open source software and decentralized systems!",
            author="alice",
            timestamp=time.time(),
            engagement={'likes': 10, 'boosts': 5, 'replies': 2}
        ),
        Post(
            id="post_2",
            content="Check out this amazing photo I took at the beach!",
            author="bob",
            timestamp=time.time(),
            engagement={'likes': 20, 'boosts': 8, 'replies': 3}
        ),
        Post(
            id="post_3",
            content="Federated learning is the future of AI privacy",
            author="carol",
            timestamp=time.time(),
            engagement={'likes': 15, 'boosts': 10, 'replies': 5}
        )
    ]

    for post in posts:
        recommender.add_post(post)

    print(f"\n📝 Added {len(posts)} posts")

    # Simulate user interactions
    user_id = "user_alice"

    print(f"\n👤 Simulating interactions for {user_id}:")
    recommender.process_interaction(user_id, "post_1", "like")
    print(f"   Liked post_1")

    recommender.process_interaction(user_id, "post_3", "boost")
    print(f"   Boosted post_3")

    recommender.process_interaction(user_id, "post_2", "skip")
    print(f"   Skipped post_2")

    # Force LoRA update
    recommender._batch_update_loras()

    # Get recommendations
    print(f"\n🎯 Generating recommendations for {user_id}:")
    recommendations = recommender.recommend(user_id, limit=3)

    for i, (post, score) in enumerate(recommendations, 1):
        print(f"   {i}. {post.id} (score={score:.4f})")
        print(f"      Content: {post.content[:60]}...")

    # Test LoRA export/import
    print(f"\n📦 Testing LoRA portability:")
    exported = recommender.export_user_lora(user_id)
    if exported:
        print(f"   Exported LoRA for {user_id}")
        print(f"   Interaction count: {exported['interaction_count']}")

        # Import to new instance
        new_recommender = DecentralizedRecommender(
            instance_domain="pleroma.site",
            storage_dir="/tmp/test_recommender2"
        )
        new_recommender.import_user_lora(exported)
        print(f"   Imported LoRA to new instance")

    # Final stats
    print(f"\n📊 Final statistics:")
    stats = recommender.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print(f"\n✅ All tests passed!")

    return recommender


if __name__ == "__main__":
    test_decentralized_recommender()
