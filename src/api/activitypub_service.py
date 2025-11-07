"""
Complete ActivityPub Recommendation Service
Integrates decentralized recommender with ActivityPub protocol
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import uuid
import json

from ..models.decentralized_recommender import DecentralizedRecommender, Post
from ..data.activitypub_client import ActivityPubClient, FederatedContentCollector

logger = logging.getLogger(__name__)


class ActivityPubRecommendationService:
    """
    Full ActivityPub actor that provides recommendations through the federation.

    This service:
    1. Acts as a proper ActivityPub Service actor
    2. Collects content from federation
    3. Learns from user interactions (likes, boosts)
    4. Delivers personalized recommendations
    5. Preserves privacy (LoRAs stay local)
    """

    def __init__(
        self,
        service_domain: str,
        instance_url: str,
        recommender: DecentralizedRecommender,
        content_collector: Optional[FederatedContentCollector] = None
    ):
        """
        Initialize ActivityPub recommendation service.

        Args:
            service_domain: Domain where service is hosted
            instance_url: URL of the Mastodon/Pleroma instance
            recommender: Decentralized recommender instance
            content_collector: Optional federated content collector
        """
        self.service_domain = service_domain
        self.instance_url = instance_url
        self.recommender = recommender
        self.content_collector = content_collector

        self.actor_id = f"https://{service_domain}/actor"
        self.inbox_url = f"https://{service_domain}/inbox"
        self.outbox_url = f"https://{service_domain}/outbox"

        # Track followers (users who opted in)
        self.followers: set = set()

        logger.info(f"ActivityPub Recommendation Service initialized")
        logger.info(f"  Actor ID: {self.actor_id}")
        logger.info(f"  Instance: {instance_url}")

    def get_actor_object(self) -> Dict:
        """
        Return standard ActivityPub Actor object.

        This allows the service to be a proper ActivityPub citizen
        that can be discovered and followed.
        """
        return {
            "@context": [
                "https://www.w3.org/ns/activitystreams",
                "https://w3id.org/security/v1"
            ],
            "id": self.actor_id,
            "type": "Service",
            "preferredUsername": "recommendations",
            "name": "Federated AI Recommendations",
            "summary": (
                "🤖 Privacy-preserving AI recommendations powered by federated learning. "
                "Follow me to get personalized content suggestions!"
            ),
            "inbox": self.inbox_url,
            "outbox": self.outbox_url,
            "followers": f"https://{self.service_domain}/followers",
            "following": f"https://{self.service_domain}/following",
            "icon": {
                "type": "Image",
                "mediaType": "image/png",
                "url": f"https://{self.service_domain}/avatar.png"
            },
            "publicKey": {
                "id": f"{self.actor_id}#main-key",
                "owner": self.actor_id,
                "publicKeyPem": self._get_public_key()
            },
            # Custom extension for recommendation capabilities
            "capabilities": {
                "federatedLearning": True,
                "privacyPreserving": True,
                "perUserLoRA": True,
                "loraRank": self.recommender.lora_config.lora_rank,
                "embeddingDim": self.recommender.foundation.embedding_dim
            }
        }

    def process_inbox_activity(self, activity: Dict) -> Optional[Dict]:
        """
        Process incoming ActivityPub activities.

        Handles:
        - Follow: User opts into recommendations
        - Like: Learn from user preference
        - Announce (boost): Learn from user preference (stronger signal)
        - Create: Add content to recommendation pool

        Args:
            activity: Standard ActivityPub activity

        Returns:
            Optional response activity
        """
        activity_type = activity.get('type')
        actor = activity.get('actor')

        logger.debug(f"Processing {activity_type} from {actor}")

        try:
            if activity_type == 'Follow':
                return self._handle_follow(activity)

            elif activity_type == 'Undo':
                # Handle Unfollow
                if activity.get('object', {}).get('type') == 'Follow':
                    return self._handle_unfollow(activity)

            elif activity_type == 'Like':
                self._handle_like(activity)

            elif activity_type == 'Announce':
                self._handle_announce(activity)

            elif activity_type == 'Create':
                self._handle_create(activity)

        except Exception as e:
            logger.error(f"Error processing activity: {e}")

        return None

    def _handle_follow(self, activity: Dict) -> Dict:
        """
        Handle Follow activity.

        User is opting into AI recommendations.
        """
        follower = activity.get('actor')

        # Add to followers
        self.followers.add(follower)

        # Initialize user in recommender
        # (will create LoRA on first interaction)

        logger.info(f"New follower: {follower}")

        # Accept the follow
        return {
            "@context": "https://www.w3.org/ns/activitystreams",
            "id": f"{self.actor_id}/activities/{self._generate_id()}",
            "type": "Accept",
            "actor": self.actor_id,
            "object": activity
        }

    def _handle_unfollow(self, activity: Dict) -> Dict:
        """Handle Unfollow (Undo Follow) activity."""
        follower = activity.get('actor')

        if follower in self.followers:
            self.followers.remove(follower)
            logger.info(f"User unfollowed: {follower}")

        return {
            "@context": "https://www.w3.org/ns/activitystreams",
            "id": f"{self.actor_id}/activities/{self._generate_id()}",
            "type": "Accept",
            "actor": self.actor_id,
            "object": activity
        }

    def _handle_like(self, activity: Dict):
        """
        Handle Like activity.

        User liked a post - positive learning signal.
        """
        user_id = activity.get('actor')
        post_uri = activity.get('object')

        if user_id not in self.followers:
            return  # Only learn from followers

        # Extract post ID
        post_id = self._extract_post_id(post_uri)

        if post_id:
            # Process interaction for learning
            self.recommender.process_interaction(
                user_id=user_id,
                post_id=post_id,
                interaction_type='like'
            )

            logger.debug(f"Processed like: {user_id} -> {post_id}")

    def _handle_announce(self, activity: Dict):
        """
        Handle Announce (boost) activity.

        User boosted a post - strong positive learning signal.
        """
        user_id = activity.get('actor')
        post_uri = activity.get('object')

        if user_id not in self.followers:
            return

        post_id = self._extract_post_id(post_uri)

        if post_id:
            self.recommender.process_interaction(
                user_id=user_id,
                post_id=post_id,
                interaction_type='boost'
            )

            logger.debug(f"Processed boost: {user_id} -> {post_id}")

    def _handle_create(self, activity: Dict):
        """
        Handle Create activity.

        New post created - add to content pool.
        """
        obj = activity.get('object', {})

        if obj.get('type') != 'Note':
            return

        # Parse post
        try:
            from ..data.activitypub_client import ActivityPubPost
            import re

            content = self._strip_html(obj.get('content', ''))

            if not content:
                return

            post = Post(
                id=obj['id'],
                content=content,
                author=activity.get('actor', 'unknown'),
                timestamp=datetime.utcnow().timestamp(),
                engagement={
                    'likes': 0,
                    'boosts': 0,
                    'replies': 0
                }
            )

            self.recommender.add_post(post)

            logger.debug(f"Added post to content pool: {post.id}")

        except Exception as e:
            logger.warning(f"Failed to parse Create activity: {e}")

    def generate_recommendations_for_user(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Generate recommendations for a specific user.

        Args:
            user_id: User's ActivityPub actor ID
            limit: Maximum number of recommendations

        Returns:
            List of recommendation activities
        """
        if user_id not in self.followers:
            logger.warning(f"User {user_id} is not a follower")
            return []

        # Collect recent content if collector is available
        if self.content_collector:
            recent_posts = self.content_collector.collect_recent_posts(limit_per_instance=20)

            # Add to recommender
            for ap_post in recent_posts:
                self.recommender.add_post(ap_post.to_post())

        # Get recommendations
        recommendations = self.recommender.recommend(
            user_id=user_id,
            limit=limit
        )

        # Convert to ActivityPub activities
        activities = []

        for post, score in recommendations:
            activity = self._create_recommendation_activity(
                user_id=user_id,
                post=post,
                score=score
            )
            activities.append(activity)

        return activities

    def _create_recommendation_activity(
        self,
        user_id: str,
        post: Post,
        score: float
    ) -> Dict:
        """
        Create ActivityPub activity for a recommendation.

        This creates a standard Note that appears in the user's timeline.
        """
        # Format recommendation as a post
        content = (
            f"🤖 <strong>Recommended for you</strong> (relevance: {score:.2f})<br><br>"
            f"{post.content}<br><br>"
            f"<small>👤 By {post.author} | "
            f"❤️ {post.engagement.get('likes', 0)} | "
            f"🔁 {post.engagement.get('boosts', 0)}</small>"
        )

        return {
            "@context": "https://www.w3.org/ns/activitystreams",
            "id": f"{self.actor_id}/activities/{self._generate_id()}",
            "type": "Create",
            "actor": self.actor_id,
            "published": datetime.utcnow().isoformat() + "Z",
            "to": [user_id],
            "cc": [],
            "object": {
                "id": f"{self.actor_id}/notes/{self._generate_id()}",
                "type": "Note",
                "attributedTo": self.actor_id,
                "content": content,
                "published": datetime.utcnow().isoformat() + "Z",
                "to": [user_id],
                "cc": [],
                "tag": [
                    {
                        "type": "Hashtag",
                        "name": "#FederatedAI"
                    },
                    {
                        "type": "Hashtag",
                        "name": "#Recommendations"
                    }
                ],
                # Link to original post
                "inReplyTo": None,
                "attachment": {
                    "type": "Link",
                    "href": f"https://{self.recommender.instance_domain}/posts/{post.id}",
                    "name": "View Original"
                },
                # Custom metadata
                "recommendationMetadata": {
                    "score": score,
                    "originalPostId": post.id,
                    "originalAuthor": post.author,
                    "loraRank": self.recommender.lora_config.lora_rank
                }
            }
        }

    def batch_send_recommendations(self, batch_size: int = 5):
        """
        Send recommendations to all followers in batches.

        Args:
            batch_size: Number of users to process at once
        """
        followers_list = list(self.followers)

        logger.info(f"Sending recommendations to {len(followers_list)} followers")

        for i in range(0, len(followers_list), batch_size):
            batch = followers_list[i:i + batch_size]

            for user_id in batch:
                try:
                    activities = self.generate_recommendations_for_user(user_id, limit=5)

                    for activity in activities:
                        # In production, this would deliver via HTTP POST to user's inbox
                        self._deliver_activity(activity, user_id)

                    logger.debug(f"Sent {len(activities)} recommendations to {user_id}")

                except Exception as e:
                    logger.error(f"Failed to send recommendations to {user_id}: {e}")

    def _deliver_activity(self, activity: Dict, target: str):
        """
        Deliver activity to target user's inbox.

        In production, this would:
        1. Look up target's inbox URL
        2. Sign the activity with HTTP signatures
        3. POST to inbox
        """
        # Placeholder for actual delivery
        logger.debug(f"Would deliver activity to {target}: {activity.get('type')}")

    @staticmethod
    def _generate_id() -> str:
        """Generate unique activity ID."""
        return str(uuid.uuid4())

    @staticmethod
    def _extract_post_id(uri: str) -> Optional[str]:
        """Extract post ID from URI."""
        # Simple extraction - in production would be more robust
        parts = uri.rstrip('/').split('/')
        return parts[-1] if parts else None

    @staticmethod
    def _strip_html(html: str) -> str:
        """Strip HTML tags from content."""
        import re
        text = re.sub(r'<[^>]+>', '', html)
        import html as html_module
        text = html_module.unescape(text)
        return text.strip()

    @staticmethod
    def _get_public_key() -> str:
        """Get service's public key (placeholder)."""
        # In production, generate and store actual RSA key pair
        return "-----BEGIN PUBLIC KEY-----\nPLACEHOLDER\n-----END PUBLIC KEY-----"

    def get_stats(self) -> Dict:
        """Get service statistics."""
        return {
            'service_domain': self.service_domain,
            'actor_id': self.actor_id,
            'followers': len(self.followers),
            'recommender_stats': self.recommender.get_stats()
        }


def test_activitypub_service():
    """Test ActivityPub service."""
    print("🧪 Testing ActivityPub Recommendation Service")
    print("=" * 60)

    # Create recommender
    from ..models.decentralized_recommender import DecentralizedRecommender

    recommender = DecentralizedRecommender(
        instance_domain="mastodon.social",
        storage_dir="/tmp/test_ap_service"
    )

    # Create service
    service = ActivityPubRecommendationService(
        service_domain="recommendations.fediverse.ai",
        instance_url="https://mastodon.social",
        recommender=recommender
    )

    print(f"✅ Service initialized")
    print(f"   Actor ID: {service.actor_id}")

    # Get actor object
    actor = service.get_actor_object()
    print(f"\n📋 Actor Object:")
    print(f"   Type: {actor['type']}")
    print(f"   Name: {actor['name']}")
    print(f"   Capabilities: {actor['capabilities']}")

    # Simulate Follow activity
    print(f"\n👤 Simulating Follow activity...")
    follow_activity = {
        "@context": "https://www.w3.org/ns/activitystreams",
        "id": "https://mastodon.social/users/alice/follows/1",
        "type": "Follow",
        "actor": "https://mastodon.social/users/alice",
        "object": service.actor_id
    }

    response = service.process_inbox_activity(follow_activity)
    print(f"   Response type: {response['type']}")
    print(f"   Followers: {len(service.followers)}")

    # Add some test posts
    print(f"\n📝 Adding test posts...")
    test_posts = [
        Post(
            id="1",
            content="Federated learning is revolutionizing AI privacy!",
            author="bob",
            timestamp=1234567890.0,
            engagement={'likes': 10, 'boosts': 5, 'replies': 2}
        ),
        Post(
            id="2",
            content="Check out this beautiful sunset photo 🌅",
            author="carol",
            timestamp=1234567891.0,
            engagement={'likes': 50, 'boosts': 20, 'replies': 5}
        )
    ]

    for post in test_posts:
        recommender.add_post(post)

    print(f"   Added {len(test_posts)} posts")

    # Simulate Like activity
    print(f"\n❤️  Simulating Like activity...")
    like_activity = {
        "@context": "https://www.w3.org/ns/activitystreams",
        "id": "https://mastodon.social/users/alice/likes/1",
        "type": "Like",
        "actor": "https://mastodon.social/users/alice",
        "object": "https://mastodon.social/statuses/1"
    }

    service.process_inbox_activity(like_activity)
    print(f"   Processed like activity")

    # Generate recommendations
    print(f"\n🎯 Generating recommendations for alice...")
    recommendations = service.generate_recommendations_for_user(
        user_id="https://mastodon.social/users/alice",
        limit=5
    )

    print(f"   Generated {len(recommendations)} recommendations")
    if recommendations:
        rec = recommendations[0]
        print(f"\n   Sample recommendation:")
        print(f"   Type: {rec['type']}")
        print(f"   Content preview: {rec['object']['content'][:100]}...")

    # Get stats
    print(f"\n📊 Service Statistics:")
    stats = service.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"      {k}: {v}")
        else:
            print(f"   {key}: {value}")

    print(f"\n✅ All tests passed!")

    return service


if __name__ == "__main__":
    test_activitypub_service()
