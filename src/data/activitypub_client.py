"""
ActivityPub Client for Federated Content Collection
Collects posts from ActivityPub-compatible instances (Mastodon, Pleroma, etc.)
"""

import requests
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
import time
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


@dataclass
class ActivityPubPost:
    """Represents a post from ActivityPub federation."""
    id: str
    uri: str
    content: str
    author: str
    author_uri: str
    created_at: datetime
    likes_count: int
    boosts_count: int
    replies_count: int
    language: Optional[str] = None
    sensitive: bool = False
    visibility: str = "public"
    media_attachments: List[Dict] = None

    def __post_init__(self):
        if self.media_attachments is None:
            self.media_attachments = []

    def to_post(self):
        """Convert to simple Post object for recommender."""
        from ..models.decentralized_recommender import Post

        return Post(
            id=self.id,
            content=self.content,
            author=self.author,
            timestamp=self.created_at.timestamp(),
            engagement={
                'likes': self.likes_count,
                'boosts': self.boosts_count,
                'replies': self.replies_count
            }
        )


class ActivityPubClient:
    """
    Client for interacting with ActivityPub instances.
    Collects public posts for recommendation.
    """

    def __init__(
        self,
        instance_url: str,
        access_token: Optional[str] = None,
        user_agent: str = "FederatedRecommendations/1.0"
    ):
        """
        Initialize ActivityPub client.

        Args:
            instance_url: Base URL of instance (e.g., https://mastodon.social)
            access_token: Optional access token for authenticated requests
            user_agent: User agent string
        """
        self.instance_url = instance_url.rstrip('/')
        self.access_token = access_token
        self.user_agent = user_agent

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 500ms between requests

        logger.info(f"ActivityPub client initialized for {instance_url}")

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Make HTTP request to instance API.

        Args:
            endpoint: API endpoint (e.g., '/api/v1/timelines/public')
            params: Query parameters

        Returns:
            JSON response or None on error
        """
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

        # Build URL
        url = f"{self.instance_url}{endpoint}"

        # Headers
        headers = {
            'User-Agent': self.user_agent
        }
        if self.access_token:
            headers['Authorization'] = f'Bearer {self.access_token}'

        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            self.last_request_time = time.time()

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logger.warning(f"Rate limited by {self.instance_url}")
                return None
            else:
                logger.warning(f"Request failed: {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None

    def get_public_timeline(
        self,
        limit: int = 40,
        local: bool = True,
        max_id: Optional[str] = None
    ) -> List[ActivityPubPost]:
        """
        Get posts from public timeline.

        Args:
            limit: Maximum number of posts to fetch
            local: Only local posts (not federated)
            max_id: Get posts older than this ID (for pagination)

        Returns:
            List of ActivityPub posts
        """
        params = {
            'limit': min(limit, 40),  # Most instances cap at 40
            'local': str(local).lower()
        }
        if max_id:
            params['max_id'] = max_id

        data = self._make_request('/api/v1/timelines/public', params)

        if not data:
            return []

        return [self._parse_status(status) for status in data if self._is_valid_status(status)]

    def get_hashtag_timeline(
        self,
        hashtag: str,
        limit: int = 40,
        local: bool = False
    ) -> List[ActivityPubPost]:
        """
        Get posts from hashtag timeline.

        Args:
            hashtag: Hashtag to search (without #)
            limit: Maximum number of posts
            local: Only local posts

        Returns:
            List of ActivityPub posts
        """
        params = {
            'limit': min(limit, 40),
            'local': str(local).lower()
        }

        data = self._make_request(f'/api/v1/timelines/tag/{hashtag}', params)

        if not data:
            return []

        return [self._parse_status(status) for status in data if self._is_valid_status(status)]

    def get_trending_posts(
        self,
        limit: int = 20
    ) -> List[ActivityPubPost]:
        """
        Get trending posts.

        Args:
            limit: Maximum number of posts

        Returns:
            List of trending ActivityPub posts
        """
        data = self._make_request('/api/v1/trends/statuses', params={'limit': limit})

        if not data:
            return []

        return [self._parse_status(status) for status in data if self._is_valid_status(status)]

    def get_user_statuses(
        self,
        user_id: str,
        limit: int = 40,
        exclude_replies: bool = True
    ) -> List[ActivityPubPost]:
        """
        Get posts from a specific user.

        Args:
            user_id: User account ID
            limit: Maximum number of posts
            exclude_replies: Exclude reply posts

        Returns:
            List of user's posts
        """
        params = {
            'limit': min(limit, 40),
            'exclude_replies': str(exclude_replies).lower()
        }

        data = self._make_request(f'/api/v1/accounts/{user_id}/statuses', params)

        if not data:
            return []

        return [self._parse_status(status) for status in data if self._is_valid_status(status)]

    def get_home_timeline(
        self,
        limit: int = 40,
        max_id: Optional[str] = None
    ) -> List[ActivityPubPost]:
        """
        Get posts from home timeline (requires authentication).

        Args:
            limit: Maximum number of posts
            max_id: Get posts older than this ID

        Returns:
            List of posts from followed accounts
        """
        if not self.access_token:
            logger.warning("Home timeline requires authentication")
            return []

        params = {
            'limit': min(limit, 40)
        }
        if max_id:
            params['max_id'] = max_id

        data = self._make_request('/api/v1/timelines/home', params)

        if not data:
            return []

        return [self._parse_status(status) for status in data if self._is_valid_status(status)]

    def _parse_status(self, status: Dict) -> ActivityPubPost:
        """Parse Mastodon status JSON into ActivityPubPost."""
        # Extract content (strip HTML tags)
        content = self._strip_html(status.get('content', ''))

        # Parse created_at
        created_at = datetime.fromisoformat(
            status['created_at'].replace('Z', '+00:00')
        )

        return ActivityPubPost(
            id=status['id'],
            uri=status['uri'],
            content=content,
            author=status['account']['acct'],
            author_uri=status['account']['url'],
            created_at=created_at,
            likes_count=status.get('favourites_count', 0),
            boosts_count=status.get('reblogs_count', 0),
            replies_count=status.get('replies_count', 0),
            language=status.get('language'),
            sensitive=status.get('sensitive', False),
            visibility=status.get('visibility', 'public'),
            media_attachments=status.get('media_attachments', [])
        )

    def _is_valid_status(self, status: Dict) -> bool:
        """Check if status is valid for recommendations."""
        # Skip if no content
        if not status.get('content'):
            return False

        # Skip if sensitive and not marked
        if status.get('sensitive') and not status.get('spoiler_text'):
            return False

        # Skip if private
        if status.get('visibility') not in ['public', 'unlisted']:
            return False

        return True

    @staticmethod
    def _strip_html(html: str) -> str:
        """Strip HTML tags from content."""
        import re
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html)
        # Decode HTML entities
        import html as html_module
        text = html_module.unescape(text)
        return text.strip()


class FederatedContentCollector:
    """
    Collects content from multiple ActivityPub instances.
    Provides diverse content pool for recommendations.
    """

    def __init__(self, instances: List[str]):
        """
        Initialize federated content collector.

        Args:
            instances: List of instance URLs to collect from
        """
        self.clients = {
            instance: ActivityPubClient(instance)
            for instance in instances
        }

        # Track seen posts to avoid duplicates
        self.seen_posts: Set[str] = set()

        logger.info(f"Federated collector initialized with {len(instances)} instances")

    def collect_recent_posts(
        self,
        limit_per_instance: int = 20,
        local_only: bool = False
    ) -> List[ActivityPubPost]:
        """
        Collect recent posts from all instances.

        Args:
            limit_per_instance: Maximum posts per instance
            local_only: Only collect local posts

        Returns:
            List of unique posts from federation
        """
        all_posts = []

        for instance_url, client in self.clients.items():
            logger.debug(f"Collecting from {instance_url}...")

            posts = client.get_public_timeline(
                limit=limit_per_instance,
                local=local_only
            )

            # Filter duplicates
            for post in posts:
                if post.uri not in self.seen_posts:
                    self.seen_posts.add(post.uri)
                    all_posts.append(post)

        logger.info(f"Collected {len(all_posts)} unique posts from {len(self.clients)} instances")

        # Sort by timestamp
        all_posts.sort(key=lambda p: p.created_at, reverse=True)

        return all_posts

    def collect_trending(
        self,
        limit_per_instance: int = 10
    ) -> List[ActivityPubPost]:
        """
        Collect trending posts from all instances.

        Args:
            limit_per_instance: Maximum posts per instance

        Returns:
            List of trending posts
        """
        all_posts = []

        for instance_url, client in self.clients.items():
            posts = client.get_trending_posts(limit=limit_per_instance)

            for post in posts:
                if post.uri not in self.seen_posts:
                    self.seen_posts.add(post.uri)
                    all_posts.append(post)

        # Sort by engagement
        all_posts.sort(
            key=lambda p: p.likes_count + p.boosts_count * 2 + p.replies_count * 1.5,
            reverse=True
        )

        return all_posts

    def clear_seen_cache(self):
        """Clear the seen posts cache."""
        self.seen_posts.clear()


def test_activitypub_client():
    """Test ActivityPub client."""
    print("🧪 Testing ActivityPub Client")
    print("=" * 60)

    # Test with mastodon.social (should work without authentication for public timeline)
    client = ActivityPubClient("https://mastodon.social")

    print(f"✅ Client initialized for {client.instance_url}")

    # Get public timeline
    print(f"\n📰 Fetching public timeline...")
    posts = client.get_public_timeline(limit=5, local=True)

    print(f"✅ Retrieved {len(posts)} posts")
    for i, post in enumerate(posts[:3], 1):
        print(f"\n   Post {i}:")
        print(f"   Author: {post.author}")
        print(f"   Content: {post.content[:80]}...")
        print(f"   Engagement: {post.likes_count} likes, {post.boosts_count} boosts")

    # Test federated collector
    print(f"\n🌐 Testing Federated Content Collector...")
    collector = FederatedContentCollector([
        "https://mastodon.social",
        "https://fosstodon.org"
    ])

    print(f"✅ Collector initialized")

    federated_posts = collector.collect_recent_posts(limit_per_instance=3, local_only=True)
    print(f"✅ Collected {len(federated_posts)} posts from federation")

    for i, post in enumerate(federated_posts[:2], 1):
        print(f"\n   Federated Post {i}:")
        print(f"   Instance: {post.uri.split('/')[2]}")
        print(f"   Content: {post.content[:60]}...")

    print(f"\n✅ All tests passed!")

    return client, collector


if __name__ == "__main__":
    test_activitypub_client()
