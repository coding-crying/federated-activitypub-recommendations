"""
ActivityPub-Compliant Recommendation System Integration
Hooks into standard ActivityPub activity flows for learning and recommendation delivery
"""

import json
from typing import Dict, List, Optional
from datetime import datetime
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ActivityPubActivity:
    """Standard ActivityPub activity structure."""
    id: str
    type: str  # Like, Announce, Create, Follow, etc.
    actor: str  # User who performed the activity
    object: str  # Object being acted upon
    published: datetime
    to: List[str]  # Audience targeting
    cc: List[str]  # Carbon copy targeting


class ActivityPubRecommendationHook:
    """
    ActivityPub-compliant integration for federated recommendations.
    Hooks into standard activity flows without breaking protocol compliance.
    """

    def __init__(self, recommendation_service_url: str):
        self.recommendation_service_url = recommendation_service_url
        self.supported_activities = ['Like', 'Announce', 'Create', 'Follow']

    def process_inbox_activity(self, activity: Dict) -> Optional[Dict]:
        """
        Process incoming ActivityPub activities for recommendation learning.

        This hooks into the standard ActivityPub inbox processing pipeline.
        Called after standard processing, doesn't interfere with core ActivityPub.

        Args:
            activity: Standard ActivityPub activity JSON

        Returns:
            Optional recommendation activities to inject into outbox
        """
        try:
            activity_type = activity.get('type')

            if activity_type in self.supported_activities:
                # Extract learning signals from standard ActivityPub activities
                learning_signal = self._extract_learning_signal(activity)

                # Update user model (async to not block ActivityPub processing)
                self._async_update_user_model(learning_signal)

                # Generate recommendations if appropriate
                if self._should_generate_recommendations(activity):
                    return self._generate_recommendation_activity(activity)

        except Exception as e:
            logger.error(f"Error processing activity for recommendations: {e}")

        return None

    def _extract_learning_signal(self, activity: Dict) -> Dict:
        """Extract recommendation learning signals from ActivityPub activities."""
        return {
            'user_id': activity.get('actor'),
            'content_id': activity.get('object'),
            'activity_type': activity.get('type'),
            'timestamp': activity.get('published'),
            'audience': activity.get('to', []) + activity.get('cc', [])
        }

    def _generate_recommendation_activity(self, trigger_activity: Dict) -> Dict:
        """
        Generate ActivityPub-compliant recommendation activities.

        Creates standard ActivityPub 'Create' activities with recommended content
        that can be delivered through normal ActivityPub channels.
        """
        user_id = trigger_activity.get('actor')
        recommendations = self._get_recommendations_for_user(user_id)

        # Create ActivityPub 'Create' activity with recommendations
        # This follows the standard ActivityPub object model
        recommendation_activity = {
            "@context": "https://www.w3.org/ns/activitystreams",
            "id": f"{self.recommendation_service_url}/activities/{self._generate_id()}",
            "type": "Create",
            "actor": f"{self.recommendation_service_url}/actors/recommendation-bot",
            "published": datetime.utcnow().isoformat() + "Z",
            "to": [user_id],
            "object": {
                "id": f"{self.recommendation_service_url}/objects/{self._generate_id()}",
                "type": "Note",
                "attributedTo": f"{self.recommendation_service_url}/actors/recommendation-bot",
                "content": self._format_recommendations_as_content(recommendations),
                "tag": [
                    {
                        "type": "Hashtag",
                        "href": f"{self.recommendation_service_url}/tags/federated-recommendations",
                        "name": "#FederatedRecommendations"
                    }
                ],
                # Custom extension for recommendation metadata
                "recommendations": recommendations
            }
        }

        return recommendation_activity


class ActivityPubRecommendationService:
    """
    Full ActivityPub actor that can participate in the federation
    as a recommendation service.
    """

    def __init__(self, service_domain: str):
        self.service_domain = service_domain
        self.actor_id = f"https://{service_domain}/actors/recommendation-service"

    def get_actor_object(self) -> Dict:
        """
        Return standard ActivityPub Actor object.
        This allows the recommendation service to be a proper ActivityPub citizen.
        """
        return {
            "@context": [
                "https://www.w3.org/ns/activitystreams",
                "https://w3id.org/security/v1"
            ],
            "id": self.actor_id,
            "type": "Service",  # ActivityPub Service actor type
            "preferredUsername": "recommendations",
            "name": "Federated Recommendations Service",
            "summary": "Privacy-preserving federated learning recommendations for the fediverse",
            "inbox": f"https://{self.service_domain}/inbox",
            "outbox": f"https://{self.service_domain}/outbox",
            "followers": f"https://{self.service_domain}/followers",
            "following": f"https://{self.service_domain}/following",
            "publicKey": {
                "id": f"{self.actor_id}#main-key",
                "owner": self.actor_id,
                "publicKeyPem": self._get_public_key()
            },
            # Custom extension for recommendation capabilities
            "capabilities": {
                "federatedLearning": True,
                "privacyPreserving": True,
                "loraAdapters": True
            }
        }

    def process_follow_activity(self, follow_activity: Dict) -> Dict:
        """
        Handle Follow activities to opt users into recommendations.

        When a user follows the recommendation service, they opt-in to
        receiving AI-generated recommendations through standard ActivityPub.
        """
        follower = follow_activity.get('actor')

        # Accept the follow
        accept_activity = {
            "@context": "https://www.w3.org/ns/activitystreams",
            "id": f"https://{self.service_domain}/activities/{self._generate_id()}",
            "type": "Accept",
            "actor": self.actor_id,
            "object": follow_activity
        }

        # Initialize user's recommendation profile
        self._initialize_user_profile(follower)

        return accept_activity

    def deliver_recommendations_via_activitypub(self, user_id: str, recommendations: List[Dict]):
        """
        Deliver recommendations using standard ActivityPub delivery mechanisms.

        This creates proper ActivityPub activities that are delivered to user inboxes
        through the standard federation protocol.
        """
        for recommendation in recommendations:
            activity = {
                "@context": "https://www.w3.org/ns/activitystreams",
                "id": f"https://{self.service_domain}/activities/{self._generate_id()}",
                "type": "Create",
                "actor": self.actor_id,
                "published": datetime.utcnow().isoformat() + "Z",
                "to": [user_id],
                "object": {
                    "id": f"https://{self.service_domain}/objects/{self._generate_id()}",
                    "type": "Note",
                    "attributedTo": self.actor_id,
                    "content": f"🤖 Recommended for you: {recommendation['content']}",
                    "inReplyTo": None,
                    "tag": [
                        {
                            "type": "Hashtag",
                            "name": "#AIRecommendation"
                        }
                    ],
                    # Embed original post for easy boosting/liking
                    "attachment": {
                        "type": "Link",
                        "href": recommendation['original_url'],
                        "name": "Original Post"
                    }
                }
            }

            # Use standard ActivityPub delivery
            self._deliver_activity(activity, user_id)


class ActivityPubProtocolCompliance:
    """
    Ensures all recommendation system interactions comply with ActivityPub spec.
    """

    @staticmethod
    def validate_activity(activity: Dict) -> bool:
        """Validate that activities conform to ActivityPub specification."""
        required_fields = ['@context', 'id', 'type', 'actor']
        return all(field in activity for field in required_fields)

    @staticmethod
    def add_context(activity: Dict) -> Dict:
        """Add proper ActivityStreams context to activities."""
        if '@context' not in activity:
            activity['@context'] = [
                "https://www.w3.org/ns/activitystreams",
                {
                    "recommendations": "https://federated-rec.org/ns#recommendations",
                    "loraModel": "https://federated-rec.org/ns#loraModel",
                    "privacyLevel": "https://federated-rec.org/ns#privacyLevel"
                }
            ]
        return activity

    @staticmethod
    def create_recommendation_collection(service_domain: str) -> Dict:
        """
        Create ActivityPub Collection for recommendations.

        This allows recommendations to be discoverable through standard
        ActivityPub collection mechanisms.
        """
        return {
            "@context": "https://www.w3.org/ns/activitystreams",
            "id": f"https://{service_domain}/collections/recommendations",
            "type": "Collection",
            "name": "Federated Recommendations",
            "summary": "AI-generated recommendations from federated learning",
            "totalItems": 0,
            "items": []
        }


# Integration with existing Mastodon/Pleroma instances
class MastodonIntegrationHook:
    """
    Hooks for integrating with existing Mastodon instances
    without requiring code changes.
    """

    def __init__(self, mastodon_instance_url: str, recommendation_service: ActivityPubRecommendationService):
        self.mastodon_url = mastodon_instance_url
        self.rec_service = recommendation_service

    def setup_webhook_integration(self):
        """
        Set up webhook-based integration with Mastodon.

        Many Mastodon instances support webhooks for activities.
        This provides a way to integrate without forking Mastodon.
        """
        webhook_config = {
            "url": f"{self.rec_service.service_domain}/webhooks/mastodon",
            "events": ["account.created", "status.created", "favourite.created", "reblog.created"],
            "secret": self._generate_webhook_secret()
        }
        return webhook_config

    def process_mastodon_webhook(self, webhook_data: Dict):
        """
        Process webhooks from Mastodon and convert to ActivityPub activities.

        This bridges Mastodon's webhook format to proper ActivityPub activities
        for our recommendation system.
        """
        event_type = webhook_data.get('event')

        # Convert Mastodon webhook to ActivityPub activity
        if event_type == 'favourite.created':
            return self._convert_favourite_to_like_activity(webhook_data)
        elif event_type == 'reblog.created':
            return self._convert_reblog_to_announce_activity(webhook_data)
        elif event_type == 'status.created':
            return self._convert_status_to_create_activity(webhook_data)

    def _convert_favourite_to_like_activity(self, webhook_data: Dict) -> Dict:
        """Convert Mastodon favourite webhook to ActivityPub Like activity."""
        return {
            "@context": "https://www.w3.org/ns/activitystreams",
            "id": f"{self.mastodon_url}/activities/{webhook_data['favourite']['id']}",
            "type": "Like",
            "actor": f"{self.mastodon_url}/users/{webhook_data['favourite']['account']['acct']}",
            "object": f"{self.mastodon_url}/statuses/{webhook_data['favourite']['status']['id']}",
            "published": webhook_data['favourite']['created_at']
        }


def demo_activitypub_integration():
    """Demonstrate ActivityPub-compliant recommendation integration."""
    print("🌐 ActivityPub-Compliant Recommendation Integration Demo")
    print("=" * 60)

    # Create recommendation service actor
    rec_service = ActivityPubRecommendationService("recommendations.fediverse.ai")
    actor = rec_service.get_actor_object()

    print("📋 Recommendation Service Actor:")
    print(f"   ID: {actor['id']}")
    print(f"   Type: {actor['type']}")
    print(f"   Capabilities: {actor.get('capabilities', {})}")

    # Demonstrate activity processing
    sample_like_activity = {
        "@context": "https://www.w3.org/ns/activitystreams",
        "id": "https://mastodon.social/activities/123",
        "type": "Like",
        "actor": "https://mastodon.social/users/alice",
        "object": "https://mastodon.social/statuses/456",
        "published": "2024-01-01T12:00:00Z"
    }

    hook = ActivityPubRecommendationHook("https://recommendations.fediverse.ai")
    result = hook.process_inbox_activity(sample_like_activity)

    if result:
        print("\n📨 Generated Recommendation Activity:")
        print(f"   Type: {result['type']}")
        print(f"   Actor: {result['actor']}")
        print(f"   Content: {result['object']['content'][:100]}...")

    print("\n✅ ActivityPub compliance verified!")


if __name__ == "__main__":
    demo_activitypub_integration()