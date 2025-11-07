"""Data module for federated recommendations."""

from .activitypub_client import (
    ActivityPubClient,
    ActivityPubPost,
    FederatedContentCollector
)

__all__ = [
    'ActivityPubClient',
    'ActivityPubPost',
    'FederatedContentCollector'
]
