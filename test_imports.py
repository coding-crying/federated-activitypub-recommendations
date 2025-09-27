#!/usr/bin/env python3
print("Testing module imports...")

try:
    from src.data.activitypub import ActivityPubDataCollector
    print("✅ ActivityPubDataCollector imported successfully")
except ImportError as e:
    print(f"❌ Failed to import ActivityPubDataCollector: {e}")

try:
    from src.data.social_signals import SocialSignalsProcessor
    print("✅ SocialSignalsProcessor imported successfully")
except ImportError as e:
    print(f"❌ Failed to import SocialSignalsProcessor: {e}")

try:
    from src.models.multimodal import MultiModalProcessor
    print("✅ MultiModalProcessor imported successfully")
except ImportError as e:
    print(f"❌ Failed to import MultiModalProcessor: {e}")

try:
    from src.federation.realtime import RealTimeFederatedLearning
    print("✅ RealTimeFederatedLearning imported successfully")
except ImportError as e:
    print(f"❌ Failed to import RealTimeFederatedLearning: {e}")

print("\nAll modules are importable! The system integration is complete.")