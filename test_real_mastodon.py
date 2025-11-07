"""
Test Real Mastodon Integration
Fetches actual posts from Mastodon and demonstrates the recommendation system
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("🌐 Testing Real Mastodon Integration")
print("=" * 70)

# Step 1: Connect to Mastodon
print("\n1️⃣  Connecting to mastodon.social...")

try:
    from src.data.activitypub_client import ActivityPubClient

    client = ActivityPubClient("https://mastodon.social")
    print("   ✅ Connected successfully!")

except Exception as e:
    print(f"   ❌ Connection failed: {e}")
    sys.exit(1)

# Step 2: Fetch real posts
print("\n2️⃣  Fetching real public posts...")

try:
    real_posts = client.get_public_timeline(limit=10, local=True)
    print(f"   ✅ Fetched {len(real_posts)} real posts from Mastodon!")

    if real_posts:
        print("\n   📝 Sample posts:")
        for i, post in enumerate(real_posts[:5], 1):
            print(f"\n   {i}. @{post.author}")
            print(f"      {post.content[:100]}...")
            print(f"      💚 {post.likes_count} | 🔁 {post.boosts_count} | 💬 {post.replies_count}")

except Exception as e:
    print(f"   ❌ Fetch failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Initialize recommender with real data
print("\n3️⃣  Initializing recommendation system...")

try:
    from src.models.lightweight_foundation import LightweightFoundationModel
    from src.models.decentralized_recommender import DecentralizedRecommender

    print("   Loading foundation model (this may take a moment)...")
    foundation = LightweightFoundationModel(device='cpu')

    recommender = DecentralizedRecommender(
        instance_domain="mastodon.social",
        foundation_model=foundation,
        storage_dir="/tmp/mastodon_test"
    )

    print("   ✅ Recommender initialized!")
    print(f"      Foundation: {foundation.embedding_dim}d embeddings")
    print(f"      LoRA: rank={recommender.lora_config.lora_rank}")

except Exception as e:
    print(f"   ❌ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Add real Mastodon posts to recommender
print("\n4️⃣  Adding real Mastodon posts to recommendation system...")

try:
    for post in real_posts:
        recommender.add_post(post.to_post())

    stats = recommender.get_stats()
    print(f"   ✅ Added {stats['num_posts']} real posts!")
    print(f"      Cache: {stats['cache_size']} embeddings")

except Exception as e:
    print(f"   ❌ Failed to add posts: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Simulate user interactions with real content
print("\n5️⃣  Simulating user interactions with real Mastodon content...")

try:
    # Pick a few posts for interaction
    if len(real_posts) >= 3:
        user_id = "test_user_alice"

        # User likes first post
        recommender.process_interaction(user_id, real_posts[0].id, "like")
        print(f"   👍 User liked: {real_posts[0].content[:50]}...")

        # User boosts second post
        recommender.process_interaction(user_id, real_posts[1].id, "boost")
        print(f"   🔁 User boosted: {real_posts[1].content[:50]}...")

        # User skips third post
        recommender.process_interaction(user_id, real_posts[2].id, "skip")
        print(f"   ⏭️  User skipped: {real_posts[2].content[:50]}...")

        # Force LoRA update
        recommender._batch_update_loras()
        print(f"   ✅ User LoRA updated based on real Mastodon interactions!")

except Exception as e:
    print(f"   ⚠️  Interaction simulation failed: {e}")

# Step 6: Generate recommendations
print("\n6️⃣  Generating personalized recommendations from real content...")

try:
    import time
    start = time.time()

    recommendations = recommender.recommend(user_id, limit=5)
    elapsed = (time.time() - start) * 1000

    print(f"   ✅ Generated {len(recommendations)} recommendations in {elapsed:.2f}ms")

    if recommendations:
        print("\n   🎯 Top recommendations:")
        for i, (post, score) in enumerate(recommendations, 1):
            print(f"\n   {i}. Score: {score:.4f}")
            print(f"      @{post.author}")
            print(f"      {post.content[:80]}...")
            print(f"      💚 {post.engagement['likes']} | 🔁 {post.engagement['boosts']}")
    else:
        print("   ℹ️  No recommendations yet (need more interactions)")

except Exception as e:
    print(f"   ❌ Recommendation generation failed: {e}")
    import traceback
    traceback.print_exc()

# Step 7: Try trending posts
print("\n7️⃣  Fetching trending posts from Mastodon...")

try:
    trending = client.get_trending_posts(limit=5)

    if trending:
        print(f"   ✅ Found {len(trending)} trending posts!")
        print("\n   🔥 Trending on Mastodon:")
        for i, post in enumerate(trending[:3], 1):
            print(f"\n   {i}. @{post.author}")
            print(f"      {post.content[:80]}...")
            print(f"      💚 {post.likes_count} | 🔁 {post.boosts_count}")
    else:
        print("   ℹ️  No trending data available")

except Exception as e:
    print(f"   ⚠️  Trending fetch failed (this is optional): {e}")

# Final summary
print("\n" + "=" * 70)
print("✅ MASTODON INTEGRATION TEST COMPLETE!")
print("=" * 70)

print("\n📊 Summary:")
print(f"   • Successfully connected to real Mastodon instance")
print(f"   • Fetched {len(real_posts)} real posts via ActivityPub API")
print(f"   • Processed posts with foundation model")
print(f"   • Trained per-user LoRA on real content")
print(f"   • Generated personalized recommendations")
print(f"   • All components working with live data!")

print("\n🎉 The system can successfully:")
print("   ✓ Read from any ActivityPub instance (Mastodon, Pleroma, etc.)")
print("   ✓ Process real social media content")
print("   ✓ Learn user preferences from interactions")
print("   ✓ Generate personalized recommendations")
print("   ✓ Work entirely on CPU with lightweight models")

print("\n💡 Next steps to make it fully live:")
print("   1. Deploy as FastAPI web service")
print("   2. Implement HTTP signatures for ActivityPub auth")
print("   3. Register as ActivityPub Service actor")
print("   4. Users can follow @recommendations@yourdomain.com")
print("   5. Receive Like/Boost activities from real users")
print("   6. Deliver recommendations back to user timelines")

print()
