"""
Simple Mastodon API Test - No ML dependencies required
Just demonstrates that we can fetch real posts from Mastodon
"""

import json

print("🌐 Simple Mastodon Connection Test")
print("=" * 70)

# Try with requests library
try:
    import requests
    print("✅ requests library available")
except ImportError:
    print("❌ requests library not available, trying urllib...")
    import urllib.request
    import urllib.error

    class SimpleRequests:
        @staticmethod
        def get(url, headers=None, timeout=10):
            req = urllib.request.Request(url, headers=headers or {})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    data = json.loads(response.read().decode())
                    return type('Response', (), {'status_code': 200, 'json': lambda: data})()
            except Exception as e:
                return type('Response', (), {'status_code': 500, 'error': str(e)})()

    requests = SimpleRequests()

print("\n1️⃣  Connecting to mastodon.social API...")

# Mastodon public API endpoint
url = "https://mastodon.social/api/v1/timelines/public"
params = "?limit=5&local=true"

headers = {
    'User-Agent': 'FederatedRecommendations/1.0 (Test)'
}

print(f"   URL: {url}")
print(f"   Making request...")

try:
    response = requests.get(url + params, headers=headers, timeout=10)

    if response.status_code == 200:
        posts = response.json()

        print(f"\n✅ SUCCESS! Fetched {len(posts)} real posts from Mastodon!")
        print("\n📝 Real posts from mastodon.social:\n")

        for i, post in enumerate(posts, 1):
            # Parse the post data
            content = post.get('content', '')
            # Strip HTML tags
            import re
            content_clean = re.sub(r'<[^>]+>', '', content)

            account = post.get('account', {})
            author = account.get('acct', 'unknown')

            likes = post.get('favourites_count', 0)
            boosts = post.get('reblogs_count', 0)
            replies = post.get('replies_count', 0)

            print(f"{i}. @{author}")
            print(f"   {content_clean[:100]}...")
            print(f"   💚 {likes} likes | 🔁 {boosts} boosts | 💬 {replies} replies")
            print()

        print("=" * 70)
        print("✅ MASTODON CONNECTION SUCCESSFUL!")
        print("=" * 70)
        print("\nThis proves:")
        print("  ✓ Can connect to real Mastodon instances")
        print("  ✓ Can fetch real ActivityPub content")
        print("  ✓ Can parse post data (content, engagement, authors)")
        print("  ✓ Ready to integrate with recommendation system")
        print("\nOnce dependencies install, the full system will:")
        print("  ✓ Process these posts with foundation model")
        print("  ✓ Learn user preferences from interactions")
        print("  ✓ Generate personalized recommendations")

    else:
        print(f"❌ Request failed with status code: {response.status_code}")

except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\nThis might be a network issue. The API endpoint is valid.")

print()
