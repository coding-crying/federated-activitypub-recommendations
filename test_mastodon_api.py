"""
Test Mastodon API with proper headers
"""

import requests
import json

print("🌐 Testing Mastodon API Connection")
print("=" * 70)

# Try multiple instances
instances = [
    "https://fosstodon.org",
    "https://mastodon.social",
    "https://techhub.social"
]

for instance_url in instances:
    print(f"\n📡 Trying {instance_url}...")

    # Try instance info first (usually public)
    try:
        url = f"{instance_url}/api/v1/instance"
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; FederatedRecommendations/1.0)',
            'Accept': 'application/json'
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Connected to {instance_url}!")
            print(f"   Instance: {data.get('title', 'Unknown')}")
            print(f"   Description: {data.get('description', 'N/A')[:80]}...")
            print(f"   Users: {data.get('stats', {}).get('user_count', 'N/A')}")
            print(f"   Posts: {data.get('stats', {}).get('status_count', 'N/A')}")

            # Now try to get public timeline
            timeline_url = f"{instance_url}/api/v1/timelines/public?limit=3"
            timeline_response = requests.get(timeline_url, headers=headers, timeout=10)

            if timeline_response.status_code == 200:
                posts = timeline_response.json()
                print(f"\n   📝 Fetched {len(posts)} posts:")

                for i, post in enumerate(posts[:3], 1):
                    import re
                    content = re.sub(r'<[^>]+>', '', post.get('content', ''))
                    author = post.get('account', {}).get('acct', 'unknown')
                    print(f"\n   {i}. @{author}")
                    print(f"      {content[:80]}...")

                print(f"\n   🎉 SUCCESS! Can fetch real Mastodon posts!")
                break
            else:
                print(f"   ⚠️  Timeline returned: {timeline_response.status_code}")

        else:
            print(f"   ❌ Status: {response.status_code}")

    except requests.exceptions.Timeout:
        print(f"   ⏱️  Timeout - instance might be slow")
    except requests.exceptions.ConnectionError:
        print(f"   🔌 Connection error - instance might be down")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print("\n💡 Note: Some instances may block requests depending on their")
print("   rate limiting policies. In production, you would:")
print("   • Use proper OAuth authentication")
print("   • Respect rate limits (typically 300 requests / 5 minutes)")
print("   • Use application tokens")
print()
