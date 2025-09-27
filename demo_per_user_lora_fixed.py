#!/usr/bin/env python3
\"\"\"
Demonstration of Per-User LoRA Architecture for ActivityPub Recommendations
This script shows how the system works with personal LoRA adapters for each user.
\"\"\"
import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.recommendation import LoRARecommendationModel, RecommendationTrainer


def demo_per_user_lora():
    \"\"\"Demonstrate per-user LoRA architecture.\"\"\"
    print(\"🚀 Per-User LoRA Architecture Demo\")
    print(\"=\" * 50)
    
    print(\"\\n1. 🌐 Global Foundation Model (Shared)\")
    print(\"   - Frozen base model that understands universal patterns\")
    print(\"   - Remains constant across all users and servers\")
    print(\"   - Updated via federated learning from all users\")
    
    print(\"\\n2. 👤 Per-User LoRA Adapters (Personal)\")
    print(\"   - Each user has their own adapter trained on their interactions\")
    print(\"   - Lightweight (~50KB) and portable between servers\") 
    print(\"   - Adapts global model to individual preferences\")
    
    # Create global model
    global_model = LoRARecommendationModel(
        foundation_model=None,  # Using dummy for demo
        embedding_dim=512,
        lora_rank=8,
        max_users=100
    )
    
    print(f\"\\n3. ✅ Global model created with {global_model.count_parameters():,} parameters\")
    
    # Simulate users with different preferences
    users = [
        {\"id\": 1, \"preferences\": \"tech, coding, AI, science\"},
        {\"id\": 2, \"preferences\": \"art, photography, design, creativity\"}, 
        {\"id\": 3, \"preferences\": \"music, concerts, bands, audio\"},
        {\"id\": 4, \"preferences\": \"sports, fitness, health, outdoor\"}
    ]
    
    print(f\"\\n4. 👥 Simulating {len(users)} users with unique preferences:\")
    for user in users:
        print(f\"   - User {user['id']}: {user['preferences']}\")
    
    print(f\"\\n5. 🔧 Per-User Parameter Management:\")
    
    # Show global parameters that are shared in federated learning
    global_params = global_model.get_lora_parameters()
    print(f\"   - Global params for federation: {list(global_params.keys())}\")
    print(f\"     (These improve for all users via federated learning)\")
    
    # Show user-specific parameters that stay local
    for user in users[:2]:  # Show first 2 as examples
        user_specific = global_model.get_user_specific_parameters(user['id'])
        print(f\"   - User {user['id']} local params: {list(user_specific.keys())}\")
        print(f\"     (Personal preferences stay with the user)\")
    
    print(f\"\\n6. 🔄 Federated Learning Process:\")
    print(\"   - Users interact with content on their local server\")
    print(\"   - Each user's LoRA adapts to their personal preferences\")
    print(\"   - Global params are shared with federated server (privacy preserved)\")
    print(\"   - Global model improves based on all users' collective learning\")
    print(\"   - Updated global model distributed back to all users/servers\")
    
    print(f\"\\n7. 🚀 Key Benefits:\")
    print(\"   ✅ True Personalization: Each user has individual preferences\")
    print(\"   ✅ Privacy: Personal data stays with user, not shared\")
    print(\"   ✅ Portability: Users take their LoRA with them between servers\")
    print(\"   ✅ Scalability: RTX 3090 can serve 20 users efficiently\")
    print(\"   ✅ Community: Users benefit from collective intelligence\")
    
    print(f\"\\n8. 🧠 Content Processing Example:\")
    
    # Simulate processing content with per-user adaptation
    device = torch.device('cpu')  # Using CPU for demo
    
    # Simulate content embeddings (e.g., from a social media post)
    content_tokens = {
        \"input_ids\": torch.randn(3, 50, 512)  # 3 example posts with 50 tokens of 512d
    }
    
    # Process for different users
    for user in users[:2]:
        user_ids = torch.tensor([user['id']] * 3)  # Same content for each user
        
        # Forward pass with user-specific adaptation
        with torch.no_grad():
            scores = global_model(content_tokens, user_ids)
        
        print(f\"   - User {user['id']}: Content scores = {scores.tolist()}\")
    
    print(f\"\\n🎉 Per-User LoRA Architecture Successfully Demonstrated!\")
    print(\"   Each user now has their own personal AI recommendation model\")
    print(\"   while contributing to and benefiting from global intelligence.\")


def explain_user_migration():
    \"\"\"Explain how users can migrate with their personal LoRA.\"\"\"
    print(f\"\\n\" + \"=\" * 50)
    print(\"🔄 User Migration with Personal LoRA\")
    print(\"=\" * 50)
    
    print(\"\"\"


    How users move between ActivityPub servers while keeping preferences:
    
    1. 🏠 User on Server A:
       - Personal LoRA adapter trained on their interactions
       - Local preferences stored and continuously updated
       - Can export their personal LoRA parameters
    
    2. ➡️  Migration Process:
       - Export personal LoRA: 50KB of parameters (portable!)
       - Join new Server B
       - Import personal LoRA to continue personalized experience
    
    3. 🏡 User on Server B:
       - Immediately gets personalized recommendations (not cold start)
       - New interactions further refine their personal LoRA
       - Still contributes to global model via federated learning
    
    4. 🌍 Global Continuity:
       - Global model continues learning from all users
       - User benefits from improvements made by other users
       - Privacy maintained throughout the process
    \"\"\")


if __name__ == \"__main__\":
    demo_per_user_lora()
    explain_user_migration()