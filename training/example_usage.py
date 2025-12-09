"""Example usage of HydraRP async reward pipeline.

Demonstrates:
- Loading configurations
- Computing rewards on mock data
- Handling missing credentials gracefully
- Order preservation in batch processing
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.reward_functions import legal_reward_fn
from training.config_loader import load_judge_config, load_prompt_config, load_reward_config


def main():
    """Run example reward pipeline."""
    
    print("=" * 70)
    print("HydraRP Async Reward Pipeline - Example Usage")
    print("=" * 70)
    print()
    
    # Load configurations
    print("Loading configurations...")
    
    try:
        reward_config = load_reward_config()
        print("✓ Loaded reward_config.yaml")
    except Exception as e:
        print(f"✗ Failed to load reward_config.yaml: {e}")
        return
    
    prompt_config = load_prompt_config()
    if prompt_config:
        print(f"✓ Loaded prompt_config.yaml ({len(prompt_config)} prompts)")
    else:
        print("⚠ No prompt_config.yaml found (will use defaults)")
    
    judge_config = load_judge_config()
    if judge_config:
        print(f"✓ Loaded judge_config.yaml (model: {judge_config.get('model', 'N/A')})")
        has_credentials = True
    else:
        print("⚠ No judge_config.yaml found (will skip external API calls)")
        has_credentials = False
    
    print()
    
    # Prepare mock data
    print("Preparing mock data...")
    queries = [
        "What is the capital of France?",
        "Explain how photosynthesis works.",
        "Write a Python function to calculate factorial.",
    ]
    
    responses = [
        "The capital of France is Paris, a beautiful city known for art and culture.",
        "Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll.",
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
    ]
    
    references = [
        "Paris",
        "Photosynthesis is the process by which green plants use sunlight to synthesize foods.",
        "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)",
    ]
    
    print(f"✓ Created {len(queries)} query-response pairs")
    print()
    
    # Display configuration info
    print("Reward Configuration:")
    weights = reward_config.get('weights', {})
    print(f"  - Base reward weight: {weights.get('base_reward', 'N/A')}")
    print(f"  - Quality reward weight: {weights.get('quality_reward', 'N/A')}")
    print(f"  - Coherence reward weight: {weights.get('coherence_reward', 'N/A')}")
    
    scaling = reward_config.get('scaling', {})
    print(f"  - Reward scale: {scaling.get('reward_scale', 'N/A')}")
    print(f"  - Clip range: [{scaling.get('reward_clip_min', 'N/A')}, {scaling.get('reward_clip_max', 'N/A')}]")
    print()
    
    # Compute rewards
    print("Computing rewards...")
    print()
    
    if not has_credentials:
        print("⚠ WARNING: No judge credentials found.")
        print("  The pipeline will run without external LLM judge calls.")
        print("  To enable judge scoring, create config/judge_config.yaml")
        print("  from config/judge_config.example.yaml with your API credentials.")
        print()
    
    try:
        # Call reward function - this uses async internally but provides sync interface
        rewards = legal_reward_fn(
            queries=queries,
            responses=responses,
            references=references,
            log_rewards=True,  # Enable CSV logging
            prompt_type="chat"  # Use chat judge prompt
        )
        
        print("✓ Rewards computed successfully!")
        print()
        
        # Display results
        print("Results (order preserved):")
        print("-" * 70)
        for i, (query, response, reward) in enumerate(zip(queries, responses, rewards)):
            print(f"\n[{i+1}] Query: {query[:50]}...")
            print(f"    Response: {response[:50]}...")
            print(f"    Reward: {reward:.4f}")
        
        print()
        print("-" * 70)
        print(f"\nReward Statistics:")
        print(f"  Mean: {sum(rewards) / len(rewards):.4f}")
        print(f"  Min:  {min(rewards):.4f}")
        print(f"  Max:  {max(rewards):.4f}")
        print()
        
        print("✓ Logs written to ./reward_logs/")
        print()
        
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        print()
        return
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return
    
    # Success message
    print("=" * 70)
    print("Example completed successfully!")
    print()
    print("Next steps:")
    print("  1. Check ./reward_logs/ for CSV log files")
    print("  2. Create config/judge_config.yaml to enable LLM judging")
    print("  3. Customize config/prompt_config.yaml for your use case")
    print("  4. Adjust config/reward_config.yaml weights and penalties")
    print("=" * 70)


if __name__ == "__main__":
    main()
