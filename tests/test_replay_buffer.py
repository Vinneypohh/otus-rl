"""Test script for ReplayBuffer implementation.

This script tests the functionality of the replay buffer including
capacity limits, sampling, and randomness.
"""
from lunar_lander.replay_buffer import ReplayBuffer


def test_replay_buffer() -> None:
    """Test replay buffer functionality."""
    # Create a buffer with capacity of 5 (small size to test overflow)
    buffer = ReplayBuffer(capacity=5)
    print(f"1. Buffer created. Size: {len(buffer)}")

    # Fill buffer with data
    print("\n2. Adding data...")
    for i in range(1, 8):  # Add 7 elements (1, 2, ..., 7)
        state = [i, i]  # Fake state
        action = i  # Fake action
        reward = i * 10  # Fake reward
        next_state = [i + 1, i + 1]
        done = False
        
        print(f"   Adding experience #{i} -> Action {action}")
        buffer.push(state, action, reward, next_state, done)

    # Check size (should be 5, not 7, since capacity=5)
    print(f"\n3. Current buffer size: {len(buffer)} (Expected 5)")
    if len(buffer) == 5:
        print("✅ Overflow test passed! (Old elements 1 and 2 were removed)")
    else:
        print("❌ Error: buffer did not remove old elements")

    # Try to sample a batch
    batch_size = 3
    print(f"\n4. Sampling random batch of {batch_size} elements...")
    states, actions, rewards, next_s, dones = buffer.sample(batch_size)

    print(f"   Retrieved Actions: {actions.tolist()}")
    print(f"   Actions data type: {type(actions)}")  # Should be torch.Tensor

    # Check randomness (sample again)
    states2, actions2, _, _, _ = buffer.sample(batch_size)
    print(f"   Second sample Actions: {actions2.tolist()}")

    if actions.tolist() != actions2.tolist():
        print("✅ Randomness test passed! (Batches are different)")
    else:
        print("⚠️ Batches matched (rare occurrence, or randomness issue)")


if __name__ == "__main__":
    test_replay_buffer()
