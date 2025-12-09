"""HydraRP Training Package - Async-first GRPO reward pipeline."""

__version__ = "0.1.0"

from training.reward_functions import legal_reward_fn

__all__ = ["legal_reward_fn"]
