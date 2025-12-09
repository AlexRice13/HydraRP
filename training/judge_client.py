"""Async OpenAI judge client singleton for HydraRP.

Provides a singleton AsyncOpenAI client configured from YAML settings.
"""

from openai import AsyncOpenAI
from typing import Optional
from training.config_loader import load_judge_config


# Global singleton instance
_judge_client: Optional[AsyncOpenAI] = None


def strip_trailing_chat_completions(endpoint: str) -> str:
    """Strip trailing /chat/completions from endpoint URL.
    
    Args:
        endpoint: API endpoint URL
        
    Returns:
        Cleaned endpoint URL
    """
    if endpoint.endswith('/chat/completions'):
        return endpoint[:-len('/chat/completions')]
    return endpoint


def get_judge_client() -> AsyncOpenAI:
    """Get or create the singleton AsyncOpenAI judge client.
    
    Returns:
        Configured AsyncOpenAI client instance
        
    Raises:
        ValueError: If judge configuration is missing or incomplete
    """
    global _judge_client
    
    if _judge_client is None:
        # Load configuration
        judge_config = load_judge_config()
        
        if not judge_config:
            raise ValueError(
                "Judge configuration not found. Please create config/judge_config.yaml "
                "from config/judge_config.example.yaml with your API credentials."
            )
        
        # Validate required fields
        required_fields = ['endpoint', 'api_key', 'model']
        missing_fields = [f for f in required_fields if f not in judge_config]
        if missing_fields:
            raise ValueError(
                f"Judge configuration missing required fields: {missing_fields}. "
                f"Please check config/judge_config.yaml"
            )
        
        # Extract and clean endpoint
        endpoint = strip_trailing_chat_completions(judge_config['endpoint'])
        api_key = judge_config['api_key']
        
        # Create client with retry configuration
        _judge_client = AsyncOpenAI(
            base_url=endpoint,
            api_key=api_key,
            max_retries=2,
        )
    
    return _judge_client


def reset_judge_client():
    """Reset the singleton client (useful for testing or reconfiguration)."""
    global _judge_client
    _judge_client = None
