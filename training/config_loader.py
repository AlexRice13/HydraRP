"""Configuration loader for HydraRP.

Loads YAML configuration files with support for defaults and environment variable overrides.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


def get_config_path(filename: str, env_var: str, default_dir: str = "config") -> Path:
    """Get configuration file path with environment variable override support.
    
    Args:
        filename: Default filename (e.g., 'judge_config.yaml')
        env_var: Environment variable name for override (e.g., 'JUDGE_CONFIG_PATH')
        default_dir: Default directory for config files
        
    Returns:
        Path object for the configuration file
    """
    if env_var in os.environ:
        return Path(os.environ[env_var])
    
    # Try to find the config directory relative to this file or the current working directory
    base_paths = [
        Path(__file__).parent.parent,  # Project root (when running from training/)
        Path.cwd(),  # Current working directory
    ]
    
    for base_path in base_paths:
        config_path = base_path / default_dir / filename
        if config_path.exists():
            return config_path
    
    # Return default path even if it doesn't exist (caller can handle)
    return Path.cwd() / default_dir / filename


def load_yaml_config(filepath: Path, required: bool = False) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        filepath: Path to YAML file
        required: If True, raise error if file doesn't exist; if False, return empty dict
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        FileNotFoundError: If required=True and file doesn't exist
    """
    if not filepath.exists():
        if required:
            raise FileNotFoundError(
                f"Required configuration file not found: {filepath}\n"
                f"Please create it from the example file or check the path."
            )
        return {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        return config if config is not None else {}


def load_judge_config() -> Dict[str, Any]:
    """Load judge configuration (API credentials and model settings).
    
    Returns:
        Dictionary with keys: endpoint, api_key, model
        Returns empty dict if file not found (not required for offline testing)
    """
    config_path = get_config_path('judge_config.yaml', 'JUDGE_CONFIG_PATH')
    return load_yaml_config(config_path, required=False)


def load_prompt_config() -> Dict[str, Any]:
    """Load prompt configuration for different judge scenarios.
    
    Returns:
        Dictionary with prompt templates for different use cases
        Returns empty dict if file not found
    """
    config_path = get_config_path('prompt_config.yaml', 'PROMPT_CONFIG_PATH')
    return load_yaml_config(config_path, required=False)


def load_reward_config() -> Dict[str, Any]:
    """Load reward function configuration (weights, penalties, thresholds).
    
    Returns:
        Dictionary with reward calculation parameters
        
    Raises:
        FileNotFoundError: If reward_config.yaml not found (this is required)
    """
    config_path = get_config_path('reward_config.yaml', 'REWARD_CONFIG_PATH')
    return load_yaml_config(config_path, required=True)


def get_prompt(prompt_config: Dict[str, Any], prompt_key: str, default: str = "") -> str:
    """Get a specific prompt from prompt configuration with fallback.
    
    Args:
        prompt_config: Loaded prompt configuration dictionary
        prompt_key: Key for the desired prompt (e.g., 'code_judge_prompt')
        default: Default value if prompt not found
        
    Returns:
        Prompt string
    """
    return prompt_config.get(prompt_key, default)
