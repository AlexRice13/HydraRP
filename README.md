# HydraRP - Async-First GRPO Reward Pipeline

An asynchronous, order-preserving reward pipeline for Group Relative Policy Optimization (GRPO) that integrates LLM-based judging with configurable reward functions.

## Features

- **Async-First Architecture**: Leverages Python's asyncio for efficient parallel reward computation
- **Order-Preserving**: Maintains input-output correspondence in batch reward calculations
- **YAML-Driven Configuration**: Secrets and prompts loaded from configuration files (never hardcoded)
- **Comprehensive Logging**: CSV-based reward logging for analysis and debugging
- **Flexible Reward Functions**: Configurable weights and penalties for different reward components

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Judge Credentials

Create `config/judge_config.yaml` from the example:

```bash
cp config/judge_config.example.yaml config/judge_config.yaml
```

Edit `config/judge_config.yaml` with your actual credentials:

```yaml
endpoint: "https://your-openai-endpoint.com"
api_key: "your-api-key-here"
model: "gpt-4"
```

**Note**: `config/judge_config.yaml` is gitignored to prevent credential leaks.

### 3. Configure Judge Prompts

Create `config/prompt_config.yaml` from the example:

```bash
cp config/prompt_config.example.yaml config/prompt_config.yaml
```

Edit the prompts as needed for your use case. Default prompts are provided in the example file.

### 4. Review Reward Configuration

The file `config/reward_config.yaml` contains hyperparameters and weights for reward calculation:
- Reward weights for different components
- Penalty multipliers
- Scoring thresholds

Adjust these values based on your training objectives.

## Usage

### Running the Example

```bash
# Run as a module (recommended)
python -m training.example_usage

# Or install the package and run
pip install -e .
python -m training.example_usage
```

This script demonstrates:
- Loading configurations from YAML files
- Initializing the reward pipeline
- Computing rewards for a small batch of mock data
- Handling missing credentials gracefully

### Using in Your Code

```python
from training.reward_functions import legal_reward_fn
from training.config_loader import load_judge_config, load_prompt_config

# Load configurations
judge_config = load_judge_config()
prompt_config = load_prompt_config()

# Prepare your data
queries = ["Your query here"]
responses = ["Your response here"]
references = ["Reference answer"]

# Compute rewards (order-preserving)
rewards = legal_reward_fn(
    queries=queries,
    responses=responses,
    references=references,
    log_rewards=True
)

print(f"Rewards: {rewards}")
```

## Configuration Files

### `config/judge_config.yaml` (Required, Gitignored)

Contains OpenAI-compatible API credentials:
- `endpoint`: API endpoint URL (trailing `/chat/completions` will be stripped automatically)
- `api_key`: Your API key
- `model`: Model name to use for judging

### `config/prompt_config.yaml` (Optional, Gitignored)

Contains judge prompts for different scenarios:
- `code_judge_prompt`: For evaluating code responses
- `chat_judge_prompt`: For evaluating chat responses
- `multi_turn_judge_prompt`: For multi-turn conversations
- `sac_judge_prompt`: For specialized SAC evaluation

### `config/reward_config.yaml` (Version Controlled)

Contains reward function hyperparameters:
- Weight coefficients for different reward components
- Penalty multipliers
- Thresholds and other scoring parameters

## Logging

Reward logs are automatically written to `./reward_logs/` as CSV files with timestamps. Each log entry includes:
- Queries, responses, and references
- Individual reward components
- Final aggregated rewards
- Metadata and timestamps

The log directory and CSV files are gitignored.

## Architecture

### Key Components

1. **`training/config_loader.py`**: YAML configuration loading with defaults and overrides
2. **`training/judge_client.py`**: Async OpenAI client singleton with retry logic
3. **`training/reward_functions.py`**: Async reward calculation with order preservation
4. **`training/reward_logger.py`**: CSV-based logging for reward tracking
5. **`training/example_usage.py`**: Example usage and integration test

### Async Design

The pipeline uses `asyncio.gather()` to parallelize LLM judge calls while preserving input order. The `legal_reward_fn` provides a synchronous interface that internally uses async execution.

## Environment Variable Overrides

You can override config file paths using environment variables:

```bash
export JUDGE_CONFIG_PATH=/path/to/custom/judge_config.yaml
export PROMPT_CONFIG_PATH=/path/to/custom/prompt_config.yaml
export REWARD_CONFIG_PATH=/path/to/custom/reward_config.yaml

python training/example_usage.py
```

## Development

### Running Tests

```bash
# Run example with mock data (no API calls if credentials missing)
python -m training.example_usage
```

### Project Structure

```
HydraRP/
├── config/
│   ├── judge_config.example.yaml    # Example credentials (version controlled)
│   ├── prompt_config.example.yaml   # Example prompts (version controlled)
│   └── reward_config.yaml           # Reward hyperparameters (version controlled)
├── training/
│   ├── __init__.py
│   ├── config_loader.py             # YAML configuration loading
│   ├── judge_client.py              # Async OpenAI client singleton
│   ├── reward_functions.py          # Async reward calculation
│   ├── reward_logger.py             # CSV logging utility
│   └── example_usage.py             # Example usage script
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package setup
├── LICENSE                           # MIT License
└── README.md                         # This file
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please ensure:
- All secrets remain in gitignored config files
- Async patterns are preserved
- Order preservation is maintained in batch operations
- Tests pass before submitting PRs
