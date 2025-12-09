"""Async reward functions for HydraRP GRPO pipeline.

Provides order-preserving async reward calculation with LLM-based judging.
"""

import asyncio
import re
from typing import List, Dict, Any, Optional

from training.config_loader import load_judge_config, load_prompt_config, load_reward_config, get_prompt
from training.reward_logger import get_reward_logger


async def call_judge_llm(
    query: str,
    response: str,
    reference: str,
    prompt_template: str,
    model: str,
    client,
    temperature: float = 0.0,
    max_tokens: int = 10,
    timeout: float = 30.0
) -> float:
    """Call the judge LLM to score a response asynchronously.
    
    Args:
        query: Input query/question
        response: Model's response to evaluate
        reference: Reference/gold answer
        prompt_template: Prompt template with {query}, {response}, {reference} placeholders
        model: Model name to use for judging
        client: AsyncOpenAI client instance
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        
    Returns:
        Numeric score from judge (0-10 scale)
    """
    try:
        # Format the prompt
        prompt = prompt_template.format(
            query=query,
            response=response,
            reference=reference
        )
        
        # Make async API call
        completion = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            timeout=timeout
        )
        
        # Extract score from response
        judge_output = completion.choices[0].message.content.strip()
        
        # Try to parse numeric score
        match = re.search(r'(\d+\.?\d*)', judge_output)
        if match:
            score = float(match.group(1))
            # Clip to 0-10 range
            return max(0.0, min(10.0, score))
        else:
            # Default to neutral score if parsing fails
            return 5.0
            
    except asyncio.TimeoutError:
        print(f"Warning: Judge LLM call timed out after {timeout}s")
        return 5.0
    except Exception as e:
        print(f"Warning: Judge LLM call failed: {e}")
        return 5.0


def calculate_length_penalty(response: str, min_length: int = 10, max_length: int = 2000) -> float:
    """Calculate penalty for response length.
    
    Args:
        response: Text response
        min_length: Minimum acceptable length
        max_length: Maximum acceptable length
        
    Returns:
        Penalty value (0 for acceptable length, negative for too short/long)
    """
    length = len(response)
    
    if length < min_length:
        # Penalize very short responses
        return -0.5 * (1.0 - length / min_length)
    elif length > max_length:
        # Penalize very long responses
        return -0.3 * (length - max_length) / max_length
    else:
        return 0.0


def calculate_repetition_penalty(response: str) -> float:
    """Calculate penalty for repetitive content.
    
    Args:
        response: Text response
        
    Returns:
        Penalty value (negative if repetitive)
    """
    words = response.lower().split()
    if len(words) < 10:
        return 0.0
    
    # Calculate unique word ratio
    unique_ratio = len(set(words)) / len(words)
    
    # Penalize if less than 60% unique words
    if unique_ratio < 0.6:
        return -0.5 * (0.6 - unique_ratio)
    
    return 0.0


def calculate_coherence_score(response: str) -> float:
    """Calculate a simple coherence score based on sentence structure.
    
    Args:
        response: Text response
        
    Returns:
        Coherence score (0-1)
    """
    # Simple heuristic: check for complete sentences
    sentences = re.split(r'[.!?]+', response)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return 0.0
    
    # Award points for having multiple sentences and reasonable length
    score = 0.0
    
    # Multiple sentences is good
    if len(sentences) > 1:
        score += 0.5
    
    # Reasonable sentence lengths
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
    if 5 <= avg_sentence_length <= 30:
        score += 0.5
    
    return score


async def legal_reward_fn_async(
    queries: List[str],
    responses: List[str],
    references: List[str],
    log_rewards: bool = True,
    prompt_type: str = "chat"  # "code", "chat", "multi_turn", "sac"
) -> List[float]:
    """Calculate rewards asynchronously with order preservation.
    
    This async function processes all inputs in parallel while maintaining
    input-output order correspondence using asyncio.gather().
    
    Args:
        queries: List of input queries
        responses: List of model responses
        references: List of reference answers
        log_rewards: Whether to log rewards to CSV
        prompt_type: Type of judge prompt to use
        
    Returns:
        List of reward scores in the same order as inputs
    """
    # Load configurations
    judge_config = load_judge_config()
    prompt_config = load_prompt_config()
    reward_config = load_reward_config()
    
    # Get judge settings
    model = judge_config.get('model', 'gpt-4')
    judge_params = reward_config.get('judge', {})
    temperature = judge_params.get('temperature', 0.0)
    max_tokens = judge_params.get('max_tokens', 10)
    timeout = judge_params.get('timeout', 30.0)
    
    # Get appropriate prompt
    prompt_key_map = {
        'code': 'code_judge_prompt',
        'chat': 'chat_judge_prompt',
        'multi_turn': 'multi_turn_judge_prompt',
        'sac': 'sac_judge_prompt'
    }
    prompt_key = prompt_key_map.get(prompt_type, 'chat_judge_prompt')
    prompt_template = get_prompt(prompt_config, prompt_key, "")
    
    # Get reward configuration
    weights = reward_config.get('weights', {})
    penalties = reward_config.get('penalties', {})
    thresholds = reward_config.get('thresholds', {})
    scaling = reward_config.get('scaling', {})
    
    # Extract parameters
    base_weight = weights.get('base_reward', 1.0)
    quality_weight = weights.get('quality_reward', 0.8)
    coherence_weight = weights.get('coherence_reward', 0.6)
    length_penalty_mult = penalties.get('length_penalty', 0.1)
    repetition_penalty_mult = penalties.get('repetition_penalty', 0.2)
    min_length = thresholds.get('min_length', 10)
    max_length = thresholds.get('max_length', 2000)
    reward_scale = scaling.get('reward_scale', 1.0)
    reward_clip_min = scaling.get('reward_clip_min', -10.0)
    reward_clip_max = scaling.get('reward_clip_max', 10.0)
    
    # Prepare async tasks for judge calls (if credentials available and prompt exists)
    judge_scores = []
    
    if judge_config and prompt_template:
        try:
            # Get judge client (may raise if config invalid)
            from training.judge_client import get_judge_client
            client = get_judge_client()
            
            # Create tasks for all judge calls - asyncio.gather preserves order
            judge_tasks = [
                call_judge_llm(
                    query=queries[i],
                    response=responses[i],
                    reference=references[i],
                    prompt_template=prompt_template,
                    model=model,
                    client=client,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout
                )
                for i in range(len(queries))
            ]
            
            # Execute all tasks in parallel, preserving order
            judge_scores = await asyncio.gather(*judge_tasks)
        except (ValueError, Exception) as e:
            # Judge client initialization failed, use default scores
            print(f"Warning: Judge client unavailable ({e}), using default scores")
            judge_scores = [5.0] * len(queries)
    else:
        # No judge available, use default scores
        judge_scores = [5.0] * len(queries)
    
    # Calculate reward components for each sample
    total_rewards = []
    reward_components_list = []
    
    for i in range(len(queries)):
        query = queries[i]
        response = responses[i]
        reference = references[i]
        judge_score = judge_scores[i]
        
        # Calculate individual components
        base_reward = (judge_score / 10.0) * base_weight
        quality_reward = (judge_score / 10.0) * quality_weight
        coherence_score = calculate_coherence_score(response)
        coherence_reward = coherence_score * coherence_weight
        length_penalty = calculate_length_penalty(response, min_length, max_length) * length_penalty_mult
        repetition_penalty = calculate_repetition_penalty(response) * repetition_penalty_mult
        
        # Aggregate reward
        total_reward = (
            base_reward +
            quality_reward +
            coherence_reward +
            length_penalty +
            repetition_penalty
        )
        
        # Apply scaling and clipping
        total_reward = total_reward * reward_scale
        total_reward = max(reward_clip_min, min(reward_clip_max, total_reward))
        
        total_rewards.append(total_reward)
        
        # Store components for logging
        reward_components = {
            'base_reward': base_reward,
            'quality_reward': quality_reward,
            'coherence_reward': coherence_reward,
            'length_penalty': length_penalty,
            'repetition_penalty': repetition_penalty,
        }
        reward_components_list.append(reward_components)
    
    # Log rewards if enabled
    if log_rewards:
        logger = get_reward_logger(enabled=True)
        logger.log_batch(
            queries=queries,
            responses=responses,
            references=references,
            judge_scores=judge_scores,
            reward_components_list=reward_components_list,
            total_rewards=total_rewards
        )
    
    return total_rewards


def legal_reward_fn(
    queries: List[str],
    responses: List[str],
    references: List[str],
    log_rewards: bool = True,
    prompt_type: str = "chat"
) -> List[float]:
    """Synchronous wrapper for async reward calculation.
    
    This function provides a synchronous interface to the async reward pipeline.
    It handles event loop management automatically.
    
    Args:
        queries: List of input queries
        responses: List of model responses
        references: List of reference answers
        log_rewards: Whether to log rewards to CSV
        prompt_type: Type of judge prompt to use
        
    Returns:
        List of reward scores in the same order as inputs
    """
    try:
        # Try to get current event loop
        asyncio.get_running_loop()
        # If we get here, we're in an async context
        # Caller should use legal_reward_fn_async directly
        raise RuntimeError(
            "legal_reward_fn called from within an async context. "
            "Use legal_reward_fn_async directly instead."
        )
    except RuntimeError as e:
        # Check if this is our own error or the "no running loop" error
        if "legal_reward_fn called from within" in str(e):
            # Re-raise our own error
            raise
        # Otherwise, no running loop, we can use asyncio.run
        return asyncio.run(
            legal_reward_fn_async(
                queries=queries,
                responses=responses,
                references=references,
                log_rewards=log_rewards,
                prompt_type=prompt_type
            )
        )
