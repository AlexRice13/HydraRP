"""Reward logging utility for HydraRP.

Logs reward calculations to CSV files for analysis and debugging.
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


class RewardLogger:
    """CSV-based logger for reward calculations."""
    
    def __init__(self, log_dir: str = "reward_logs", enabled: bool = True):
        """Initialize reward logger.
        
        Args:
            log_dir: Directory to store log files
            enabled: Whether logging is enabled
        """
        self.log_dir = Path(log_dir)
        self.enabled = enabled
        self.current_log_file: Optional[Path] = None
        
        if self.enabled:
            # Create log directory if it doesn't exist
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create new log file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_log_file = self.log_dir / f"rewards_{timestamp}.csv"
            
            # Initialize CSV with headers
            self._write_header()
    
    def _write_header(self):
        """Write CSV header row."""
        if not self.enabled or self.current_log_file is None:
            return
        
        headers = [
            'timestamp',
            'query',
            'response',
            'reference',
            'judge_score',
            'base_reward',
            'quality_reward',
            'coherence_reward',
            'length_penalty',
            'repetition_penalty',
            'total_reward',
            'metadata'
        ]
        
        with open(self.current_log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_reward(
        self,
        query: str,
        response: str,
        reference: str,
        judge_score: float,
        reward_components: Dict[str, float],
        total_reward: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a single reward calculation.
        
        Args:
            query: Input query
            response: Model response
            reference: Reference answer
            judge_score: Score from judge LLM
            reward_components: Dictionary of individual reward components
            total_reward: Final aggregated reward
            metadata: Optional additional metadata
        """
        if not self.enabled or self.current_log_file is None:
            return
        
        timestamp = datetime.now().isoformat()
        
        # Extract components with defaults
        row = [
            timestamp,
            query,
            response,
            reference,
            judge_score,
            reward_components.get('base_reward', 0.0),
            reward_components.get('quality_reward', 0.0),
            reward_components.get('coherence_reward', 0.0),
            reward_components.get('length_penalty', 0.0),
            reward_components.get('repetition_penalty', 0.0),
            total_reward,
            str(metadata) if metadata else ''
        ]
        
        with open(self.current_log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def log_batch(
        self,
        queries: List[str],
        responses: List[str],
        references: List[str],
        judge_scores: List[float],
        reward_components_list: List[Dict[str, float]],
        total_rewards: List[float],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ):
        """Log a batch of reward calculations.
        
        Args:
            queries: List of input queries
            responses: List of model responses
            references: List of reference answers
            judge_scores: List of scores from judge LLM
            reward_components_list: List of reward component dictionaries
            total_rewards: List of final aggregated rewards
            metadata_list: Optional list of metadata dictionaries
        """
        if not self.enabled:
            return
        
        if metadata_list is None:
            metadata_list = [None] * len(queries)
        
        for i in range(len(queries)):
            self.log_reward(
                query=queries[i],
                response=responses[i],
                reference=references[i],
                judge_score=judge_scores[i],
                reward_components=reward_components_list[i],
                total_reward=total_rewards[i],
                metadata=metadata_list[i]
            )


# Global logger instance
_global_logger: Optional[RewardLogger] = None


def get_reward_logger(log_dir: str = "reward_logs", enabled: bool = True) -> RewardLogger:
    """Get or create the global reward logger instance.
    
    Args:
        log_dir: Directory to store log files
        enabled: Whether logging is enabled
        
    Returns:
        RewardLogger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = RewardLogger(log_dir=log_dir, enabled=enabled)
    
    return _global_logger


def reset_reward_logger():
    """Reset the global logger instance (useful for testing)."""
    global _global_logger
    _global_logger = None
