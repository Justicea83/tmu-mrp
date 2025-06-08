"""
Base classes for the matching engine.
"""
import abc
import logging
from typing import Dict, Any, List, Optional

from core.models import ResumeJobMatch
from core.utils import log_message

# Define the segment key for storing matching engine results in extra_data
MATCHING_ENGINE_SEGMENT = 'matching_engine'


class BaseMatcher(abc.ABC):
    """
    Abstract base class for all matchers in the matching engine.
    """

    def __init__(self, match: ResumeJobMatch):
        """
        Initialize the matcher with a resume-job match.

        Args:
            match: The resume-job match to process.
        """
        self.match = match
        self.resume = match.resume
        self.job_description = match.job_description
        self.user = match.user
        self.extra_data = match.extra_data or {}

    @abc.abstractmethod
    def compute_score(self) -> float:
        """
        Compute a score for the match between the job application and the job post.

        Returns:
            A score between 0 and 100.
        """
        pass

    def get_score_details(self) -> Dict[str, Any]:
        """
        Get detailed information about the score computation.

        Returns:
            A dictionary with details about the score computation.
        """
        return {}

    def _convert_numpy_types(self, obj):
        """Recursively convert NumPy types to native Python types."""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(i) for i in obj]
        elif hasattr(obj, 'item'):  # Check for NumPy scalar types
            return obj.item()
        else:
            return obj

    def save_score(self, score: float, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Save the score and details to the job application's extra_data.

        Args:
            score: The score to save.
            details: Optional details about the score computation.
        """
        # Always work with the match's current extra_data to avoid race conditions
        extra_data = self.match.extra_data or {}
        if MATCHING_ENGINE_SEGMENT not in extra_data:
            extra_data[MATCHING_ENGINE_SEGMENT] = {}

        # Convert NumPy types to native Python types
        if details:
            details = self._convert_numpy_types(details)

        matcher_name = self.__class__.__name__.lower().replace('matcher', '')
        extra_data[MATCHING_ENGINE_SEGMENT][matcher_name] = {
            'score': score,
            'details': details or {}
        }

        # Update the match object's extra_data
        self.match.extra_data = extra_data
        self.match.save(update_fields=['extra_data'])
        log_message(logging.INFO, f"Saved {matcher_name} score: {score}", self.user)


class MatchingEngine:
    """
    Main class for the matching engine.
    """

    def __init__(self, match: ResumeJobMatch):
        """
        Initialize the matching engine with a resume-job match.

        Args:
            match: The resume-job match to process.
        """
        self.match = match
        self.matchers: List[BaseMatcher] = []

    def register_matcher(self, matcher: BaseMatcher) -> None:
        """
        Register a matcher with the matching engine.

        Args:
            matcher: The matcher to register.
        """
        self.matchers.append(matcher)

    def compute_scores(self) -> Dict[str, float]:
        """
        Compute scores for all registered matchers.

        Returns:
            A dictionary mapping matcher names to scores.
        """
        scores = {}
        for matcher in self.matchers:
            matcher_name = matcher.__class__.__name__.lower().replace('matcher', '')
            score = matcher.compute_score()
            scores[matcher_name] = score
        return scores

    def compute_overall_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Compute an overall score based on all matcher scores.

        Args:
            weights: Optional dictionary mapping matcher names to weights.
                    If not provided, equal weights will be used.

        Returns:
            An overall score between 0 and 100.
        """
        if not self.matchers:
            return 0.0

        # Get scores for all matchers
        scores = {}
        for matcher in self.matchers:
            matcher_name = matcher.__class__.__name__.lower().replace('matcher', '')
            scores[matcher_name] = matcher.compute_score()

        # If no weights are provided, use equal weights
        if not weights:
            total_score = sum(scores.values())
            return total_score / len(scores) if scores else 0.0

        # Apply custom weights
        weighted_scores = []
        total_weight = 0.0

        for matcher_name, score in scores.items():
            weight = weights.get(matcher_name, 1.0)  # Default weight is 1.0
            weighted_scores.append(score * weight)
            total_weight += weight

        # Avoid division by zero
        if total_weight == 0.0:
            return 0.0

        return sum(weighted_scores) / total_weight

    def run(self, compute_overall: bool = True, weights: Optional[Dict[str, float]] = None) -> None:
        """
        Run the matching engine, computing and saving all scores.

        Args:
            compute_overall: Whether to compute and save an overall score.
            weights: Optional dictionary mapping matcher names to weights for overall score calculation.
                    If not provided and compute_overall is True, equal weights will be used.
        """
        if not self.matchers:
            log_message(logging.WARNING, "No matchers registered with the matching engine", self.match.user)
            return

        # Compute and save individual scores
        matcher_scores = {}
        for matcher in self.matchers:
            score = matcher.compute_score()
            details = matcher.get_score_details()
            matcher.save_score(score, details)
            matcher_name = matcher.__class__.__name__.lower().replace('matcher', '')
            matcher_scores[matcher_name] = score

        # Get the match's extra_data (should already contain individual scores)
        extra_data = self.match.extra_data or {}
        if MATCHING_ENGINE_SEGMENT not in extra_data:
            extra_data[MATCHING_ENGINE_SEGMENT] = {}

        # Save matcher names for reference (preserve existing individual scores)
        extra_data[MATCHING_ENGINE_SEGMENT]['matchers'] = [m.__class__.__name__ for m in self.matchers]

        # Compute and save overall score if requested
        if compute_overall:
            overall_score = self.compute_overall_score(weights)

            extra_data[MATCHING_ENGINE_SEGMENT]['overall'] = {
                'score': overall_score,
                'details': {
                    'matchers': [m.__class__.__name__ for m in self.matchers],
                    'weights': weights or 'equal'
                }
            }

            log_message(logging.INFO, f"Computed overall matching score: {overall_score}", self.match.user)
        else:
            # If not computing overall score, just note that individual scores are available
            log_message(logging.INFO, "Individual matcher scores saved without computing overall score", self.match.user)

        # Update extra_data reference (individual scores should already be there)
        self.match.extra_data = extra_data
        self.match.save(update_fields=['extra_data'])
