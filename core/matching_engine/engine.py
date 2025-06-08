"""
Main matching engine implementation.
"""
import logging
from typing import Dict, Any, Optional

from core.matching_engine.base import MatchingEngine
from core.matching_engine.experience import ExperienceMatcher
from core.matching_engine.general import GeneralMatcher
from core.matching_engine.location import LocationMatcher
from core.matching_engine.skills import SkillsMatcher
from core.models import ResumeJobMatch
from core.utils import log_message


def create_matching_engine(match: ResumeJobMatch, general_model: str = None, skills_model: str = None) -> MatchingEngine:
    """
    Create and configure a matching engine for a resume-job match.

    Args:
        match: The resume-job match to process.
        general_model: The sentence transformer model to use for general matching.
        skills_model: The sentence transformer model to use for skills matching.

    Returns:
        A configured matching engine.
    """
    engine = MatchingEngine(match)

    # Register matchers with specified models
    engine.register_matcher(GeneralMatcher(match, model_name=general_model))
    engine.register_matcher(ExperienceMatcher(match))
    engine.register_matcher(LocationMatcher(match))
    engine.register_matcher(SkillsMatcher(match, model_name=skills_model))

    return engine


def compute_resume_job_match(match: ResumeJobMatch, compute_overall: bool = True, 
                           weights: Optional[Dict[str, float]] = None,
                           general_model: str = None, skills_model: str = None) -> Dict[str, Any]:
    """
    Compute match scores for a resume-job match.

    Args:
        match: The resume-job match to process.
        compute_overall: Whether to compute and save an overall score.
        weights: Optional dictionary mapping matcher names to weights for overall score calculation.
                If not provided and compute_overall is True, equal weights will be used.
        general_model: The sentence transformer model to use for general matching.
        skills_model: The sentence transformer model to use for skills matching.

    Returns:
        A dictionary containing the match scores.
    """
    try:
        log_message(logging.INFO, f"Computing match scores for resume {match.resume.ID} vs job {match.job_description.Position}", match.user)

        # Create and run the matching engine
        engine = create_matching_engine(match, general_model=general_model, skills_model=skills_model)
        engine.run(compute_overall=compute_overall, weights=weights)

        # Get the scores from the match's extra_data
        extra_data = match.extra_data or {}
        match_data = extra_data.get('matching_engine', {})

        return match_data
    except Exception as e:
        log_message(logging.ERROR, f"Error computing match scores: {str(e)}", match.user, exception=e)
        return {}


# Keep old function for backward compatibility
def compute_application_match(application, compute_overall: bool = True, 
                           weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Legacy compatibility function"""
    return compute_resume_job_match(application, compute_overall, weights)
