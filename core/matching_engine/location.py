"""
Location matcher for the matching engine.
"""
import logging
from typing import Dict, Any, List

from sentence_transformers import SentenceTransformer

from core.matching_engine.base import BaseMatcher
from core.models import ResumeJobMatch
from core.utils import log_message, cosine_similarity, key_or_default
from core.preprocessors.download_sentence_transformer import MODEL_DIR


class LocationMatcher(BaseMatcher):
    """
    Matcher that computes a score based on the match between the job's location requirements
    and the applicant's location.
    """

    def __init__(self, match: ResumeJobMatch):
        """
        Initialize the location matcher.

        Args:
            match: The resume-job match to process.
        """
        super().__init__(match)
        self.job_locations = []
        self.applicant_location = {}
        self.location_details = {}
        self.model = SentenceTransformer(MODEL_DIR)  # Using the pre-downloaded model

    def extract_job_locations(self) -> List[Dict[str, Any]]:
        """
        Extract the job locations from the job post.

        Returns:
            A list of job locations.
        """
        job_locations = []

        # Extract location from job description
        description = self.job_description.Long_Description
        
        # Check for common location indicators in the description
        location_patterns = [
            "location:", "based in", "located in", "office in", "work from",
            "remote", "onsite", "hybrid", "telecommute"
        ]
        
        import re
        for pattern in location_patterns:
            matches = re.finditer(pattern + r"\s*([^.\n]*)", description, re.IGNORECASE)
            for match in matches:
                location_text = match.group(1).strip()
                if location_text:
                    job_locations.append({
                        'text': location_text,
                        'type': 'extracted'
                    })

        # If no locations found, use a default
        if not job_locations:
            job_locations.append({
                'text': 'Not specified',
                'type': 'default'
            })

        return job_locations

    def extract_applicant_location(self) -> Dict[str, Any]:
        """
        Extract the applicant's location from the resume and generated location data.

        Returns:
            A dictionary containing the applicant's location information.
        """
        applicant_location = {}

        # Get the generated location data from the resume
        geo_location = self.resume.get_location_data()
        if geo_location:
            applicant_location['geo'] = geo_location

        # Extract location from the parsed resume
        parsed_results = self.resume.parsed_data
        if parsed_results:
            basics = key_or_default(parsed_results, 'basics', {})
            location_data = key_or_default(basics, 'location', {})

            if location_data:
                applicant_location['resume'] = location_data

        return applicant_location

    def compute_location_match_score(self, job_locations: List[Dict[str, Any]],
                                     applicant_location: Dict[str, Any]) -> float:
        """
        Compute a score based on the match between job locations and applicant location.

        Args:
            job_locations: List of job locations.
            applicant_location: Applicant's location information.

        Returns:
            A score between 0 and 100.
        """
        if not job_locations or not applicant_location:
            log_message(logging.WARNING, "Missing location information for matching", self.user)
            return 50.0  # Default score when information is missing

        # Check for remote work
        print('locations')
        print(job_locations)
        is_remote_job = any(
            'remote' in loc['text'].lower() for loc in job_locations
            if isinstance(loc, dict) and 'text' in loc
        )

        if is_remote_job:
            return 100.0  # Perfect score for remote jobs

        # Get applicant's country and region
        applicant_country = None
        applicant_region = None

        if 'geo' in applicant_location:
            geo = applicant_location['geo']
            applicant_country = geo.get('country')
            applicant_region = geo.get('region')

        if not applicant_country and 'resume' in applicant_location:
            resume_loc = applicant_location['resume']
            applicant_country = resume_loc.get('countryCode')

            # Try to extract region from address
            address = resume_loc.get('address', '')
            if address and applicant_region is None:
                # Simple heuristic: last part of address might be region/state
                parts = [p.strip() for p in address.split(',')]
                if len(parts) > 1:
                    applicant_region = parts[-1]

        # If we still don't have location information, return a neutral score
        if not applicant_country and not applicant_region:
            return 50.0

        # Check for country and region matches in job locations
        best_score = 0.0

        for job_loc in job_locations:
            if not isinstance(job_loc, dict) or 'text' not in job_loc:
                continue

            job_loc_text = job_loc['text'].lower()

            # Country match
            if applicant_country and applicant_country.lower() in job_loc_text:
                # If we have an exact country match, score at least 80
                score = 80.0

                # Region match (bonus)
                if applicant_region and applicant_region.lower() in job_loc_text:
                    score = 100.0  # Perfect match

                best_score = max(best_score, score)

            # Region match without country match
            elif applicant_region and applicant_region.lower() in job_loc_text:
                score = 70.0  # Good match but not perfect
                best_score = max(best_score, score)

            # Partial text match
            elif applicant_region:
                # Check if any part of the applicant's region is in the job location
                region_parts = applicant_region.lower().split()
                for part in region_parts:
                    if len(part) > 3 and part in job_loc_text:  # Only consider substantial parts
                        score = 60.0  # Partial match
                        best_score = max(best_score, score)

        # If no match found but we have location information, use sentence transformers for semantic matching
        if best_score == 0.0 and (applicant_country or applicant_region):
            # Prepare location texts for semantic matching
            job_location_texts = [loc['text'] for loc in job_locations if isinstance(loc, dict) and 'text' in loc]

            if not job_location_texts:
                return 30.0  # Base score if no job location texts available

            # Prepare applicant location text
            applicant_location_parts = []
            if applicant_country:
                applicant_location_parts.append(applicant_country)
            if applicant_region:
                applicant_location_parts.append(applicant_region)

            applicant_location_text = " ".join(applicant_location_parts)

            # Encode location texts using the sentence transformer
            applicant_embedding = self.model.encode(applicant_location_text)
            job_embeddings = [self.model.encode(loc) for loc in job_location_texts]

            # Find the best semantic match
            best_similarity = 0.0
            for job_embedding in job_embeddings:
                similarity = cosine_similarity(applicant_embedding, job_embedding)
                best_similarity = max(best_similarity, similarity)

            # Convert similarity to a score between 30 and 80
            # A similarity of 0.7 or higher is considered good
            if best_similarity >= 0.7:
                best_score = 50.0 + (best_similarity * 30.0)  # Score between 50 and 80
            else:
                best_score = 30.0 + (best_similarity * 20.0)  # Score between 30 and 50

            log_message(logging.INFO, f"Semantic location match score: {best_score} (similarity: {best_similarity})",
                        self.user)

        return best_score

    def compute_score(self) -> float:
        """
        Compute a score based on the match between the job's location requirements
        and the applicant's location.

        Returns:
            A score between 0 and 100.
        """
        # Extract job locations
        self.job_locations = self.extract_job_locations()

        # Extract applicant location
        self.applicant_location = self.extract_applicant_location()

        # Store details for later retrieval
        self.location_details = {
            'job_locations': self.job_locations,
            'applicant_location': self.applicant_location,
            'semantic_matching_enabled': True  # Indicate that semantic matching is enabled
        }

        # Compute match score
        score = self.compute_location_match_score(self.job_locations, self.applicant_location)

        return score

    def get_score_details(self) -> Dict[str, Any]:
        """
        Get detailed information about the location score computation.

        Returns:
            A dictionary with details about the location score computation.
        """
        return self.location_details
