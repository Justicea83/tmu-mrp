"""
Experience matcher for the matching engine.
"""
import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple

from core.matching_engine.base import BaseMatcher
from core.models import ResumeJobMatch
from core.utils import log_message, APPLICATION_SEGMENT_PARSED_RESULTS, key_or_default


class ExperienceMatcher(BaseMatcher):
    """
    Matcher that computes a score based on the match between the job's experience requirements
    and the applicant's experience.
    """

    def __init__(self, match: ResumeJobMatch):
        """
        Initialize the experience matcher.

        Args:
            match: The resume-job match to process.
        """
        super().__init__(match)
        self.required_experience_months = 0
        self.applicant_experience_months = 0
        self.experience_details = {}

    def extract_required_experience(self) -> int:
        """
        Extract the required experience in months from the job description.

        Returns:
            The required experience in months.
        """
        # Get the required experience from the job description
        required_years = self.job_description.required_experience_years
        return int(required_years * 12)  # Convert years to months

    def extract_applicant_experience(self) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Extract the applicant's experience in months from the resume.

        Returns:
            A tuple containing:
            - The total experience in months
            - A list of work experiences with details
        """
        parsed_results = self.resume.parsed_data
        if not parsed_results:
            log_message(logging.WARNING, "No parsed resume results found", self.user)
            return 0, []

        work_experiences = []
        total_months = 0

        # Extract work experience from the parsed resume
        work = key_or_default(parsed_results, 'work', [])

        for job in work:
            start_date = key_or_default(job, 'startDate')
            end_date = key_or_default(job, 'endDate')
            position = key_or_default(job, 'position', '')
            company = key_or_default(job, 'name', '')

            if not start_date:
                continue

            try:
                start_year, start_month = self._parse_date(start_date)

                if end_date and end_date.lower() != 'present' and end_date != 'null' and end_date is not None:
                    end_year, end_month = self._parse_date(end_date)
                else:
                    # If end_date is not provided or is 'present', use current date
                    now = datetime.now()
                    end_year, end_month = now.year, now.month

                # Calculate duration in months
                duration_months = (end_year - start_year) * 12 + (end_month - start_month)
                if duration_months < 0:
                    duration_months = 0

                total_months += duration_months

                work_experiences.append({
                    'company': company,
                    'position': position,
                    'start_date': start_date,
                    'end_date': end_date if end_date else 'Present',
                    'duration_months': duration_months
                })

            except (ValueError, TypeError) as e:
                log_message(logging.WARNING, f"Error parsing work experience dates: {e}", self.user)
                continue

        return total_months, work_experiences

    def _parse_date(self, date_str: str) -> Tuple[int, int]:
        """
        Parse a date string into year and month.

        Args:
            date_str: The date string to parse.

        Returns:
            A tuple containing the year and month.
        """
        # Try different date formats
        formats = ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m', '%Y/%m', '%m/%Y', '%Y']

        for fmt in formats:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.year, date_obj.month
            except ValueError:
                continue

        # If all formats fail, try to extract year and month using regex
        year_match = re.search(r'(\d{4})', date_str)
        if year_match:
            year = int(year_match.group(1))
            # Try to find month
            month_match = re.search(r'(\d{1,2})[/-]', date_str)
            month = int(month_match.group(1)) if month_match else 1
            return year, month

        raise ValueError(f"Could not parse date: {date_str}")

    def compute_score(self) -> float:
        """
        Compute a score based on the match between the job's experience requirements
        and the applicant's experience.

        Returns:
            A score between 0 and 100.
        """
        # Extract required experience from job post
        self.required_experience_months = self.extract_required_experience()

        # Extract applicant's experience from resume
        self.applicant_experience_months, work_experiences = self.extract_applicant_experience()

        # Store details for later retrieval
        self.experience_details = {
            'required_experience_months': self.required_experience_months,
            'applicant_experience_months': self.applicant_experience_months,
            'work_experiences': work_experiences
        }

        # If no experience is required, return a perfect score
        if self.required_experience_months == 0:
            return 100.0

        # Calculate the ratio of applicant's experience to required experience
        ratio = self.applicant_experience_months / self.required_experience_months

        # Score calculation:
        # - If applicant has exactly the required experience, score is 80
        # - If applicant has more experience, score increases up to 100
        # - If applicant has less experience, score decreases proportionally
        if ratio >= 1.0:
            # More experience than required
            additional_score = min(20, (ratio - 1.0) * 40)  # Up to 20 additional points
            score = 80.0 + additional_score
        else:
            # Less experience than required
            score = 80.0 * ratio

        return min(100.0, max(0.0, score))

    def get_score_details(self) -> Dict[str, Any]:
        """
        Get detailed information about the experience score computation.

        Returns:
            A dictionary with details about the experience score computation.
        """
        return self.experience_details
