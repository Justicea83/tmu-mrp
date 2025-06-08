"""
Education Matcher for Resume-Job Matching System

This module implements education-based matching between resumes and job descriptions,
analyzing degree levels, fields of study, and educational requirements.
"""

import json
import re
from typing import Dict, List, Any, Optional
from .base import BaseMatcher
from ..models import ResumeJobMatch


class EducationMatcher(BaseMatcher):
    """
    Matches resumes to jobs based on educational background.
    
    Considers:
    - Degree level alignment (Associate, Bachelor's, Master's, PhD)
    - Field of study relevance to job requirements
    - Educational institution quality (basic heuristics)
    - Professional certifications
    """

    def __init__(self, match: ResumeJobMatch):
        super().__init__(match)
        
        # Degree level hierarchy (higher values = higher degrees)
        self.degree_levels = {
            'phd': 5, 'doctorate': 5, 'doctoral': 5,
            'master': 4, 'masters': 4, 'mba': 4, 'ms': 4, 'ma': 4,
            'bachelor': 3, 'bachelors': 3, 'bs': 3, 'ba': 3,
            'associate': 2, 'associates': 2,
            'diploma': 1, 'certificate': 1, 'certification': 1
        }
        
        # Field mappings for relevance scoring
        self.field_mappings = {
            'technology': ['computer science', 'information technology', 'software engineering', 
                          'data science', 'cybersecurity', 'information systems'],
            'business': ['business administration', 'management', 'marketing', 'finance', 
                        'economics', 'accounting'],
            'engineering': ['engineering', 'mechanical', 'electrical', 'civil', 'industrial'],
            'science': ['mathematics', 'statistics', 'physics', 'chemistry', 'biology'],
            'healthcare': ['medicine', 'nursing', 'healthcare', 'medical', 'health'],
            'education': ['education', 'teaching', 'pedagogy', 'curriculum']
        }
        
        # Common degree abbreviations
        self.degree_patterns = [
            r'\b(ph\.?d|doctorate|doctoral)\b',
            r'\b(m\.?s\.?|master|mba|m\.?a\.?)\b',
            r'\b(b\.?s\.?|b\.?a\.?|bachelor)\b',
            r'\b(associate|a\.?s\.?|a\.?a\.?)\b',
            r'\b(diploma|certificate|cert\.?)\b'
        ]

    def compute_score(self) -> float:
        """
        Compute education matching score between resume and job.
        
        Returns:
            float: Education matching score (0.0 to 1.0)
        """
        try:
            # Convert resume and job to dict format for processing
            resume_dict = self.resume.__dict__
            job_dict = self.job_description.__dict__
            
            # Parse resume education data
            resume_education = self._extract_resume_education(resume_dict)
            
            # Extract job education requirements
            job_requirements = self._extract_job_education_requirements(job_dict)
            
            if not resume_education and not job_requirements:
                return 0.5  # Neutral score when no education data available
            
            if not resume_education:
                return 0.3  # Lower score if resume has no education data
                
            # Compute various education match components
            degree_score = self._compute_degree_level_match(resume_education, job_requirements)
            field_score = self._compute_field_relevance(resume_education, job_dict)
            experience_education_alignment = self._compute_experience_education_alignment(resume_dict, job_dict)
            
            # Weighted combination
            final_score = (
                degree_score * 0.4 +
                field_score * 0.4 +
                experience_education_alignment * 0.2
            )
            
            return min(final_score, 1.0)
            
        except Exception as e:
            # Use logging from base class
            import logging
            logging.getLogger(__name__).warning(f"Error computing education score: {e}")
            return 0.5

    def _extract_resume_education(self, resume: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract education information from resume parsed JSON."""
        try:
            if not isinstance(resume.get('parsed_json'), str):
                return []
                
            parsed_data = json.loads(resume['parsed_json'])
            education_data = parsed_data.get('education', [])
            
            if not isinstance(education_data, list):
                return []
                
            return education_data
            
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    def _extract_job_education_requirements(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Extract education requirements from job description."""
        job_text = (job.get('Long Description', '') + ' ' + 
                   job.get('Position', '')).lower()
        
        # Look for degree requirements in job description
        required_degrees = []
        for pattern in self.degree_patterns:
            matches = re.findall(pattern, job_text, re.IGNORECASE)
            required_degrees.extend(matches)
        
        # Look for field requirements
        required_fields = []
        for field_category, keywords in self.field_mappings.items():
            for keyword in keywords:
                if keyword.lower() in job_text:
                    required_fields.append(field_category)
                    break
        
        return {
            'required_degrees': required_degrees,
            'required_fields': required_fields,
            'years_required': self._extract_years_required(job)
        }

    def _extract_years_required(self, job: Dict[str, Any]) -> Optional[int]:
        """Extract years of experience required from job."""
        exp_years = job.get('Exp Years', '')
        if isinstance(exp_years, str) and exp_years:
            # Extract number from strings like "2y", "5 years", etc.
            match = re.search(r'(\d+)', exp_years)
            if match:
                return int(match.group(1))
        return None

    def _compute_degree_level_match(self, resume_education: List[Dict], 
                                   job_requirements: Dict) -> float:
        """Compute how well resume degree levels match job requirements."""
        if not resume_education:
            return 0.3
            
        # Get highest degree level from resume
        resume_max_level = 0
        for edu in resume_education:
            study_type = edu.get('studyType', '').lower()
            for degree_name, level in self.degree_levels.items():
                if degree_name in study_type:
                    resume_max_level = max(resume_max_level, level)
                    break
        
        # Get required degree level from job
        required_degrees = job_requirements.get('required_degrees', [])
        if not required_degrees:
            return 0.7  # No specific requirements, good match
            
        required_max_level = 0
        for degree in required_degrees:
            degree_lower = degree.lower()
            for degree_name, level in self.degree_levels.items():
                if degree_name in degree_lower:
                    required_max_level = max(required_max_level, level)
                    break
        
        if required_max_level == 0:
            return 0.7  # No clear requirement found
            
        # Score based on how resume degree compares to requirement
        if resume_max_level >= required_max_level:
            return 1.0  # Meets or exceeds requirement
        elif resume_max_level == required_max_level - 1:
            return 0.8  # Close to requirement
        elif resume_max_level > 0:
            return 0.6  # Has some education but below requirement
        else:
            return 0.3  # No relevant degree found

    def _compute_field_relevance(self, resume_education: List[Dict], job: Dict) -> float:
        """Compute relevance of education field to job requirements."""
        if not resume_education:
            return 0.3
            
        # Extract fields from resume education
        resume_fields = []
        for edu in resume_education:
            area = edu.get('area', '').lower()
            if area:
                resume_fields.append(area)
        
        if not resume_fields:
            return 0.4
            
        # Get job field requirements
        job_text = (job.get('Long Description', '') + ' ' + 
                   job.get('Position', '') + ' ' +
                   job.get('Primary Keyword', '')).lower()
        
        # Score field relevance
        max_relevance = 0.0
        
        for resume_field in resume_fields:
            for field_category, keywords in self.field_mappings.items():
                # Check if resume field matches job field category
                field_match_score = 0.0
                
                # Direct keyword match in resume field
                for keyword in keywords:
                    if keyword in resume_field:
                        field_match_score = max(field_match_score, 1.0)
                        break
                
                # Check if job requires this field category
                job_needs_field = any(keyword in job_text for keyword in keywords)
                
                if job_needs_field and field_match_score > 0:
                    max_relevance = max(max_relevance, 1.0)
                elif job_needs_field:
                    # Job needs this field but resume doesn't have exact match
                    # Check for partial matches
                    for keyword in keywords:
                        if any(word in resume_field for word in keyword.split()):
                            max_relevance = max(max_relevance, 0.6)
        
        return max_relevance if max_relevance > 0 else 0.4

    def _compute_experience_education_alignment(self, resume: Dict, job: Dict) -> float:
        """Compute how well education aligns with job experience requirements."""
        try:
            # Parse resume work experience
            parsed_data = json.loads(resume.get('parsed_json', '{}'))
            work_experience = parsed_data.get('work', [])
            
            if not isinstance(work_experience, list) or not work_experience:
                return 0.5
                
            # Calculate years of experience from resume
            total_experience_years = 0
            for work in work_experience:
                start_date = work.get('startDate')
                end_date = work.get('endDate')
                
                if start_date:
                    # Simple calculation - extract years
                    start_year = self._extract_year(start_date)
                    end_year = self._extract_year(end_date) if end_date else 2024
                    
                    if start_year and end_year:
                        total_experience_years += max(0, end_year - start_year)
            
            # Get job experience requirements
            required_years = self._extract_years_required(job)
            
            if required_years is None:
                return 0.7  # No specific requirement
                
            # Score based on experience vs education level
            resume_education = self._extract_resume_education(resume)
            if not resume_education:
                return 0.5
                
            # Higher education can compensate for less experience
            max_degree_level = 0
            for edu in resume_education:
                study_type = edu.get('studyType', '').lower()
                for degree_name, level in self.degree_levels.items():
                    if degree_name in study_type:
                        max_degree_level = max(max_degree_level, level)
            
            # Adjusted experience based on education
            adjusted_experience = total_experience_years
            if max_degree_level >= 4:  # Master's or higher
                adjusted_experience += 2
            elif max_degree_level >= 3:  # Bachelor's
                adjusted_experience += 1
            
            # Score alignment
            if adjusted_experience >= required_years:
                return 1.0
            elif adjusted_experience >= required_years * 0.8:
                return 0.8
            elif adjusted_experience >= required_years * 0.6:
                return 0.6
            else:
                return 0.4
                
        except Exception as e:
            self.logger.warning(f"Error computing experience-education alignment: {e}")
            return 0.5

    def _extract_year(self, date_string: str) -> Optional[int]:
        """Extract year from date string."""
        if not date_string:
            return None
            
        # Look for 4-digit year
        match = re.search(r'(\d{4})', str(date_string))
        if match:
            return int(match.group(1))
            
        return None

    def get_explanation(self, resume: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed explanation of education matching."""
        resume_education = self._extract_resume_education(resume)
        job_requirements = self._extract_job_education_requirements(job)
        
        explanation = {
            'resume_education': resume_education,
            'job_requirements': job_requirements,
            'degree_match': self._compute_degree_level_match(resume_education, job_requirements),
            'field_relevance': self._compute_field_relevance(resume_education, job),
            'experience_alignment': self._compute_experience_education_alignment(resume, job)
        }
        
        return explanation 