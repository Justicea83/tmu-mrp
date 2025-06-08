"""
General matcher for the matching engine.
"""
import logging
import re
import spacy
from typing import Dict, Any

from sentence_transformers import SentenceTransformer

from core.matching_engine.base import BaseMatcher
from core.models import ResumeJobMatch
from core.utils import log_message, text_to_vector, cosine_similarity, key_or_default
from core.preprocessors.download_sentence_transformer import CAREER_BERT_MODEL_DIR


class GeneralMatcher(BaseMatcher):
    """
    Matcher that computes a score based on the general semantic similarity between
    the job description and the applicant's resume, with sensitive information removed.
    """

    def __init__(self, match: ResumeJobMatch, model_name: str = None):
        """
        Initialize the general matcher.

        Args:
            match: The resume-job match to process.
            model_name: The sentence transformer model to use. If None, uses CareerBERT.
        """
        super().__init__(match)
        
        # Set the model to use
        if model_name is None:
            self.model_name = CAREER_BERT_MODEL_DIR
        elif model_name.lower() == 'careerbert':
            self.model_name = CAREER_BERT_MODEL_DIR
        else:
            self.model_name = model_name
            
        self.model = SentenceTransformer(self.model_name)
        self.nlp = spacy.load('en_core_web_lg')  # Load the spaCy model for NER
        self.general_details = {}
        self.sanitized_resume_text = ""
        self.sanitized_job_description = ""

    def sanitize_resume_data(self, parsed_results: Dict[str, Any]) -> str:
        """
        Sanitize resume data by removing sensitive information like names, company names, and gender.

        Args:
            parsed_results: The parsed resume data.

        Returns:
            A sanitized string representation of the resume data.
        """
        sanitized_sections = []

        # Extract and sanitize basics section
        basics = key_or_default(parsed_results, 'basics', {})
        if basics:
            # Skip name, email, phone, and other personal identifiers
            summary = key_or_default(basics, 'summary', '')
            if summary:
                sanitized_sections.append(f"Summary: {summary}")

            # Add location but remove specific address
            location = key_or_default(basics, 'location', {})
            if location:
                country = key_or_default(location, 'countryCode', '')
                if country:
                    sanitized_sections.append(f"Country: {country}")

        # Extract and sanitize work experience
        work_experiences = key_or_default(parsed_results, 'work', [])
        if work_experiences:
            work_section = ["Work Experience:"]
            for job in work_experiences:
                # Skip company names
                position = key_or_default(job, 'position', '')
                if position:
                    work_section.append(f"Position: {position}")

                # Add job summaries
                summaries = key_or_default(job, 'summary', [])
                if isinstance(summaries, list):
                    for summary in summaries:
                        # Remove company names from summaries
                        sanitized_summary = self._remove_company_names(summary)
                        work_section.append(f"- {sanitized_summary}")
                elif isinstance(summaries, str):
                    sanitized_summary = self._remove_company_names(summaries)
                    work_section.append(f"- {sanitized_summary}")

            sanitized_sections.append("\n".join(work_section))

        # Extract and sanitize education
        education = key_or_default(parsed_results, 'education', [])
        if education:
            education_section = ["Education:"]
            for edu in education:
                # Skip institution names
                area = key_or_default(edu, 'area', '')
                study_type = key_or_default(edu, 'studyType', '')
                if area or study_type:
                    education_section.append(f"{study_type} in {area}")

            sanitized_sections.append("\n".join(education_section))

        # Extract skills
        skills = key_or_default(parsed_results, 'skills', [])
        if skills:
            skills_section = ["Skills:"]
            for skill in skills:
                if isinstance(skill, dict) and 'name' in skill:
                    skills_section.append(f"- {skill['name']}")
                elif isinstance(skill, str):
                    skills_section.append(f"- {skill}")

            sanitized_sections.append("\n".join(skills_section))

        # Extract and sanitize projects
        projects = key_or_default(parsed_results, 'projects', [])
        if projects:
            projects_section = ["Projects:"]
            for project in projects:
                # Skip project names
                summaries = key_or_default(project, 'summary', [])
                if isinstance(summaries, list):
                    for summary in summaries:
                        # Remove company names from summaries
                        sanitized_summary = self._remove_company_names(summary)
                        projects_section.append(f"- {sanitized_summary}")
                elif isinstance(summaries, str):
                    sanitized_summary = self._remove_company_names(summaries)
                    projects_section.append(f"- {sanitized_summary}")

            sanitized_sections.append("\n".join(projects_section))

        return "\n\n".join(sanitized_sections)

    def _remove_company_names(self, text: str) -> str:
        """
        Remove company names from text using spaCy NER.

        Args:
            text: The text to sanitize.

        Returns:
            The sanitized text.
        """
        # Use spaCy NER to identify and replace organization names
        doc = self.nlp(text)
        sanitized_text = text

        # Collect all organization entities
        org_entities = [ent for ent in doc.ents if ent.label_ == "ORG"]

        # Sort entities by their start position in reverse order to avoid index issues when replacing
        for ent in sorted(org_entities, key=lambda x: x.start_char, reverse=True):
            # Check if the entity is preceded by prepositions like "at", "for", etc.
            start_idx = max(0, ent.start_char - 10)  # Look up to 10 chars before the entity
            prefix_text = text[start_idx:ent.start_char].lower()

            if any(prep in prefix_text for prep in [" at ", " for ", " with ", " by ", " from "]):
                # Replace the entity with "a company"
                sanitized_text = sanitized_text[:ent.start_char] + "a company" + sanitized_text[ent.end_char:]

        # Also apply the regex for cases that might be missed by NER
        sanitized_text = re.sub(r'(?i)\b(at|for|with|by|from)\s+[A-Z][a-zA-Z0-9\s&]+', r'\1 a company', sanitized_text)

        return sanitized_text

    def sanitize_job_description(self, description: str) -> str:
        """
        Sanitize job description by removing company names and other sensitive information
        using spaCy NER.

        Args:
            description: The job description.

        Returns:
            The sanitized job description.
        """
        # Use spaCy NER to identify and replace organization and location names
        doc = self.nlp(description)
        sanitized = description

        # Process entities in reverse order to avoid index issues when replacing
        entities = sorted(doc.ents, key=lambda x: x.start_char, reverse=True)

        for ent in entities:
            if ent.label_ == "ORG":  # Organization entity
                # Check if the entity is preceded by prepositions
                start_idx = max(0, ent.start_char - 10)
                prefix_text = description[start_idx:ent.start_char].lower()

                if any(prep in prefix_text for prep in [" at ", " for ", " with ", " by ", " from "]):
                    # Replace the organization with "a company"
                    sanitized = sanitized[:ent.start_char] + "a company" + sanitized[ent.end_char:]

            elif ent.label_ in ["GPE", "LOC"]:  # Geopolitical entity or location
                # Check if the entity is preceded by location prepositions
                start_idx = max(0, ent.start_char - 10)
                prefix_text = description[start_idx:ent.start_char].lower()

                if any(prep in prefix_text for prep in [" in ", " at ", " near ", " around "]):
                    # Replace the location with "a location"
                    sanitized = sanitized[:ent.start_char] + "a location" + sanitized[ent.end_char:]

        # Apply regex as fallback for cases that might be missed by NER
        sanitized = re.sub(r'(?i)\b(at|for|with|by|from)\s+[A-Z][a-zA-Z0-9\s&]+', r'\1 a company', sanitized)
        sanitized = re.sub(r'(?i)\b(in|at|near|around)\s+[A-Z][a-zA-Z0-9\s,]+', r'\1 a location', sanitized)

        return sanitized

    def compute_score(self) -> float:
        """
        Compute a score based on the general semantic similarity between
        the job description and the applicant's resume.

        Returns:
            A score between 0 and 100.
        """
        # Get parsed resume data directly from resume
        parsed_results = self.resume.parsed_data
        if not parsed_results:
            log_message(logging.WARNING, "No parsed resume results found", self.user)
            return 0.0

        # Get job description
        job_description = self.job_description.description or ""
        if not job_description:
            log_message(logging.WARNING, "No job description found", self.user)
            return 0.0

        # Sanitize resume data and job description
        self.sanitized_resume_text = self.sanitize_resume_data(parsed_results)
        self.sanitized_job_description = self.sanitize_job_description(job_description)

        # Vectorize sanitized texts
        resume_vector = text_to_vector(self.sanitized_resume_text, self.model)
        job_vector = text_to_vector(self.sanitized_job_description, self.model)

        # Calculate similarity
        similarity = cosine_similarity(resume_vector, job_vector)

        # Convert similarity to a score between 0 and 100
        score = similarity * 100

        # Store details for later retrieval
        self.general_details = {
            'similarity': similarity,
            'sanitized_resume_length': len(self.sanitized_resume_text),
            'sanitized_job_description_length': len(self.sanitized_job_description),
            'model_used': self.model_name
        }

        return score

    def get_score_details(self) -> Dict[str, Any]:
        """
        Get detailed information about the general score computation.

        Returns:
            A dictionary with details about the general score computation.
        """
        return self.general_details
