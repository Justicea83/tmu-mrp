"""
Skills matcher for the matching engine.
"""
import logging
from typing import Dict, Any, List

from sentence_transformers import SentenceTransformer

from core.matching_engine.base import BaseMatcher
from core.models import ResumeJobMatch
from core.utils import cosine_similarity, key_or_default
from core.preprocessors.download_sentence_transformer import CAREER_BERT_MODEL_DIR
from core.resume_parser import ResumeParser
from core.engine.jd_parser import JdParser


class SkillsMatcher(BaseMatcher):
    """
    Matcher that computes a score based on the match between the job's skill requirements
    and the applicant's skills.
    """

    def __init__(self, match: ResumeJobMatch, model_name: str = None):
        """
        Initialize the skills matcher.

        Args:
            match: The resume-job match to process.
            model_name: The sentence transformer model to use. If None, uses CareerBERT.
        """
        super().__init__(match)
        self.job_skills = []
        self.applicant_skills = []
        self.skills_details = {}
        
        # Set the model to use
        if model_name is None:
            self.model_name = CAREER_BERT_MODEL_DIR
        elif model_name.lower() == 'careerbert':
            self.model_name = CAREER_BERT_MODEL_DIR
        else:
            self.model_name = model_name
            
        self.model = SentenceTransformer(self.model_name)  # Using the specified model

    def extract_job_skills(self) -> List[str]:
        """
        Extract the job skills from the job description using JdParser.

        Returns:
            A list of job skills.
        """
        try:
            # Use JdParser to extract skills from job description
            jd_parser = JdParser(self.job_description.Long_Description)
            extracted_data = jd_parser.get_extracted_data()
            
            # Get skills from the parser results
            job_skills = extracted_data.get('skills', [])
            
            # Ensure we return a list of strings
            if isinstance(job_skills, list):
                return [str(skill) for skill in job_skills if skill]
            else:
                return []
                
        except Exception as e:
            logging.warning(f"Failed to extract job skills using JdParser: {e}")
            return []

    def extract_applicant_skills(self) -> List[str]:
        """
        Extract the applicant's skills from the resume using ResumeParser.

        Returns:
            A list of applicant skills.
        """
        try:
            # Use ResumeParser to extract skills from resume text
            resume_parser = ResumeParser(self.resume.Resume_str)
            extracted_data = resume_parser.get_extracted_data()
            
            # Get skills from the parser results
            applicant_skills = extracted_data.get('skills', [])
            
            # Ensure we return a list of strings
            if isinstance(applicant_skills, list):
                return [str(skill) for skill in applicant_skills if skill]
            else:
                return []
                
        except Exception as e:
            logging.warning(f"Failed to extract applicant skills using ResumeParser: {e}")
            
            # Fallback to using parsed_data if available
            applicant_skills = []
            parsed_results = self.resume.parsed_data
            if parsed_results:
                skills_data = key_or_default(parsed_results, 'skills', [])

                for skill_item in skills_data:
                    if isinstance(skill_item, str) and skill_item not in applicant_skills:
                        applicant_skills.append(skill_item)
                    elif isinstance(skill_item, dict) and 'name' in skill_item and skill_item['name'] not in applicant_skills:
                        applicant_skills.append(skill_item['name'])

            return applicant_skills

    def compute_exact_match_score(self, job_skills: List[str], applicant_skills: List[str]) -> float:
        """
        Compute a score based on exact matches between job skills and applicant skills.

        Args:
            job_skills: List of job skills.
            applicant_skills: List of applicant skills.

        Returns:
            A score between 0 and 100.
        """
        if not job_skills:
            return 100.0  # Perfect score if no skills are required

        # Convert to lowercase for case-insensitive matching
        job_skills_lower = [s.lower() for s in job_skills]
        applicant_skills_lower = [s.lower() for s in applicant_skills]

        # Count exact matches
        matching_skills = sum(1 for skill in job_skills_lower if skill in applicant_skills_lower)

        # Calculate score as a percentage
        score = (matching_skills / len(job_skills)) * 100

        return score

    def compute_semantic_match_score(self, job_skills: List[str], applicant_skills: List[str]) -> Dict[str, Any]:
        """
        Compute a score based on semantic similarity between job skills and applicant skills.

        Args:
            job_skills: List of job skills.
            applicant_skills: List of applicant skills.

        Returns:
            A dictionary containing the score and semantic matches.
        """
        if not job_skills or not applicant_skills:
            return {'score': 0.0, 'matches': []}

        # Encode skills using the sentence transformer
        job_skill_embeddings = {skill: self.model.encode(skill) for skill in job_skills}
        applicant_skill_embeddings = {skill: self.model.encode(skill) for skill in applicant_skills}

        # Find semantic matches
        semantic_matches = []
        total_similarity = 0.0

        for job_skill, job_embedding in job_skill_embeddings.items():
            best_match = None
            best_similarity = 0.0

            for applicant_skill, applicant_embedding in applicant_skill_embeddings.items():
                similarity = cosine_similarity(job_embedding, applicant_embedding)

                if similarity > best_similarity and similarity > 0.7:  # Threshold for semantic similarity
                    best_similarity = similarity
                    best_match = applicant_skill

            if best_match:
                semantic_matches.append({
                    'job_skill': job_skill,
                    'applicant_skill': best_match,
                    'similarity': best_similarity
                })
                total_similarity += best_similarity

        # Calculate score
        if not job_skills:
            semantic_score = 100.0
        else:
            semantic_score = (total_similarity / len(job_skills)) * 100

        return {
            'score': semantic_score,
            'matches': semantic_matches
        }

    def compute_score(self) -> float:
        """
        Compute a score based on the match between the job's skill requirements
        and the applicant's skills.

        Returns:
            A score between 0 and 100.
        """
        # Extract job skills
        self.job_skills = self.extract_job_skills()

        # Extract applicant skills
        self.applicant_skills = self.extract_applicant_skills()

        # Compute exact match score
        exact_match_score = self.compute_exact_match_score(self.job_skills, self.applicant_skills)

        # Compute semantic match score
        semantic_match_result = self.compute_semantic_match_score(self.job_skills, self.applicant_skills)
        semantic_match_score = semantic_match_result['score']

        # Combine scores (giving more weight to exact matches)
        combined_score = (exact_match_score * 0.7) + (semantic_match_score * 0.3)

        # Store details for later retrieval
        self.skills_details = {
            'job_skills': self.job_skills,
            'applicant_skills': self.applicant_skills,
            'exact_match_score': exact_match_score,
            'semantic_match_score': semantic_match_score,
            'semantic_matches': semantic_match_result['matches'],
            'model_used': self.model_name
        }

        return combined_score

    def get_score_details(self) -> Dict[str, Any]:
        """
        Get detailed information about the skills score computation.

        Returns:
            A dictionary with details about the skills score computation.
        """
        return self.skills_details
