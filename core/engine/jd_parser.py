"""
Job Description parser for extracting skills, experience, and other information from job description text.
"""
import spacy
import re
import csv
import os
from typing import Dict, Any, List, Optional


class JdParser:
    """
    Parser for extracting structured information from job description text.
    """
    
    def __init__(self, jd_text: str):
        """
        Initialize the job description parser.
        
        Args:
            jd_text: The job description text to parse
        """
        print('Loading spaCy model for JD parsing...')
        self.nlp = spacy.load('en_core_web_lg')
        self.jd_text = jd_text
        self.doc = self.nlp(jd_text)
        self.details = {
            'skills': [],
            'experience': None,
            'occupation': None,
            'requirements': [],
            'domain': None,
        }
        self.skill_keywords = self._load_skills_from_csv()
        self._extract_details()
    
    def get_extracted_data(self) -> Dict[str, Any]:
        """
        Get the extracted job description data.
        
        Returns:
            Dictionary containing extracted information
        """
        return self.details
    
    def _load_skills_from_csv(self) -> List[str]:
        """
        Load skills from the skills.csv file.
        
        Returns:
            List of skills from the CSV file
        """
        skills = []
        try:
            # Get the path to the skills.csv file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            skills_file = os.path.join(current_dir, 'skills.csv')
            
            with open(skills_file, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    # The CSV file has skills in a single row, comma-separated
                    skills.extend([skill.strip() for skill in row if skill.strip()])
                    
        except FileNotFoundError:
            print(f"Warning: skills.csv not found at {skills_file}. Using fallback skills.")
            # Fallback to a minimal set of skills if file not found
            skills = ['python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'angular', 'node.js']
        except Exception as e:
            print(f"Error loading skills from CSV: {e}. Using fallback skills.")
            skills = ['python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'angular', 'node.js']
            
        return skills

    def _extract_details(self):
        """Extract all details from the job description"""
        self._extract_skills()
        self._extract_experience()
        self._extract_occupation()
        self._extract_requirements()
        self._extract_domain()
    
    def _extract_skills(self) -> List[str]:
        """
        Extract skills from job description text using keyword matching and NER.
        
        Returns:
            List of identified skills
        """
        skills = []
        
        # Extract skills using keyword matching (case-insensitive) from CSV file
        text_lower = self.jd_text.lower()
        for skill in self.skill_keywords:
            if skill.lower() in text_lower:
                skills.append(skill.title() if ' ' not in skill else skill)
        
        # Use spaCy to extract additional technical terms
        for token in self.doc:
            # Look for capitalized words that might be technologies/skills
            if (token.text.isupper() or token.text.istitle()) and len(token.text) > 2:
                if token.pos_ in ['NOUN', 'PROPN'] and token.text.lower() not in ['the', 'and', 'for', 'with', 'you', 'our']:
                    potential_skill = token.text
                    if potential_skill not in skills and len(potential_skill) > 2:
                        skills.append(potential_skill)
        
        # Look for skills in common JD patterns
        skill_patterns = [
            r'experience (?:with|in) ([^,.]+)',
            r'knowledge of ([^,.]+)',
            r'proficient in ([^,.]+)',
            r'familiar with ([^,.]+)',
            r'expertise in ([^,.]+)',
            r'skills?:?\s*([^.]+)',
            r'requirements?:?\s*([^.]+)'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                # Split by common delimiters and clean
                potential_skills = re.split(r'[,;&/\n]', match)
                for potential_skill in potential_skills:
                    cleaned_skill = potential_skill.strip()
                    if len(cleaned_skill) > 2 and cleaned_skill not in skills:
                        skills.append(cleaned_skill.title())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in skills:
            skill_lower = skill.lower()
            if skill_lower not in seen:
                seen.add(skill_lower)
                unique_skills.append(skill)
        
        self.details['skills'] = unique_skills[:25]  # Limit to top 25 skills
        return unique_skills
    
    def _extract_experience(self) -> Optional[int]:
        """
        Extract required years of experience from job description.
        
        Returns:
            Number of years of experience required or None if not found
        """
        # Patterns to match experience requirements
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*in',
            r'(\d+)\+?\s*years?\s*working',
            r'minimum\s*(?:of\s*)?(\d+)\+?\s*years?',
            r'at least\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*yr[s]?\s*(?:of\s*)?experience',
            r'(\d+)-(\d+)\s*years?\s*(?:of\s*)?experience',  # Range pattern
        ]
        
        max_experience = 0
        text_lower = self.jd_text.lower()
        
        for pattern in experience_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        # Handle range patterns - take the minimum
                        years = int(match[0])
                    else:
                        years = int(match)
                    max_experience = max(max_experience, years)
                except ValueError:
                    continue
        
        experience = max_experience if max_experience > 0 else None
        self.details['experience'] = experience
        return experience
    
    def _extract_occupation(self) -> Optional[str]:
        """
        Extract job title/occupation from job description.
        
        Returns:
            Job title or None if not found
        """
        # Common job title patterns at the beginning of JDs
        title_patterns = [
            r'^([^.\n]+?)(?:\s*-\s*|\n)',  # First line before dash or newline
            r'position:?\s*([^.\n]+)',
            r'role:?\s*([^.\n]+)',
            r'job title:?\s*([^.\n]+)',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, self.jd_text.strip(), re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if len(title) > 3 and len(title) < 100:  # Reasonable title length
                    self.details['occupation'] = title
                    return title
        
        # Look for common job titles in the text
        job_titles = [
            'software engineer', 'data scientist', 'product manager', 'designer',
            'developer', 'analyst', 'consultant', 'manager', 'director',
            'senior', 'junior', 'lead', 'principal', 'architect',
            'frontend developer', 'backend developer', 'fullstack developer',
            'devops engineer', 'qa engineer', 'ui/ux designer'
        ]
        
        text_lower = self.jd_text.lower()
        for title in job_titles:
            if title in text_lower:
                self.details['occupation'] = title.title()
                return title.title()
        
        return None
    
    def _extract_requirements(self) -> List[str]:
        """
        Extract job requirements from the description.
        
        Returns:
            List of requirements
        """
        requirements = []
        
        # Look for requirements sections
        req_patterns = [
            r'requirements?:?\s*(.+?)(?:\n\s*\n|\n[A-Z]|$)',
            r'qualifications?:?\s*(.+?)(?:\n\s*\n|\n[A-Z]|$)',
            r'must have:?\s*(.+?)(?:\n\s*\n|\n[A-Z]|$)',
            r'desired skills?:?\s*(.+?)(?:\n\s*\n|\n[A-Z]|$)',
        ]
        
        for pattern in req_patterns:
            matches = re.findall(pattern, self.jd_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Split by bullet points or newlines
                req_items = re.split(r'[•\-*\n]', match)
                for item in req_items:
                    cleaned_item = item.strip()
                    if len(cleaned_item) > 10:  # Filter out very short items
                        requirements.append(cleaned_item)
        
        self.details['requirements'] = requirements[:10]  # Limit to 10 requirements
        return requirements
    
    def _extract_domain(self) -> Optional[str]:
        """
        Extract industry domain from job description.
        
        Returns:
            Industry domain or None if not found
        """
        # Common industry domains
        domains = [
            'information technology', 'finance', 'healthcare', 'education',
            'retail', 'manufacturing', 'consulting', 'media', 'gaming',
            'fintech', 'edtech', 'healthtech', 'e-commerce', 'saas',
            'automotive', 'aerospace', 'telecommunications', 'banking'
        ]
        
        text_lower = self.jd_text.lower()
        for domain in domains:
            if domain in text_lower:
                self.details['domain'] = domain.title()
                return domain.title()
        
        return None


def parse_job_description(jd_text: str) -> Dict[str, Any]:
    """
    Convenience function to parse a job description and return extracted data.
    
    Args:
        jd_text: The job description text to parse
        
    Returns:
        Dictionary containing extracted information
    """
    parser = JdParser(jd_text)
    return parser.get_extracted_data()


if __name__ == '__main__':
    # Example usage
    sample_jd = """
    Senior Software Engineer - Python/Django
    
    We are looking for an experienced Senior Software Engineer to join our team.
    
    Requirements:
    - 5+ years of experience in Python development
    - Strong knowledge of Django and Flask frameworks
    - Experience with React and JavaScript
    - Proficiency in SQL and PostgreSQL
    - Familiarity with AWS and Docker
    - Knowledge of Agile methodologies
    
    Skills:
    - Python, JavaScript, HTML, CSS
    - Django, Flask, React
    - PostgreSQL, Redis
    - Git, Docker, AWS
    - Problem-solving and teamwork
    """
    
    result = parse_job_description(sample_jd)
    import pprint
    pprint.pprint(result) 