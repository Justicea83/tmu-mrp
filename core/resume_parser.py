"""
Resume parser for extracting skills, experience, and other information from resume text.
"""
import spacy
import re
from typing import Dict, Any, List, Optional


class ResumeParser:
    """
    Parser for extracting structured information from resume text.
    """
    
    def __init__(self, resume_text: str):
        """
        Initialize the resume parser.
        
        Args:
            resume_text: The resume text to parse
        """
        print('Loading spaCy model...')
        self.nlp = spacy.load('en_core_web_lg')
        self.resume_text = resume_text
        self.doc = self.nlp(resume_text)
        self.details = {
            'skills': [],
            'experience': None,
            'occupation': None,
            'education': [],
            'contact_info': {},
        }
        self._extract_details()
    
    def get_extracted_data(self) -> Dict[str, Any]:
        """
        Get the extracted resume data.
        
        Returns:
            Dictionary containing extracted information
        """
        return self.details
    
    def _extract_details(self):
        """Extract all details from the resume"""
        self._extract_skills()
        self._extract_experience()
        self._extract_occupation()
        self._extract_education()
        self._extract_contact_info()
    
    def _extract_skills(self) -> List[str]:
        """
        Extract skills from resume text using keyword matching and NER.
        
        Returns:
            List of identified skills
        """
        skills = []
        
        # Common skill keywords
        skill_keywords = [
            # Programming languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
            'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'html', 'css', 'dart', 'perl',
            
            # Frameworks and libraries
            'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'laravel',
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'jquery', 'bootstrap',
            'node.js', 'nest.js', 'next.js', 'gatsby', 'svelte',
            
            # Databases
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
            'oracle', 'sqlite', 'dynamodb', 'neo4j',
            
            # Cloud platforms and DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible',
            'jenkins', 'gitlab', 'circleci', 'travis',
            
            # Tools and technologies
            'git', 'jira', 'confluence', 'slack', 'figma', 'photoshop', 'illustrator',
            'tableau', 'power bi', 'excel', 'powerpoint',
            
            # Soft skills
            'leadership', 'communication', 'teamwork', 'problem-solving', 'analytical',
            'project management', 'agile', 'scrum', 'kanban', 'waterfall',
            
            # Other technical skills
            'machine learning', 'artificial intelligence', 'data science', 'web development',
            'mobile development', 'devops', 'cybersecurity', 'blockchain', 'microservices',
            'api development', 'rest api', 'graphql', 'testing', 'unit testing'
        ]
        
        # Extract skills using keyword matching (case-insensitive)
        text_lower = self.resume_text.lower()
        for skill in skill_keywords:
            if skill.lower() in text_lower:
                skills.append(skill.title() if ' ' not in skill else skill)
        
        # Use spaCy to extract additional technical terms
        for token in self.doc:
            # Look for capitalized words that might be technologies/skills
            if (token.text.isupper() or token.text.istitle()) and len(token.text) > 2:
                if token.pos_ in ['NOUN', 'PROPN'] and token.text.lower() not in ['the', 'and', 'for', 'with']:
                    potential_skill = token.text
                    if potential_skill not in skills and len(potential_skill) > 2:
                        skills.append(potential_skill)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in skills:
            skill_lower = skill.lower()
            if skill_lower not in seen:
                seen.add(skill_lower)
                unique_skills.append(skill)
        
        self.details['skills'] = unique_skills[:30]  # Limit to top 30 skills
        return unique_skills
    
    def _extract_experience(self) -> Optional[int]:
        """
        Extract years of experience from resume text.
        
        Returns:
            Number of years of experience or None if not found
        """
        # Patterns to match experience
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*in',
            r'(\d+)\+?\s*years?\s*working',
            r'experience\s*:?\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*yr[s]?\s*(?:of\s*)?experience',
        ]
        
        max_experience = 0
        text_lower = self.resume_text.lower()
        
        for pattern in experience_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    years = int(match)
                    max_experience = max(max_experience, years)
                except ValueError:
                    continue
        
        experience = max_experience if max_experience > 0 else None
        self.details['experience'] = experience
        return experience
    
    def _extract_occupation(self) -> Optional[str]:
        """
        Extract primary occupation/job title from resume.
        
        Returns:
            Primary occupation or None if not found
        """
        # Common job titles
        job_titles = [
            'software engineer', 'data scientist', 'product manager', 'designer',
            'developer', 'analyst', 'consultant', 'manager', 'director',
            'senior', 'junior', 'lead', 'principal', 'architect',
            'frontend', 'backend', 'fullstack', 'devops', 'qa engineer'
        ]
        
        text_lower = self.resume_text.lower()
        
        for title in job_titles:
            if title in text_lower:
                # Find the full title in context
                pattern = rf'\b[\w\s]*{re.escape(title)}[\w\s]*\b'
                matches = re.findall(pattern, text_lower)
                if matches:
                    # Return the first reasonable match
                    occupation = matches[0].strip().title()
                    self.details['occupation'] = occupation
                    return occupation
        
        # Use NER to find potential job titles
        for ent in self.doc.ents:
            if ent.label_ in ['PERSON', 'ORG']:
                continue
            if any(word in ent.text.lower() for word in ['engineer', 'developer', 'manager', 'analyst']):
                self.details['occupation'] = ent.text
                return ent.text
        
        return None
    
    def _extract_education(self) -> List[str]:
        """
        Extract education information from resume.
        
        Returns:
            List of education entries
        """
        education = []
        
        # Education keywords
        education_keywords = [
            'bachelor', 'master', 'phd', 'doctorate', 'degree',
            'university', 'college', 'institute', 'school',
            'b.s.', 'b.a.', 'm.s.', 'm.a.', 'mba', 'ph.d.'
        ]
        
        text_lower = self.resume_text.lower()
        
        for keyword in education_keywords:
            if keyword in text_lower:
                # Find surrounding context
                pattern = rf'[^.]*{re.escape(keyword)}[^.]*'
                matches = re.findall(pattern, text_lower)
                for match in matches:
                    education.append(match.strip().title())
        
        # Remove duplicates
        education = list(set(education))
        self.details['education'] = education[:5]  # Limit to 5 entries
        return education
    
    def _extract_contact_info(self) -> Dict[str, str]:
        """
        Extract contact information from resume.
        
        Returns:
            Dictionary containing contact information
        """
        contact_info = {}
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, self.resume_text)
        if emails:
            contact_info['email'] = emails[0]
        
        # Phone pattern
        phone_pattern = r'[\+]?[1-9]?[0-9]{3}[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
        phones = re.findall(phone_pattern, self.resume_text)
        if phones:
            contact_info['phone'] = phones[0]
        
        # LinkedIn pattern
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin = re.findall(linkedin_pattern, self.resume_text.lower())
        if linkedin:
            contact_info['linkedin'] = linkedin[0]
        
        self.details['contact_info'] = contact_info
        return contact_info


def parse_resume(resume_text: str) -> Dict[str, Any]:
    """
    Convenience function to parse a resume and return extracted data.
    
    Args:
        resume_text: The resume text to parse
        
    Returns:
        Dictionary containing extracted information
    """
    parser = ResumeParser(resume_text)
    return parser.get_extracted_data()


if __name__ == '__main__':
    # Example usage
    sample_resume = """
    John Doe
    Software Engineer
    john.doe@email.com
    +1-555-123-4567
    
    Experience:
    - 5 years of experience in Python development
    - 3 years working with Django and Flask
    - Experience with React and JavaScript
    - Knowledge of SQL and PostgreSQL
    
    Skills:
    - Python, JavaScript, HTML, CSS
    - Django, Flask, React
    - SQL, PostgreSQL, MongoDB
    - Git, Docker, AWS
    
    Education:
    - Bachelor's in Computer Science from MIT
    """
    
    result = parse_resume(sample_resume)
    import pprint
    pprint.pprint(result) 