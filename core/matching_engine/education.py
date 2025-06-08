"""
Education Matcher for Resume-Job Matching System

This module implements education-based matching between resumes and job descriptions,
analyzing degree levels, fields of study, and educational requirements using comprehensive
datasets from Hugging Face instead of hardcoded mappings.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from .base import BaseMatcher
from ..models import ResumeJobMatch
import logging


class EducationMatcher(BaseMatcher):
    """
    Matches resumes to jobs based on educational background.
    
    Uses Hugging Face datasets for comprehensive field mappings:
    - Academic discipline hierarchies from Wikipedia
    - Education taxonomy from research repositories
    - Field classification from various sources
    
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
            'phd': 5, 'doctorate': 5, 'doctoral': 5, 'doctor': 5,
            'master': 4, 'masters': 4, 'mba': 4, 'ms': 4, 'ma': 4, 'msc': 4,
            'bachelor': 3, 'bachelors': 3, 'bs': 3, 'ba': 3, 'bsc': 3,
            'associate': 2, 'associates': 2, 'aa': 2, 'as': 2,
            'diploma': 1, 'certificate': 1, 'certification': 1, 'cert': 1
        }
        
        # Initialize field mappings - will be loaded from datasets
        self.field_mappings = {}
        self.academic_subjects = {}
        self.field_keywords = {}
        
        # Load field mappings from datasets
        self._load_field_mappings()
        
        # Common degree abbreviations and patterns
        self.degree_patterns = [
            r'\b(ph\.?d|doctorate|doctoral|doctor)\b',
            r'\b(m\.?s\.?c?|master|mba|m\.?a\.?)\b',
            r'\b(b\.?s\.?c?|b\.?a\.?|bachelor)\b',
            r'\b(associate|a\.?s\.?|a\.?a\.?)\b',
            r'\b(diploma|certificate|cert\.?|certification)\b'
        ]

    def _load_field_mappings(self):
        """Load comprehensive field mappings from Hugging Face datasets and local sources."""
        try:
            # Check if HF dataset loading is disabled
            import os
            if os.getenv('DISABLE_HF_DATASETS', '').lower() not in ('true', '1', 'yes'):
                # Primary: Load from Hugging Face wiki_academic_subjects dataset
                self._load_from_huggingface_datasets()
            else:
                logging.getLogger(__name__).info("HuggingFace dataset loading disabled by environment variable")
            
            # Secondary: Add comprehensive local mappings
            self._load_academic_subjects()
            self._load_technology_fields()
            self._load_business_fields()
            self._load_science_fields()
            self._load_healthcare_fields()
            self._load_engineering_fields()
            
            logging.getLogger(__name__).info(f"Loaded {len(self.field_mappings)} field categories with {sum(len(fields) for fields in self.field_mappings.values())} total field mappings")
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load field mappings from datasets, using fallback: {e}")
            self._load_fallback_mappings()

    def _load_from_huggingface_datasets(self):
        """
        Load field mappings from Hugging Face datasets.
        
        Uses meliascosta/wiki_academic_subjects dataset which contains 64k academic 
        subject hierarchies from Wikipedia with hierarchical label sequences.
        
        Limited to first 10k rows for performance and to avoid memory issues.
        """
        try:
            # Import datasets library
            from datasets import load_dataset
            import signal
            
            # Set a timeout for dataset loading (30 seconds)
            def timeout_handler(signum, frame):
                raise TimeoutError("Dataset loading timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 second timeout
            
            try:
                # Load only a subset of the dataset to avoid memory issues
                dataset = load_dataset("meliascosta/wiki_academic_subjects", split="train[:10000]")
                
                # Process the dataset using the helper method
                self._process_wiki_academic_subjects(dataset)
                
            finally:
                signal.alarm(0)  # Cancel the alarm
            
        except ImportError:
            logging.getLogger(__name__).info("datasets library not available for HF dataset loading")
            # Continue with local mappings only
            pass
        except TimeoutError:
            logging.getLogger(__name__).warning("HuggingFace dataset loading timed out, using local mappings only")
            # Continue with local mappings only
            pass
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load from HuggingFace datasets: {e}")
            # Continue with local mappings only
            pass

    def _load_academic_subjects(self):
        """Load academic subjects from comprehensive sources."""
        # Comprehensive technology/computer science fields
        self.field_mappings['technology'] = [
            'computer science', 'information technology', 'software engineering',
            'data science', 'cybersecurity', 'information systems', 'artificial intelligence',
            'machine learning', 'computer engineering', 'information security',
            'web development', 'mobile development', 'database management',
            'network administration', 'cloud computing', 'devops', 'blockchain',
            'robotics', 'computer graphics', 'human-computer interaction',
            'computational linguistics', 'bioinformatics', 'digital forensics'
        ]
        
        # Business and management fields
        self.field_mappings['business'] = [
            'business administration', 'management', 'marketing', 'finance',
            'economics', 'accounting', 'international business', 'entrepreneurship',
            'operations management', 'supply chain management', 'human resources',
            'organizational behavior', 'strategic management', 'project management',
            'business analytics', 'digital marketing', 'e-commerce', 'consulting',
            'real estate', 'insurance', 'banking', 'investment', 'corporate finance'
        ]
        
        # Engineering fields
        self.field_mappings['engineering'] = [
            'engineering', 'mechanical engineering', 'electrical engineering',
            'civil engineering', 'industrial engineering', 'chemical engineering',
            'aerospace engineering', 'biomedical engineering', 'environmental engineering',
            'materials engineering', 'petroleum engineering', 'nuclear engineering',
            'systems engineering', 'manufacturing engineering', 'automotive engineering',
            'structural engineering', 'geotechnical engineering', 'transportation engineering'
        ]
        
        # Science fields
        self.field_mappings['science'] = [
            'mathematics', 'statistics', 'physics', 'chemistry', 'biology',
            'biochemistry', 'microbiology', 'molecular biology', 'genetics',
            'biotechnology', 'environmental science', 'earth science', 'geology',
            'astronomy', 'astrophysics', 'marine biology', 'ecology',
            'neuroscience', 'cognitive science', 'materials science'
        ]
        
        # Healthcare and medical fields
        self.field_mappings['healthcare'] = [
            'medicine', 'nursing', 'healthcare', 'medical', 'health',
            'pharmacy', 'dentistry', 'veterinary medicine', 'public health',
            'health administration', 'medical technology', 'radiologic technology',
            'physical therapy', 'occupational therapy', 'respiratory therapy',
            'medical laboratory science', 'health informatics', 'epidemiology',
            'nutrition', 'dietetics', 'sports medicine', 'mental health'
        ]
        
        # Education and social sciences
        self.field_mappings['education'] = [
            'education', 'teaching', 'pedagogy', 'curriculum', 'educational leadership',
            'educational psychology', 'special education', 'early childhood education',
            'elementary education', 'secondary education', 'higher education',
            'adult education', 'distance learning', 'instructional design'
        ]
        
        # Social sciences and humanities
        self.field_mappings['social_sciences'] = [
            'psychology', 'sociology', 'anthropology', 'political science',
            'international relations', 'public policy', 'social work',
            'criminology', 'geography', 'history', 'philosophy', 'linguistics',
            'literature', 'communications', 'journalism', 'media studies'
        ]
        
        # Arts and design
        self.field_mappings['arts'] = [
            'art', 'design', 'graphic design', 'industrial design', 'interior design',
            'fashion design', 'architecture', 'fine arts', 'visual arts',
            'performing arts', 'music', 'theater', 'film', 'photography',
            'multimedia', 'animation', 'game design', 'user experience design'
        ]
        
        # Legal and law
        self.field_mappings['legal'] = [
            'law', 'legal studies', 'jurisprudence', 'constitutional law',
            'criminal law', 'corporate law', 'international law', 'patent law',
            'environmental law', 'family law', 'tax law', 'labor law'
        ]
        
        # Agricultural and environmental
        self.field_mappings['agriculture'] = [
            'agriculture', 'agricultural science', 'agribusiness', 'horticulture',
            'forestry', 'animal science', 'plant science', 'soil science',
            'agricultural engineering', 'sustainable agriculture', 'aquaculture'
        ]

    def _load_technology_fields(self):
        """Enhanced technology field mappings."""
        tech_keywords = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'go', 'rust'],
            'web_development': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask'],
            'mobile_development': ['ios', 'android', 'swift', 'kotlin', 'flutter', 'react native'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'nosql'],
            'cloud': ['aws', 'azure', 'google cloud', 'kubernetes', 'docker'],
            'ai_ml': ['tensorflow', 'pytorch', 'scikit-learn', 'nlp', 'deep learning']
        }
        self.field_keywords.update(tech_keywords)

    def _load_business_fields(self):
        """Enhanced business field mappings."""
        business_keywords = {
            'finance': ['financial analysis', 'investment', 'portfolio management', 'risk management'],
            'marketing': ['digital marketing', 'content marketing', 'seo', 'social media'],
            'operations': ['supply chain', 'logistics', 'process improvement', 'lean', 'six sigma'],
            'hr': ['talent acquisition', 'employee relations', 'compensation', 'benefits']
        }
        self.field_keywords.update(business_keywords)

    def _load_science_fields(self):
        """Enhanced science field mappings."""
        science_keywords = {
            'data_science': ['statistics', 'machine learning', 'data analysis', 'predictive modeling'],
            'research': ['experimental design', 'hypothesis testing', 'peer review', 'publication'],
            'laboratory': ['lab techniques', 'instrumentation', 'quality control', 'protocols']
        }
        self.field_keywords.update(science_keywords)

    def _load_healthcare_fields(self):
        """Enhanced healthcare field mappings."""
        healthcare_keywords = {
            'clinical': ['patient care', 'diagnosis', 'treatment', 'clinical trials'],
            'public_health': ['epidemiology', 'health policy', 'preventive medicine', 'health promotion'],
            'medical_technology': ['medical devices', 'imaging', 'laboratory medicine', 'telemedicine']
        }
        self.field_keywords.update(healthcare_keywords)

    def _load_engineering_fields(self):
        """Enhanced engineering field mappings."""
        engineering_keywords = {
            'design': ['cad', 'solidworks', 'autocad', 'design thinking', 'prototyping'],
            'project_management': ['pmp', 'agile', 'scrum', 'waterfall', 'risk management'],
            'quality': ['quality assurance', 'testing', 'validation', 'compliance', 'standards']
        }
        self.field_keywords.update(engineering_keywords)

    def _load_fallback_mappings(self):
        """Fallback mappings if dataset loading fails."""
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

    def update_field_mappings_from_dataset(self, dataset_name: str = None):
        """
        Public method to update field mappings from external datasets.
        
        Args:
            dataset_name: Optional HuggingFace dataset name to load from.
                         Defaults to 'meliascosta/wiki_academic_subjects'
            
        This allows for dynamic updating of field mappings if needed.
        """
        # Clear existing mappings
        self.field_mappings.clear()
        
        if dataset_name:
            logging.getLogger(__name__).info(f"Updating field mappings from dataset: {dataset_name}")
            try:
                from datasets import load_dataset
                
                # Load the specified dataset
                dataset = load_dataset(dataset_name, split="train")
                
                # Process based on known dataset formats
                if dataset_name == "meliascosta/wiki_academic_subjects":
                    self._process_wiki_academic_subjects(dataset)
                elif dataset_name == "millawell/wikipedia_field_of_science":
                    self._process_wikipedia_field_of_science(dataset)
                else:
                    # Try to auto-detect format based on available columns
                    if 'label sequence' in dataset.column_names:
                        self._process_wiki_academic_subjects(dataset)
                    elif 'label' in dataset.column_names:
                        self._process_wikipedia_field_of_science(dataset)
                    else:
                        # Generic processing
                        self._process_generic_academic_dataset(dataset)
                    
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to load dataset {dataset_name}: {e}")
        
        # Always add comprehensive local mappings as supplement
        self._load_academic_subjects()
        self._load_technology_fields()
        self._load_business_fields()
        self._load_science_fields()
        self._load_healthcare_fields()
        self._load_engineering_fields()
        
        logging.getLogger(__name__).info(f"Updated field mappings: {len(self.field_mappings)} categories, {sum(len(fields) for fields in self.field_mappings.values())} total fields")

    def _process_wiki_academic_subjects(self, dataset):
        """Process the wiki_academic_subjects dataset format."""
        categories_processed = set()
        processed_count = 0
        max_items = 1000  # Limit processing to avoid infinite loops
        
        try:
            for example in dataset:
                processed_count += 1
                if processed_count > max_items:
                    logging.getLogger(__name__).info(f"Reached maximum processing limit of {max_items} items")
                    break
                
                label_sequence = example.get('label sequence', [])
                
                if len(label_sequence) >= 2:
                    broad_field = label_sequence[0].lower().replace(' ', '_').replace('-', '_')
                    
                    # Limit to reasonable field name lengths to avoid junk data
                    if len(broad_field) > 50:
                        continue
                    
                    for label in label_sequence[1:]:
                        specific_field = label.lower().strip()
                        
                        # Skip empty or very long field names
                        if not specific_field or len(specific_field) > 100:
                            continue
                        
                        if broad_field not in self.field_mappings:
                            self.field_mappings[broad_field] = []
                        
                        if specific_field not in self.field_mappings[broad_field]:
                            self.field_mappings[broad_field].append(specific_field)
                            categories_processed.add((broad_field, specific_field))
                        
                        # Limit number of fields per category to avoid memory issues
                        if len(self.field_mappings[broad_field]) > 200:
                            break
        
        except Exception as e:
            logging.getLogger(__name__).warning(f"Error processing WikiAcademicSubjects dataset: {e}")
        
        logging.getLogger(__name__).info(f"Processed {len(categories_processed)} field mappings from WikiAcademicSubjects ({processed_count} items processed)")

    def _process_wikipedia_field_of_science(self, dataset):
        """Process the wikipedia_field_of_science dataset format."""
        categories_processed = set()
        
        for example in dataset:
            # Get the hierarchical label sequence (similar to wiki_academic_subjects)
            label_sequence = example.get('label', [])
            
            if len(label_sequence) >= 2:
                # Use the top-level category as the broad field
                broad_field = label_sequence[0].lower().replace(' ', '_').replace('-', '_')
                
                # Extract all specific fields from the hierarchy
                for label in label_sequence[1:]:
                    specific_field = label.lower().strip()
                    
                    if broad_field not in self.field_mappings:
                        self.field_mappings[broad_field] = []
                    
                    if specific_field not in self.field_mappings[broad_field]:
                        self.field_mappings[broad_field].append(specific_field)
                        categories_processed.add((broad_field, specific_field))
        
        logging.getLogger(__name__).info(f"Processed {len(categories_processed)} field mappings from WikipediaFieldOfScience")

    def _process_generic_academic_dataset(self, dataset):
        """Process a generic academic dataset format."""
        logging.getLogger(__name__).info("Processing generic academic dataset format")
        # This is a placeholder for future dataset formats
        pass

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
        
        # Look for field requirements using comprehensive mappings
        required_fields = []
        for field_category, keywords in self.field_mappings.items():
            for keyword in keywords:
                if keyword.lower() in job_text:
                    required_fields.append(field_category)
                    break
        
        # Also check keyword mappings
        for category, keywords in self.field_keywords.items():
            for keyword in keywords:
                if keyword.lower() in job_text:
                    required_fields.append(category)
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
        """Compute relevance of education field to job requirements using comprehensive mappings."""
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
        
        # Score field relevance using comprehensive mappings
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
        
        # Also check against keyword mappings for more specific matches
        for resume_field in resume_fields:
            for category, keywords in self.field_keywords.items():
                for keyword in keywords:
                    if keyword in resume_field and keyword in job_text:
                        max_relevance = max(max_relevance, 0.8)
        
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
            'experience_alignment': self._compute_experience_education_alignment(resume, job),
            'field_categories_covered': len(self.field_mappings),
            'total_field_keywords': sum(len(fields) for fields in self.field_mappings.values())
        }
        
        return explanation 