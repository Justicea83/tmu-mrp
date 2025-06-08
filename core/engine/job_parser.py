import os
import spacy
import pprint
from spacy.matcher import Matcher
from . import utils


class ResumeParser(object):
    def __init__(self, resume_text, skills_file=None, custom_regex=None):
        print('Spacy model is loading...')
        nlp = spacy.load('en_core_web_lg')

        current_directory = os.path.dirname(os.path.abspath(__file__))

        # Try to load custom model, fallback to base model if not available
        try:
            custom_nlp = spacy.load(os.path.join(current_directory, 'models', 'resume_model'))
        except OSError:
            print('Custom resume model not found, using base model...')
            custom_nlp = nlp

        self.__skills_file = skills_file
        self.__custom_regex = custom_regex
        self.__matcher = Matcher(nlp.vocab)
        self.__details = {
            'domain': None,
            'all_skills': None,
            'skills': None,
            'occupation': None,
            'experience': None,
        }
        
        # Process resume text directly
        self.__resume_text = resume_text
        self.__text_raw = resume_text
        self.__text = ' '.join(self.__text_raw.split())
        self.__nlp = nlp(self.__text)
        self.__custom_nlp = custom_nlp(self.__text_raw)
        self.__noun_chunks = list(self.__nlp.noun_chunks)
        self.__get_basic_details()

    def get_extracted_data(self):
        return self.__details

    def __get_basic_details(self):
        cust_tags = utils.extract_tags_with_custom_model(self.__custom_nlp)

        all_skills = utils.clean_skills(utils.key_or_default(cust_tags, 'SKILL', []))

        skills = utils.extract_skills_from_all(
            all_skills,
            self.__noun_chunks,
            self.__skills_file
        )

        if 'EXPERIENCE' in cust_tags and len(cust_tags['EXPERIENCE']) > 0:
            experience = utils.extract_years_of_experience(cust_tags['EXPERIENCE'][0])
        else:
            experience = None

        try:
            self.__details['occupation'] = cust_tags['OCCUPATION'][0]
        except (IndexError, KeyError):
            self.__details['occupation'] = utils.key_or_default(cust_tags, 'OCCUPATION')

        try:
            self.__details['all_skills'] = all_skills
        except (IndexError, KeyError):
            self.__details['all_skills'] = None

        self.__details['experience'] = experience
        self.__details['skills'] = skills

        self.__details['domain'] = utils.key_or_default(cust_tags, 'DOMAIN')

        return


def resume_result_wrapper(resume_text):
    parser = ResumeParser(resume_text)
    return parser.get_extracted_data()


if __name__ == '__main__':
    # Example usage with resume text
    sample_resume = """
    John Doe
    Software Engineer
    
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
    - Bachelor's in Computer Science
    """
    
    parser = ResumeParser(sample_resume)
    result = parser.get_extracted_data()
    pprint.pprint(result)
