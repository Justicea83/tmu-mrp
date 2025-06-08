"""
Data models for the resume-job matching and ranking system.
"""
import json
import random
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class Resume:
    """Resume representation from processed CSV data"""
    ID: str
    Resume_str: str
    Resume_html: str
    Category: str
    hash: str
    char_len: str
    sent_len: str
    type_token_ratio: str
    gender_term_count: str
    html_len: str
    text_from_html: str
    html_strip_diff: str
    parsed_json: str = ""
    location_data: Dict[str, Any] = field(default_factory=dict)
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def parsed_data(self) -> Dict[str, Any]:
        """Get parsed resume data as dictionary"""
        try:
            return json.loads(self.parsed_json) if self.parsed_json else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def generate_random_location(self) -> Dict[str, Any]:
        """Generate random location data similar to ipinfo format"""
        locations = [
            {
                "city": "Toronto", "region": "Ontario", "country": "CA",
                "loc": "43.6532,-79.3832", "postal": "M5V", "timezone": "America/Toronto"
            },
            {
                "city": "Vancouver", "region": "British Columbia", "country": "CA", 
                "loc": "49.2827,-123.1207", "postal": "V6B", "timezone": "America/Vancouver"
            },
            {
                "city": "New York", "region": "New York", "country": "US",
                "loc": "40.7128,-74.0060", "postal": "10001", "timezone": "America/New_York"
            },
            {
                "city": "San Francisco", "region": "California", "country": "US",
                "loc": "37.7749,-122.4194", "postal": "94102", "timezone": "America/Los_Angeles"
            },
            {
                "city": "London", "region": "England", "country": "GB",
                "loc": "51.5074,-0.1278", "postal": "SW1A", "timezone": "Europe/London"
            },
            {
                "city": "Berlin", "region": "Berlin", "country": "DE",
                "loc": "52.5200,13.4050", "postal": "10115", "timezone": "Europe/Berlin"
            },
            {
                "city": "Sydney", "region": "New South Wales", "country": "AU",
                "loc": "-33.8688,151.2093", "postal": "2000", "timezone": "Australia/Sydney"
            },
            {
                "city": "Mumbai", "region": "Maharashtra", "country": "IN",
                "loc": "19.0760,72.8777", "postal": "400001", "timezone": "Asia/Kolkata"
            }
        ]
        return random.choice(locations)
    
    def get_location_data(self) -> Dict[str, Any]:
        """Get location data, generating if not exists"""
        if not self.location_data:
            self.location_data = self.generate_random_location()
        return self.location_data


@dataclass 
class JobDescription:
    """Job description representation from CSV data"""
    id: str
    Position: str
    Long_Description: str
    Company_Name: str
    Exp_Years: str
    Primary_Keyword: str
    English_Level: str
    Published: str
    Long_Description_lang: str
    __index_level_0__: str
    char_len: str
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def description(self) -> str:
        """Alias for Long_Description for compatibility"""
        return self.Long_Description
    
    @property
    def required_experience_years(self) -> float:
        """Extract required experience years from Exp_Years field"""
        try:
            return float(self.Exp_Years)
        except (ValueError, TypeError):
            return 0.0


@dataclass
class ResumeJobMatch:
    """Represents a match between a resume and job description"""
    resume: Resume
    job_description: JobDescription
    match_id: str
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def save(self, update_fields: Optional[List[str]] = None):
        """Save method for compatibility - persists extra_data"""
        # In this simple implementation, extra_data is already updated by reference
        # so no additional action is needed since we're not using a database
        pass
    
    @property
    def user(self):
        """Mock user for compatibility - return resume ID"""
        return MockUser(self.resume.ID)


@dataclass
class MockUser:
    """Mock user class for compatibility"""
    id: str
    
    def __str__(self):
        return f"Resume_{self.id}"


@dataclass
class Skill:
    """Skill representation"""
    name: str


class MockSkillsManager:
    """Mock Django skills manager"""
    def __init__(self, skills: List[str]):
        self._skills = [Skill(name=skill) for skill in skills]
    
    def all(self):
        return self._skills 