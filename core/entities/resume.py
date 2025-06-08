from pydantic import BaseModel, Field
from typing import Optional


class Resume(BaseModel):
    """
    Resume entity representing a resume with all its associated metadata.
    All fields are strings as specified.
    """
    
    ID: str = Field(..., description="Unique identifier for the resume")
    Resume_str: str = Field(..., description="String representation of the resume")
    Resume_html: str = Field(..., description="HTML representation of the resume")
    Category: str = Field(..., description="Category classification of the resume")
    hash: str = Field(..., description="Hash value of the resume")
    char_len: str = Field(..., description="Character length of the resume")
    sent_len: str = Field(..., description="Sentence length of the resume")
    type_token_ratio: str = Field(..., description="Type token ratio of the resume")
    gender_term_count: str = Field(..., description="Count of gender-related terms")
    html_len: str = Field(..., description="Length of HTML content")
    text_from_html: str = Field(..., description="Text extracted from HTML")
    html_strip_diff: str = Field(..., description="Difference after HTML stripping")

    class Config:
        """Pydantic configuration"""
        str_strip_whitespace = True
        validate_assignment = True 