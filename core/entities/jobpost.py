from pydantic import BaseModel, Field


class JobPost(BaseModel):
    """
    JobPost entity representing a job posting with all its associated metadata.
    All fields are strings as specified.
    """
    
    Position: str = Field(..., description="Job position title")
    Long_Description: str = Field(..., alias="Long Description", description="Detailed job description")
    Company_Name: str = Field(..., alias="Company Name", description="Name of the hiring company")
    Exp_Years: str = Field(..., alias="Exp Years", description="Required years of experience")
    Primary_Keyword: str = Field(..., alias="Primary Keyword", description="Primary keyword for the job")
    English_Level: str = Field(..., alias="English Level", description="Required English proficiency level")
    Published: str = Field(..., description="Publication date/status")
    Long_Description_lang: str = Field(..., alias="Long Description_lang", description="Language of the job description")
    id: str = Field(..., description="Unique identifier for the job post")
    __index_level_0__: str = Field(..., description="Index level identifier")
    char_len: str = Field(..., description="Character length of the description")

    class Config:
        """Pydantic configuration"""
        str_strip_whitespace = True
        validate_assignment = True
        allow_population_by_field_name = True 