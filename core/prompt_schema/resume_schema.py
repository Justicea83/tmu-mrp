from core.prompt_schema import BaseSchema


def get_resume_schema():
    """
    Create a schema for parsing resumes into a JSON draft7 format.

    Returns:
        BaseSchema: A schema for parsing resumes.
    """
    schema = BaseSchema()
    schema.input_text = """
    You are provided with a resume. Parse it into a standardized JSON format following the JSON Schema Draft 7 specification.
    Extract all relevant information including personal details, work experience, education, skills, projects, etc.
    DO NOT GUESS ANYTHING. If you cannot find information for a field, set it to null.
    """
    schema.template_type = "Resume Parser"

    # Define the schema based on the example resume in the issue description
    resume_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Resume",
        "type": "object",
        "properties": {
            "basics": {
                "type": "object",
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "label": {"type": ["string", "null"]},
                    "email": {"type": ["string", "null"]},
                    "phone": {"type": ["string", "null"]},
                    "summary": {"type": ["string", "null"]},
                    "location": {
                        "type": "object",
                        "properties": {
                            "address": {"type": ["string", "null"]},
                            "countryCode": {"type": ["string", "null"]}
                        }
                    },
                    "profiles": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "network": {"type": ["string", "null"]},
                                "username": {"type": ["string", "null"]},
                                "url": {"type": ["string", "null"]}
                            }
                        }
                    }
                }
            },
            "work": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": ["string", "null"]},
                        "position": {"type": ["string", "null"]},
                        "startDate": {"type": ["string", "null"]},
                        "endDate": {"type": ["string", "null"]},
                        "summary": {
                            "type": ["array", "string", "null"],
                            "items": {"type": "string"}
                        },
                        "location": {"type": ["string", "null"]},
                        "url": {"type": ["string", "null"]}
                    }
                }
            },
            "education": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "institution": {"type": ["string", "null"]},
                        "area": {"type": ["string", "null"]},
                        "studyType": {"type": ["string", "null"]},
                        "startDate": {"type": ["string", "null"]},
                        "endDate": {"type": ["string", "null"]},
                        "score": {"type": ["string", "null"]}
                    }
                }
            },
            "skills": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": ["string", "null"]}
                    }
                }
            },
            "projects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": ["string", "null"]},
                        "summary": {
                            "type": ["array", "string", "null"],
                            "items": {"type": "string"}
                        },
                        "startDate": {"type": ["string", "null"]},
                        "endDate": {"type": ["string", "null"]},
                        "url": {"type": ["string", "null"]}
                    }
                }
            }
        }
    }

    schema.example_output_json = resume_schema

    return schema