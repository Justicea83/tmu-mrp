import json
import logging
from .helpers import (
    num_tokens_from_string, generate_template_str, generate_synthetic_output_json
)
from pydantic import BaseModel

from core.exceptions import CrossJobsError
from core.utils import log_message


class OpenAIPromptBuilder:
    def __init__(self):
        self.prompt = ""

    def add_text(self, text):
        self.prompt += text

    def add_example(self, template_type, input_template, output_json):
        self.add_text(f"EXAMPLE {template_type} TEXT:")
        self.add_line_break()
        self.add_text(f"EXAMPLE {template_type} TEMPLATE:")
        self.add_line_break()
        self.add_text(input_template)
        self.add_line_break()
        self.add_text(f"EXAMPLE OUTPUT JSON:")
        self.add_line_break()
        self.add_text(output_json)

    def add_separator(self):
        self.prompt += "\n---\n"

    def add_line_break(self):
        self.prompt += "\n"

    def get_prompt(self):
        return self.prompt.strip()

    def get_num_tokens(self):
        return num_tokens_from_string(self.prompt)


def dict_to_json_str(d, indent=None):
    json_str = json.dumps(d, indent=indent)
    return f"```json\n{json_str}\n```"


def get_user_prompt(
        doc,
        template,
        add_example=True,
        minimize_tokens=False,
        json7schema=False
):
    indent = None if minimize_tokens else 2

    builder = OpenAIPromptBuilder()
    template_type = template.template_type.upper()
    if add_example:
        example_template = dict_to_json_str(template.example_input_template, indent=indent)
        example_output = dict_to_json_str(template.example_output_json, indent=indent)
        builder.add_example(
            template_type,
            example_template,
            example_output
        )
        builder.add_separator()
    builder.add_text(f"ACTUAL {template_type} TEXT")
    builder.add_separator()
    builder.add_text(doc)
    builder.add_separator()
    if json7schema:
        builder.add_text(
            f"INSTRUCTION: RETURN JSON EXTRACT OF {template_type} FROM ACTUAL TEXT GIVEN THE JSON DRAFT 7 SCHEMA PROVIDED BELOW.")
    else:
        builder.add_text(
            f"INSTRUCTION: RETURN JSON OUTPUT OF {template_type} FROM ACTUAL TEXT USING JSON TEMPLATE FORMAT PROVIDED BELOW.")
    builder.add_separator()
    template_str = generate_template_str(template.example_output_json, indent=indent, json7schema=json7schema)
    builder.add_text(template_str)
    return builder


def get_system_prompt(template_type, detected_language=None):
    builder = OpenAIPromptBuilder()
    builder.add_text(
        f"Extract structured information from text of {template_type} document in the JSON format provided.")
    if detected_language:
        builder.add_text(f"Detected language of the document is: {detected_language}")
    builder.add_line_break()
    builder.add_text(
        f"""Rules:
    - Convert {template_type} text into JSON using the provided template.
    - Ensure the output is valid JSON with no extra text.
    - Exclude non-documented, incomprehensible, or inferred information.
    - Refrain from adding or calculating values not specified in the document.
    """
    )
    return builder


class QueryParams(BaseModel):
    query: list = None
    query_tokens: int = 0
    document_tokens: int = 0
    parser_tokens: int = 0


def general_llm_prompt(
        document_content,
):
    SYSTEM_PROMPT = """
    You are now acting as a versatile assistant on a job search platform, equipped to handle a wide array of requests from users. Your role is multifaceted, encompassing the creation of sample job descriptions, resume advice, interview preparation tips, and any other job search-related inquiries. You must:
    Generate Sample Job Descriptions: Upon request, craft detailed job descriptions for a variety of roles across different industries. These descriptions should include job responsibilities, required qualifications, preferred skills, and any other relevant information.
    Provide Resume Writing Assistance: Offer guidance on how to structure a resume, highlight important elements to include, and tailor a resume to specific job listings.
    Offer Interview Preparation Tips: Share advice on how to prepare for job interviews, including potential questions, how to present oneself, and strategies for answering common interview questions.
    Answer Job Search-Related Queries: Respond to a broad range of questions related to job searching, including how to navigate job search platforms, networking strategies, and how to negotiate job offers.
    Custom Requests: Be prepared to tackle custom requests that may not fit neatly into the above categories but are related to job searching and career advancement.
    Your responses should be informative, supportive, and tailored to the individual needs of the users. Remember to maintain a professional tone and provide actionable advice that users can apply to their job search efforts.
    
    FORMAT YOUR RESPONSE AS MARKDOWN
    """
    prompt = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": document_content
        }
    ]

    query_tokens = len(document_content) + len(SYSTEM_PROMPT)

    document_tokens = num_tokens_from_string(document_content)

    return QueryParams(
        query=prompt,
        query_tokens=query_tokens,
        document_tokens=document_tokens,
    )


def generate_llm_prompt(
        document_content,
        parser,
        add_example=True,
        minimize_tokens=False,
        detected_language=None,
        json7schema=False
):
    """
    Generate the prompt for the LLM
    """
    try:
        # prepare prompts
        user_prompt = get_user_prompt(
            document_content, parser, add_example=add_example,
            minimize_tokens=minimize_tokens, json7schema=json7schema
        )
        system_prompt = get_system_prompt(
            parser.template_type, detected_language=detected_language
        )
        prompt = [
            {"role": "system", "content": system_prompt.get_prompt()},
            {"role": "user", "content": user_prompt.get_prompt()}
        ]

        query_tokens = user_prompt.get_num_tokens() + system_prompt.get_num_tokens()
        indent = None if minimize_tokens else 2

        parser_tokens = num_tokens_from_string(
            generate_synthetic_output_json(parser.example_output_json, indent=indent)
        )
        document_tokens = num_tokens_from_string(document_content)

        return QueryParams(
            query=prompt,
            query_tokens=query_tokens,
            document_tokens=document_tokens,
            parser_tokens=parser_tokens
        )
    except Exception as e:
        log_message(logging.ERROR, "Error generating prompt", exception=e)
        raise CrossJobsError(
            "Error generating prompt",
            "GPT-Parser error"
        )
