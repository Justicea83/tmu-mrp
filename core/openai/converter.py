import logging
import re
import json
import jsonschema
from typing import Optional
from genson import SchemaBuilder
from copy import deepcopy

from core.exceptions import CrossJobsException
from core.utils import log_message
from core.prompt_schema import BaseSchema


class Converter:
    def __init__(self, response_text: str, template):
        self.response_text: str = response_text
        self.template: BaseSchema = template
        self.extracted_json: dict = None
        self.valid = False
        self.errors = []
        self.new_json_schema = None

    def extract_json(self) -> Optional[dict]:
        # First, try to find the JSON with triple backticks and the `json` word
        json_pattern = re.compile(r'```json\s*({[\s\S]*?})\s*```', re.MULTILINE)
        match = json_pattern.search(self.response_text)

        if not match:
            # If not found, try to find a JSON object without triple backticks and the `json` word
            json_pattern = re.compile(r'({[\s\S]*})', re.MULTILINE)
            match = json_pattern.search(self.response_text)

        if match:
            raw_json = match.group(1)
            try:
                extracted_json = json.loads(raw_json)
                return extracted_json
            except json.JSONDecodeError as e:
                log_message(logging.ERROR, "Error decoding JSO", exception=e)
                raise CrossJobsException(
                    "Error decoding JSON in OpenAI response",
                    "GPT-Parser bad output"
                )
        else:
            raise CrossJobsException(
                "No JSON found in OpenAI response",
                "GPT-Parser empty output"
            )

    def validate_json_against_template(self):
        """Validate a JSON object against a JSON schema"""
        valid, errors = True, []
        try:
            validator = jsonschema.Draft7Validator(self.template.example_output_json)
            validator.validate(self.extracted_json)
            valid, errors = True, []
        except json.JSONDecodeError as e:
            valid, errors = False, [str(e)]
        except jsonschema.ValidationError as e:
            valid, errors = False, [str(e)]
        finally:
            return valid, errors

    def generate_schema_from_json(self) -> dict:
        """Generate a draft-7 schema from a JSON object"""
        builder = SchemaBuilder()
        builder.add_object(self.extracted_json)
        generated_schema = builder.to_schema()

        # Make all fields optional and allow nulls
        for property_name in generated_schema.get('properties', {}):
            generated_schema['properties'][property_name]['type'] \
                = ['null', generated_schema['properties'][property_name]['type']]

        generated_schema.pop('required', None)
        return generated_schema

    def parse_json(self):
        self.extracted_json: dict = self.extract_json()
        try:
            if self.extracted_json:
                self.valid, self.errors = self.validate_json_against_template()
                if not self.valid:
                    self.new_json_schema = self.generate_schema_from_json()
        except Exception as e:
            log_message(logging.ERROR, "Error decoding JSO", exception=e)
            raise CrossJobsException(
                "Error validating JSON with template",
                "GPT-Parser invalid output"
            )

    def process(self):
        self.parse_json()
        extracted_json = deepcopy(self.extracted_json)

        self.extracted_json = prune_json(extracted_json, self.template.example_output_json)

        if check_empty(self.extracted_json):
            raise CrossJobsException(
                "All empty strings in JSON",
                "GPT-Parser empty output"
            )
        """
        TODO: Note that disabling the dummy check does two things:
        1. If a user actually uploads a dummy invoice, it won't throw an error (which is what we want)
        2. If the GPT-Parser output is a dummy invoice, it won't throw an error (which is what we don't want, but it's a tradeoff)
        """
        # if check_invoice_json_not_dummy(self.extracted_json):
        #     raise CrossJobsException(
        #         "Dummy response detected",
        #         "GPT-Parser inaccurate output"
        #     )


def check_invoice_json_not_dummy(obj, dummy_strings=None):
    """Check if the JSON is an OpenAI dummy"""
    if dummy_strings is None:
        dummy_strings = ["ABC Company", "123 Main Street"]

    if isinstance(obj, str):
        return any(dummy_string in obj for dummy_string in dummy_strings)
    elif isinstance(obj, dict):
        return any(check_invoice_json_not_dummy(v, dummy_strings) for v in obj.values())
    elif isinstance(obj, list):
        return any(check_invoice_json_not_dummy(item, dummy_strings) for item in obj)
    else:
        return False


def check_arrays_not_empty(item, array_fields: list = []):
    """check if array fields are empty"""
    if array_fields:
        return all([check_empty(item.get(field)) for field in array_fields])
    return False


def check_empty(item):
    """Check if an item is empty"""
    if isinstance(item, str):
        return item == ''
    elif isinstance(item, list):
        return all(check_empty(subitem) for subitem in item)
    elif isinstance(item, dict):
        return all(check_empty(subitem) for subitem in item.values())
    else:
        return False


def prune_json(json_obj, schema):
    original_obj = json_obj

    if isinstance(json_obj, dict):
        json_obj = {
            k: prune_json(v, schema.get('properties', {}).get(k, {}))
            for k, v in json_obj.items()
            if k in schema.get('properties', {})
        }

        # If we don't find anything
        if not bool(json_obj):
            json_obj = {
                k: v
                for k, v in original_obj.get('properties', {}).items()
                if k in schema.get('properties', {})
            }
    elif isinstance(json_obj, list) and 'items' in schema:
        json_obj = [prune_json(v, schema['items']) for v in json_obj]
    return json_obj
