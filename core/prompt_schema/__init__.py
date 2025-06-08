from pydantic import BaseModel, Field, field_validator
from pydantic.types import constr
from typing import List, Dict, Any, Optional
import json

FieldType = constr(pattern=r'^(table|text|number|date|array|string)$')
SubFieldType = constr(pattern=r'^(text|number|date|array|string)$')
field_types = {
    'table': 'array',
    'array': 'array',
    'text': 'string',
    'string': 'string',
    'number': 'number',
    'date': 'string'
}

sub_field_types = ['text', 'number', 'date']

simple_field_types = {
    'array': 'table',
    'string': 'text',
    'number': 'number',
    'date': 'date'
}


class BaseSchema(BaseModel):
    example_input_template: Optional[Dict] = Field(default=None)
    example_output_json: Optional[Dict] = Field(default=None)
    template_type: Optional[str] = Field(default=None)
    input_text: Optional[str] = Field(default=None)

    def set_examples_from_config(self, input_config_path: str, output_config_path: str):
        self.set_input(input_config_path)
        self.set_output(output_config_path)

    def set_input(self, input_config_path: str):
        with open(input_config_path, 'r') as f:
            self.example_input_template = PromptSchemaValidator(**json.load(f)).to_draft7_schema()

    def set_output(self, output_config_path: str):
        with open(output_config_path, 'r') as f:
            self.example_output_json = PromptSchemaValidator(**json.load(f)).to_draft7_schema()


class SubField(BaseModel):
    class Config:
        pass

    name: str = Field(..., pattern=r'^[a-z0-9_ ]*$', max_length=30)
    type: SubFieldType
    description: Optional[str] = Field(None, max_length=200)
    required: Optional[bool] = Field(False)

    @field_validator('name')
    def validate_name(cls, value):
        if not value.isalnum() and not value.replace('_', '').replace(' ', '').isalnum():
            raise ValueError('Name must match the pattern: [a-z0-9_ ]')
        return value

    @field_validator('type')
    def validate_type(cls, value):
        if value not in ['text', 'number']:
            raise ValueError('Type must be one of text or number')
        return value

    @field_validator('description')
    def validate_description(cls, value):
        if value is not None and len(value) > 200:
            raise ValueError('Description must not exceed 200 characters')
        return value




class FieldModel(BaseModel):
    class Config:
        pass

    # name: str = Field(..., pattern=r'^[a-z0-9_ ]*$', max_length=30)
    name: str = Field()
    type: FieldType
    description: Optional[str] = Field(None)
    subFields: Optional[List[SubField]] = Field([])
    required: Optional[bool] = Field(False)

    @field_validator('name')
    def validate_name(cls, value):
        if not value.isalnum() and not value.replace('_', '').replace(' ', '').isalnum():
            raise ValueError('Name must match the pattern: [a-z0-9_ ]')
        return value

    @field_validator('description')
    def validate_description(cls, value):
        if value is not None and len(value) > 200:
            raise ValueError('Description must not exceed 200 characters')
        return value


class PromptSchemaValidator(BaseModel):
    fields: List[FieldModel] = Field(..., min_items=1)

    def to_draft7_schema(self) -> Dict[str, Any]:
        properties = {}
        for field in self.fields:
            if field.type == "table" and field.subFields:
                sub_properties = {}
                for sub_field in field.subFields:
                    data = {
                        "type": [field_types[sub_field.type], "null"]
                    }
                    if sub_field.description:
                        data["description"] = sub_field.description
                    sub_properties[sub_field.name] = data
                field_schema = {
                    "type": ["array", "null"],
                    "items": {
                        "type": "object",
                        "properties": sub_properties,
                        "description": field.description,
                        "additionalProperties": False
                    }
                }
            else:
                field_schema = {
                    "type": [field_types[field.type], "null"]
                }
                if field.description:
                    field_schema["description"] = field.description
            properties[field.name] = field_schema

        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": properties,
            "additionalProperties": False
        }
        return schema
