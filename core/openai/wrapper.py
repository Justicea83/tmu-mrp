import logging
import os
import time
from enum import Enum
from timeit import default_timer as timer
from core import openai
from openai import OpenAI
from pydantic import BaseModel, Field

from core.exceptions import CrossJobsError
from core.utils import log_message

OPENAI_SECRET = os.environ["OPENAI_SECRET"]
OPENAI_BASEURL = os.environ["OPENAI_BASE_URL"]


class OpenAIModels(str, Enum):
    GPT_35_TURBO = 'gpt-3.5-turbo'
    GPT_35_TURBO_16k = 'gpt-3.5-turbo-16k'  # 16k context but not limited completion tokens
    GPT_35_TURBO_1106 = 'gpt-3.5-turbo-1106'  # DEFAULT - 16k context but 4096 completion tokens
    GPT_4_1106_PREVIEW = 'gpt-4o-mini-2024-07-18'
    GPT_4_0125_PREVIEW = 'gpt-4o-mini-2024-07-18'


class OpenAIRetryWrapper:
    def __init__(self, max_retries=1, retry_codes=(429, 500), delay=5):
        self.client = OpenAI(api_key=OPENAI_SECRET, base_url=OPENAI_BASEURL)
        self.max_retries = max_retries
        self.retry_codes = retry_codes
        self.delay = delay
        self.response = None
        self.model = None
        self.runtime = 0

    def create_completion(self, messages, **kwargs):
        start = timer()
        self.model = kwargs['model']
        retries = 0
        while retries <= self.max_retries:
            try:
                self.response = self.client.chat.completions.create(
                    messages=messages,
                    temperature=0,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    **kwargs
                )
                self.runtime = int(timer() - start)
                return self.response
            except openai.APIConnectionError as e:
                log_message(logging.ERROR, "OpenAI APIError", exception=e)
                retries += 1
                if retries <= self.max_retries:
                    time.sleep(self.delay)
                else:
                    self.runtime = int(timer() - start)
                    raise CrossJobsError(
                        str(e),
                        'Unable to parse document at this time.'
                    )
            except openai.RateLimitError as e:
                self.runtime = int(timer() - start)
                log_message(logging.ERROR, "OpenAI RateLimitError", exception=e)
                raise CrossJobsError(
                    str(e),
                    'Server is overloaded.'
                )
            except openai.APIStatusError as e:
                log_message(logging.ERROR, "OpenAI Status Error", exception=e)
                if e.status_code in self.retry_codes:
                    retries += 1
                    time.sleep(self.delay)
                else:
                    self.runtime = int(timer() - start)
                    raise CrossJobsError(
                        str(e),
                        'Unable to parse document at this time.'
                    )
            except Exception as e:
                self.runtime = int(timer() - start)
                log_message(logging.ERROR, "OpenAI Exception", exception=e)
                raise CrossJobsError(
                    str(e),
                    'Unable to parse document at this time.'
                )

        self.runtime = int(timer() - start)

    @property
    def first_choice_text(self):
        if self.response is not None:
            return self.response.choices[0].message.content
        return None

    @property
    def first_choice_stop_reason(self):
        if self.response is not None:
            return self.response.choices[0].finish_reason
        return None

    @property
    def total_tokens_used(self):
        if self.response is not None and hasattr(self.response, "usage"):
            return self.response.usage.total_tokens
        return 0

    @property
    def prompt_tokens_used(self):
        if self.response is not None and hasattr(self.response, "usage"):
            return self.response.usage.prompt_tokens
        return 0

    @property
    def completion_tokens_used(self):
        if self.response is not None and hasattr(self.response, "usage"):
            return self.response.usage.completion_tokens
        return 0

    @property
    def total_cost(self) -> str:
        """
        Calculate the total cost and convert to cents
        :return:
        """
        cost = 0
        if self.model == OpenAIModels.GPT_4_0125_PREVIEW:
            cost = ((self.prompt_tokens_used * 0.03) + (self.completion_tokens_used * 0.06)) / 1000
        elif self.model == OpenAIModels.GPT_35_TURBO.value:
            cost = ((self.prompt_tokens_used * 0.001) + (self.completion_tokens_used * 0.002)) / 1000
        elif self.model == OpenAIModels.GPT_35_TURBO_16k.value:
            cost = ((self.prompt_tokens_used * 0.003) + (self.completion_tokens_used * 0.004)) / 1000
        elif self.model == OpenAIModels.GPT_35_TURBO_1106.value:
            cost = ((self.prompt_tokens_used * 0.001) + (self.completion_tokens_used * 0.002)) / 1000
        elif self.model == OpenAIModels.GPT_4_1106_PREVIEW.value:
            cost = ((self.prompt_tokens_used * 0.01) + (self.completion_tokens_used * 0.03)) / 1000
        return str(round(cost, 4))


class ModelParams(BaseModel):
    class Config:
        use_enum_values = True

    model: OpenAIModels = Field(default=OpenAIModels.GPT_4_0125_PREVIEW)
    response_format: dict = Field(default={'type': 'json_object'})
    max_tokens: int = None


def select_model(
        document_tokens,
        parser_tokens,
        query_tokens,
        preconfigured_model=None,
        set_max_output_tokens=False
):
    """
    Select OpenAI Model, Model Parameters and Max Tokens
    :param document_tokens: Number of tokens in the text
    :param parser_tokens: Number of tokens in the template
    :param query_tokens: Number of tokens in the prompt
    :param preconfigured_model: Model selected in parser
    :param set_max_output_tokens: Whether to estimate tokens in parser
    """
    model_params = ModelParams()

    if query_tokens >= (16384 * 0.9) and preconfigured_model != OpenAIModels.GPT_4_1106_PREVIEW:
        raise CrossJobsError(
            "Prompt is too long.",
            "Document is too long."
        )
    if query_tokens >= (128000 * 0.9):
        raise CrossJobsError(
            "Prompt is too long.",
            "Document is too long."
        )

    if preconfigured_model:
        model_params.model = preconfigured_model

    else:
        max_tokens = parser_tokens + document_tokens
        if max_tokens > 4096:
            model_params.model = OpenAIModels.GPT_35_TURBO

        if set_max_output_tokens:
            if query_tokens + max_tokens < 16384:
                model_params.max_tokens = max_tokens

    if model_params.model == OpenAIModels.GPT_35_TURBO:
        model_params.response_format = None

    return model_params


def call_openai(
        query,
        model_params
):
    """
    Call OpenAI and return the response
    :param query: the prompt to send to OpenAI with system and user messages
    :param model_params: the model parameters to use
    :return:
    """
    openai_wrapper = OpenAIRetryWrapper()
    openai_wrapper.create_completion(query, **model_params)
    return openai_wrapper
