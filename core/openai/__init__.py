from core.openai.prompt_builder import generate_llm_prompt
from core.openai.wrapper import select_model, call_openai


class QueryEngine:

    @staticmethod
    def build_query(
            prompt,
            parser,
            include_example=False,
            minimize_tokens=True,
            selected_language=None,
            json7schema=True
    ):
        """
        Build natural language query
        :param prompt: string of document content
        :param parser: the parser object
        :param include_example: if true, include example response in the query
        :param minimize_tokens: if true, remove indents, newlines, etc. from the query
        :param selected_language: the language selected by the user
        :param json7schema: if true, the json schema will be in json draft7 format
        :return: a query object
        """
        query_params = generate_llm_prompt(
            prompt,
            parser,
            add_example=include_example,
            minimize_tokens=minimize_tokens,
            detected_language=selected_language,
            json7schema=json7schema
        )
        return query_params

    @staticmethod
    def select_model(query_params):
        """
        Select the model and model params to use for the query
        :param query_params: QueryParams object
        :return:
        """
        model_params = select_model(
            document_tokens=query_params.document_tokens,
            parser_tokens=query_params.parser_tokens,
            query_tokens=query_params.query_tokens,
            set_max_output_tokens=False
        )
        return model_params

    @staticmethod
    def execute_query(query_params, model_params):
        return call_openai(
            query=query_params.query,
            model_params=model_params.dict(exclude_none=True)
        )
