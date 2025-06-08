from core.openai.converter import Converter


class PostProcessor:

    @staticmethod
    def convert_to_dict(output: str, parser):
        _converter = Converter(output, parser)
        _converter.process()
        return _converter.extracted_json
