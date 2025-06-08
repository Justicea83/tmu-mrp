import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Constants for application segments
APPLICATION_SEGMENT_PARSED_RESULTS = 'parsed_results'


def log_message(level, message=None, extra_info=None, exception=None):
    """
    Enhanced utility function for logging.

    :param level: Logging level (e.g., logging.INFO, logging.ERROR)
    :param message: Message to log
    :param extra_info: Additional information to log (optional, dict format)
    :param exception: Exception instance to log (optional)
    """

    log_context = []

    # Additional context
    if extra_info and isinstance(extra_info, dict):
        extra_context = ", ".join([f"{key}: {value}" for key, value in extra_info.items()])
        log_context.append(extra_context)

    # Combine contexts with the main message
    if log_context:
        message = f"{message} | Context: {' | '.join(log_context)}"

    # Log the message with exception traceback if an exception is provided
    if exception:
        logger.exception(message, exc_info=exception)
    else:
        logger.log(level, message)


def cosine_similarity(vector_a, vector_b):
    if vector_b is None or vector_a is None:
        return 0

    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


def text_to_vector(text: str, model: SentenceTransformer = None):
    """
    Convert text to vector using sentence transformer.
    
    Args:
        text: The text to convert
        model: Optional pre-loaded model
        
    Returns:
        Vector representation of the text
    """
    if model is None:
        # Use default model if none provided
        from core.preprocessors.download_sentence_transformer import APPLICATION_SENTENCE_TRANSFORMER_MODEL
        model = SentenceTransformer(APPLICATION_SENTENCE_TRANSFORMER_MODEL)
    
    return model.encode(text)


def key_or_default(dictionary: dict, key: str, default_value=None):
    """
    Get a value from a dictionary with a default fallback.
    
    Args:
        dictionary: The dictionary to search
        key: The key to look for
        default_value: Default value if key is not found
        
    Returns:
        The value or default_value
    """
    return dictionary.get(key, default_value) if dictionary else default_value
