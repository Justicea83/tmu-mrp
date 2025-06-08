class CrossJobsError(Exception):
    """
    Base class for all custom errors
    """

    def __init__(self, message, usr_msg=None):
        self.message = message
        self.usr_message = usr_msg

    def __str__(self):
        return self.message

    def get_user_msg(self):
        return self.usr_message


class CrossJobsException(CrossJobsError):
    """
    Exceptions occur due to textract or gpt failures that can be retried
    """

    def __init__(self, message, usr_msg=None):
        super().__init__(message, usr_msg)


class UnauthorizedError(Exception):
    def __init__(self, message="Forbidden"):
        super().__init__(message)


class ItemNotFoundError(Exception):
    def __init__(self, message="The requested item was not found"):
        super().__init__(message)