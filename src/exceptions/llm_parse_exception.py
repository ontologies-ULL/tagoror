"""
Define a custom exception for errors that occur during the parsing of LLM responses.
"""

class LLMParseException(Exception):
    """Exception raised for errors that occur during the parsing of LLM responses.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="An error occurred while parsing the LLM response."):
        self.message = message
        super().__init__(self.message)
