"""
"""

class TransientNetworkException(Exception):
    """Exception raised for transient network errors during LLM requests.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="An error occurred while communicating with the LLM provider."):
        self.message = message
        super().__init__(self.message)
