"""
Define a custom exception for critical system errors.
"""

class CriticalSystemException(Exception):
    """Exception raised for critical system errors.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="A critical system error has occurred."):
        self.message = message
        super().__init__(self.message)
