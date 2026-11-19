class UnknownModelException(Exception):
    """Exception raised when a requested model cannot be resolved by the ModelCatalog.
 
    Attributes:
        message -- explanation of the error
    """
 
    def __init__(self, message="The requested model is not in the catalog."):
        self.message = message
        super().__init__(self.message)
 
