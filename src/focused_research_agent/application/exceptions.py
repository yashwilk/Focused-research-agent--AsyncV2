
class ApplicationError(Exception):
    """
    Represent an expected application/use-case failure.

    This exception is raised by the application layer when the research use
    case cannot proceed because of an expected input or business-level
    problem.

    Args:
        message: Human-readable description of the application error.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
