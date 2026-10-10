"""Application-specific exceptions."""


class LLMClientError(RuntimeError):
    """A model request failed or returned an unusable response."""
