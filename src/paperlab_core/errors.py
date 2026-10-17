"""Exceptions raised by paperlab-core."""


class PaperLabError(Exception):
    """Base class for all paperlab-core errors."""


class DataError(PaperLabError):
    """The input data or column mapping is invalid."""


class ConfigError(PaperLabError):
    """A run configuration is invalid."""


class TemplateError(PaperLabError):
    """A prompt template is invalid or references unknown columns."""
