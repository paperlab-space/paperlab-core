"""Exceptions raised by paperlab-space."""


class PaperLabError(Exception):
    """Base class for all paperlab-space errors."""


class DataError(PaperLabError):
    """The input data or column mapping is invalid."""


class ConfigError(PaperLabError):
    """A run configuration is invalid."""


class TemplateError(PaperLabError):
    """A prompt template is invalid or references unknown columns."""
