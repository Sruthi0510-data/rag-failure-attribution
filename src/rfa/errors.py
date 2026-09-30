"""Failure modes that need distinct exit codes, one exception type each."""


class RfaError(Exception):
    """Base class. Anything raised deliberately by this package is a subclass."""


class ConfigError(RfaError):
    """The evaluation was not runnable as specified: bad flags, unknown chunker, bad k."""


class CorpusError(RfaError):
    """The corpus or question set could not be read, or an annotation does not fit its document."""


class UnderpoweredError(RfaError):
    """The evaluation set is too small to support the rate the caller asked to be guaranteed."""
