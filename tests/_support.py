"""Minimal assertion helpers, so the test suite needs nothing but numpy.

The tests are written as plain ``test_*`` functions with bare ``assert``\\ s, which means
``pytest`` collects and runs them unchanged if you have it. These two helpers exist only
so that ``python tests/run_tests.py`` also works on a bare interpreter: a verification
suite that requires installing a test framework is a suite that does not get run.
"""

from __future__ import annotations

import re


class raises:
    """Context manager asserting that a block raises a given exception.

    ``match`` is treated as a regular expression searched in the message, which is what
    ``pytest.raises(match=...)`` does.
    """

    def __init__(self, expected: type[BaseException], match: str | None = None) -> None:
        self.expected = expected
        self.match = match
        self.value: BaseException | None = None

    def __enter__(self) -> "raises":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            raise AssertionError(f"expected {self.expected.__name__} to be raised, nothing was")
        if not issubclass(exc_type, self.expected):
            return False
        if self.match is not None and not re.search(self.match, str(exc)):
            raise AssertionError(
                f"{exc_type.__name__} message {str(exc)!r} does not match {self.match!r}"
            )
        self.value = exc
        return True


class approx:
    """Float comparison with a tolerance, mirroring ``pytest.approx``."""

    def __init__(self, expected: float, abs_tol: float = 1e-9) -> None:
        self.expected = float(expected)
        self.abs_tol = abs_tol

    def __eq__(self, other) -> bool:
        try:
            return abs(float(other) - self.expected) <= self.abs_tol
        except (TypeError, ValueError):
            return NotImplemented

    def __repr__(self) -> str:
        return f"approx({self.expected!r}, abs_tol={self.abs_tol!r})"
