"""Tests for the neuroclamp package's public API (__all__)."""

import neuroclamp


def test_all_exports_are_accessible():
    """Every name listed in __all__ must be accessible via getattr(neuroclamp, name)."""
    for name in neuroclamp.__all__:
        assert hasattr(neuroclamp, name), (
            f"{name!r} is in __all__ but not accessible on neuroclamp"
        )
        obj = getattr(neuroclamp, name)
        assert obj is not None, f"{name!r} resolved to None"
