import importlib

import pytest


@pytest.mark.parametrize("package", ["pipeline", "eval"])
def test_package_importable(package: str) -> None:
    assert importlib.import_module(package).__doc__
